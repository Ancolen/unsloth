# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pinned ONNX assets for Live Voice.

The VAD and turn-endpoint models are small files with an explicit source,
revision and SHA-256.  They are downloaded on demand from the Settings UI, not
at import time: an offline install keeps the energy VAD and the extended silence
window, and the status endpoint explains the fallback.  Downloads are atomic and
hash-checked before the file becomes visible under its final name.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from loggers import get_logger

logger = get_logger(__name__)

SILERO_COMMIT = "5cd7945676eb32225748052e2e6a0580e4686a08"
SMART_TURN_COMMIT = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"


class AssetError(RuntimeError):
    """A live asset could not be downloaded or verified."""


@dataclass(frozen = True)
class LiveAsset:
    """A small pinned asset (VAD/endpoint) the live engine can use."""

    role: str
    name: str
    source: str
    repo: str
    revision: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    env_override: str

    def to_dict(self, *, installed: bool, active: bool) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "source": self.source,
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "license": self.license,
            "installed": installed,
            "active": active,
            "envOverride": self.env_override,
        }


LIVE_ASSETS: tuple[LiveAsset, ...] = (
    LiveAsset(
        role = "vad",
        name = "Silero VAD",
        source = "github",
        repo = "snakers4/silero-vad",
        revision = SILERO_COMMIT,
        filename = "silero_vad.onnx",
        url = (
            "https://raw.githubusercontent.com/snakers4/silero-vad/"
            f"{SILERO_COMMIT}/src/silero_vad/data/silero_vad.onnx"
        ),
        sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        size_bytes = 2_327_524,
        license = "MIT",
        env_override = "UNSLOTH_LIVE_VAD_MODEL",
    ),
    LiveAsset(
        role = "endpoint",
        name = "Smart Turn v3.2",
        source = "huggingface",
        repo = "pipecat-ai/smart-turn-v3",
        revision = SMART_TURN_COMMIT,
        filename = "smart-turn-v3.2-cpu.onnx",
        url = (
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/"
            f"{SMART_TURN_COMMIT}/smart-turn-v3.2-cpu.onnx"
        ),
        sha256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
        size_bytes = 8_679_182,
        license = "BSD-2-Clause",
        env_override = "UNSLOTH_LIVE_SMART_TURN_MODEL",
    ),
)


def asset_by_role(role: str) -> Optional[LiveAsset]:
    wanted = (role or "").strip().lower()
    for asset in LIVE_ASSETS:
        if asset.role == wanted:
            return asset
    return None


def _live_root() -> Path:
    override = (os.environ.get("UNSLOTH_LIVE_ASSET_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from utils.paths.storage_roots import cache_root

        return cache_root() / "live"
    except Exception:  # noqa: BLE001 - a minimal test environment may lack utils
        return Path.home() / ".unsloth" / "studio" / "cache" / "live"


def asset_path(asset: LiveAsset) -> Path:
    """The env override wins; otherwise the managed download location."""
    override = (os.environ.get(asset.env_override) or "").strip()
    if override:
        return Path(override).expanduser()
    return _live_root() / asset.filename


def asset_installed(asset: LiveAsset) -> bool:
    override = (os.environ.get(asset.env_override) or "").strip()
    path = asset_path(asset)
    try:
        if override:
            # A user-selected voice is authoritative; its size need not match the
            # catalog's pinned default.
            return path.is_file()
        return path.is_file() and path.stat().st_size == asset.size_bytes
    except OSError:
        return False


def available_asset_path(asset: LiveAsset) -> Optional[Path]:
    """The path a detector should load, when it exists; ``None`` keeps the fallback."""
    return asset_path(asset) if asset_installed(asset) else None


_onnxruntime: Optional[bool] = None


def onnxruntime_available() -> bool:
    """An installed ONNX asset is inert without the runtime; report it honestly."""
    global _onnxruntime
    if _onnxruntime is None:
        try:
            _onnxruntime = importlib.util.find_spec("onnxruntime") is not None
        except (ImportError, ValueError):
            _onnxruntime = False
    return _onnxruntime


def asset_active(asset: LiveAsset) -> bool:
    return asset_installed(asset) and onnxruntime_available()


def asset_states() -> tuple[dict[str, Any], ...]:
    return tuple(
        asset.to_dict(
            installed = asset_installed(asset),
            active = asset_active(asset),
        )
        for asset in LIVE_ASSETS
    )


_locks: dict[str, asyncio.Lock] = {}


def _lock_for(role: str) -> asyncio.Lock:
    lock = _locks.get(role)
    if lock is None:
        lock = asyncio.Lock()
        _locks[role] = lock
    return lock


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def download_asset(
    asset: LiveAsset,
    *,
    client: Optional[httpx.AsyncClient] = None,
    force: bool = False,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Download one pinned asset atomically and verify its SHA-256.

    Returns the same shape as `/api/live/status` asset entries.  A failed
    download leaves no partial file at the final path.
    """
    if asset_installed(asset) and not force:
        return asset.to_dict(installed = True, active = asset_active(asset))

    async with _lock_for(asset.role):
        if asset_installed(asset) and not force:
            return asset.to_dict(installed = True, active = asset_active(asset))

        path = asset_path(asset)
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            if path.parent == _live_root():
                path.parent.chmod(0o700)
        except OSError as exc:
            raise AssetError(f"Could not create the live asset directory: {exc}") from exc

        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.part"
        )
        own_client = client is None
        http = client or httpx.AsyncClient(
            follow_redirects = True,
            timeout = httpx.Timeout(timeout_s, connect = 10.0),
        )
        digest = hashlib.sha256()
        total = 0
        try:
            async with http.stream(
                "GET",
                asset.url,
                follow_redirects = True,
                headers = {"User-Agent": "unsloth-studio-live/1"},
            ) as response:
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes(64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > asset.size_bytes:
                            raise AssetError(
                                f"{asset.name} is larger than its pinned size "
                                f"({asset.size_bytes} bytes)."
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if total != asset.size_bytes:
                raise AssetError(
                    f"{asset.name} was {total} bytes; expected {asset.size_bytes}."
                )
            actual = digest.hexdigest()
            if actual.lower() != asset.sha256.lower():
                raise AssetError(
                    f"{asset.name} failed its SHA-256 check; expected "
                    f"{asset.sha256}, got {actual}."
                )
            os.replace(temp_path, path)
            try:
                path.chmod(0o644)
            except OSError:
                pass
            logger.info(
                "live.asset_installed",
                role = asset.role,
                bytes = total,
                path = str(path),
            )
            return asset.to_dict(installed = True, active = asset_active(asset))
        except AssetError:
            raise
        except httpx.HTTPError as exc:
            raise AssetError(f"Could not download {asset.name}: {exc}") from exc
        except OSError as exc:
            raise AssetError(f"Could not write {asset.name}: {exc}") from exc
        finally:
            try:
                temp_path.unlink(missing_ok = True)
            except OSError:
                pass
            if own_client:
                await http.aclose()


def verify_installed_asset(asset: LiveAsset) -> bool:
    """Re-hash an installed asset; used by tests and the install path."""
    path = asset_path(asset)
    if not path.is_file():
        return False
    return _sha256_file(path).lower() == asset.sha256.lower()
