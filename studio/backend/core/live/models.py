# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Feature flag, platform capability and the live model catalog surface.

The live engine deliberately reuses Studio's model manager: STT and TTS ids are
the ids the Voice settings and model hub already understand.  This module only
reports readiness/capabilities and keeps the pinned VAD/endpoint asset metadata
in one place.  Download actions reuse the existing Hugging Face pipeline.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from core.live.protocol import PROTOCOL_VERSION, SAMPLE_RATE


def live_enabled() -> bool:
    """Live Voice is on unless an operator explicitly disables it."""
    value = os.environ.get("UNSLOTH_STUDIO_LIVE_VOICE", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def screen_capture_supported() -> bool:
    """Tauri's portal capture is Linux-first; other platforms hide the toggle."""
    return sys.platform.startswith("linux")


def _engine_kind(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip().lower()
    return value or "studio"


@dataclass(frozen = True)
class LiveAsset:
    """A small pinned asset (VAD/endpoint) the live engine needs."""

    role: str
    repo: str
    revision: str
    filename: str
    sha256: str = ""
    size_bytes: Optional[int] = None
    installed: bool = False
    env_override: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "installed": self.installed,
            "envOverride": self.env_override,
        }


# These are the same pinned families the Screen Companion reference used.  They
# are intentionally declared here (not downloaded at import) so the catalog can
# grow without changing the socket protocol.
LIVE_ASSETS: tuple[LiveAsset, ...] = (
    LiveAsset(
        role = "vad",
        repo = "snakers4/silero-vad",
        revision = "v6.2.3",
        filename = "silero_vad.onnx",
        env_override = "UNSLOTH_LIVE_VAD_MODEL",
    ),
    LiveAsset(
        role = "endpoint",
        repo = "pipecat-ai/smart-turn",
        revision = "v3.2",
        filename = "smart_turn.onnx",
        env_override = "UNSLOTH_LIVE_SMART_TURN_MODEL",
    ),
)


def _asset_installed(asset: LiveAsset) -> bool:
    value = os.environ.get(asset.env_override, "").strip()
    return bool(value) and Path(value).expanduser().is_file()


def _assets() -> tuple[LiveAsset, ...]:
    return tuple(
        replace(asset, installed = _asset_installed(asset)) for asset in LIVE_ASSETS
    )


@dataclass(frozen = True)
class LiveStatus:
    enabled: bool
    protocol: int
    sample_rate: int
    platform: str
    screen_capture: str
    stt_engine: str
    tts_engine: str
    assets: tuple[LiveAsset, ...] = field(default_factory = tuple)

    def to_dict(self) -> dict[str, Any]:
        by_role = {asset.role: asset for asset in self.assets}
        return {
            "enabled": self.enabled,
            "protocol": self.protocol,
            "sampleRate": self.sample_rate,
            "platform": self.platform,
            "capabilities": {
                "audioCapture": "frontend",
                "screenCapture": self.screen_capture,
                "stt": self.stt_engine,
                "tts": self.tts_engine,
                "vad": (
                    "silero"
                    if by_role.get("vad") is not None and by_role["vad"].installed
                    else "energy"
                ),
                "endpoint": (
                    "smart_turn"
                    if by_role.get("endpoint") is not None
                    and by_role["endpoint"].installed
                    else "silence"
                ),
            },
            "assets": [asset.to_dict() for asset in self.assets],
        }


def status() -> LiveStatus:
    return LiveStatus(
        enabled = live_enabled(),
        protocol = PROTOCOL_VERSION,
        sample_rate = SAMPLE_RATE,
        platform = platform.system().lower(),
        screen_capture = "tauri" if screen_capture_supported() else "unsupported",
        stt_engine = _engine_kind("UNSLOTH_LIVE_STT"),
        tts_engine = _engine_kind("UNSLOTH_LIVE_TTS"),
        assets = _assets(),
    )
