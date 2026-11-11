# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speech-to-text adapters for Live Voice.

The production adapter reuses Studio's OpenAI-compatible transcription route over
an in-process ASGI transport.  That keeps model discovery, model loading,
account checks and the existing sidecar lifecycle in one place; the live engine
does not duplicate any of it.  Tests inject `FakeStt` instead.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol

import httpx

from loggers import get_logger

logger = get_logger(__name__)


class SttError(RuntimeError):
    """A transcription failed in a way the live UI should surface."""


class SttEngine(Protocol):
    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str: ...

    async def close(self) -> None: ...


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON upstream error is still an error
        text = (response.text or "").strip()
        return text[:300] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()[:300]
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()[:300]
        if isinstance(error, str) and error.strip():
            return error.strip()[:300]
    return f"HTTP {response.status_code}"


class FakeStt:
    """Deterministic test/development adapter."""

    def __init__(self, text: str = "hello from the live test adapter") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        self.calls.append(
            {"bytes": len(wav_bytes), "model": model, "language": language}
        )
        return self.text

    async def close(self) -> None:
        return None


class AsgiStt:
    """Calls Studio's own /v1/audio/transcriptions route without a socket."""

    def __init__(
        self,
        app: Any,
        token: str,
        *,
        model: Optional[str] = None,
        timeout_s: float = 180.0,
    ) -> None:
        self._app = app
        self._token = token
        self._model = (model or "").strip() or None
        self._client = httpx.AsyncClient(
            transport = httpx.ASGITransport(app = app),
            base_url = "http://studio.live",
            timeout = httpx.Timeout(timeout_s, connect = 5.0),
        )

    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        if not wav_bytes:
            raise SttError("The recorded utterance was empty.")
        selected = (model or self._model or "").strip()
        data: dict[str, str] = {"response_format": "json"}
        if selected and selected != "whisper-1":
            data["model"] = selected
        if language and language.strip():
            data["language"] = language.strip()
        files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
        headers = {
            "Authorization": f"Bearer {self._token}",
            # Live utterances are internal; keep them out of the API activity list.
            "X-Unsloth-Live-Voice": "1",
        }
        try:
            response = await self._client.post(
                "/v1/audio/transcriptions",
                data = data,
                files = files,
                headers = headers,
            )
        except httpx.HTTPError as exc:
            raise SttError(f"Could not reach the local STT engine: {exc}") from exc
        if response.status_code >= 400:
            raise SttError(_error_detail(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise SttError("The local STT engine returned a non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise SttError("The local STT engine returned an unexpected response.")
        text = payload.get("text")
        return str(text).strip() if isinstance(text, str) else ""

    async def close(self) -> None:
        await self._client.aclose()


def build_stt(
    app: Any,
    token: str,
    *,
    model: Optional[str] = None,
) -> SttEngine:
    """Select the adapter from the environment.

    `UNSLOTH_LIVE_STT=fake` is for protocol/UI development and tests.  The default
    uses Studio's real transcription engine.
    """
    if os.environ.get("UNSLOTH_LIVE_STT", "").strip().lower() == "fake":
        return FakeStt()
    return AsgiStt(app, token, model = model)
