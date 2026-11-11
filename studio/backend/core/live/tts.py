# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-to-speech adapters for Live Voice.

`AsgiTts` reuses Studio's OpenAI-compatible /v1/audio/speech route in-process, so
the live engine inherits the existing model manager, voice handling and license
gates.  `ToneTts` is the dependency-free development/test adapter; the UI labels
it as a test tone rather than pretending it is a voice.
"""

from __future__ import annotations

import os
import wave
from typing import Any, Optional, Protocol

import httpx

from core.live import audio
from loggers import get_logger

logger = get_logger(__name__)


class TtsError(RuntimeError):
    """A synthesis failed in a way the live UI should surface."""


class TtsEngine(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[bytes, int]: ...

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


class ToneTts:
    """Development adapter: a short tone proportional to the sentence length."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[bytes, int]:
        self.calls.append(text)
        duration = 180 + min(1200, max(0, len(text)) * 12)
        return audio.tone_wav(duration_ms = duration), audio.SAMPLE_RATE

    async def close(self) -> None:
        return None


class AsgiTts:
    """Calls Studio's own /v1/audio/speech route without a socket."""

    def __init__(
        self,
        app: Any,
        token: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        timeout_s: float = 180.0,
    ) -> None:
        self._app = app
        self._token = token
        self._model = (model or "").strip() or None
        self._voice = (voice or "").strip() or None
        self._language = (language or "").strip() or None
        self._client = httpx.AsyncClient(
            transport = httpx.ASGITransport(app = app),
            base_url = "http://studio.live",
            timeout = httpx.Timeout(timeout_s, connect = 5.0),
        )

    async def synthesize(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[bytes, int]:
        if not text.strip():
            raise TtsError("There is nothing to speak.")
        selected_model = (model or self._model or "").strip()
        payload: dict[str, Any] = {
            "input": text,
            "response_format": "wav",
        }
        if selected_model:
            payload["model"] = selected_model
        selected_voice = (voice or self._voice or "").strip()
        if selected_voice:
            payload["voice"] = selected_voice
        selected_language = (language or self._language or "").strip()
        if selected_language:
            payload["language"] = selected_language
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            response = await self._client.post(
                "/v1/audio/speech",
                json = payload,
                headers = headers,
            )
        except httpx.HTTPError as exc:
            raise TtsError(f"Could not reach the local TTS engine: {exc}") from exc
        if response.status_code >= 400:
            detail = _error_detail(response)
            if response.status_code == 400 and "no model loaded" in detail.lower():
                detail = (
                    "No TTS model is loaded. Install and select a voice model "
                    "in Settings → Voice."
                )
            raise TtsError(detail)
        wav_bytes = response.content
        if not wav_bytes:
            raise TtsError("The local TTS engine returned empty audio.")
        try:
            sample_rate = audio.wav_sample_rate(wav_bytes)
        except (wave.Error, EOFError) as exc:
            raise TtsError("The local TTS engine returned a non-WAV response.") from exc
        return wav_bytes, sample_rate

    async def close(self) -> None:
        await self._client.aclose()


def build_tts(
    app: Any,
    token: str,
    *,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    language: Optional[str] = None,
) -> TtsEngine:
    """Select the adapter from the environment.

    `UNSLOTH_LIVE_TTS=fake` returns a test tone; the default uses Studio's real
    speech engine.
    """
    if os.environ.get("UNSLOTH_LIVE_TTS", "").strip().lower() == "fake":
        return ToneTts()
    return AsgiTts(
        app,
        token,
        model = model,
        voice = voice,
        language = language,
    )
