# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-to-speech adapters for Live Voice.

`AsgiTts` reuses Studio's OpenAI-compatible /v1/audio/speech route in-process, so
the live engine inherits the existing model manager, voice handling and license
gates.  `PiperTts` is an optional local CPU path for a pinned Piper voice (the
Turkish `tr_TR-fettah-medium` voice is the reference).  `ToneTts` is the
dependency-free development/test adapter; the UI labels it as a test tone rather
than pretending it is a voice.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import wave
from pathlib import Path
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


class PiperTts:
    """Local Piper ONNX voice, CPU-first and lazily loaded.

    Piper is optional: the adapter reports an actionable install/configuration
    error instead of silently changing engines.  The live session's TTS worker is
    sequential, but a lock keeps a shared instance safe if it is ever reused.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        *,
        speaker: Optional[str] = None,
        use_cuda: bool = False,
        deterministic: bool = False,
        voice_instance: Any = None,
    ) -> None:
        self.model_path = (model_path or "").strip()
        self.config_path = (config_path or "").strip() or None
        self.speaker = (speaker or "").strip() or None
        self.use_cuda = bool(use_cuda)
        self.deterministic = bool(deterministic)
        self._voice = voice_instance
        self._load_lock = threading.Lock()
        self._synth_lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        if self._voice is not None:
            return self._voice
        with self._load_lock:
            if self._voice is not None:
                return self._voice
            if not self.model_path:
                raise TtsError(
                    "Piper is selected but no voice is configured. Set "
                    "UNSLOTH_LIVE_PIPER_MODEL to a Piper .onnx voice."
                )
            path = Path(self.model_path)
            if not path.is_file():
                raise TtsError(
                    f"Piper voice not found at {path}. Set UNSLOTH_LIVE_PIPER_MODEL "
                    "to the downloaded .onnx file."
                )
            config_path = (
                Path(self.config_path)
                if self.config_path
                else path.with_name(f"{path.name}.json")
            )
            try:
                from piper.voice import PiperVoice  # type: ignore
            except Exception as exc:  # noqa: BLE001 - optional dependency
                raise TtsError(
                    "Piper TTS is not installed in the Studio Python environment. "
                    "Install 'piper-tts' and retry."
                ) from exc
            try:
                self._voice = PiperVoice.load(
                    str(path),
                    config_path=str(config_path) if config_path.is_file() else None,
                    use_cuda = self.use_cuda,
                )
            except Exception as exc:  # noqa: BLE001 - a bad voice must be actionable
                raise TtsError(f"Piper voice could not load: {exc}") from exc
            return self._voice

    def _synthesis_config(self) -> Any:
        if not (self.speaker or self.deterministic):
            return None
        try:
            from piper.config import SynthesisConfig  # type: ignore
        except Exception:  # noqa: BLE001 - older piper has no config class
            return None
        values: dict[str, Any] = {}
        if self.deterministic:
            values["noise_scale"] = 0.0
            values["noise_w_scale"] = 0.0
        if self.speaker:
            speaker_map = (
                getattr(getattr(self._voice, "config", None), "speaker_id_map", None)
                or {}
            )
            speaker_id = (
                speaker_map.get(self.speaker)
                if isinstance(speaker_map, dict)
                else None
            )
            if speaker_id is None:
                try:
                    speaker_id = int(self.speaker)
                except ValueError:
                    speaker_id = None
            if speaker_id is not None:
                values["speaker_id"] = speaker_id
        try:
            return SynthesisConfig(**values)
        except TypeError:
            return None

    def _synthesize_blocking(self, text: str) -> tuple[bytes, int]:
        voice = self._ensure_loaded()
        with self._synth_lock:
            config = self._synthesis_config()
            chunks: list[bytes] = []
            try:
                if hasattr(voice, "synthesize_stream_raw"):
                    iterator = (
                        voice.synthesize_stream_raw(text, syn_config = config)
                        if config is not None
                        else voice.synthesize_stream_raw(text)
                    )
                    for raw in iterator:
                        if raw:
                            chunks.append(bytes(raw))
                elif hasattr(voice, "synthesize"):
                    iterator = (
                        voice.synthesize(text, syn_config = config)
                        if config is not None
                        else voice.synthesize(text)
                    )
                    for chunk in iterator:
                        raw = getattr(chunk, "audio_int16_bytes", None)
                        if raw:
                            chunks.append(bytes(raw))
                        elif isinstance(chunk, (bytes, bytearray)) and chunk:
                            chunks.append(bytes(chunk))
                elif hasattr(voice, "synthesize_wav"):
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as wav:
                        voice.synthesize_wav(text, wav)
                    buffer.seek(0)
                    with wave.open(buffer, "rb") as wav:
                        return wav.readframes(wav.getnframes()), int(
                            wav.getframerate()
                        )
                else:
                    raise TtsError("The loaded Piper voice cannot synthesize speech.")
            except TtsError:
                raise
            except Exception as exc:  # noqa: BLE001 - synthesis must be actionable
                raise TtsError(f"Piper synthesis failed: {exc}") from exc
            sample_rate = int(
                getattr(getattr(voice, "config", None), "sample_rate", 22050)
            )
            return b"".join(chunks), sample_rate

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
        pcm, sample_rate = await asyncio.to_thread(self._synthesize_blocking, text)
        if not pcm:
            raise TtsError("Piper returned no audio.")
        return audio.pcm16_wav(pcm, sample_rate), sample_rate

    async def close(self) -> None:
        with self._load_lock:
            self._voice = None


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

    `UNSLOTH_LIVE_TTS=fake` returns a test tone, `piper` uses the local Piper
    voice from `UNSLOTH_LIVE_PIPER_MODEL`, `xtts` uses the opt-in XTTS-v2
    adapter, and the default uses Studio's real speech engine.
    """
    engine = os.environ.get("UNSLOTH_LIVE_TTS", "").strip().lower()
    if engine == "fake":
        return ToneTts()
    if engine == "piper":
        return PiperTts(
            os.environ.get("UNSLOTH_LIVE_PIPER_MODEL", "").strip(),
            os.environ.get("UNSLOTH_LIVE_PIPER_CONFIG", "").strip() or None,
            speaker = os.environ.get("UNSLOTH_LIVE_PIPER_SPEAKER", "").strip()
            or None,
            use_cuda = os.environ.get("UNSLOTH_LIVE_PIPER_CUDA", "").strip().lower()
            in ("1", "true", "yes", "on"),
        )
    if engine == "xtts":
        from core.live.xtts import build_xtts_tts

        return build_xtts_tts()
    return AsgiTts(
        app,
        token,
        model = model,
        voice = voice,
        language = language,
    )
