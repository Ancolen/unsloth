# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One Live Voice WebSocket session.

The session is audio-only on purpose: it owns the microphone frame pipeline, VAD,
endpointing, STT and the TTS sentence queue.  Chat remains in the frontend's
assistant-ui runtime, so every existing Studio tool, model and history feature
keeps working.  The session emits typed events; the route only moves bytes.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Optional

from core.live import audio
from core.live.endpoint import EndpointConfig, EndpointDetector, SmartTurnScorer
from core.live.protocol import (
    FRAME_MS,
    MODE_BARGE_IN,
    MODE_TURN,
    PROTOCOL_VERSION,
    VALID_MODES,
    error_event,
    event,
    parse_client_message,
    validate_hello,
)
from core.live.stt import SttEngine
from core.live.tts import TtsEngine
from core.live.vad import EnergyVad, build_vad
from loggers import get_logger

logger = get_logger(__name__)

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


@dataclass
class LiveConfig:
    language: str = "tr"
    mode: str = MODE_TURN
    stt_model: Optional[str] = None
    tts_model: Optional[str] = None
    voice: Optional[str] = None
    echo_cancellation: bool = False
    sample_rate: int = audio.SAMPLE_RATE
    vad_threshold: float = 0.015
    endpoint: EndpointConfig = field(default_factory = EndpointConfig)

    def with_hello(self, message: Any) -> "LiveConfig":
        language = (message.language or self.language or "tr").strip() or "tr"
        mode = message.mode if message.mode in VALID_MODES else self.mode
        endpoint = self.endpoint
        if message.endpointFastMs is not None or message.endpointExtendedMs is not None:
            endpoint = replace(
                endpoint,
                fast_silence_ms = (
                    float(message.endpointFastMs)
                    if message.endpointFastMs is not None
                    else endpoint.fast_silence_ms
                ),
                extended_silence_ms = (
                    float(message.endpointExtendedMs)
                    if message.endpointExtendedMs is not None
                    else endpoint.extended_silence_ms
                ),
                max_utterance_ms = (
                    float(message.maxUtteranceMs)
                    if message.maxUtteranceMs is not None
                    else endpoint.max_utterance_ms
                ),
            )
        return replace(
            self,
            language = language,
            mode = mode,
            stt_model = (message.sttModel or "").strip() or self.stt_model,
            tts_model = (message.ttsModel or "").strip() or self.tts_model,
            voice = (message.voice or "").strip() or self.voice,
            echo_cancellation = (
                bool(message.echoCancellation)
                if message.echoCancellation is not None
                else self.echo_cancellation
            ),
            endpoint = endpoint,
        )


@dataclass(frozen = True)
class SpeakItem:
    id: str
    seq: int
    text: str
    generation: int


class LiveSession:
    """Audio/VAD/STT/TTS session bound to one WebSocket."""

    def __init__(
        self,
        *,
        send_json: SendJson,
        send_bytes: SendBytes,
        stt: SttEngine,
        tts: TtsEngine,
        config: Optional[LiveConfig] = None,
        vad: Optional[Any] = None,
        endpoint: Optional[EndpointDetector] = None,
        smart_turn: Optional[SmartTurnScorer] = None,
    ) -> None:
        self._send_json_raw = send_json
        self._send_bytes_raw = send_bytes
        self.stt = stt
        self.tts = tts
        self.config = config or LiveConfig()
        self._vad = vad or build_vad(threshold = self.config.vad_threshold)
        self._endpoint = endpoint or EndpointDetector(self.config.endpoint)
        smart_turn_env = (os.environ.get("UNSLOTH_LIVE_SMART_TURN_MODEL") or "").strip()
        self._smart_turn = smart_turn or SmartTurnScorer(
            model_path = Path(smart_turn_env) if smart_turn_env else None
        )
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize = 96)
        self._stt_queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue(maxsize = 8)
        self._tts_queue: asyncio.Queue[SpeakItem] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._closed = False
        self._started = False
        self._muted = False
        self._state = "idle"
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance = bytearray()
        self._pre_roll: Deque[bytes] = deque()
        self._frame_ms = float(FRAME_MS)
        self._last_level_at = 0.0
        self._last_speech_start_ms = 0
        self._utterance_counter = 0
        self._playback_active = False
        self._barge_in_sent = False
        self._tts_generation = 0
        self._ready_sent = False
        self._last_smart_turn_silence_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._tasks or self._closed:
            return
        self._tasks = [
            asyncio.create_task(self._audio_worker(), name = "live-audio"),
            asyncio.create_task(self._stt_worker(), name = "live-stt"),
            asyncio.create_task(self._tts_worker(), name = "live-tts"),
        ]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown is best effort
                pass
        self._tasks.clear()
        try:
            await self.stt.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask the session end
            logger.debug("live.stt_close_failed", error = str(exc))
        try:
            await self.tts.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("live.tts_close_failed", error = str(exc))

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    def push_audio(self, pcm: bytes) -> None:
        if self._closed or not self._started or self._muted:
            return
        normalized = audio.normalize_pcm(pcm)
        if normalized is None:
            return
        try:
            self._audio_queue.put_nowait(normalized)
        except asyncio.QueueFull:
            # Keep the newest audio: a slow VAD must not make the user's next word
            # disappear behind seconds of stale frames.
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._audio_queue.put_nowait(normalized)
            except asyncio.QueueFull:
                pass

    async def handle_text(self, raw: str) -> None:
        message = parse_client_message(raw)
        if message is None:
            await self._send(error_event("bad_message", "The live message was not valid JSON."))
            return
        kind = message.type.strip()
        if kind == "hello":
            await self._handle_hello(message)
        elif kind == "start":
            self._started = True
            self._reset_audio_state()
            await self._set_state("listening")
        elif kind == "stop":
            self._started = False
            self._reset_audio_state()
            self._cancel_tts()
            await self._set_state("idle")
        elif kind == "mute":
            self._muted = bool(message.muted)
            self._reset_audio_state()
            if not self._muted and self._started:
                await self._set_state("listening")
        elif kind == "mode":
            mode = (message.mode or "").strip()
            if mode in VALID_MODES:
                self.config = replace(self.config, mode = mode)
                if mode == MODE_TURN:
                    self._barge_in_sent = False
            else:
                await self._send(error_event("bad_mode", f"Unknown live mode {mode!r}."))
        elif kind == "speak":
            self._handle_speak(message)
        elif kind == "tts_cancel":
            self._cancel_tts()
        elif kind == "playback_started":
            self._playback_active = True
            self._barge_in_sent = False
            await self._set_state("speaking")
        elif kind == "playback_finished":
            self._playback_active = False
            self._barge_in_sent = False
            if self._tts_queue.empty() and not self._speaking:
                await self._set_state("listening")
        elif kind == "ping":
            await self._send(event("pong", atMs = int(time.monotonic() * 1000)))
        else:
            logger.debug("live.unknown_message", kind = kind)

    async def _handle_hello(self, message: Any) -> None:
        problem = validate_hello(message)
        if problem:
            await self._send(error_event("bad_hello", problem))
            return
        self.config = self.config.with_hello(message)
        if not self._ready_sent:
            self._ready_sent = True
            await self._send(
                event(
                    "ready",
                    protocol = PROTOCOL_VERSION,
                    sampleRate = self.config.sample_rate,
                    echoCancellation = self.config.echo_cancellation,
                    models = {
                        "stt": self.config.stt_model or "default",
                        "tts": self.config.tts_model or "resident",
                        "voice": self.config.voice or "",
                    },
                    mode = self.config.mode,
                )
            )

    def _handle_speak(self, message: Any) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        item_id = (message.id or f"tts-{time.monotonic_ns()}").strip()
        seq = int(message.seq or 0)
        self._tts_queue.put_nowait(
            SpeakItem(
                id = item_id,
                seq = seq,
                text = text,
                generation = self._tts_generation,
            )
        )

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------

    async def _audio_worker(self) -> None:
        while not self._closed:
            try:
                frame = await self._audio_queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._process_frame(frame)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end live
                logger.exception("live.audio_frame_failed", error = str(exc))
                await self._send(
                    error_event(
                        "audio_processing_failed",
                        "Live audio processing failed; the session is still running.",
                        retryable = True,
                    )
                )

    async def _process_frame(self, frame: bytes) -> None:
        if self._closed or not self._started or self._muted:
            return
        # Turn mode gates the microphone while the assistant speaks.  If a client
        # keeps streaming frames anyway, they must not open an utterance.
        if self._playback_active and self.config.mode == MODE_TURN:
            return

        decision = self._vad.process(frame, self.config.sample_rate)
        self._frame_ms = audio.frame_duration_ms(frame, self.config.sample_rate)
        now = time.monotonic()
        if now - self._last_level_at >= 0.1:
            self._last_level_at = now
            await self._send(event("level", rms = round(float(decision.rms), 4)))

        if not self._speaking:
            self._pre_roll.append(frame)
            max_frames = max(1, int(math.ceil(self.config.endpoint.pre_roll_ms / self._frame_ms)))
            while len(self._pre_roll) > max_frames:
                self._pre_roll.popleft()
            if not decision.is_speech:
                return
            self._speaking = True
            self._speech_ms = self._frame_ms
            self._silence_ms = 0.0
            self._utterance = bytearray(b"".join(self._pre_roll))
            self._pre_roll.clear()
            self._last_smart_turn_silence_ms = None
            self._last_speech_start_ms = int(now * 1000)
            await self._set_state("speech")
            await self._send(event("speech_started", atMs = self._last_speech_start_ms))
            await self._maybe_barge_in()
            return

        self._utterance.extend(frame)
        self._speech_ms += self._frame_ms
        if decision.is_speech:
            self._silence_ms = 0.0
        else:
            self._silence_ms += self._frame_ms

        if self._playback_active and self.config.mode == MODE_BARGE_IN and decision.is_speech:
            await self._maybe_barge_in()

        # Without Smart Turn, wait for the extended window: a 300 ms pause is
        # often just a breath, and cutting the user off is worse than a slower
        # endpoint.  With the scorer installed, the fast path is available.
        semantic_complete = False
        if self._silence_ms >= self.config.endpoint.fast_silence_ms:
            # A Smart Turn inference every 32 ms would burn a CPU core while the
            # user pauses.  Score at most every 120 ms of silence, on the speech
            # segment without the trailing silence.
            last_scored = self._last_smart_turn_silence_ms
            if last_scored is None or self._silence_ms - last_scored >= 120.0:
                self._last_smart_turn_silence_ms = self._silence_ms
                segment = audio.trim_trailing_silence(
                    bytes(self._utterance),
                    self._silence_ms,
                    self.config.sample_rate,
                )
                score = self._smart_turn.score(segment)
                if score is not None:
                    semantic_complete = score >= self._smart_turn.threshold
        endpoint = self._endpoint.decide(
            silence_ms = self._silence_ms,
            utterance_ms = self._speech_ms,
            semantic_complete = semantic_complete,
        )
        if not endpoint.endpoint:
            return

        utterance = audio.trim_trailing_silence(
            bytes(self._utterance),
            self._silence_ms,
            self.config.sample_rate,
        )
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance.clear()
        self._vad.reset()
        self._barge_in_sent = False
        self._last_smart_turn_silence_ms = None
        await self._send(event("speech_ended", atMs = int(time.monotonic() * 1000)))
        await self._send(event("endpoint", reason = endpoint.reason))
        if not utterance:
            await self._set_state("listening")
            return
        self._utterance_counter += 1
        utterance_id = f"utt-{self._utterance_counter}"
        try:
            self._stt_queue.put_nowait((utterance_id, utterance))
        except asyncio.QueueFull:
            try:
                self._stt_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._stt_queue.put_nowait((utterance_id, utterance))
            except asyncio.QueueFull:
                pass
        await self._set_state("transcribing")

    async def _maybe_barge_in(self) -> None:
        if self._barge_in_sent or not self._playback_active:
            return
        self._barge_in_sent = True
        self._cancel_tts()
        self._playback_active = False
        await self._send(
            event(
                "barge_in",
                utteranceId = f"utt-{self._utterance_counter + 1}",
                atMs = int(time.monotonic() * 1000),
            )
        )

    def _reset_audio_state(self) -> None:
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance.clear()
        self._pre_roll.clear()
        self._barge_in_sent = False
        self._last_smart_turn_silence_ms = None
        try:
            self._vad.reset()
        except Exception:  # noqa: BLE001 - a reset must not break a mode change
            pass
        self._endpoint.reset()

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    async def _stt_worker(self) -> None:
        while not self._closed:
            try:
                utterance_id, pcm = await self._stt_queue.get()
            except asyncio.CancelledError:
                return
            wav = audio.pcm16_wav(pcm, self.config.sample_rate)
            try:
                text = await self.stt.transcribe(
                    wav,
                    model = self.config.stt_model,
                    language = self.config.language,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - surface the engine's reason
                logger.warning("live.stt_failed", error = str(exc))
                await self._send(
                    error_event(
                        "stt_failed",
                        str(exc) or "Speech-to-text failed.",
                        retryable = True,
                    )
                )
                await self._set_state("listening")
                continue
            text = (text or "").strip()
            if not text:
                await self._send(
                    error_event(
                        "no_speech",
                        "I could not hear a complete phrase. Try again.",
                        retryable = True,
                    )
                )
                await self._set_state("listening")
                continue
            await self._send(
                event(
                    "transcript",
                    text = text,
                    utteranceId = utterance_id,
                    language = self.config.language,
                    durationMs = int(audio.frame_duration_ms(pcm, self.config.sample_rate)),
                )
            )
            if not self._speaking and not self._playback_active and self._tts_queue.empty():
                await self._set_state("listening")

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _cancel_tts(self) -> None:
        self._tts_generation += 1
        while True:
            try:
                self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _tts_worker(self) -> None:
        while not self._closed:
            try:
                item = await self._tts_queue.get()
            except asyncio.CancelledError:
                return
            if item.generation != self._tts_generation:
                continue
            try:
                wav_bytes, sample_rate = await self.tts.synthesize(
                    item.text,
                    model = self.config.tts_model,
                    voice = self.config.voice,
                    language = self.config.language,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - a TTS failure must not kill listening
                logger.warning("live.tts_failed", error = str(exc))
                await self._send(
                    error_event(
                        "tts_failed",
                        str(exc) or "Text-to-speech failed.",
                        retryable = True,
                    )
                )
                if self._tts_queue.empty() and not self._playback_active:
                    await self._set_state("listening")
                continue
            if item.generation != self._tts_generation:
                continue
            await self._send(
                event(
                    "tts_start",
                    id = item.id,
                    seq = item.seq,
                    format = "wav",
                    sampleRate = int(sample_rate),
                    text = item.text,
                )
            )
            await self._send_bytes(wav_bytes)
            await self._send(event("tts_end", id = item.id, seq = item.seq))
            # The client owns playback.  Treating it as active until it says
            # playback_finished gives barge-in a correct window and keeps turn
            # mode from accepting a second utterance mid-sentence.
            self._playback_active = True
            await self._set_state("speaking")

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    async def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        await self._send(event("state", state = state))

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self._send_json_raw(payload)
        except Exception as exc:  # noqa: BLE001 - a closed socket ends the session
            logger.debug("live.send_json_failed", error = str(exc))

    async def _send_bytes(self, payload: bytes) -> None:
        if self._closed:
            return
        try:
            await self._send_bytes_raw(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("live.send_bytes_failed", error = str(exc))
