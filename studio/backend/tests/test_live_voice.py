# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic Live Voice engine tests: protocol, VAD, endpoint and one full turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from core.live import audio
from core.live.endpoint import EndpointConfig, EndpointDetector
from core.live.protocol import (
    PROTOCOL_VERSION,
    parse_client_message,
    validate_hello,
)
from core.live.session import LiveSession
from core.live.stt import FakeStt
from core.live.tts import ToneTts
from core.live.vad import EnergyVad


def _frame_for(ms: int, speech: bool) -> bytes:
    pcm = audio.speech_like_pcm(ms) if speech else audio.silence_pcm(ms)
    return pcm


def test_protocol_parses_and_validates_hello() -> None:
    message = parse_client_message(
        json.dumps(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "sampleRate": 16000,
                "channels": 1,
                "language": "tr",
                "mode": "barge_in",
            }
        )
    )
    assert message is not None
    assert message.type == "hello"
    assert validate_hello(message) == ""


def test_protocol_rejects_bad_json_and_bad_hello() -> None:
    assert parse_client_message("{not-json") is None
    message = parse_client_message(json.dumps({"type": "hello", "protocol": 99}))
    assert message is not None
    assert "Unsupported live protocol" in validate_hello(message)


def test_energy_vad_opens_and_releases() -> None:
    vad = EnergyVad(threshold = 0.015, min_speech_ms = 96, release_ms = 250)
    speech = audio.speech_like_pcm(32)
    silence = audio.silence_pcm(32)
    decisions = [vad.process(speech) for _ in range(4)]
    assert decisions[-1].is_speech is True
    released = [vad.process(silence) for _ in range(10)]
    assert released[-1].is_speech is False


def test_endpoint_policy() -> None:
    detector = EndpointDetector(
        EndpointConfig(
            fast_silence_ms = 300,
            extended_silence_ms = 900,
            min_utterance_ms = 250,
            max_utterance_ms = 5000,
        )
    )
    assert detector.decide(silence_ms = 100, utterance_ms = 400).endpoint is False
    fast = detector.decide(silence_ms = 320, utterance_ms = 800, semantic_complete = True)
    assert fast.endpoint is True and fast.reason == "semantic"
    extended = detector.decide(silence_ms = 320, utterance_ms = 800, semantic_complete = False)
    assert extended.endpoint is False
    forced = detector.decide(silence_ms = 0, utterance_ms = 5000)
    assert forced.endpoint is True and forced.reason == "max"


def test_full_turn_transcribes_and_speaks() -> None:
    async def scenario() -> tuple[list[dict[str, Any]], list[bytes]]:
        events: list[dict[str, Any]] = []
        audio_frames: list[bytes] = []

        async def send_json(payload: dict[str, Any]) -> None:
            events.append(payload)

        async def send_bytes(payload: bytes) -> None:
            audio_frames.append(payload)

        session = LiveSession(
            send_json = send_json,
            send_bytes = send_bytes,
            stt = FakeStt("ekranda ne var"),
            tts = ToneTts(),
        )
        await session.start()
        await session.handle_text(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "sampleRate": 16000,
                    "channels": 1,
                    "language": "tr",
                    "mode": "turn",
                }
            )
        )
        await session.handle_text(json.dumps({"type": "start"}))

        # ~800 ms of speech, then enough silence for the endpoint detector.
        for _ in range(25):
            session.push_audio(_frame_for(32, True))
        for _ in range(40):
            session.push_audio(_frame_for(32, False))
        for _ in range(40):
            await asyncio.sleep(0.01)
            if any(event.get("type") == "transcript" for event in events):
                break

        await session.handle_text(
            json.dumps({"type": "speak", "id": "tts-1", "seq": 0, "text": "merhaba"})
        )
        for _ in range(60):
            await asyncio.sleep(0.01)
            if any(event.get("type") == "tts_end" for event in events):
                break
        await session.close()
        return events, audio_frames

    events, audio_frames = asyncio.run(scenario())
    kinds = [event.get("type") for event in events]
    assert "ready" in kinds
    assert "speech_started" in kinds
    assert "endpoint" in kinds
    assert "transcript" in kinds
    transcript = next(event for event in events if event.get("type") == "transcript")
    assert transcript["text"] == "ekranda ne var"
    assert "tts_start" in kinds
    assert "tts_end" in kinds
    assert audio_frames and audio_frames[0].startswith(b"RIFF")


def test_barge_in_while_playback_is_active() -> None:
    async def scenario() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        async def send_json(payload: dict[str, Any]) -> None:
            events.append(payload)

        async def send_bytes(payload: bytes) -> None:
            return None

        session = LiveSession(
            send_json = send_json,
            send_bytes = send_bytes,
            stt = FakeStt("yeni soru"),
            tts = ToneTts(),
        )
        await session.start()
        await session.handle_text(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "sampleRate": 16000,
                    "channels": 1,
                    "language": "tr",
                    "mode": "barge_in",
                }
            )
        )
        await session.handle_text(json.dumps({"type": "start"}))
        await session.handle_text(json.dumps({"type": "playback_started", "id": "tts-0"}))
        for _ in range(12):
            session.push_audio(_frame_for(32, True))
        for _ in range(30):
            await asyncio.sleep(0.01)
            if any(event.get("type") == "barge_in" for event in events):
                break
        await session.close()
        return events

    events = asyncio.run(scenario())
    assert any(event.get("type") == "barge_in" for event in events)
