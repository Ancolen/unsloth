# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic Live Voice engine tests: protocol, VAD, endpoint and one full turn."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from core.live import audio
from core.live.endpoint import EndpointConfig, EndpointDetector, SmartTurnScorer
from core.live.models import status as live_status
from core.live.protocol import (
    PROTOCOL_VERSION,
    parse_client_message,
    validate_hello,
)
from core.live.session import LiveSession
from core.live.stt import FakeStt
from core.live.tts import PiperTts, TtsError, ToneTts
from core.live.vad import EnergyVad
from core.live.xtts import XttsTts


def _frame_for(ms: int, speech: bool) -> bytes:
    pcm = audio.speech_like_pcm(ms) if speech else audio.silence_pcm(ms)
    return pcm


def test_status_reports_detector_fallbacks(monkeypatch) -> None:
    monkeypatch.delenv("UNSLOTH_LIVE_VAD_MODEL", raising = False)
    monkeypatch.delenv("UNSLOTH_LIVE_SMART_TURN_MODEL", raising = False)
    capabilities = live_status().to_dict()["capabilities"]
    assert capabilities["vad"] == "energy"
    assert capabilities["endpoint"] == "silence"


def test_status_only_claims_onnx_detectors_when_the_runtime_is_available(
    tmp_path, monkeypatch
) -> None:
    vad_path = tmp_path / "vad.onnx"
    turn_path = tmp_path / "turn.onnx"
    vad_path.write_bytes(b"vad")
    turn_path.write_bytes(b"turn")
    monkeypatch.setenv("UNSLOTH_LIVE_VAD_MODEL", str(vad_path))
    monkeypatch.setenv("UNSLOTH_LIVE_SMART_TURN_MODEL", str(turn_path))

    import core.live.models as models

    monkeypatch.setattr(models, "_onnxruntime_available", lambda: False)
    degraded = live_status().to_dict()
    assert degraded["capabilities"]["vad"] == "energy"
    assert degraded["capabilities"]["endpoint"] == "silence"
    assert all(asset["installed"] and not asset["active"] for asset in degraded["assets"])

    monkeypatch.setattr(models, "_onnxruntime_available", lambda: True)
    ready = live_status().to_dict()
    assert ready["capabilities"]["vad"] == "silero"
    assert ready["capabilities"]["endpoint"] == "smart_turn"
    assert all(asset["active"] for asset in ready["assets"])


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


def test_smart_turn_scores_whisper_features_from_the_utterance_end() -> None:
    import numpy as np

    class _Session:
        def __init__(self) -> None:
            self.feeds: list[dict[str, Any]] = []

        def get_inputs(self):
            return [SimpleNamespace(name = "input_features")]

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            return [np.array([[0.91]], dtype = np.float32)]

    class _Extractor:
        def __init__(self) -> None:
            self.lengths: list[int] = []
            self.kwargs: dict[str, Any] = {}

        def __call__(self, values, **kwargs):
            self.lengths.append(len(values))
            self.kwargs = kwargs
            return {"input_features": np.zeros((1, 128, 3000), dtype = np.float32)}

    session = _Session()
    extractor = _Extractor()
    scorer = SmartTurnScorer(session = session, feature_extractor = extractor)
    pcm = audio.speech_like_pcm(10_000)
    score = scorer.score(pcm)
    assert score is not None and abs(score - 0.91) < 1e-6
    assert extractor.lengths == [8 * 16000]
    assert extractor.kwargs["sampling_rate"] == 16000
    assert "input_features" in session.feeds[0]
    assert session.feeds[0]["input_features"].shape == (1, 128, 3000)
    assert scorer.available is True


def test_session_throttles_smart_turn_scoring() -> None:
    class _Scorer:
        threshold = 0.80

        def __init__(self) -> None:
            self.calls = 0

        def score(self, _pcm: bytes) -> float:
            self.calls += 1
            return 0.0

    async def scenario() -> int:
        events: list[dict[str, Any]] = []

        async def send_json(payload: dict[str, Any]) -> None:
            events.append(payload)

        async def send_bytes(_payload: bytes) -> None:
            return None

        scorer = _Scorer()
        session = LiveSession(
            send_json = send_json,
            send_bytes = send_bytes,
            stt = FakeStt("tamam"),
            tts = ToneTts(),
            smart_turn = scorer,
        )
        await session.start()
        await session.handle_text(json.dumps({"type": "hello", "protocol": 1}))
        await session.handle_text(json.dumps({"type": "start"}))
        for _ in range(25):
            session.push_audio(_frame_for(32, True))
        for _ in range(40):
            session.push_audio(_frame_for(32, False))
        for _ in range(60):
            await asyncio.sleep(0.01)
            if any(event.get("type") == "endpoint" for event in events):
                break
        await session.close()
        return scorer.calls

    calls = asyncio.run(scenario())
    # 900 ms of extended silence, scored every 120 ms, is ~6 calls; the previous
    # per-frame version made ~19.
    assert 1 <= calls <= 7


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


class _FakePiperVoice:
    class config:
        sample_rate = 22050

    def synthesize_stream_raw(self, text: str):
        yield b"\x00\x01" * 256


def test_piper_tts_wraps_streamed_pcm() -> None:
    async def scenario() -> tuple[bytes, int]:
        engine = PiperTts("unused.onnx", voice_instance = _FakePiperVoice())
        try:
            return await engine.synthesize("merhaba")
        finally:
            await engine.close()

    wav_bytes, sample_rate = asyncio.run(scenario())
    assert sample_rate == 22050
    assert wav_bytes.startswith(b"RIFF")
    assert audio.wav_sample_rate(wav_bytes) == 22050


def test_piper_tts_reports_a_missing_voice() -> None:
    async def scenario() -> None:
        engine = PiperTts("/definitely/missing/voice.onnx")
        await engine.synthesize("merhaba")

    try:
        asyncio.run(scenario())
    except TtsError as exc:
        assert "Piper voice not found" in str(exc)
    else:
        raise AssertionError("expected TtsError for a missing Piper voice")


class _FakeXttsModel:
    class config:
        class audio:
            output_sample_rate = 24000

    speaker_manager = SimpleNamespace(
        speakers = {
            "Asya Anara": {
                "gpt_cond_latent": [[0.0]],
                "speaker_embedding": [[0.0]],
            }
        }
    )

    def inference_stream(self, text: str, language: str, gpt: Any, spk: Any, **_: Any):
        yield [0.0, 0.25, -0.25, 1.0]


def test_xtts_requires_the_cpml_acceptance_flag() -> None:
    async def scenario() -> None:
        engine = XttsTts(model_instance = _FakeXttsModel())
        await engine.synthesize("merhaba")

    try:
        asyncio.run(scenario())
    except TtsError as exc:
        assert "CPML" in str(exc)
    else:
        raise AssertionError("expected the XTTS license gate")


def test_xtts_wraps_streamed_pcm() -> None:
    async def scenario() -> tuple[bytes, int]:
        engine = XttsTts(
            model_instance = _FakeXttsModel(),
            accept_license = True,
        )
        try:
            return await engine.synthesize("merhaba", voice = "Asya Anara")
        finally:
            await engine.close()

    wav_bytes, sample_rate = asyncio.run(scenario())
    assert sample_rate == 24000
    assert wav_bytes.startswith(b"RIFF")
    assert audio.wav_sample_rate(wav_bytes) == 24000
