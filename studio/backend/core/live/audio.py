# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PCM helpers for Live Voice.

The live socket carries little-endian signed 16-bit mono PCM at 16 kHz.  These
helpers stay dependency-free (stdlib only) so the live engine can be imported by
tests without a model or an audio device.
"""

from __future__ import annotations

import array
import io
import math
import sys
import wave
from typing import Optional

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1


def _samples(pcm: bytes) -> array.array:
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH_BYTES)
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def rms_int16(pcm: bytes) -> float:
    """Root-mean-square level in 0..1 for a PCM frame."""
    samples = _samples(pcm)
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(samples)) / 32768.0


def peak_int16(pcm: bytes) -> float:
    samples = _samples(pcm)
    if not samples:
        return 0.0
    return max(abs(sample) for sample in samples) / 32768.0


def frame_duration_ms(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    if sample_rate <= 0:
        return 0.0
    samples = len(pcm) // SAMPLE_WIDTH_BYTES
    return samples * 1000.0 / float(sample_rate)


def pcm16_wav(
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> bytes:
    """Wrap PCM in a WAV container for the existing transcription route."""
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH_BYTES)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm[:usable])
    return out.getvalue()


def wav_sample_rate(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return int(wav.getframerate())


def trim_trailing_silence(
    pcm: bytes,
    silence_ms: float,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Drop the silence that triggered endpointing, keeping at least one frame.

    The caller has already decided the utterance ended; the exact sample boundary
    is not audio-critical, so this is a cheap byte trim rather than a second VAD
    pass.  It never returns an empty buffer when the input had audio.
    """
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH_BYTES)
    if usable <= 0 or silence_ms <= 0:
        return pcm[:usable]
    silence_bytes = int(sample_rate * (silence_ms / 1000.0)) * SAMPLE_WIDTH_BYTES
    minimum_bytes = int(sample_rate * 0.25) * SAMPLE_WIDTH_BYTES
    keep = max(minimum_bytes, usable - silence_bytes)
    keep -= keep % SAMPLE_WIDTH_BYTES
    return pcm[:keep]


def tone_wav(
    *,
    duration_ms: int = 400,
    frequency: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.12,
) -> bytes:
    """A short, quiet tone used by the development TTS adapter and tests."""
    duration_ms = max(60, min(int(duration_ms), 5000))
    sample_count = int(sample_rate * duration_ms / 1000.0)
    samples = array.array("h")
    for index in range(sample_count):
        # Fade in/out so the tone cannot click on speaker output.
        fade = min(1.0, index / max(1.0, sample_rate * 0.02))
        fade = min(fade, (sample_count - index) / max(1.0, sample_rate * 0.02))
        value = amplitude * fade * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    if sys.byteorder == "big":
        samples.byteswap()
    return pcm16_wav(samples.tobytes(), sample_rate = sample_rate)


def silence_pcm(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    sample_count = max(0, int(sample_rate * duration_ms / 1000.0))
    return b"\x00\x00" * sample_count


def speech_like_pcm(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Deterministic non-silent PCM for VAD/session tests (no microphone needed)."""
    sample_count = max(0, int(sample_rate * duration_ms / 1000.0))
    samples = array.array("h")
    for index in range(sample_count):
        value = 0.20 * math.sin(2.0 * math.pi * 180.0 * index / sample_rate)
        value += 0.08 * math.sin(2.0 * math.pi * 360.0 * index / sample_rate)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def normalize_pcm(pcm: bytes) -> Optional[bytes]:
    """Trim an odd trailing byte; None when the frame carries no complete sample."""
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH_BYTES)
    return pcm[:usable] if usable else None
