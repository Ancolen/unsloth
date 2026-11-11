# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice activity detection for Live Voice.

`EnergyVad` is the deterministic fallback and the default until the pinned Silero
ONNX asset is installed.  `SileroVad` is the production adapter; it falls back to
the energy detector if onnxruntime or the model file is unavailable, and reports
that state so the UI can surface it instead of failing silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.live import audio
from core.live.assets import asset_by_role, available_asset_path
from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class VadFrame:
    """One VAD decision for one audio frame."""

    is_speech: bool
    probability: float
    rms: float


class EnergyVad:
    """RMS threshold with onset/release hysteresis.

    The default threshold is deliberately conservative: a fan, keyboard or room
    tone should not open an utterance.  The live settings expose it, and the
    Silero adapter replaces it in production.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.015,
        release_ratio: float = 0.65,
        min_speech_ms: float = 120.0,
        release_ms: float = 250.0,
    ) -> None:
        self.threshold = max(0.0005, float(threshold))
        self.release_threshold = self.threshold * max(0.0, min(1.0, release_ratio))
        self.min_speech_ms = max(0.0, float(min_speech_ms))
        self.release_ms = max(0.0, float(release_ms))
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def reset(self) -> None:
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def process(self, pcm: bytes, sample_rate: int = audio.SAMPLE_RATE) -> VadFrame:
        rms = audio.rms_int16(pcm)
        frame_ms = audio.frame_duration_ms(pcm, sample_rate)
        above = rms >= self.threshold
        if above:
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
            if not self._speaking and self._speech_ms >= self.min_speech_ms:
                self._speaking = True
        else:
            self._silence_ms += frame_ms
            if self._speaking and rms < self.release_threshold and self._silence_ms >= self.release_ms:
                self._speaking = False
                self._speech_ms = 0.0
        return VadFrame(
            is_speech = self._speaking,
            probability = 1.0 if above else 0.0,
            rms = rms,
        )


class SileroVad:
    """Silero VAD wrapper with a deterministic fallback.

    The pinned model is an ONNX asset downloaded by the live model catalog.  Until
    it exists (and onnxruntime is importable) this class behaves exactly like the
    energy detector, but records why it degraded so /api/live/status can explain
    it.
    """

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        threshold: float = 0.5,
        fallback: Optional[EnergyVad] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = float(threshold)
        self.fallback = fallback or EnergyVad()
        self._session: Any = None
        self._state: Any = None
        self._degraded_reason = ""
        self._load_attempted = False

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self.model_path is None or not self.model_path.exists():
            self._degraded_reason = "Silero VAD model is not installed"
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._degraded_reason = f"onnxruntime is unavailable: {exc}"
            return
        try:
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options = options,
                providers = ["CPUExecutionProvider"],
            )
            # Silero v5/v6 LSTM state: [2, batch, 128]
            import numpy as np  # type: ignore

            self._state = np.zeros((2, 1, 128), dtype = np.float32)
        except Exception as exc:  # noqa: BLE001 - a bad asset must not kill live
            self._session = None
            self._degraded_reason = f"Silero VAD could not load: {exc}"

    def reset(self) -> None:
        self.fallback.reset()
        self._state = None
        self._load_attempted = False

    def process(self, pcm: bytes, sample_rate: int = audio.SAMPLE_RATE) -> VadFrame:
        self._load()
        if self._session is None or sample_rate != 16000:
            return self.fallback.process(pcm, sample_rate)
        try:
            import numpy as np  # type: ignore

            samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype = "<i2")
            if samples.size != 512:
                # The session feeds 512-sample frames; anything else goes through the
                # fallback rather than silently changing the model's contract.
                return self.fallback.process(pcm, sample_rate)
            values = (samples.astype(np.float32) / 32768.0).reshape(1, -1)
            if self._state is None:
                self._state = np.zeros((2, 1, 128), dtype = np.float32)
            outputs = self._session.run(
                None,
                {"input": values, "state": self._state, "sr": np.array(sample_rate, dtype = np.int64)},
            )
            probability = float(outputs[0].reshape(-1)[0])
            self._state = outputs[1]
            return VadFrame(
                is_speech = probability >= self.threshold,
                probability = probability,
                rms = audio.rms_int16(pcm),
            )
        except Exception as exc:  # noqa: BLE001 - inference must never crash audio
            logger.warning("live.silero_failed", error = str(exc))
            self._session = None
            self._degraded_reason = f"Silero inference failed: {exc}"
            return self.fallback.process(pcm, sample_rate)


def build_vad(*, threshold: float = 0.015) -> Any:
    """Pick the best available VAD without requiring optional dependencies."""
    model_path = os.environ.get("UNSLOTH_LIVE_VAD_MODEL", "").strip()
    if model_path:
        return SileroVad(model_path = Path(model_path), fallback = EnergyVad(threshold = threshold))
    asset = asset_by_role("vad")
    installed = available_asset_path(asset) if asset is not None else None
    if installed is not None:
        return SileroVad(model_path = installed, fallback = EnergyVad(threshold = threshold))
    return EnergyVad(threshold = threshold)
