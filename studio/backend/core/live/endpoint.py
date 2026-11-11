# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn endpointing for Live Voice.

The first implementation is silence-driven with two windows:

* a fast path when the caller says the utterance sounds complete (300 ms),
* an extended path when it may still be unfinished (900 ms).

`SmartTurnScorer` is the seam for the pinned Smart Turn ONNX model.  It is
optional: without it the session passes `semantic_complete=True` and the fast
window applies, which keeps the live mode usable before the asset is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class EndpointDecision:
    endpoint: bool
    reason: str = ""


@dataclass(frozen = True)
class EndpointConfig:
    fast_silence_ms: float = 300.0
    extended_silence_ms: float = 900.0
    max_utterance_ms: float = 30000.0
    min_utterance_ms: float = 250.0
    pre_roll_ms: float = 300.0


class EndpointDetector:
    """Decides when a spoken utterance is over.

    Pure and deterministic so the whole live turn can be tested with synthetic
    frames.  The session feeds silence and utterance durations; the detector owns
    the policy and the reason strings.
    """

    def __init__(self, config: Optional[EndpointConfig] = None) -> None:
        self.config = config or EndpointConfig()

    def decide(
        self,
        *,
        silence_ms: float,
        utterance_ms: float,
        semantic_complete: bool = True,
    ) -> EndpointDecision:
        if utterance_ms >= self.config.max_utterance_ms:
            return EndpointDecision(True, "max")
        if utterance_ms < self.config.min_utterance_ms:
            return EndpointDecision(False, "")
        if silence_ms >= self.config.extended_silence_ms:
            return EndpointDecision(True, "silence")
        if semantic_complete and silence_ms >= self.config.fast_silence_ms:
            return EndpointDecision(True, "semantic")
        return EndpointDecision(False, "")

    def reset(self) -> None:
        """Stateless today; kept so a future scorer can clear its own buffers."""


class SmartTurnScorer:
    """Optional Smart Turn v3.2 adapter.

    Smart Turn is a Whisper-Tiny classifier: it consumes 8-second log-mel
    features (`input_features`), not raw PCM, and returns a completion
    probability.  Until the pinned ONNX asset, onnxruntime and the Whisper
    feature extractor are available this class reports `available=False`; the
    endpoint detector then uses the extended silence window.
    """

    MAX_SECONDS = 8.0

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        threshold: float = 0.80,
        session: Any = None,
        feature_extractor: Any = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self._session = session
        self._feature_extractor = feature_extractor
        self._input_name = "input_features"
        self._load_attempted = False
        self._degraded_reason = ""

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def available(self) -> bool:
        self._load()
        return self._session is not None and self._feature_extractor is not None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self._session is None:
            if self.model_path is None or not self.model_path.exists():
                self._degraded_reason = "Smart Turn model is not installed"
                return
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self._degraded_reason = f"onnxruntime is unavailable: {exc}"
                return
            try:
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    providers = ["CPUExecutionProvider"],
                )
            except Exception as exc:  # noqa: BLE001 - an asset failure is not fatal
                logger.warning("live.smart_turn_failed", error = str(exc))
                self._degraded_reason = f"Smart Turn could not load: {exc}"
                self._session = None
                return
        if self._feature_extractor is None:
            try:
                from transformers import WhisperFeatureExtractor  # type: ignore

                self._feature_extractor = WhisperFeatureExtractor(
                    chunk_length = int(self.MAX_SECONDS)
                )
            except Exception as exc:  # noqa: BLE001 - optional dependency
                logger.warning("live.smart_turn_features_failed", error = str(exc))
                self._degraded_reason = f"Whisper feature extractor is unavailable: {exc}"
                self._session = None
                return
        try:
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = str(inputs[0].name)
        except Exception:  # noqa: BLE001 - an injected test session may not expose it
            self._input_name = "input_features"

    def score(self, pcm: bytes) -> Optional[float]:
        """Return a completion probability, or None when the scorer is unavailable."""
        self._load()
        if self._session is None or self._feature_extractor is None:
            return None
        try:
            import numpy as np  # type: ignore

            samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype = "<i2")
            if samples.size == 0:
                return None
            values = samples.astype(np.float32) / 32768.0
            max_samples = int(self.MAX_SECONDS * 16000)
            if values.size > max_samples:
                # Whisper sees the end of the utterance; the beginning is the
                # least useful part for a completion decision.
                values = values[-max_samples:]
            features = self._feature_extractor(
                values,
                sampling_rate = 16000,
                return_tensors = "np",
                padding = "max_length",
                max_length = max_samples,
                truncation = True,
                do_normalize = True,
            )
            input_features = np.asarray(features["input_features"], dtype = np.float32)
            if input_features.ndim == 2:
                input_features = input_features[None, ...]
            outputs = self._session.run(None, {self._input_name: input_features})
            return float(outputs[0].reshape(-1)[0])
        except Exception as exc:  # noqa: BLE001 - inference must never crash audio
            logger.warning("live.smart_turn_inference_failed", error = str(exc))
            # Disable the scorer instead of retrying (and logging) every frame;
            # the session falls back to the extended silence window.
            self._session = None
            self._degraded_reason = f"Smart Turn inference failed: {exc}"
            return None

    def is_complete(self, pcm: bytes) -> bool:
        score = self.score(pcm)
        return True if score is None else score >= self.threshold
