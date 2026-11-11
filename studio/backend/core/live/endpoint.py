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

    The model scores a trimmed speech segment and returns a completion
    probability.  Until the pinned ONNX asset and onnxruntime are available this
    class reports `available=False`; the endpoint detector then uses the fast
    silence window and the UI can show the reduced-quality hint.
    """

    def __init__(self, *, model_path: Optional[Path] = None, threshold: float = 0.80) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self._session: Any = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        self._load()
        return self._session is not None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self.model_path is None or not self.model_path.exists():
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception:  # noqa: BLE001 - optional dependency
            return
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers = ["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001 - an asset failure is not fatal
            logger.warning("live.smart_turn_failed", error = str(exc))
            self._session = None

    def score(self, pcm: bytes) -> Optional[float]:
        """Return a completion probability, or None when the scorer is unavailable."""
        self._load()
        if self._session is None:
            return None
        try:
            import numpy as np  # type: ignore

            samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype = "<i2")
            values = (samples.astype(np.float32) / 32768.0).reshape(1, -1)
            outputs = self._session.run(None, {"input": values})
            return float(outputs[0].reshape(-1)[0])
        except Exception as exc:  # noqa: BLE001 - inference must never crash audio
            logger.warning("live.smart_turn_inference_failed", error = str(exc))
            return None

    def is_complete(self, pcm: bytes) -> bool:
        score = self.score(pcm)
        return True if score is None else score >= self.threshold
