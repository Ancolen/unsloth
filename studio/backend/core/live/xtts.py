# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional XTTS-v2 voice for Live Voice.

XTTS-v2 is the higher-quality Turkish voice from the Screen Companion design:
a zero-shot neural model that runs locally through the optional ``coqui-tts``
package.  It is deliberately opt-in: the default stays on Studio's speech route,
because XTTS needs PyTorch, roughly 1.8 GB of weights and the CPML license.

The model is downloaded from Hugging Face with a pinned revision on first use.
A built-in speaker or a reference WAV path can be selected; unknown Studio voice
ids fall back to the configured XTTS speaker so switching the Live voice does
not break the session.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Optional

from core.live import audio
from core.live.tts import TtsError

XTTS_REPO_ID = "coqui/XTTS-v2"
XTTS_PINNED_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
XTTS_FILES = (
    "config.json",
    "model.pth",
    "vocab.json",
    "speakers_xtts.pth",
    "mel_stats.pth",
)
XTTS_DEFAULT_SPEAKER = "Asya Anara"
XTTS_DEFAULT_LANGUAGE = "tr"
XTTS_SAMPLE_RATE = 24_000

_INSTALL_HINT = (
    "XTTS-v2 requires the optional 'coqui-tts' package (PyTorch). Install it "
    "in Studio's Python environment and restart the backend."
)
_LICENSE_HINT = (
    "XTTS-v2 is licensed under CPML (non-commercial). Review the license and "
    "set UNSLOTH_LIVE_XTTS_ACCEPT_LICENSE=1 to enable it."
)


class XttsTts:
    """Implement the live TTS protocol with XTTS-v2 on CPU or CUDA."""

    def __init__(
        self,
        *,
        model_dir: Optional[Path] = None,
        speaker: str = XTTS_DEFAULT_SPEAKER,
        language: str = XTTS_DEFAULT_LANGUAGE,
        device: str = "auto",
        deterministic: bool = False,
        accept_license: bool = False,
        model_instance: Any = None,
    ) -> None:
        self.model_dir = model_dir
        self.speaker = (speaker or "").strip() or XTTS_DEFAULT_SPEAKER
        self.language = (language or "").strip() or XTTS_DEFAULT_LANGUAGE
        self.device = (device or "auto").strip().lower()
        self.deterministic = bool(deterministic)
        self.accept_license = bool(accept_license)
        self._model = model_instance
        self._load_lock = threading.Lock()
        self._synth_lock = threading.Lock()
        self._conditioning_lock = threading.Lock()
        self._conditioning: dict[str, tuple[Any, Any]] = {}

    def _ensure_license(self) -> None:
        if not self.accept_license:
            raise TtsError(_LICENSE_HINT)

    def _ensure_loaded(self) -> Any:
        self._ensure_license()
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
            except ImportError as exc:  # pragma: no cover - minimal install
                raise TtsError(_INSTALL_HINT) from exc

            model_dir = self._resolve_model_dir()
            config = XttsConfig()
            config.load_json(str(model_dir / "config.json"))
            model = Xtts(config)
            model.load_checkpoint(config, checkpoint_dir = str(model_dir), eval = True)
            model.to(_torch_device(self.device, torch))
            self._model = model
            return model

    def _resolve_model_dir(self) -> Path:
        model_dir = self.model_dir or Path(
            os.environ.get("UNSLOTH_LIVE_XTTS_MODEL_DIR", "").strip()
            or (Path.home() / ".unsloth" / "studio" / "live-models" / "xtts-v2")
        )
        if all((model_dir / name).is_file() for name in XTTS_FILES):
            return model_dir
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # noqa: BLE001 - Studio normally ships this
            raise TtsError(
                "huggingface_hub is required to download the XTTS-v2 weights."
            ) from exc
        model_dir.mkdir(parents = True, exist_ok = True)
        for filename in XTTS_FILES:
            if (model_dir / filename).is_file():
                continue
            hf_hub_download(
                repo_id = XTTS_REPO_ID,
                revision = XTTS_PINNED_REVISION,
                filename = filename,
                local_dir = model_dir,
            )
        return model_dir

    def _voice_name(self, model: Any, requested: Optional[str]) -> str:
        candidates = [requested, self.speaker]
        for candidate in candidates:
            value = (candidate or "").strip()
            if not value:
                continue
            if _reference_path(value) is not None:
                return value
            speakers = _speakers(model)
            if value in speakers:
                return value
        speakers = _speakers(model)
        if self.speaker in speakers:
            return self.speaker
        if speakers:
            return sorted(speakers)[0]
        return self.speaker

    def _resolve_conditioning(self, model: Any, voice: str) -> tuple[Any, Any]:
        with self._conditioning_lock:
            cached = self._conditioning.get(voice)
            if cached is not None:
                return cached
            reference = _reference_path(voice)
            if reference is not None:
                latents = model.get_conditioning_latents(
                    [str(reference)], sound_norm_refs = True
                )
            else:
                entry = _speakers(model).get(voice)
                if entry is None:
                    available = ", ".join(sorted(_speakers(model))) or "none"
                    raise ValueError(
                        f"Unknown XTTS voice '{voice}'. Use a built-in speaker "
                        f"({available}) or a reference WAV path."
                    )
                latents = _speaker_latents(entry)
            resolved = (latents[0], latents[1])
            self._conditioning[voice] = resolved
            return resolved

    def _synthesize_blocking(
        self,
        text: str,
        requested_voice: Optional[str],
        requested_language: Optional[str],
    ) -> tuple[bytes, int]:
        model = self._ensure_loaded()
        voice = self._voice_name(model, requested_voice)
        gpt_cond_latent, speaker_embedding = self._resolve_conditioning(model, voice)
        language = (requested_language or self.language or "").strip().lower()
        if not language or language == "auto":
            language = XTTS_DEFAULT_LANGUAGE
        with self._synth_lock:
            try:
                if hasattr(model, "inference_stream"):
                    chunks = list(
                        model.inference_stream(
                            text,
                            language,
                            gpt_cond_latent,
                            speaker_embedding,
                            stream_chunk_size = 20,
                            enable_text_splitting = False,
                            do_sample = not self.deterministic,
                        )
                    )
                else:
                    result = model.inference(
                        text,
                        language,
                        gpt_cond_latent,
                        speaker_embedding,
                        enable_text_splitting = False,
                        do_sample = not self.deterministic,
                    )
                    chunks = [result.get("wav", result)]
            except Exception as exc:  # noqa: BLE001 - surfaced as a TTS error
                raise TtsError(f"XTTS synthesis failed: {exc}") from exc
        pcm = b"".join(_to_pcm16(chunk) for chunk in chunks)
        return pcm, _output_sample_rate(model)

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
        try:
            pcm, sample_rate = await asyncio.to_thread(
                self._synthesize_blocking, text, voice, language
            )
        except Exception as exc:  # noqa: BLE001 - keep one actionable error type
            if isinstance(exc, TtsError):
                raise
            raise TtsError(str(exc) or "XTTS synthesis failed.") from exc
        if not pcm:
            raise TtsError("XTTS returned no audio.")
        return audio.pcm16_wav(pcm, sample_rate), sample_rate

    async def close(self) -> None:
        with self._load_lock:
            self._model = None
        with self._conditioning_lock:
            self._conditioning.clear()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - optional cleanup
            pass


def build_xtts_tts() -> XttsTts:
    """Build the opt-in XTTS engine from the live environment variables."""
    model_dir = os.environ.get("UNSLOTH_LIVE_XTTS_MODEL_DIR", "").strip()
    return XttsTts(
        model_dir = Path(model_dir) if model_dir else None,
        speaker = os.environ.get("UNSLOTH_LIVE_XTTS_SPEAKER", "").strip()
        or XTTS_DEFAULT_SPEAKER,
        language = os.environ.get("UNSLOTH_LIVE_XTTS_LANGUAGE", "").strip()
        or XTTS_DEFAULT_LANGUAGE,
        device = os.environ.get("UNSLOTH_LIVE_XTTS_DEVICE", "").strip() or "auto",
        deterministic = os.environ.get("UNSLOTH_LIVE_XTTS_DETERMINISTIC", "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        accept_license = os.environ.get("UNSLOTH_LIVE_XTTS_ACCEPT_LICENSE", "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
    )


def _torch_device(value: str, torch: Any) -> str:
    value = (value or "auto").strip().lower()
    if value in ("auto", "cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("gpu:"):
        return f"cuda:{value.split(':', 1)[1]}"
    return value


def _speakers(model: Any) -> dict[str, Any]:
    speakers = getattr(getattr(model, "speaker_manager", None), "speakers", None)
    return speakers if isinstance(speakers, dict) else {}


def _reference_path(voice: str) -> Optional[Path]:
    candidate = voice[5:] if voice.startswith("file:") else voice
    path = Path(candidate).expanduser()
    return path if path.is_file() else None


def _speaker_latents(entry: Any) -> tuple[Any, Any]:
    if isinstance(entry, dict):
        gpt = entry.get("gpt_cond_latent", entry.get("gpt_conditioning_latents"))
        embedding = entry.get("speaker_embedding")
        if gpt is not None and embedding is not None:
            return gpt, embedding
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return entry[0], entry[1]
    values = list(getattr(entry, "values", lambda: [])())
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("XTTS speaker entry does not contain conditioning latents")


def _output_sample_rate(model: Any) -> int:
    audio_config = getattr(getattr(model, "config", None), "audio", None)
    return int(getattr(audio_config, "output_sample_rate", XTTS_SAMPLE_RATE))


def _to_pcm16(chunk: Any) -> bytes:
    """Convert an XTTS float waveform (torch tensor or array) to 16-bit PCM."""
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - XTTS already requires NumPy/PyTorch
        raise TtsError("NumPy is required for XTTS audio conversion.") from exc
    if hasattr(chunk, "detach"):
        chunk = chunk.detach().float().cpu().numpy()
    samples = np.asarray(chunk, dtype = np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()
