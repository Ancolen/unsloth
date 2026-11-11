# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Feature flag, platform capability and the live model catalog surface.

The live engine deliberately reuses Studio's model manager: STT and TTS ids are
the ids the Voice settings and model hub already understand.  This module reports
readiness/capabilities; the pinned VAD/endpoint assets and their download path
live in `core.live.assets`.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from core.live import assets as live_assets
from core.live.protocol import PROTOCOL_VERSION, SAMPLE_RATE


def live_enabled() -> bool:
    """Live Voice is on unless an operator explicitly disables it."""
    value = os.environ.get("UNSLOTH_STUDIO_LIVE_VOICE", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def screen_capture_supported() -> bool:
    """Tauri's portal capture is Linux-first; other platforms hide the toggle."""
    return sys.platform.startswith("linux")


def _engine_kind(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip().lower()
    return value or "studio"


@dataclass(frozen = True)
class LiveStatus:
    enabled: bool
    protocol: int
    sample_rate: int
    platform: str
    screen_capture: str
    stt_engine: str
    tts_engine: str
    onnxruntime: bool
    assets: tuple[dict[str, Any], ...] = field(default_factory = tuple)

    def to_dict(self) -> dict[str, Any]:
        by_role = {asset["role"]: asset for asset in self.assets}
        return {
            "enabled": self.enabled,
            "protocol": self.protocol,
            "sampleRate": self.sample_rate,
            "platform": self.platform,
            "capabilities": {
                "audioCapture": "frontend",
                "screenCapture": self.screen_capture,
                "stt": self.stt_engine,
                "tts": self.tts_engine,
                "onnxruntime": "available" if self.onnxruntime else "missing",
                "vad": (
                    "silero"
                    if by_role.get("vad") is not None and by_role["vad"]["active"]
                    else "energy"
                ),
                "endpoint": (
                    "smart_turn"
                    if by_role.get("endpoint") is not None
                    and by_role["endpoint"]["active"]
                    else "silence"
                ),
            },
            "assets": list(self.assets),
        }


def status() -> LiveStatus:
    return LiveStatus(
        enabled = live_enabled(),
        protocol = PROTOCOL_VERSION,
        sample_rate = SAMPLE_RATE,
        platform = platform.system().lower(),
        screen_capture = "tauri" if screen_capture_supported() else "unsupported",
        stt_engine = _engine_kind("UNSLOTH_LIVE_STT"),
        tts_engine = _engine_kind("UNSLOTH_LIVE_TTS"),
        onnxruntime = live_assets.onnxruntime_available(),
        assets = live_assets.asset_states(),
    )
