# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed WebSocket protocol for Live Voice.

The protocol is intentionally small: JSON control messages plus binary PCM audio
frames.  Both sides are versioned so a stale frontend cannot silently drive a new
backend.  Unknown fields are ignored; unknown message types are reported as a
non-fatal protocol error.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROTOCOL_VERSION = 1

# Audio format exchanged on the socket.  The frontend resamples from the
# AudioContext rate before sending; the backend never guesses.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
FRAME_MS = 32
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 512
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES

MODE_TURN = "turn"
MODE_BARGE_IN = "barge_in"
VALID_MODES = (MODE_TURN, MODE_BARGE_IN)


class ClientMessage(BaseModel):
    """One JSON message from the frontend.

    A single optional-fields model keeps the wire format forgiving: older clients
    that omit a field keep working, and newer clients can add fields without a
    backend release.
    """

    model_config = ConfigDict(extra = "ignore")

    type: str = Field("", description = "Message type")
    protocol: Optional[int] = None
    sampleRate: Optional[int] = None
    channels: Optional[int] = None
    language: Optional[str] = None
    mode: Optional[str] = None
    sttModel: Optional[str] = None
    ttsModel: Optional[str] = None
    voice: Optional[str] = None
    muted: Optional[bool] = None
    id: Optional[str] = None
    seq: Optional[int] = None
    text: Optional[str] = None
    final: Optional[bool] = None
    reason: Optional[str] = None
    echoCancellation: Optional[bool] = None
    endpointFastMs: Optional[int] = None
    endpointExtendedMs: Optional[int] = None
    maxUtteranceMs: Optional[int] = None


def parse_client_message(raw: str) -> Optional[ClientMessage]:
    """Parse one text frame; None means it was not a valid JSON object."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    try:
        return ClientMessage.model_validate(value)
    except ValidationError:
        return None


def event(kind: str, **payload: Any) -> dict[str, Any]:
    """Build one outbound event.  Keeps the wire shape in one place."""
    return {"type": kind, **payload}


def error_event(
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    return event(
        "error",
        code = code,
        message = message,
        retryable = retryable,
    )


def validate_hello(message: ClientMessage) -> str:
    """Return a human-readable reason when a hello is unusable, else an empty string."""
    if message.protocol is not None and message.protocol != PROTOCOL_VERSION:
        return (
            f"Unsupported live protocol {message.protocol}; "
            f"this Studio expects {PROTOCOL_VERSION}."
        )
    if message.sampleRate is not None and message.sampleRate != SAMPLE_RATE:
        return f"Live audio must be {SAMPLE_RATE} Hz; got {message.sampleRate}."
    if message.channels is not None and message.channels != CHANNELS:
        return f"Live audio must be mono; got {message.channels} channels."
    if message.mode is not None and message.mode not in VALID_MODES:
        return f"Unknown live mode {message.mode!r}; use 'turn' or 'barge_in'."
    return ""
