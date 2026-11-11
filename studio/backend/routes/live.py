# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live Voice API: status plus one bidirectional audio WebSocket.

The route is deliberately thin.  It authenticates, rejects non-UI callers, moves
text/binary frames, and lets `LiveSession` own the audio pipeline.  Chat remains
in the frontend's normal assistant-ui runtime.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from auth.authentication import get_current_subject, subject_for_header_or_query_token
from core.live.models import live_enabled, status
from core.live.session import LiveSession
from core.live.stt import build_stt
from core.live.tts import build_tts
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()

LOCAL_ORIGIN_HOSTS = frozenset({"localhost", "127.0.0.1", "tauri.localhost"})


def _origin_allowed(origin: Optional[str]) -> bool:
    """Live audio is local UI only: same-machine pages plus the Tauri shell."""
    if not origin:
        # Non-browser clients (tests, native helpers) carry a token but no Origin.
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme.lower() == "tauri":
        return True
    return (parsed.hostname or "").lower() in LOCAL_ORIGIN_HOSTS


def _bearer_token(websocket: WebSocket) -> Optional[str]:
    token = (websocket.query_params.get("token") or "").strip()
    if token:
        return token
    header = websocket.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


@router.get("/status")
async def live_status(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Capabilities and the feature flag; safe to poll before opening a session."""
    return status().to_dict()


@router.websocket("/ws")
async def live_websocket(websocket: WebSocket) -> None:
    if not live_enabled():
        await websocket.close(code = 4403)
        return
    if not _origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code = 4403)
        return

    token = _bearer_token(websocket)
    if not token:
        await websocket.close(code = 4401)
        return
    try:
        await subject_for_header_or_query_token(websocket, token)
    except HTTPException:
        await websocket.close(code = 4401)
        return
    except Exception as exc:  # noqa: BLE001 - auth failure must not leak internals
        logger.warning("live.auth_failed", error = str(exc))
        await websocket.close(code = 4401)
        return

    await websocket.accept()

    async def send_json(payload: dict) -> None:
        await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        await websocket.send_bytes(payload)

    stt = build_stt(websocket.app, token)
    tts = build_tts(websocket.app, token)
    session = LiveSession(
        send_json = send_json,
        send_bytes = send_bytes,
        stt = stt,
        tts = tts,
    )
    await session.start()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            binary = message.get("bytes")
            if binary is not None:
                session.push_audio(binary)
                continue
            text = message.get("text")
            if text is not None:
                await session.handle_text(text)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - the session boundary owns the error
        logger.exception("live.session_failed", error = str(exc))
        try:
            await websocket.close(code = 1011)
        except Exception:  # noqa: BLE001 - already closed
            pass
    finally:
        await session.close()
