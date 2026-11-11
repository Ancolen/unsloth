# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live Voice engine.

Imports are intentionally kept out of this package initializer: the VAD, STT and
TTS modules import sibling modules, and an eager re-export would make that a
cycle.  Import the concrete module you need (``core.live.session``, etc.).
"""
