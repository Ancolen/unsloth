// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Live Voice screen capture.
//!
//! Phase 0 is a deterministic fixture: the commands, state machine and frontend
//! contract are real, so the live bar, preview and attachment path can be built
//! and tested.  Phase 3 replaces the fixture with the Linux xdg-desktop-portal
//! ScreenCast + PipeWire/GStreamer capture path.  The public command surface
//! (`live_screen_start/stop/pause/latest/status`) is intentionally stable so the
//! frontend does not change when the capture backend lands.

use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A small dark placeholder JPEG; only used until the portal backend lands.
const STUB_JPEG_BASE64: &str = "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";

#[derive(Default)]
struct LiveScreenInner {
    active: bool,
    paused: bool,
    source: Option<String>,
    error: Option<String>,
    frame: Option<LiveFramePayload>,
}

static LIVE_SCREEN: LazyLock<Mutex<LiveScreenInner>> =
    LazyLock::new(|| Mutex::new(LiveScreenInner::default()));

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveFramePayload {
    pub jpeg_base64: String,
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveScreenStatus {
    pub active: bool,
    pub paused: bool,
    pub source: Option<String>,
    pub error: Option<String>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn status_of(inner: &LiveScreenInner) -> LiveScreenStatus {
    LiveScreenStatus {
        active: inner.active,
        paused: inner.paused,
        source: inner.source.clone(),
        error: inner.error.clone(),
    }
}

#[tauri::command]
pub fn live_screen_status() -> LiveScreenStatus {
    match LIVE_SCREEN.lock() {
        Ok(inner) => status_of(&inner),
        Err(_) => LiveScreenStatus {
            active: false,
            paused: false,
            source: None,
            error: Some("The live screen state is unavailable.".to_string()),
        },
    }
}

#[tauri::command]
pub fn live_screen_start() -> Result<LiveScreenStatus, String> {
    let mut inner = LIVE_SCREEN
        .lock()
        .map_err(|_| "The live screen state is unavailable.".to_string())?;
    inner.active = true;
    inner.paused = false;
    inner.source = Some("fixture".to_string());
    inner.error = None;
    inner.frame = Some(LiveFramePayload {
        jpeg_base64: STUB_JPEG_BASE64.to_string(),
        width: 320,
        height: 180,
        timestamp_ms: now_ms(),
    });
    Ok(status_of(&inner))
}

#[tauri::command]
pub fn live_screen_stop() -> LiveScreenStatus {
    match LIVE_SCREEN.lock() {
        Ok(mut inner) => {
            inner.active = false;
            inner.paused = false;
            inner.source = None;
            inner.error = None;
            inner.frame = None;
            status_of(&inner)
        }
        Err(_) => LiveScreenStatus {
            active: false,
            paused: false,
            source: None,
            error: Some("The live screen state is unavailable.".to_string()),
        },
    }
}

#[tauri::command]
pub fn live_screen_pause(paused: bool) -> LiveScreenStatus {
    match LIVE_SCREEN.lock() {
        Ok(mut inner) => {
            inner.paused = paused;
            status_of(&inner)
        }
        Err(_) => LiveScreenStatus {
            active: false,
            paused,
            source: None,
            error: Some("The live screen state is unavailable.".to_string()),
        },
    }
}

#[tauri::command]
pub fn live_screen_latest() -> Option<LiveFramePayload> {
    let inner = LIVE_SCREEN.lock().ok()?;
    if !inner.active || inner.paused {
        return None;
    }
    inner.frame.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_pause_and_stop_are_reflected_in_status() {
        let _ = live_screen_stop();
        let started = live_screen_start().expect("start");
        assert!(started.active);
        assert!(live_screen_latest().is_some());
        let paused = live_screen_pause(true);
        assert!(paused.paused);
        assert!(live_screen_latest().is_none());
        let stopped = live_screen_stop();
        assert!(!stopped.active);
        assert!(live_screen_latest().is_none());
    }
}
