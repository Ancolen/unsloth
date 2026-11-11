// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Live Voice screen capture.
//!
//! Linux uses the `org.freedesktop.portal.ScreenCast` portal for consent and a
//! persistent PipeWire stream, then `pipewiresrc`/GStreamer turns it into one
//! JPEG frame per second.  The frontend only ever sees the stable command
//! surface below (`live_screen_start/stop/pause/latest/status`), so the capture
//! backend can change without touching the chat UI.
//!
//! The portal session and the restore token are intentionally process-local:
//! the token lets a later session re-select the same monitor without asking the
//! user again, while closing the app releases the PipeWire stream.  Frames stay
//! in RAM and are only uploaded when the user sends a live turn with the
//! "attach screen" setting enabled.

use serde::{Deserialize, Serialize};

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

#[cfg(target_os = "linux")]
mod imp {
    use std::collections::VecDeque;
    use std::fs;
    use std::io::Write;
    use std::os::fd::{AsRawFd, OwnedFd};
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::PathBuf;
    use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
    use std::sync::{LazyLock, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use ashpd::desktop::screencast::{
        CursorMode, OpenPipeWireRemoteOptions, Screencast, SelectSourcesOptions, SourceType,
    };
    use ashpd::desktop::PersistMode;
    use base64::Engine as _;
    use gst::prelude::*;
    use gstreamer as gst;
    use gstreamer_app as gst_app;
    use tauri::Manager as _;
    use tokio::sync::oneshot;

    use super::{LiveFramePayload, LiveScreenStatus};

    /// Keep a short history so a future change detector can compare frames
    /// without changing the capture thread.  The UI only reads the newest one.
    const MAX_FRAMES: usize = 6;
    const CAPTURE_LONG_SIDE: i32 = 1280;
    const JPEG_QUALITY: i32 = 80;
    const RESTORE_TOKEN_FILE: &str = "live-screencast-token";
    const STATE_UNAVAILABLE: &str = "The live screen state is unavailable.";

    enum CaptureCommand {
        Stop,
    }

    struct CaptureHandle {
        commands: Sender<CaptureCommand>,
        stopped: Receiver<()>,
        thread: Option<JoinHandle<()>>,
    }

    struct PortalHandle {
        shutdown: oneshot::Sender<()>,
    }

    struct LiveScreenInner {
        active: bool,
        starting: bool,
        paused: bool,
        generation: u64,
        source: Option<String>,
        error: Option<String>,
        frame: Option<LiveFramePayload>,
        frames: VecDeque<LiveFramePayload>,
        capture: Option<CaptureHandle>,
        portal: Option<PortalHandle>,
    }

    impl Default for LiveScreenInner {
        fn default() -> Self {
            Self {
                active: false,
                starting: false,
                paused: false,
                generation: 0,
                source: None,
                error: None,
                frame: None,
                frames: VecDeque::with_capacity(MAX_FRAMES),
                capture: None,
                portal: None,
            }
        }
    }

    static LIVE_SCREEN: LazyLock<Mutex<LiveScreenInner>> =
        LazyLock::new(|| Mutex::new(LiveScreenInner::default()));

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

    fn unavailable_status(error: &str) -> LiveScreenStatus {
        LiveScreenStatus {
            active: false,
            paused: false,
            source: None,
            error: Some(error.to_string()),
        }
    }

    fn restore_token_path(app: &tauri::AppHandle) -> Option<PathBuf> {
        app.path()
            .app_config_dir()
            .ok()
            .map(|directory| directory.join(RESTORE_TOKEN_FILE))
    }

    fn load_restore_token(app: &tauri::AppHandle) -> Option<String> {
        let path = restore_token_path(app)?;
        fs::read_to_string(path)
            .ok()
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty())
    }

    fn save_restore_token(app: &tauri::AppHandle, token: &str) {
        let Some(path) = restore_token_path(app) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(mut file) = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
        else {
            return;
        };
        let _ = file.write_all(token.as_bytes());
    }

    /// Read the dimensions out of a JPEG start-of-frame segment.  `jpegenc`
    /// caps are not guaranteed to carry width/height, and reading the header
    /// avoids a second buffer decode in the UI thread.
    fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
        if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
            return None;
        }
        let mut index = 2;
        while index + 3 < bytes.len() {
            if bytes[index] != 0xFF {
                index += 1;
                continue;
            }
            while index < bytes.len() && bytes[index] == 0xFF {
                index += 1;
            }
            if index >= bytes.len() {
                break;
            }
            let marker = bytes[index];
            index += 1;
            if marker == 0xD8 || marker == 0xD9 {
                continue;
            }
            if index + 1 >= bytes.len() {
                break;
            }
            let length = u16::from_be_bytes([bytes[index], bytes[index + 1]]) as usize;
            if length < 2 || index + length > bytes.len() {
                break;
            }
            let is_start_of_frame = matches!(
                marker,
                0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
            );
            if is_start_of_frame && index + 6 < bytes.len() {
                let height = u16::from_be_bytes([bytes[index + 3], bytes[index + 4]]) as u32;
                let width = u16::from_be_bytes([bytes[index + 5], bytes[index + 6]]) as u32;
                if width > 0 && height > 0 {
                    return Some((width, height));
                }
            }
            index += length;
        }
        None
    }

    fn store_frame(generation: u64, jpeg: &[u8]) {
        let Some((width, height)) = jpeg_dimensions(jpeg) else {
            return;
        };
        let payload = LiveFramePayload {
            jpeg_base64: base64::engine::general_purpose::STANDARD.encode(jpeg),
            width,
            height,
            timestamp_ms: now_ms(),
        };
        let Ok(mut inner) = LIVE_SCREEN.lock() else {
            return;
        };
        if inner.generation != generation || !inner.active || inner.paused {
            return;
        }
        inner.frames.push_back(payload.clone());
        while inner.frames.len() > MAX_FRAMES {
            inner.frames.pop_front();
        }
        inner.frame = Some(payload);
    }

    /// Called by the capture thread when the stream fails.  It also releases
    /// the portal session: the frontend sees `active: false` on its next
    /// status poll and can offer a retry.
    fn mark_error(generation: u64, message: String) {
        let (portal, capture) = {
            let Ok(mut inner) = LIVE_SCREEN.lock() else {
                return;
            };
            if inner.generation != generation {
                return;
            }
            inner.active = false;
            inner.starting = false;
            inner.paused = false;
            inner.error = Some(message);
            inner.frame = None;
            inner.frames.clear();
            (inner.portal.take(), inner.capture.take())
        };
        if let Some(portal) = portal {
            let _ = portal.shutdown.send(());
        }
        // Dropping the handle detaches the capture thread that is calling us;
        // it exits immediately after `mark_error`.
        drop(capture);
    }

    fn target_frame_size(size: Option<(i32, i32)>) -> (i32, i32) {
        let Some((width, height)) = size else {
            return (1280, 720);
        };
        let width = width.max(1);
        let height = height.max(1);
        let long_side = width.max(height);
        let scale = (CAPTURE_LONG_SIDE as f64 / long_side as f64).min(1.0);
        let target_width = (((width as f64 * scale).round() as i32).max(2)) & !1;
        let target_height = (((height as f64 * scale).round() as i32).max(2)) & !1;
        (target_width, target_height)
    }

    fn build_pipeline(
        fd: &OwnedFd,
        node_id: u32,
        generation: u64,
        target_width: i32,
        target_height: i32,
    ) -> Result<(gst::Pipeline, gst::Bus), String> {
        gst::init().map_err(|error| format!("GStreamer could not start: {error}"))?;

        if gst::ElementFactory::find("pipewiresrc").is_none() {
            return Err("The GStreamer 'pipewiresrc' element is not installed. \
                 Install the GStreamer PipeWire plugin to share the screen."
                .to_string());
        }
        let source = gst::ElementFactory::make("pipewiresrc")
            .property("fd", fd.as_raw_fd())
            .property("path", node_id.to_string())
            .property("do-timestamp", true)
            .build()
            .map_err(|error| format!("Could not create the PipeWire source: {error}"))?;
        build_pipeline_from_source(source, generation, target_width, target_height)
    }

    /// Assemble the shared encode path.  Kept separate from `build_pipeline` so
    /// a `videotestsrc` smoke test can exercise it without a portal session.
    fn build_pipeline_from_source(
        source: gst::Element,
        generation: u64,
        target_width: i32,
        target_height: i32,
    ) -> Result<(gst::Pipeline, gst::Bus), String> {
        for factory in [
            "videorate",
            "videoscale",
            "videoconvert",
            "jpegenc",
            "appsink",
        ] {
            if gst::ElementFactory::find(factory).is_none() {
                return Err(format!(
                    "The GStreamer '{factory}' element is not installed. \
                     Install the GStreamer base/good plugins to share the screen."
                ));
            }
        }

        let rate = gst::ElementFactory::make("videorate")
            .property("max-rate", 1i32)
            .build()
            .map_err(|error| format!("Could not create the frame limiter: {error}"))?;
        let scale = gst::ElementFactory::make("videoscale")
            .build()
            .map_err(|error| format!("Could not create the frame scaler: {error}"))?;
        let caps = gst::Caps::builder("video/x-raw")
            .field("width", target_width)
            .field("height", target_height)
            .field("pixel-aspect-ratio", gst::Fraction::new(1, 1))
            .build();
        let filter = gst::ElementFactory::make("capsfilter")
            .property("caps", caps)
            .build()
            .map_err(|error| format!("Could not create the frame size filter: {error}"))?;
        let convert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|error| format!("Could not create the frame converter: {error}"))?;
        let encoder = gst::ElementFactory::make("jpegenc")
            .property("quality", JPEG_QUALITY)
            .build()
            .map_err(|error| format!("Could not create the JPEG encoder: {error}"))?;
        let sink = gst::ElementFactory::make("appsink")
            .property("max-buffers", 2u32)
            .property("drop", true)
            .property("sync", false)
            .property("emit-signals", false)
            .build()
            .map_err(|error| format!("Could not create the frame sink: {error}"))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| "The GStreamer appsink element has an unexpected type.".to_string())?;

        sink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    if let Some(buffer) = sample.buffer() {
                        if let Ok(map) = buffer.map_readable() {
                            store_frame(generation, map.as_slice());
                        }
                    }
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        let pipeline = gst::Pipeline::new();
        pipeline
            .add_many([
                &source,
                &rate,
                &scale,
                &filter,
                &convert,
                &encoder,
                sink.upcast_ref(),
            ])
            .map_err(|error| format!("Could not assemble the screen pipeline: {error}"))?;
        gst::Element::link_many([
            &source,
            &rate,
            &scale,
            &filter,
            &convert,
            &encoder,
            sink.upcast_ref(),
        ])
        .map_err(|error| format!("Could not link the screen pipeline: {error}"))?;

        let bus = pipeline
            .bus()
            .ok_or_else(|| "The screen pipeline has no bus.".to_string())?;
        Ok((pipeline, bus))
    }

    fn capture_loop(
        pipeline: gst::Pipeline,
        bus: gst::Bus,
        commands: Receiver<CaptureCommand>,
        generation: u64,
    ) {
        loop {
            match commands.recv_timeout(Duration::from_millis(100)) {
                Ok(CaptureCommand::Stop) => break,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }

            if let Some(message) =
                bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos])
            {
                let text = match message.view() {
                    gst::MessageView::Error(error) => {
                        let debug = error.debug().unwrap_or_default();
                        if debug.is_empty() {
                            format!("The screen stream failed: {}", error.error())
                        } else {
                            format!("The screen stream failed: {} ({debug})", error.error())
                        }
                    }
                    _ => "The screen stream ended.".to_string(),
                };
                mark_error(generation, text);
                break;
            }
        }
        let _ = pipeline.set_state(gst::State::Null);
    }

    pub fn status() -> LiveScreenStatus {
        match LIVE_SCREEN.lock() {
            Ok(inner) => status_of(&inner),
            Err(_) => unavailable_status(STATE_UNAVAILABLE),
        }
    }

    pub fn latest() -> Option<LiveFramePayload> {
        let inner = LIVE_SCREEN.lock().ok()?;
        if !inner.active || inner.paused {
            return None;
        }
        inner.frame.clone()
    }

    pub fn pause(paused: bool) -> LiveScreenStatus {
        let Ok(mut inner) = LIVE_SCREEN.lock() else {
            return unavailable_status(STATE_UNAVAILABLE);
        };
        if inner.active {
            inner.paused = paused;
            if paused {
                inner.frame = None;
            }
        }
        status_of(&inner)
    }

    pub fn stop() -> LiveScreenStatus {
        let (status, capture, portal) = {
            let Ok(mut inner) = LIVE_SCREEN.lock() else {
                return unavailable_status(STATE_UNAVAILABLE);
            };
            inner.active = false;
            inner.starting = false;
            inner.paused = false;
            inner.generation = inner.generation.wrapping_add(1);
            inner.source = None;
            inner.error = None;
            inner.frame = None;
            inner.frames.clear();
            (status_of(&inner), inner.capture.take(), inner.portal.take())
        };

        if let Some(capture) = capture {
            let _ = capture.commands.send(CaptureCommand::Stop);
            // A healthy pipeline reaches NULL in a few milliseconds.  If it
            // does not, detaching is safer than blocking the Tauri command
            // thread; the generation check makes a stale pipeline harmless.
            let _ = capture.stopped.recv_timeout(Duration::from_secs(2));
            if let Some(thread) = capture.thread {
                if thread.is_finished() {
                    let _ = thread.join();
                }
            }
        }
        if let Some(portal) = portal {
            let _ = portal.shutdown.send(());
        }
        status
    }

    pub async fn start(app: tauri::AppHandle) -> Result<LiveScreenStatus, String> {
        let generation = {
            let mut inner = LIVE_SCREEN
                .lock()
                .map_err(|_| STATE_UNAVAILABLE.to_string())?;
            if inner.active {
                return Ok(status_of(&inner));
            }
            if inner.starting {
                return Err("Screen sharing is already starting.".to_string());
            }
            inner.starting = true;
            inner.paused = false;
            inner.generation = inner.generation.wrapping_add(1);
            inner.error = None;
            inner.source = None;
            inner.frame = None;
            inner.frames.clear();
            // Normally these are already empty; dropping them still closes a
            // previous portal session because the oneshot receiver treats a
            // dropped sender as shutdown.
            inner.capture.take();
            inner.portal.take();
            inner.generation
        };

        let restore_token = load_restore_token(&app);
        let proxy = match Screencast::new().await {
            Ok(proxy) => proxy,
            Err(error) => {
                return fail_start(
                    generation,
                    format!("Screen sharing is unavailable: {error}"),
                );
            }
        };
        let session = match proxy.create_session(Default::default()).await {
            Ok(session) => session,
            Err(error) => {
                return fail_start(
                    generation,
                    format!("Screen sharing could not create a portal session: {error}"),
                );
            }
        };

        let mut options = SelectSourcesOptions::default()
            .set_sources(SourceType::MONITOR | SourceType::WINDOW)
            .set_multiple(false)
            .set_cursor_mode(CursorMode::EMBEDDED)
            .set_persist_mode(PersistMode::ExplicitlyRevoked);
        if let Some(token) = restore_token.as_deref() {
            options = options.set_restore_token(token);
        }
        if let Err(error) = proxy.select_sources(&session, options).await {
            let _ = session.close().await;
            return fail_start(
                generation,
                format!("Screen sharing could not select a source: {error}"),
            );
        }

        let streams = match proxy.start(&session, None, Default::default()).await {
            Ok(request) => match request.response() {
                Ok(streams) => streams,
                Err(error) => {
                    let _ = session.close().await;
                    return fail_start(
                        generation,
                        format!("Screen sharing was cancelled: {error}"),
                    );
                }
            },
            Err(error) => {
                let _ = session.close().await;
                return fail_start(
                    generation,
                    format!("Screen sharing could not start: {error}"),
                );
            }
        };

        let Some(stream) = streams.streams().first() else {
            let _ = session.close().await;
            return fail_start(
                generation,
                "The portal started without returning a screen stream.".to_string(),
            );
        };
        let node_id = stream.pipe_wire_node_id();
        let stream_size = stream.size();
        let (target_width, target_height) = target_frame_size(stream_size);
        let source = stream_size
            .map(|(width, height)| format!("screen {width}x{height}"))
            .unwrap_or_else(|| "screen".to_string());
        if let Some(token) = streams.restore_token() {
            save_restore_token(&app, token);
        }

        let fd = match proxy
            .open_pipe_wire_remote(&session, OpenPipeWireRemoteOptions::default())
            .await
        {
            Ok(fd) => fd,
            Err(error) => {
                let _ = session.close().await;
                return fail_start(
                    generation,
                    format!("Screen sharing could not open the PipeWire stream: {error}"),
                );
            }
        };

        let cancelled = LIVE_SCREEN
            .lock()
            .map(|inner| inner.generation != generation || !inner.starting)
            .unwrap_or(true);
        if cancelled {
            let _ = session.close().await;
            return Err("Screen sharing was cancelled.".to_string());
        }

        let (commands_tx, commands_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let (started_tx, started_rx) = oneshot::channel::<Result<(), String>>();
        let thread = match thread::Builder::new()
            .name("live-screen-capture".to_string())
            .spawn(move || {
                match build_pipeline(&fd, node_id, generation, target_width, target_height) {
                    Ok((pipeline, bus)) => match pipeline.set_state(gst::State::Playing) {
                        Ok(_) => {
                            let _ = started_tx.send(Ok(()));
                            capture_loop(pipeline, bus, commands_rx, generation);
                        }
                        Err(error) => {
                            let message = format!("The screen pipeline could not start: {error}");
                            let _ = started_tx.send(Err(message.clone()));
                            mark_error(generation, message);
                        }
                    },
                    Err(message) => {
                        let _ = started_tx.send(Err(message.clone()));
                        mark_error(generation, message);
                    }
                }
                drop(fd);
                let _ = stopped_tx.send(());
            }) {
            Ok(thread) => thread,
            Err(error) => {
                let _ = session.close().await;
                return fail_start(
                    generation,
                    format!("Could not start the screen capture thread: {error}"),
                );
            }
        };

        match started_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(message)) => {
                let _ = session.close().await;
                return Err(message);
            }
            Err(_) => {
                let _ = session.close().await;
                return Err("The screen capture thread stopped before it started.".to_string());
            }
        }

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let error = {
            let Ok(mut inner) = LIVE_SCREEN.lock() else {
                let _ = commands_tx.send(CaptureCommand::Stop);
                let _ = session.close().await;
                return Err(STATE_UNAVAILABLE.to_string());
            };
            if inner.generation != generation || inner.error.is_some() {
                let error = inner
                    .error
                    .clone()
                    .unwrap_or_else(|| "Screen sharing was cancelled.".to_string());
                inner.starting = false;
                Some(error)
            } else {
                inner.active = true;
                inner.starting = false;
                inner.source = Some(source);
                inner.capture = Some(CaptureHandle {
                    commands: commands_tx,
                    stopped: stopped_rx,
                    thread: Some(thread),
                });
                inner.portal = Some(PortalHandle {
                    shutdown: shutdown_tx,
                });
                None
            }
        };

        if let Some(error) = error {
            let _ = session.close().await;
            return Err(error);
        }

        tauri::async_runtime::spawn(async move {
            let _ = shutdown_rx.await;
            let _ = session.close().await;
        });

        Ok(status())
    }

    fn fail_start(generation: u64, message: String) -> Result<LiveScreenStatus, String> {
        if let Ok(mut inner) = LIVE_SCREEN.lock() {
            if inner.generation == generation {
                inner.starting = false;
                inner.active = false;
                inner.error = Some(message.clone());
            }
        }
        Err(message)
    }

    /// Deterministic state tests do not need a portal, a desktop or GStreamer.
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        static TEST_LOCK: Mutex<()> = Mutex::new(());

        fn mini_jpeg(width: u16, height: u16) -> Vec<u8> {
            let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
            bytes.extend_from_slice(&height.to_be_bytes());
            bytes.extend_from_slice(&width.to_be_bytes());
            bytes.extend_from_slice(&[0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
            bytes.extend_from_slice(&[0xFF, 0xD9]);
            bytes
        }

        #[test]
        fn frame_store_respects_generation_and_pause() {
            let _guard = TEST_LOCK.lock().expect("test lock");
            let jpeg = mini_jpeg(320, 180);
            assert_eq!(jpeg_dimensions(&jpeg), Some((320, 180)));
            assert_eq!(jpeg_dimensions(&[0xFF, 0xD8, 0x00]), None);

            let generation = {
                let mut inner = LIVE_SCREEN.lock().expect("state");
                inner.active = true;
                inner.starting = false;
                inner.paused = false;
                inner.generation = 41;
                inner.frames.clear();
                inner.frame = None;
                inner.generation
            };

            store_frame(generation - 1, &jpeg);
            assert!(latest().is_none());
            store_frame(generation, &jpeg);
            let frame = latest().expect("current frame");
            assert_eq!((frame.width, frame.height), (320, 180));
            assert!(frame.jpeg_base64.starts_with("/9j/"));

            pause(true);
            assert!(latest().is_none());
            pause(false);
            let _ = stop();
            assert!(latest().is_none());
        }

        #[test]
        #[ignore = "requires the xdg-desktop-portal ScreenCast interface"]
        fn portal_creates_and_closes_a_screencast_session() {
            let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
            runtime.block_on(async {
                let proxy = Screencast::new().await.expect("ScreenCast portal");
                let session = proxy
                    .create_session(Default::default())
                    .await
                    .expect("portal session");
                proxy
                    .select_sources(
                        &session,
                        SelectSourcesOptions::default()
                            .set_sources(SourceType::MONITOR | SourceType::WINDOW)
                            .set_multiple(false)
                            .set_cursor_mode(CursorMode::EMBEDDED)
                            .set_persist_mode(PersistMode::ExplicitlyRevoked),
                    )
                    .await
                    .expect("source selection");
                session.close().await.expect("session close");
            });
        }

        #[test]
        #[ignore = "requires the GStreamer video and JPEG plugins"]
        fn gstreamer_pipeline_encodes_jpeg_frames() {
            let _guard = TEST_LOCK.lock().expect("test lock");
            gst::init().expect("GStreamer init");
            let generation = {
                let mut inner = LIVE_SCREEN.lock().expect("state");
                inner.active = true;
                inner.starting = false;
                inner.paused = false;
                inner.generation = 77;
                inner.frames.clear();
                inner.frame = None;
                inner.generation
            };

            let source = gst::ElementFactory::make("videotestsrc")
                .property("num-buffers", 2i32)
                .build()
                .expect("videotestsrc");
            let (pipeline, bus) =
                build_pipeline_from_source(source, generation, 1280, 960).expect("pipeline");
            pipeline
                .set_state(gst::State::Playing)
                .expect("pipeline playing");
            let message = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(8),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            );
            let _ = pipeline.set_state(gst::State::Null);
            assert_eq!(
                message.as_ref().map(|message| message.type_()),
                Some(gst::MessageType::Eos),
                "the test pipeline did not finish cleanly"
            );

            let frame = latest().expect("an encoded frame");
            assert_eq!((frame.width, frame.height), (1280, 960));
            assert!(frame.jpeg_base64.starts_with("/9j/"));
            let _ = stop();
        }

        #[test]
        fn target_size_caps_the_long_side_and_keeps_the_aspect() {
            assert_eq!(target_frame_size(Some((2560, 1440))), (1280, 720));
            assert_eq!(target_frame_size(Some((1920, 1080))), (1280, 720));
            assert_eq!(target_frame_size(Some((1080, 1920))), (720, 1280));
            assert_eq!(target_frame_size(Some((800, 600))), (800, 600));
            assert_eq!(target_frame_size(None), (1280, 720));
        }
    }
}

#[cfg(target_os = "linux")]
use imp as platform;

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::{LiveFramePayload, LiveScreenStatus};

    const UNSUPPORTED: &str = "Screen sharing is only available in Unsloth Desktop on Linux.";

    fn unsupported() -> LiveScreenStatus {
        LiveScreenStatus {
            active: false,
            paused: false,
            source: None,
            error: Some(UNSUPPORTED.to_string()),
        }
    }

    pub fn status() -> LiveScreenStatus {
        unsupported()
    }

    pub fn latest() -> Option<LiveFramePayload> {
        None
    }

    pub fn pause(_paused: bool) -> LiveScreenStatus {
        unsupported()
    }

    pub fn stop() -> LiveScreenStatus {
        unsupported()
    }

    pub async fn start(_app: tauri::AppHandle) -> Result<LiveScreenStatus, String> {
        Err(UNSUPPORTED.to_string())
    }
}

#[tauri::command]
pub fn live_screen_status() -> LiveScreenStatus {
    platform::status()
}

#[tauri::command]
pub async fn live_screen_start(app: tauri::AppHandle) -> Result<LiveScreenStatus, String> {
    platform::start(app).await
}

#[tauri::command]
pub fn live_screen_stop() -> LiveScreenStatus {
    platform::stop()
}

#[tauri::command]
pub fn live_screen_pause(paused: bool) -> LiveScreenStatus {
    platform::pause(paused)
}

#[tauri::command]
pub fn live_screen_latest() -> Option<LiveFramePayload> {
    platform::latest()
}
