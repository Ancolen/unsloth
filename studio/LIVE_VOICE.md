# Live Voice in Unsloth Studio

Status: design + implementation plan
Owner: Ancolen
Target: Unsloth Desktop (Linux first), Studio web UI where possible
Feature flag: `UNSLOTH_STUDIO_LIVE_VOICE=1`

**Implementation status (2026-11-12):**

- Backend live engine (`core/live/`), `/api/live/ws`, energy VAD, endpointing,
  in-process STT/TTS, barge-in and the protocol tests are implemented.
- Frontend Live button/bar, microphone capture, 16 kHz framing, WebAudio playback,
  streamed sentence TTS and the assistant-ui bridge are implemented.
- Linux screen capture is implemented with xdg-desktop-portal ScreenCast +
  `OpenPipeWireRemote` + GStreamer (`pipewiresrc ! videorate ! videoscale ! jpegenc`),
  including the restore token, a latest-frame ring buffer and pause/stop. It still
  needs hardware-in-the-loop validation on GNOME Wayland and installer/packaging
  checks.
- Pending: Silero/Smart Turn asset auto-download, the Live model catalog/settings
  UI, partial transcripts, and Phase 5 hardening.

## 1. Goal

Add a first-class, hands-free voice conversation mode to the normal Studio chat:

- A **Live** button next to the existing composer microphone.
- Pressing it opens a live bar over the composer: waveform, state, live transcript,
  screen-sharing thumbnail, mute / pause / stop.
- Speech is transcribed locally, sent as a normal chat message, and the assistant
  reply is spoken sentence by sentence while it streams.
- The user can interrupt the assistant (barge-in). The default mode is turn-based;
  barge-in is opt-in and gated by echo-cancellation quality.
- Screen sharing is a first-class toggle: the latest screen frame is attached to
  the next chat turn when the selected model supports vision.
- Models (chat/vision, STT, TTS, VAD/endpoint) are downloaded through Studio's
  existing Hugging Face model manager. Nothing depends on an external application.

The feature is implemented **inside** Studio:

| Layer | Location | Responsibility |
|---|---|---|
| UI | `studio/frontend/src/features/live/` | Live button/bar, audio capture/playback, chat glue, screen preview, settings |
| Engine | `studio/backend/core/live/` + `routes/live.py` | WebSocket session, VAD/endpointing, STT, TTS queue, model readiness |
| Screen | `studio/src-tauri/src/live_screen.rs` | Linux portal + PipeWire capture, JPEG frames |
| Chat | existing `features/chat/` | Unchanged; live turns are ordinary chat messages |

## 2. Non-goals (v1)

- True simultaneous full-duplex conversation (both sides talking at once). v1 ships
  turn-based listening and barge-in, which is the practical "natural conversation"
  mode. A native realtime omni provider is a later phase.
- Word-by-word partial transcripts. v1 shows listening state, waveform and the final
  transcript after endpointing. Partial captions are a follow-up.
- Windows/macOS screen capture. v1 screen capture is Linux/GNOME Wayland first.
- Cloud STT/TTS as defaults. Local engines are the default; custom providers may be
  added later.
- Replacing Studio's chat runtime, tool loop, MCP, RAG, history or model manager.

## 3. User experience

### 3.1 Composer

```text
[ + ]  [ image ]  [ mic ]  [ Live ]  [ send ]
```

- `mic` remains one-shot dictation.
- `Live` starts the continuous session. It is disabled while dictation is active and
  hidden in Compare mode and on unsupported platforms.
- When live is active the mic button is disabled and `AutoReadAloud` is suppressed so
  the live TTS controller is the only speaker.
- If the composer has a draft when live starts, the draft is saved and restored when
  live stops.

### 3.2 Live bar

```text
+------------------------------------------------------------------+
| * Listening   ~ waveform ~   00:04   [screen on]   [mute] [stop] |
| "summarize what is on my screen..."                              |
+------------------------------------------------------------------+
```

States:

| State | UI | Notes |
|---|---|---|
| `setup` | Setup card | Missing STT/TTS models; download buttons |
| `connecting` | Spinner | Mic permission, WebSocket, model warmup |
| `listening` | Waveform, "Listening" | VAD armed |
| `speech` | Red dot, timer | User speaking |
| `transcribing` | Spinner | Endpoint reached, STT in flight |
| `thinking` | Three dots | Chat request running |
| `speaking` | Speaker icon | TTS playing; barge-in mode keeps the mic armed |
| `interrupted` | "Interrupted" badge | User barged in; chat run cancelled |
| `error` | Actionable error | Mic/model/portal failure |

### 3.3 Modes

1. **Turn-based (default):** mic frames are ignored while TTS is playing. No echo risk.
2. **Barge-in:** mic stays open while TTS plays. Sustained user speech stops playback,
   cancels the in-flight chat run and starts a new utterance. Recommended with
   headphones or working echo cancellation.
3. **True full-duplex (later):** simultaneous speech, handled by a native realtime
   audio model or backend AEC. Not in v1.

### 3.4 Settings

`Settings -> Live`:

- **Voice:** microphone device, STT language (`tr` default), STT model, TTS engine /
  voice, rate, volume, test microphone, test speaker.
- **Conversation:** default mode, endpoint silence, VAD sensitivity, max utterance,
  "interrupt while thinking".
- **Screen:** default screen sharing, capture rate (1 fps), cloud frame consent
  (off by default), privacy pause.
- **Models:** live readiness checklist, download/progress, VRAM estimate, license
  acceptance.
- **Diagnostics:** last-turn latency, barge-in count, echo test result.

## 4. Architecture

```text
+--------------------------- Unsloth Studio (single app) ----------------------------+
|                                                                                    |
| React frontend                                                                     |
|   existing chat thread / composer / attachments / model picker                     |
|   features/live/                                                                   |
|     LiveButton  LiveBar  LiveController  LiveAudioCapture  LiveAudioPlayer         |
|     LiveTts  LiveScreen  LiveStore  LiveSettings                                   |
|                         |  WebSocket /api/live/ws                                  |
|                         v                                                          |
| FastAPI backend                                                                    |
|   routes/live.py                                                                   |
|   core/live/{session,vad,endpoint,stt,tts,models,audio,protocol,events}.py         |
|   existing chat /api/inference/chat/completions        (unchanged)                 |
|   existing whisper.cpp / audio.cpp engines             (reused)                    |
|   existing Hugging Face download manager               (reused)                    |
|                         |                                                          |
| Tauri (Rust)                                                                       |
|   live_screen.rs: xdg-desktop-portal ScreenCast + PipeWire + JPEG                  |
|                         | invoke                                                   |
|                         v                                                          |
|   frontend preview / chat attachment                                               |
+------------------------------------------------------------------------------------+
```

### 4.1 Why this split

- **Audio I/O in the frontend.** Studio already captures microphone audio with
  `PcmRecorder` (`ScriptProcessorNode` fallback included) and requests
  `echoCancellation` / `noiseSuppression`. Keeping capture and playback in the
  WebView avoids adding Python audio-device dependencies to Studio's managed
  environment and works in both the desktop and web UI.
- **VAD/STT/TTS in the backend.** These need models, GPU/CPU scheduling and the
  existing audio engines. The backend also owns model readiness and lifecycle.
- **Screen capture in Tauri/Rust.** WebKitGTK does not implement `getDisplayMedia`
  on Linux. The portal + PipeWire stack belongs to the desktop shell, where GTK and
  GStreamer dependencies already live. This keeps PyGObject and GStreamer out of the
  managed Python environment.
- **Chat stays as-is.** Live transcripts enter the normal assistant-ui runtime as
  user messages. All existing Studio features (tools, MCP, RAG, history, model
  manager, prompt queue) keep working.

### 4.2 Alternatives rejected

| Alternative | Reason |
|---|---|
| Separate companion service | Requirement is one integrated Studio app; no second runtime or package |
| Python backend audio I/O (sounddevice/PipeWire) | Adds device permissions and dependencies; frontend capture already works. May return later as an AEC fallback |
| Python screen capture | PyGObject/GStreamer in the managed env is a packaging burden |
| Backend-owned chat loop | Loses Studio's tool/MCP/history/model infrastructure; creates two brains |
| Browser `getDisplayMedia` | Not available in WebKitGTK on Linux |

## 5. Backend live engine

### 5.1 Files

```text
studio/backend/routes/live.py
studio/backend/core/live/
  __init__.py
  protocol.py     typed WebSocket messages
  session.py      LiveSession lifecycle and state
  audio.py        PCM framing, resampling, RMS, WAV encoding
  vad.py          Silero VAD v6.2.3 ONNX + energy fallback
  endpoint.py     Smart Turn v3.2 ONNX + silence/semantic endpointing
  stt.py          adapter over Studio's STT engines
  tts.py          sentence queue, prefetch, cancellation
  models.py       live model catalog and warmup/readiness
  events.py       event types and serialization
```

`main.py` registers the router once:

```python
app.include_router(live_router, prefix="/api/live", tags=["live"])
```

### 5.2 WebSocket protocol

Endpoint: `WS /api/live/ws`

Authentication: UI session only. The frontend appends its access token as
`?token=...`; the backend validates it with the existing
`subject_for_header_or_query_token()` helper. Origin is checked. API keys are
rejected for live routes. Remote/LAN access is off by default.

Client -> server:

| Message | Body |
|---|---|
| `hello` | `{protocol, sampleRate: 16000, channels: 1, language, mode, sttModel, ttsModel, voice}` |
| `start` / `stop` | session control |
| `mute` | `{muted}` |
| `mode` | `{mode: "turn" \| "barge_in"}` |
| `speak` | `{id, seq, text, final}` |
| `tts_cancel` | `{id?}` |
| `playback_started` / `playback_finished` | `{id}` |
| `ping` | keepalive |
| binary frames | 16 kHz mono PCM int16, 32 ms (512 samples) |

Server -> client:

| Message | Body |
|---|---|
| `ready` | `{models, echoCancellation, sampleRate}` |
| `state` | `{state}` |
| `level` | `{rms}` at 10 Hz |
| `speech_started` / `speech_ended` | `{atMs}` |
| `partial` | `{text}` (reserved) |
| `transcript` | `{text, utteranceId, language, durationMs}` |
| `endpoint` | `{reason}` |
| `barge_in` | `{utteranceId}` |
| `tts_start` / `tts_end` | `{id}` |
| `tts_audio` | JSON header `{id, seq, format, sampleRate, text}` + binary body |
| `screen_state` | `{active, paused, source}` when screen capture is backend-owned |
| `error` | `{code, message, retryable}` |

All messages are JSON text except binary audio. Unknown message types are ignored
with a warning, never a disconnect.

### 5.3 Audio pipeline

```text
PCM 16 kHz frames
  -> pre-roll ring buffer (300 ms)
  -> Silero VAD
       speech_started
       utterance buffer
       Smart Turn + silence
         -> endpoint (300 ms fast / 900 ms extended)
         -> trim trailing silence
         -> STT
         -> transcript
```

- Pre-roll: 300 ms before speech onset so word starts are not clipped.
- Endpoint: Smart Turn probability >= 0.80 plus 300 ms silence is the fast path;
  otherwise wait up to 900 ms. A max utterance guard (default 30 s) forces a cut.
- VAD fallback: if the ONNX model cannot load, use an energy VAD and surface a
  visible warning; never fail silently.
- Barge-in: when `mode=barge_in` and playback is active, VAD continues. Sustained
  speech for 120 ms above the threshold emits `barge_in`. In `turn` mode, frames
  during playback are discarded.
- Echo policy: the frontend reports `track.getSettings().echoCancellation` in
  `hello`. If false, the UI defaults to turn mode or warns that headphones are
  recommended. The backend raises barge-in thresholds when playback is active.
- Bounded queues: if the client outruns the server, drop the oldest audio frames
  and reset the utterance buffer rather than growing without bound.

### 5.4 STT

- Reuse Studio's existing engines:
  - whisper.cpp `large-v3-turbo` (default, Turkish),
  - audio.cpp Qwen3-ASR 0.6B / Parakeet (alternatives),
  - Transformers ASR for custom repos.
- `LiveStt` calls the engine functions underneath the existing
  `/api/inference/audio/transcriptions` route directly; no loopback HTTP.
- Warm the model at session start, in a worker thread; the event loop must not block.
- Language defaults to `tr`; `auto` is available.
- Partial transcripts are not part of v1. The UI shows listening/transcribing states.

### 5.5 TTS

The frontend converts streamed assistant text into sentences and sends `speak`
messages. The backend:

- keeps a sentence queue keyed by `id`/`seq`,
- synthesizes the first sentence immediately,
- prefetches the next sentence while the current one plays,
- cancels queued and in-flight work on `tts_cancel` (audio.cpp exposes
  `cancel_event`; Piper/XTTS are aborted between sentences),
- sends complete WAV payloads as binary WebSocket messages.

Engines:

| Engine | Default | Notes |
|---|---|---|
| Piper Turkish (`tr_TR-fettah-medium`) | yes | CPU, low latency, ~0 VRAM |
| XTTS-v2 | optional | GPU, ~1.8 GB VRAM, CPML non-commercial license gate |
| Studio audio.cpp models | optional | Kokoro/Piper EN/Qwen3-TTS/CosyVoice/IndexTTS2; test Turkish first |

Text is cleaned in the frontend (`markdownToSpeechText`): code blocks are skipped,
tool-call JSON and reasoning are never spoken. v1 uses WAV; Opus can follow.

### 5.6 Model readiness and VRAM

`LiveModelManager`:

- reports STT/TTS/VAD/endpoint readiness in `ready` or `needs_setup`,
- returns download actions for missing models,
- warms STT/TTS in parallel at session start and emits progress,
- optionally unloads idle models after N minutes,
- integrates with Studio's existing model/VRAM status surfaces.

Recommended 8 GB configuration:

| Role | Model | VRAM |
|---|---|---|
| Chat + vision | Qwen3-VL 4B Instruct GGUF Q4_K_M + mmproj | ~3.1 GB |
| STT | whisper.cpp large-v3-turbo (q5) | ~1.1 GB GPU or 0 (CPU) |
| TTS | Piper `tr_TR-fettah-medium` | ~0 GB (CPU) |
| VAD + endpoint | Silero + Smart Turn ONNX | ~0.1 GB (CPU) |
| **Total** | | **~4.2 GB** |
| TTS alternative | XTTS-v2 | +1.8 GB GPU (total ~6.0 GB) |

If the chat model is larger, move STT/TTS to CPU and show a fit warning.

### 5.7 Security

- UI session only; API keys cannot open a live session.
- Origin checked; remote/LAN access disabled by default.
- Raw audio is never written to disk. Only transcripts become chat messages.
- Screen frames become chat attachments only when the user sends them. Cloud frame
  upload requires an explicit setting.
- Mic/screen indicators are always visible while active.
- No audio or screen content in logs.

## 6. Frontend live controller

### 6.1 Files

```text
studio/frontend/src/features/live/
  live-button.tsx
  live-bar.tsx
  live-controller.ts
  live-audio-capture.ts
  live-audio-player.ts
  live-tts.ts
  live-screen.ts
  live-store.ts
  live-settings.tsx
  live-types.ts
  live-i18n.ts (or keys in locales)
```

Small anchor edits:

- `components/assistant-ui/thread.tsx`:
  - `ComposerRightControls` (`~line 7623`) gets `<LiveButton />` next to the mic,
    inside the `ComposerPrimitive.If dictation={false}` block.
  - the composer area gets `<LiveBar />` next to `ChatDictationBar`.
- `components/assistant-ui/auto-read-aloud.tsx`: skip auto-read while live is active.
- `features/settings/tabs/voice-tab.tsx`: add the Live settings section/tab.
- `features/settings/settings-search.ts`: register new keys.
- `i18n/locales/en.ts` and `tr.ts`: new strings. Other locales may fall back.

### 6.2 State machine

```text
IDLE
 -> CONNECTING (permission + WS + model warmup)
 -> READY
 -> LISTENING
 -> SPEECH
 -> TRANSCRIBING
 -> THINKING
 -> SPEAKING
 -> LISTENING
any -> ERROR -> IDLE
SPEAKING + barge_in -> INTERRUPTED -> LISTENING
```

- Turn mode: capture frames are not sent (or are ignored) while speaking.
- Barge-in: on `barge_in`, stop playback first, then `thread.cancelRun()`, then
  listen for the new utterance.
- Thinking: if the user speaks, the setting decides whether to wait or cancel the
  run and send the new utterance.

### 6.3 Audio capture

- `getUserMedia({ audio: { deviceId, echoCancellation: true,
  noiseSuppression: true, autoGainControl: true } })`.
- Reuse Studio's `PcmRecorder` pattern (`ScriptProcessorNode`, with the existing
  WebKitGTK fallback behavior).
- Resample the AudioContext rate (44.1/48 kHz) to 16 kHz.
- Send 32 ms (512 sample) int16 frames over the WebSocket.
- Read `track.getSettings().echoCancellation` and include it in `hello`.
- Surface permission/device errors with actions.

### 6.4 Audio playback

- `LiveAudioPlayer` uses WebAudio `AudioBufferSourceNode` with a `nextStartTime`
  queue for gapless sentence playback.
- Decode WAV payloads with `decodeAudioData`.
- `cancel()` stops all sources, clears the queue and applies a 20 ms fade-out.
- Send `playback_started` / `playback_finished` for barge-in thresholds.
- v1 uses the default output device; device selection can follow.

### 6.5 Chat integration

- Transcript -> message:
  `aui.composer().setText(transcript)`; if screen sharing is on and a frame is
  available, `await aui.composer().addAttachment(file)`; then
  `aui.composer().send()`.
- Streamed reply -> TTS:
  observe the last assistant message with `useAuiState`; use
  `markdownToSpeechText` + `splitIntoSentences` + `planTtsChunk` to send complete
  sentences. Do not speak reasoning, tool-call JSON or tool results.
- Barge-in: `aui.thread().cancelRun()`, stop playback, `tts_cancel`.
- Auto-read is temporarily disabled while live is active; restore the previous value.
- Dictation is disabled while live is active.
- Composer draft is saved before live starts and restored after it stops.
- Live turns are normal messages. Optional metadata: `{live: true, interrupted,
  audioMs}`. Studio's chat history supports custom metadata.
- If the chat model is not loaded, the Live button prompts to load one. If screen
  sharing is on and the model has no vision input, `VisionImageAdapter` rejects the
  attachment; show an actionable error and send text only or offer a vision model.
- External providers: frame upload requires `allow_cloud_screen`; otherwise attach
  nothing and show a badge.

### 6.6 Screen sharing (frontend)

- Live bar screen toggle calls Tauri `invoke("live_screen_start")`.
- GNOME portal picker appears on first use.
- Preview polls `invoke("live_screen_latest")` at 1 fps.
- On send, the same JPEG becomes a `File` and is attached to the message.
- Vision capability and cloud consent checks are performed before attaching.
- "What changed?" can attach the previous keyframe plus the current frame later.

## 7. Screen capture (Tauri/Rust)

File: `studio/src-tauri/src/live_screen.rs`.

Commands:

```rust
live_screen_start() -> { active, source }
live_screen_stop()
live_screen_pause(paused: bool)
live_screen_latest() -> { jpegBase64, width, height, timestampMs }
live_screen_status() -> { active, paused, source, error }
```

Registered in `generate_handler!` in `main.rs` (`~line 2501`).

Pipeline:

```text
xdg-desktop-portal ScreenCast (ashpd / D-Bus)
 -> user picks monitor/window
 -> PipeWire stream node
 -> GStreamer: pipewiresrc ! videorate max-rate=1 ! videoscale ! jpegenc ! appsink
 -> latest JPEG ring buffer
```

- First use shows the GNOME picker. The restore token is written with mode 0600
  to `<app config dir>/live-screencast-token`, so later sessions reuse the same
  source without a picker (GNOME suppresses the dialog with `persist_mode=2`).
- The portal returns a `restore_token` and a PipeWire node; `OpenPipeWireRemote`
  supplies the remote fd, which `pipewiresrc` consumes alongside `path=<node>`.
- Frames are downscaled to 1280 px wide, JPEG quality 80, and kept in a six-frame
  in-memory ring buffer. No frame is written to disk.
- `pause` drops frames without tearing down the portal stream, so resume is
  immediate; `stop` sets the pipeline to NULL, joins the capture thread (with a
  2 s escape hatch) and closes the portal session.
- A capture error marks the session inactive and publishes the message through
  `live_screen_status`; the frontend stops its preview timer and offers a retry.
- v1 privacy: portal picker + pause + persistent indicator. Window-title/app
  filtering is a follow-up (D-Bus focused-window probe).
- Linux only. Other platforms compile a stub with the same commands and hide the
  screen button; the voice engine itself still works everywhere.

Dependencies and packaging:

- Rust: `ashpd` (tokio portal client), `gstreamer`, `gstreamer-app` (plus the
  existing `base64` and `tokio`).
- System: GStreamer base/good (`pipewiresrc`, `videorate`, `videoscale`,
  `videoconvert`, `jpegenc`, `appsink`), PipeWire, xdg-desktop-portal. The
  capture thread checks every element and returns an actionable error when one is
  missing. Add the checks to `studio/setup.sh` and the Linux installer.
- AppImage must bundle the required GStreamer plugins; this is the main packaging
  risk. Fallback: portal handshake plus a `gst-launch-1.0` helper managed by the
  backend, still inside Studio.

## 8. Model plan

| Role | Recommended | Engine | Size / VRAM | License note |
|---|---|---|---|---|
| Chat + vision | Qwen3-VL 4B Instruct GGUF Q4_K_M + mmproj | llama.cpp | ~3.1 GB | model card; Studio license gate |
| Chat alternative | any installed vision-capable GGUF | llama.cpp | varies | - |
| STT | Whisper large-v3-turbo | whisper.cpp | ~1.1 GB GPU / CPU | MIT |
| STT alternative | Qwen3-ASR 0.6B | audio.cpp | small | model card |
| TTS default | Piper `tr_TR-fettah-medium` | Piper ONNX adapter | CPU, ~0 GB | Piper MIT; voice license check |
| TTS quality | XTTS-v2 | coqui/transformers adapter | ~1.8 GB GPU | CPML non-commercial; acceptance gate |
| TTS alternative | Studio audio.cpp models | audio.cpp | varies | test Turkish quality |
| VAD | Silero VAD v6.2.3 | onnxruntime | ~2 MB CPU | MIT |
| Endpoint | Smart Turn v3.2 | onnxruntime | small CPU | license metadata |
| Vision bridge (optional) | Qwen3-VL 4B | llama.cpp | extra | only for text-only chat + screen |

Download flow:

- Use Studio's existing HF hub/download manager (queue, progress, cancel, checksum).
- `LiveModelCatalog` entries carry role, repo, revision, files, size, license,
  engine and recommended combinations.
- Small VAD/endpoint ONNX assets are pinned and auto-downloaded with SHA-256
  verification.
- License acceptance is bound to repo + revision (XTTS especially).
- After install, the UI shows "Live ready" and offers mic/speaker/screen tests.

## 9. Implementation phases

### Phase 0 - Spike (3-5 days)

- Live button + empty live bar.
- getUserMedia + ScriptProcessor + 16 kHz resample -> stub WebSocket.
- Backend stub: `ready`, `level`, fake `transcript`, fake `tts_audio`.
- Tauri stub: fixture JPEG for the screen preview.
- Chat glue: send a fake transcript as a message.
- Exit: audio frames arrive, fake transcript sends a message, test audio plays,
  fixture screen preview renders.

### Phase 1 - Turn-based live (1.5-2 weeks)

- Backend: framing/resample, energy VAD -> Silero VAD, endpointing, STT adapter,
  model readiness, full WS protocol.
- Frontend: full state machine, transcript -> chat send, live bar states, mic
  device/language settings, draft preservation.
- Chat: dictation disabled, auto-read suppressed.
- Exit: hands-free turn-taking with final transcripts; setup flow for missing models.
- Tests: VAD/endpoint units, WS integration with WAV fixtures, controller units.

### Phase 2 - Streaming TTS + barge-in (2 weeks)

- Backend: TTS queue, sentence synthesis, prefetch, cancellation; Piper adapter
  first, audio.cpp second.
- Frontend: streamed-text sentence extraction, WebAudio player queue, gapless
  playback, cancellation, half-duplex mic gating.
- Barge-in: `playback_started` + VAD -> `barge_in` -> stop TTS + `cancelRun()`;
  interrupted metadata.
- Exit: first audio p50 <= 1.5 s; barge-in stops audio <= 200 ms; sentence gaps
  <= 300 ms; no double speaking.

### Phase 3 - Screen sharing (1.5-2 weeks) - implemented, HIL validation pending

- Tauri portal + PipeWire + JPEG capture, restore token, pause, Linux packaging:
  implemented in `src-tauri/src/live_screen.rs` (ashpd + GStreamer, 1 fps, 1280 px,
  six-frame ring, generation-guarded restart/stop).
- Frontend screen toggle, preview, frame attachment, vision checks, cloud consent:
  implemented (Linux desktop detection, start-state spinner, status-driven error
  recovery, attached JPEG on the next live turn).
- Backend optional window filter; optional VLM bridge for text-only chat: not yet
  (follow-up).
- Exit: toggle -> picker -> preview <= 2 s; the sent message carries the frame; no
  cloud upload without consent. Needs a GNOME Wayland run and packaged-build check.

### Phase 4 - Model catalog and Turkish voice (1-2 weeks)

- Piper Turkish + optional XTTS, STT alternatives, live catalog, downloads,
  licenses, VRAM estimates.
- Settings -> Live: readiness checklist, tests, mode selection, thresholds.
- Exit: clean install can download and run the stack; Turkish STT/TTS quality
  checked; errors actionable.

### Phase 5 - Hardening and polish (1-2 weeks)

- Reconnect/error recovery, idle unload, metrics, i18n, accessibility, shortcuts,
  documentation.
- 30-minute soak, VRAM/CPU, leaks, Studio regression suites.
- Exit: all metrics met; existing Studio tests green.

### Phase 6 - Optional true full-duplex

- Native realtime omni provider (OpenAI Realtime / Gemini Live) or backend
  PipeWire AEC. Not required for v1.

Total: ~7-11 weeks for one developer; ~5-7 weeks with frontend/backend/Rust in
parallel.

## 10. Tests and acceptance

Automated:

- Backend unit: VAD (synthetic speech/silence), endpoint decisions, resampler, WAV
  encode, TTS queue cancellation, fake STT.
- Backend integration: feed a WAV fixture into a WS session; assert `transcript`,
  `endpoint`, `barge_in`, `tts_cancel` events.
- Frontend unit: controller transitions, sentence extraction, TTS queue cancel,
  attachment flow, composer send mocks.
- Component: Live button visibility, bar states, error cards, settings.
- HIL: real mic/speaker, GNOME portal, 30-minute session.

Acceptance metrics:

| Metric | Target |
|---|---|
| End of speech -> transcript visible | p50 <= 800 ms, p90 <= 1.5 s |
| End of speech -> first TTS audio | p50 <= 1.5 s, p90 <= 2.5 s |
| Barge-in -> audio stops | <= 200 ms |
| Barge-in -> chat cancelled | <= 300 ms |
| Gap between sentences | <= 300 ms |
| First screen frame after picker | <= 2 s |
| Preview rate | >= 1 fps |
| VRAM (recommended stack) | <= 6.8 GB steady state |
| Turkish STT WER | at or better than the current baseline |
| Privacy | no raw audio on disk; no unconsented cloud frame |
| Stability | 30-minute session, no leak, no per-turn model reload |

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Weak echo cancellation in WebKitGTK | False barge-in | Turn mode default; use `getSettings().echoCancellation`; headphone guidance; backend PipeWire AEC later |
| GStreamer/portal packaging | Screen feature broken in releases | Compile the Rust module on Linux; installer dependency checks; AppImage test; `gst-launch` fallback |
| Slow/serialized TTS engine | High first-audio latency | Piper CPU default; one-sentence first chunk; prefetch; live queue priority; audio.cpp cancel_event |
| Turkish TTS quality | Poor UX | Start with Piper `tr_TR-fettah`; optional XTTS; test Studio audio.cpp models |
| VRAM pressure | Model load failures | Live VRAM estimate; CPU fallback; disable XTTS; idle unload |
| Upstream conflicts | Maintenance burden | New directories + feature flag + minimal anchor edits; regular rebase |
| Licensing | Distribution risk | Studio AGPL-3.0-only; XTTS CPML; model license gates; branding review |
| Privacy | Audio/frame leaks | Local defaults; explicit consent; indicators; no raw audio persistence |
| UI conflicts | Live/dictation overlap | Mutual exclusion; hide in Compare/web; clear states |
| Scope creep | Delay | True full-duplex and partial transcripts are out of v1 |

## 12. Relationship to Screen Companion

This feature does **not** import or depend on the Screen Companion package. Its
code is a reference for algorithms and UX only:

- `conversation/voice_loop.py` - VAD/endpoint/barge-in flow and thresholds.
- `audio/processors.py`, `audio/aec.py` - Silero/Smart Turn/echo policy.
- `media/screen/portal_cast.py` - portal handshake and restore token behavior.
- `media/buffer.py`, `media/changes.py`, `media/privacy.py` - frame pipeline,
  change detection, privacy policy.
- `runtimes/piper.py`, `runtimes/xtts.py` - Turkish TTS adapter behavior.
- `runtimes/faster_whisper.py` - STT adapter behavior if needed.
- `ui/pages/chat.py` - PTT/continuous/mute/screen toggle and waveform UX.
- Turkish i18n copy.

Everything is reimplemented in Studio's stack: React/TypeScript for the UI, Python
for the live engine, Rust for screen capture.

## 13. Upstream strategy

- Fork: `Ancolen/unsloth`, branch `studio-live-voice`, `upstream` remote points at
  `unslothai/unsloth`.
- Keep changes additive:
  - new `studio/backend/core/live/`, `studio/backend/routes/live.py`,
  - new `studio/frontend/src/features/live/`,
  - new `studio/src-tauri/src/live_screen.rs`,
  - only small anchor edits in existing files (`main.py`, `thread.tsx`,
    `auto-read-aloud.tsx`, settings, i18n, Tauri `main.rs`).
- Guard the feature with `UNSLOTH_STUDIO_LIVE_VOICE`; default off in release builds
  until the feature is stable.
- Rebase on `upstream/main` regularly; run Studio's existing typecheck, tests and
  `i18n:check` after each rebase.

## 14. Open questions

1. Screen capture: Rust `gstreamer-rs` vs `gst-launch-1.0` helper. Rust is the
   long-term path; the helper is the packaging fallback.
2. Turkish TTS: Piper first; XTTS quality vs VRAM; test Studio audio.cpp models.
3. STT: whisper.cpp large-v3-turbo vs Qwen3-ASR 0.6B on Turkish benchmarks.
4. Partial transcripts: streaming ASR vs rolling re-transcription; not v1.
5. True full-duplex: native realtime provider vs backend AEC; later.
6. Whether live turns should carry `live` metadata in chat history by default.

## License

This document and the Live Voice implementation are part of Unsloth Studio and are
licensed under AGPL-3.0-only, consistent with `/studio/LICENSE.AGPL-3.0`.
