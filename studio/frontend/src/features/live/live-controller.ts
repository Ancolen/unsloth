// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Live Voice orchestration.
 *
 * This controller owns the microphone, socket, playback queue and the bridge to
 * the normal assistant-ui chat runtime.  Chat stays in Studio's existing
 * runtime: a transcript is sent as an ordinary user message, and the streamed
 * assistant text is split into sentences for TTS.  There is no second chat
 * brain here.
 */

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { LiveAudioCapture } from "./live-audio-capture";
import { LiveAudioPlayer } from "./live-audio-player";
import { LiveSocket, fetchLiveStatus, getLiveToken } from "./live-client";
import {
  LiveScreenUnsupportedError,
  frameToDataUrl,
  frameToFile,
  latestLiveScreenFrame,
  liveScreenStatus,
  liveScreenSupported,
  startLiveScreen,
  stopLiveScreen,
} from "./live-screen";
import { useLiveStore } from "./live-store";
import { LiveSentenceStream } from "./live-tts";
import type {
  LiveFramePayload,
  LiveMode,
  LiveServerEvent,
  LiveStatus,
} from "./live-types";

export interface LiveBridge {
  sendUserMessage: (text: string, attachment: File | null) => Promise<void>;
  cancelRun: () => void;
  saveDraft: () => string;
  restoreDraft: (draft: string) => void;
  isRunActive: () => boolean;
  currentAssistantMessageId: () => string | null;
}

const PROTOCOL_VERSION = 1;
const SAMPLE_RATE = 16000;

function errorMessage(error: unknown, fallback: string): string {
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  return error instanceof Error && error.message ? error.message : fallback;
}

class LiveController {
  private socket = new LiveSocket();
  private capture = new LiveAudioCapture();
  private player = new LiveAudioPlayer();
  private bridge: LiveBridge | null = null;
  private sentences = new LiveSentenceStream();
  private token: string | null = null;
  private status: LiveStatus | null = null;
  private starting = false;
  private startToken = 0;
  private savedDraft = "";
  private savedAutoRead: boolean | null = null;
  private assistantMessageId: string | null = null;
  private baselineAssistantId: string | null = null;
  private liveTurnSent = false;
  private pendingTtsId: string | null = null;
  private currentPlaybackId: string | null = null;
  private ttsSeq = 0;
  private screenTimer: number | null = null;
  private lastScreenFrame: LiveFramePayload | null = null;
  private pendingTranscript: string | null = null;
  private runActive = false;
  private echoCancellation = false;

  registerBridge(bridge: LiveBridge): () => void {
    this.bridge = bridge;
    return () => {
      if (this.bridge === bridge) {
        this.bridge = null;
      }
    };
  }

  async ensureStatus(force = false): Promise<LiveStatus | null> {
    if (this.status && !force) {
      return this.status;
    }
    const status = await fetchLiveStatus();
    this.status = status;
    if (status) {
      useLiveStore.getState().setStatus(status);
    }
    return status;
  }

  async start(): Promise<void> {
    const store = useLiveStore.getState();
    if (store.active || this.starting) {
      return;
    }
    const startToken = ++this.startToken;
    this.starting = true;
    store.setError(null);
    store.setActive(true);
    store.setState("connecting");
    store.setTranscript("");
    store.setPartial("");
    store.setLevel(0);
    store.setModelsReady(false);
    store.setScreenActive(false);
    store.setScreenStarting(false);
    store.setScreenFrame(null);
    this.liveTurnSent = false;
    try {
      const status = await this.ensureStatus(true);
      if (!status?.enabled) {
        throw new Error(
          status
            ? "Live voice is disabled on this server."
            : "Live voice is unavailable on this server.",
        );
      }
      const token = getLiveToken();
      if (!token) {
        throw new Error("Sign in again before starting live voice.");
      }
      if (this.startCancelled(startToken)) {
        this.starting = false;
        return;
      }
      this.token = token;
      if (this.bridge) {
        this.savedDraft = this.bridge.saveDraft();
        // Do not speak an assistant answer that was already on screen before
        // Live started; only a new message that follows a live transcript is
        // eligible for TTS.
        this.baselineAssistantId = this.bridge.currentAssistantMessageId();
        this.runActive = this.bridge.isRunActive();
      }
      this.savedAutoRead = useVoiceSettingsStore.getState().autoReadAloud;
      useVoiceSettingsStore.getState().setAutoReadAloud(false);

      this.player.setHandlers({
        onStarted: () => {
          const id = this.pendingTtsId ?? this.currentPlaybackId;
          if (id) {
            this.socket.sendJson({ type: "playback_started", id });
          }
          useLiveStore.getState().setState("speaking");
        },
        onFinished: () => {
          const id = this.currentPlaybackId;
          if (id) {
            this.socket.sendJson({ type: "playback_finished", id });
          }
          const state = useLiveStore.getState().state;
          if (state === "speaking") {
            useLiveStore.getState().setState("listening");
          }
        },
        onError: (message) => {
          useLiveStore.getState().setError(message);
          // A decode/context failure can leave the backend believing audio is
          // still playing.  If nothing is queued locally, close that window.
          if (!this.player.isPlaying) {
            const id = this.currentPlaybackId ?? this.pendingTtsId;
            if (id) {
              this.socket.sendJson({ type: "playback_finished", id });
            }
            const store = useLiveStore.getState();
            if (store.state === "speaking") {
              store.setState("listening");
            }
          }
        },
      });
      await this.player.prepare();
      if (this.startCancelled(startToken)) {
        this.starting = false;
        this.player.dispose();
        return;
      }

      const settings = useLiveStore.getState();
      const voiceSettings = useVoiceSettingsStore.getState();
      const micDeviceId = voiceSettings.micDeviceId;
      const sttModel = settings.sttModel || voiceSettings.sttModel || undefined;
      const voice =
        settings.voice || voiceSettings.ttsStudioVoiceId || undefined;
      const captureInfo = await this.capture.start(micDeviceId || "default", {
        onFrame: (frame) => {
          this.socket.sendBinary(frame);
        },
        onLevel: (level) => {
          useLiveStore.getState().setLevel(level);
        },
        onError: (message) => {
          useLiveStore.getState().setError(message);
        },
      });
      this.echoCancellation = captureInfo.echoCancellation;
      useLiveStore.getState().setEchoCancellation(captureInfo.echoCancellation);
      if (this.startCancelled(startToken)) {
        this.starting = false;
        this.capture.stop();
        this.player.dispose();
        return;
      }

      this.socket.connect(token, {
        onOpen: () => {
          this.socket.sendJson({
            type: "hello",
            protocol: PROTOCOL_VERSION,
            sampleRate: SAMPLE_RATE,
            channels: 1,
            language: settings.language || "tr",
            mode: settings.mode,
            sttModel,
            ttsModel: settings.ttsModel || undefined,
            voice,
            echoCancellation: captureInfo.echoCancellation,
          });
          this.socket.sendJson({ type: "start" });
          this.capture.setEnabled(true);
          useLiveStore.getState().setState("listening");
        },
        onEvent: (event) => this.handleEvent(event),
        onBinary: (data) => {
          if (!this.pendingTtsId) {
            return;
          }
          void this.player.enqueue(data);
        },
        onClose: (code, reason) => {
          if (!useLiveStore.getState().active) {
            return;
          }
          if (code === 4401) {
            this.fail("Live voice authentication failed. Sign in again.");
          } else if (code === 4403) {
            this.fail("Live voice is not enabled for this window.");
          } else {
            this.fail(reason || "The live connection closed.");
          }
        },
        onError: (message) => {
          this.fail(message);
        },
      });
      this.starting = false;
    } catch (error) {
      this.starting = false;
      if (this.startCancelled(startToken)) {
        this.capture.stop();
        this.player.dispose();
        return;
      }
      this.fail(errorMessage(error, "Could not start live voice."));
    }
  }

  private startCancelled(token: number): boolean {
    return token !== this.startToken || !useLiveStore.getState().active;
  }

  stop(): void {
    this.teardown();
    this.restoreAfterLive();
    const store = useLiveStore.getState();
    store.resetRuntime();
    store.setActive(false);
    store.setState("idle");
  }

  setMuted(muted: boolean): void {
    const store = useLiveStore.getState();
    store.setMuted(muted);
    // This is the microphone mute control; the assistant's playback stays audible.
    this.capture.setEnabled(!muted);
    this.socket.sendJson({ type: "mute", muted });
  }

  setMode(mode: LiveMode): void {
    useLiveStore.getState().setMode(mode);
    this.socket.sendJson({ type: "mode", mode });
  }

  setLanguage(language: string): void {
    useLiveStore.getState().setLanguage(language);
  }

  async toggleScreen(): Promise<void> {
    const store = useLiveStore.getState();
    if (store.screenActive) {
      await this.stopScreen();
      return;
    }
    if (store.screenStarting) {
      return;
    }
    if (!liveScreenSupported()) {
      store.setError(new LiveScreenUnsupportedError().message);
      return;
    }
    store.setScreenStarting(true);
    try {
      await startLiveScreen();
      // Live voice may have been stopped while the portal dialog was open.
      if (!useLiveStore.getState().active) {
        await stopLiveScreen().catch(() => undefined);
        return;
      }
      const current = useLiveStore.getState();
      current.setScreenActive(true);
      current.setError(null);
      this.lastScreenFrame = await latestLiveScreenFrame();
      if (this.lastScreenFrame) {
        current.setScreenFrame(frameToDataUrl(this.lastScreenFrame));
      }
      this.screenTimer = window.setInterval(() => {
        void this.refreshScreenFrame();
      }, 1000);
    } catch (error) {
      if (!useLiveStore.getState().active) {
        return;
      }
      const current = useLiveStore.getState();
      current.setScreenActive(false);
      current.setScreenFrame(null);
      current.setError(errorMessage(error, "Could not start screen sharing."));
    } finally {
      useLiveStore.getState().setScreenStarting(false);
    }
  }

  private async stopScreen(): Promise<void> {
    if (this.screenTimer !== null) {
      window.clearInterval(this.screenTimer);
      this.screenTimer = null;
    }
    const store = useLiveStore.getState();
    store.setScreenActive(false);
    store.setScreenStarting(false);
    store.setScreenFrame(null);
    this.lastScreenFrame = null;
    try {
      await stopLiveScreen();
    } catch {
      // Desktop may not be available (web UI); the local state is already off.
    }
  }

  private async refreshScreenFrame(): Promise<void> {
    if (!useLiveStore.getState().screenActive) {
      return;
    }
    try {
      const frame = await latestLiveScreenFrame();
      if (frame) {
        this.lastScreenFrame = frame;
        useLiveStore.getState().setScreenFrame(frameToDataUrl(frame));
        return;
      }
      // A null frame can mean "paused" as well as "the stream failed".  The
      // status distinguishes the two; only tear down on a real stop/error.
      const status = await liveScreenStatus();
      if (status && !status.active) {
        await this.stopScreen();
        if (status.error) {
          useLiveStore.getState().setError(status.error);
        }
      }
    } catch (error) {
      await this.stopScreen();
      useLiveStore
        .getState()
        .setError(errorMessage(error, "The screen preview stopped."));
    }
  }

  observeAssistant(
    messageId: string | null,
    text: string,
    running: boolean,
  ): void {
    if (!useLiveStore.getState().active) {
      return;
    }
    if (!this.liveTurnSent) {
      // Live has not sent a transcript yet; do not read old chat history aloud,
      // but still flush a transcript that waited for the previous run to end.
      this.runActive = running;
      if (!running) {
        this.flushPendingTranscript();
      }
      return;
    }
    if (messageId && messageId === this.baselineAssistantId) {
      // This is the answer that was already on screen when Live started.
      this.runActive = running;
      if (!running) {
        this.flushPendingTranscript();
      }
      return;
    }
    if (messageId !== null) {
      this.baselineAssistantId = null;
    }
    if (messageId !== this.assistantMessageId) {
      // A new assistant message replaces the old answer.  Stop any audio still
      // queued for the old one and tell the backend that playback ended, or
      // turn mode would keep gating the microphone forever.
      if (this.currentPlaybackId || this.pendingTtsId) {
        this.socket.sendJson({ type: "tts_cancel" });
        this.player.cancel();
        const id = this.currentPlaybackId ?? this.pendingTtsId;
        if (id) {
          this.socket.sendJson({ type: "playback_finished", id });
        }
      }
      this.assistantMessageId = messageId;
      this.pendingTtsId = null;
      this.sentences.reset();
      this.ttsSeq = 0;
      this.currentPlaybackId = null;
    }
    this.runActive = running;
    const sentences = this.sentences.update(text, !running);
    for (const sentence of sentences) {
      this.sendSentence(sentence);
    }
    if (running && sentences.length === 0) {
      const state = useLiveStore.getState().state;
      if (
        state === "listening" ||
        state === "transcribing" ||
        state === "ready"
      ) {
        useLiveStore.getState().setState("thinking");
      }
    }
    if (!running) {
      this.flushPendingTranscript();
    }
  }

  private flushPendingTranscript(): void {
    const pending = this.pendingTranscript;
    this.pendingTranscript = null;
    if (pending) {
      void this.sendTranscript(pending);
      return;
    }
    const store = useLiveStore.getState();
    if (store.state === "thinking") {
      store.setState("listening");
    }
  }

  onRunActivity(active: boolean): void {
    this.runActive = active;
  }

  private sendSentence(text: string): void {
    const clean = text.trim();
    if (!clean) {
      return;
    }
    this.ttsSeq += 1;
    const id = `tts-${Date.now()}-${this.ttsSeq}`;
    this.currentPlaybackId = id;
    this.socket.sendJson({
      type: "speak",
      id,
      seq: this.ttsSeq,
      text: clean,
      final: false,
    });
  }

  private async sendTranscript(text: string): Promise<void> {
    const clean = text.trim();
    if (!clean) {
      return;
    }
    const bridge = this.bridge;
    if (!bridge) {
      this.pendingTranscript = clean;
      return;
    }
    if (this.runActive && useLiveStore.getState().mode === "turn") {
      this.pendingTranscript = clean;
      return;
    }
    let attachment: File | null = null;
    if (
      useLiveStore.getState().screenActive &&
      useLiveStore.getState().attachScreen
    ) {
      try {
        const frame = this.lastScreenFrame ?? (await latestLiveScreenFrame());
        if (frame) {
          this.lastScreenFrame = frame;
          attachment = frameToFile(frame);
        }
      } catch {
        attachment = null;
      }
    }
    this.liveTurnSent = true;
    useLiveStore.getState().setState("thinking");
    try {
      await bridge.sendUserMessage(clean, attachment);
    } catch (error) {
      useLiveStore
        .getState()
        .setError(
          errorMessage(
            error,
            "The live transcript could not be sent to the chat.",
          ),
        );
    }
  }

  // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: one flat switch over the wire protocol keeps the event mapping auditable.
  private handleEvent(event: LiveServerEvent): void {
    const store = useLiveStore.getState();
    switch (event.type) {
      case "ready": {
        store.setModelsReady(true);
        if (typeof event.echoCancellation === "boolean") {
          this.echoCancellation = event.echoCancellation;
          store.setEchoCancellation(event.echoCancellation);
        }
        if (store.state === "connecting") {
          store.setState("listening");
        }
        break;
      }
      case "state": {
        const state = typeof event.state === "string" ? event.state : "";
        if (
          state === "listening" ||
          state === "speech" ||
          state === "transcribing" ||
          state === "speaking" ||
          state === "idle"
        ) {
          // The chat may be thinking while the audio engine is listening; keep
          // the more specific chat state in that window.
          if (
            state === "listening" &&
            store.state === "thinking" &&
            this.runActive
          ) {
            break;
          }
          store.setState(state);
        }
        break;
      }
      case "level":
        if (typeof event.rms === "number") {
          store.setLevel(event.rms);
        }
        break;
      case "speech_started": {
        store.setState("speech");
        if (
          store.interruptWhileThinking &&
          this.runActive &&
          store.state !== "speaking"
        ) {
          this.bridge?.cancelRun();
          this.socket.sendJson({ type: "tts_cancel" });
          this.player.cancel();
          store.setState("interrupted");
        }
        break;
      }
      case "transcript":
        if (typeof event.text === "string" && event.text.trim()) {
          store.setTranscript(event.text);
          void this.sendTranscript(event.text);
        }
        break;
      case "barge_in": {
        this.player.cancel();
        this.socket.sendJson({ type: "tts_cancel" });
        this.bridge?.cancelRun();
        store.setState("interrupted");
        window.setTimeout(() => {
          if (useLiveStore.getState().state === "interrupted") {
            useLiveStore.getState().setState("listening");
          }
        }, 400);
        break;
      }
      case "tts_start": {
        this.pendingTtsId = typeof event.id === "string" ? event.id : null;
        if (typeof event.id === "string") {
          this.currentPlaybackId = event.id;
        }
        store.setState("speaking");
        break;
      }
      case "tts_end":
        this.pendingTtsId = null;
        break;
      case "error": {
        const code = typeof event.code === "string" ? event.code : "live_error";
        const message =
          typeof event.message === "string" && event.message
            ? event.message
            : "Live voice reported an error.";
        store.setError(message);
        if (code === "no_speech") {
          store.setState("listening");
        }
        break;
      }
      case "pong":
        break;
      default:
        break;
    }
  }

  private fail(message: string): void {
    const store = useLiveStore.getState();
    store.setError(message);
    store.setState("error");
    this.teardown();
    this.restoreAfterLive();
    store.setActive(false);
  }

  private teardown(): void {
    this.startToken += 1;
    this.starting = false;
    this.socket.close();
    this.capture.stop();
    this.player.cancel();
    this.player.dispose();
    if (this.screenTimer !== null) {
      window.clearInterval(this.screenTimer);
      this.screenTimer = null;
    }
    if (
      useLiveStore.getState().screenActive ||
      useLiveStore.getState().screenStarting
    ) {
      void stopLiveScreen().catch(() => {});
    }
    useLiveStore.getState().setScreenActive(false);
    useLiveStore.getState().setScreenStarting(false);
    useLiveStore.getState().setScreenFrame(null);
    this.assistantMessageId = null;
    this.baselineAssistantId = null;
    this.liveTurnSent = false;
    this.pendingTtsId = null;
    this.currentPlaybackId = null;
    this.pendingTranscript = null;
    this.runActive = false;
    this.sentences.reset();
    this.lastScreenFrame = null;
  }

  private restoreAfterLive(): void {
    if (this.savedAutoRead !== null) {
      useVoiceSettingsStore.getState().setAutoReadAloud(this.savedAutoRead);
      this.savedAutoRead = null;
    }
    if (this.bridge && this.savedDraft) {
      this.bridge.restoreDraft(this.savedDraft);
    }
    this.savedDraft = "";
  }
}

export const liveController = new LiveController();
