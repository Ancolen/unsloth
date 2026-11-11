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
}

const PROTOCOL_VERSION = 1;
const SAMPLE_RATE = 16000;

function errorMessage(error: unknown, fallback: string): string {
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
  private savedDraft = "";
  private savedAutoRead: boolean | null = null;
  private assistantMessageId: string | null = null;
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
    this.starting = true;
    store.setError(null);
    store.setActive(true);
    store.setState("connecting");
    store.setTranscript("");
    store.setPartial("");
    store.setLevel(0);
    store.setModelsReady(false);
    try {
      const status = await this.ensureStatus();
      if (!status?.enabled) {
        throw new Error("Live voice is disabled on this server.");
      }
      const token = getLiveToken();
      if (!token) {
        throw new Error("Sign in again before starting live voice.");
      }
      this.token = token;
      if (this.bridge) {
        this.savedDraft = this.bridge.saveDraft();
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
        },
      });
      await this.player.prepare();

      const settings = useLiveStore.getState();
      const micDeviceId = useVoiceSettingsStore.getState().micDeviceId;
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

      this.socket.connect(token, {
        onOpen: () => {
          this.socket.sendJson({
            type: "hello",
            protocol: PROTOCOL_VERSION,
            sampleRate: SAMPLE_RATE,
            channels: 1,
            language: settings.language || "tr",
            mode: settings.mode,
            sttModel: settings.sttModel || undefined,
            ttsModel: settings.ttsModel || undefined,
            voice: settings.voice || undefined,
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
      this.fail(errorMessage(error, "Could not start live voice."));
    }
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
    this.capture.setEnabled(!muted);
    this.player.setMuted(muted);
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
    if (!liveScreenSupported()) {
      store.setError(new LiveScreenUnsupportedError().message);
      return;
    }
    try {
      await startLiveScreen();
      store.setScreenActive(true);
      store.setError(null);
      this.lastScreenFrame = await latestLiveScreenFrame();
      if (this.lastScreenFrame) {
        store.setScreenFrame(frameToDataUrl(this.lastScreenFrame));
      }
      this.screenTimer = window.setInterval(() => {
        void this.refreshScreenFrame();
      }, 1000);
    } catch (error) {
      store.setScreenActive(false);
      store.setScreenFrame(null);
      store.setError(errorMessage(error, "Could not start screen sharing."));
    }
  }

  private async stopScreen(): Promise<void> {
    if (this.screenTimer !== null) {
      window.clearInterval(this.screenTimer);
      this.screenTimer = null;
    }
    const store = useLiveStore.getState();
    store.setScreenActive(false);
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
      if (!frame) {
        return;
      }
      this.lastScreenFrame = frame;
      useLiveStore.getState().setScreenFrame(frameToDataUrl(frame));
    } catch (error) {
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
    if (messageId !== this.assistantMessageId) {
      this.assistantMessageId = messageId;
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
      const pending = this.pendingTranscript;
      this.pendingTranscript = null;
      if (pending) {
        void this.sendTranscript(pending);
      } else {
        const state = useLiveStore.getState().state;
        if (state === "thinking") {
          useLiveStore.getState().setState("listening");
        }
      }
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
    this.socket.close();
    this.capture.stop();
    this.player.cancel();
    this.player.dispose();
    if (this.screenTimer !== null) {
      window.clearInterval(this.screenTimer);
      this.screenTimer = null;
    }
    if (useLiveStore.getState().screenActive) {
      void stopLiveScreen().catch(() => {});
    }
    useLiveStore.getState().setScreenActive(false);
    useLiveStore.getState().setScreenFrame(null);
    this.assistantMessageId = null;
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
