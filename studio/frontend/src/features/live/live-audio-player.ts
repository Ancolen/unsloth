// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Gapless WebAudio playback for Live Voice.
 *
 * The backend sends one complete WAV per sentence.  We decode each into an
 * AudioBuffer and schedule it at `nextStartTime`, so sentence boundaries do not
 * introduce an `Audio` element's load gap.  `cancel()` stops the current source
 * and everything queued, which is the barge-in path.
 */

const START_LEAD_S = 0.02;
const CANCEL_FADE_S = 0.02;

export interface LivePlayerHandlers {
  onStarted: () => void;
  onFinished: () => void;
  onError: (message: string) => void;
}

function audioContextCtor(): typeof AudioContext | null {
  if (typeof window === "undefined") {
    return null;
  }
  return (
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext ||
    null
  );
}

export class LiveAudioPlayer {
  private context: AudioContext | null = null;
  private gain: GainNode | null = null;
  private handlers: LivePlayerHandlers | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextStartTime = 0;
  private playing = false;
  private muted = false;

  get isPlaying(): boolean {
    return this.playing;
  }

  setHandlers(handlers: LivePlayerHandlers): void {
    this.handlers = handlers;
  }

  async prepare(): Promise<void> {
    if (this.context) {
      if (this.context.state === "suspended") {
        await this.context.resume();
      }
      return;
    }
    const Ctor = audioContextCtor();
    if (!Ctor) {
      throw new Error("This browser cannot play live audio.");
    }
    this.context = new Ctor();
    if (this.context.state === "suspended") {
      await this.context.resume();
    }
    this.gain = this.context.createGain();
    this.gain.gain.value = this.muted ? 0 : 1;
    this.gain.connect(this.context.destination);
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.gain && this.context) {
      this.gain.gain.setTargetAtTime(
        muted ? 0 : 1,
        this.context.currentTime,
        0.015,
      );
    }
  }

  async enqueue(wav: ArrayBuffer): Promise<void> {
    if (!(this.context && this.gain)) {
      try {
        await this.prepare();
      } catch (error) {
        this.handlers?.onError(
          error instanceof Error
            ? error.message
            : "Could not start live audio playback.",
        );
        return;
      }
    }
    const context = this.context;
    const gain = this.gain;
    if (!(context && gain)) {
      return;
    }
    let buffer: AudioBuffer;
    try {
      buffer = await context.decodeAudioData(wav.slice(0));
    } catch (error) {
      this.handlers?.onError(
        error instanceof Error
          ? error.message
          : "Could not decode live speech audio.",
      );
      return;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(gain);
    const startAt = Math.max(
      context.currentTime + START_LEAD_S,
      this.nextStartTime,
    );
    this.nextStartTime = startAt + buffer.duration;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      try {
        source.disconnect();
      } catch {
        // Already disconnected.
      }
      if (this.sources.size === 0) {
        this.playing = false;
        this.nextStartTime = 0;
        this.handlers?.onFinished();
      }
    };
    if (!this.playing) {
      this.playing = true;
      this.handlers?.onStarted();
    }
    source.start(startAt);
  }

  cancel(): void {
    const wasPlaying = this.playing;
    this.playing = false;
    this.nextStartTime = 0;
    const context = this.context;
    if (context && this.gain) {
      // A short gain dip avoids a click; the sources stop immediately after.
      this.gain.gain.cancelScheduledValues(context.currentTime);
      this.gain.gain.setValueAtTime(this.gain.gain.value, context.currentTime);
      this.gain.gain.linearRampToValueAtTime(
        0,
        context.currentTime + CANCEL_FADE_S,
      );
    }
    for (const source of this.sources) {
      // cancel() reports the finish itself; an onended callback arriving later
      // must not report it a second time.
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Already stopped.
      }
      try {
        source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this.sources.clear();
    if (context && this.gain) {
      this.gain.gain.setTargetAtTime(
        this.muted ? 0 : 1,
        context.currentTime + CANCEL_FADE_S,
        0.01,
      );
    }
    if (wasPlaying) {
      this.handlers?.onFinished();
    }
  }

  dispose(): void {
    this.cancel();
    this.handlers = null;
    if (this.gain) {
      try {
        this.gain.disconnect();
      } catch {
        // Already disconnected.
      }
      this.gain = null;
    }
    if (this.context) {
      try {
        void this.context.close();
      } catch {
        // Already closed.
      }
      this.context = null;
    }
  }
}
