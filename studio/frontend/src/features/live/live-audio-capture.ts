// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Microphone capture for Live Voice.
 *
 * Uses the same getUserMedia + ScriptProcessor path Studio already relies on for
 * dictation (WebKitGTK's MediaRecorder can produce zero-byte recordings), then
 * resamples the AudioContext rate to the 16 kHz mono PCM the live backend
 * expects.  Frames are 512 samples (32 ms), matching Silero's window.
 */

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;
const PROCESSOR_BUFFER = 2048;
const LEVEL_INTERVAL_MS = 100;

export type LiveCaptureErrorCode =
  | "unsupported"
  | "permission_denied"
  | "no_device"
  | "audio_failed";

export class LiveCaptureError extends Error {
  readonly code: LiveCaptureErrorCode;

  constructor(code: LiveCaptureErrorCode, message: string) {
    super(message);
    this.name = "LiveCaptureError";
    this.code = code;
  }
}

export interface LiveCaptureHandlers {
  onFrame: (frame: ArrayBuffer) => void;
  onLevel: (level: number) => void;
  onError: (message: string) => void;
}

export interface LiveCaptureInfo {
  sampleRate: number;
  echoCancellation: boolean;
  deviceLabel: string;
}

function audioContextCtor(): typeof AudioContext | null {
  if (typeof window === "undefined") {
    return null;
  }
  const candidate =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  return candidate ?? null;
}

function appendFloat32(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length === 0) {
    return right;
  }
  const merged = new Float32Array(left.length + right.length);
  merged.set(left, 0);
  merged.set(right, left.length);
  return merged;
}

function resampleLinear(
  input: Float32Array,
  outputSamples: number,
): Float32Array {
  const output = new Float32Array(outputSamples);
  if (input.length === 0) {
    return output;
  }
  if (input.length === 1) {
    output.fill(input[0]);
    return output;
  }
  const step = (input.length - 1) / Math.max(1, outputSamples - 1);
  for (let index = 0; index < outputSamples; index += 1) {
    const position = index * step;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const fraction = position - left;
    output[index] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function floatToInt16(samples: Float32Array): Int16Array {
  const output = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    output[index] =
      value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
  }
  return output;
}

function rms(samples: Float32Array): number {
  if (samples.length === 0) {
    return 0;
  }
  let total = 0;
  for (let index = 0; index < samples.length; index += 1) {
    total += samples[index] * samples[index];
  }
  return Math.sqrt(total / samples.length);
}

export class LiveAudioCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private pending: Float32Array = new Float32Array(0);
  private enabled = true;
  private lastLevelAt = 0;
  private handlers: LiveCaptureHandlers | null = null;
  private ratio = 1;

  get isCapturing(): boolean {
    return this.stream !== null;
  }

  async start(
    deviceId: string,
    handlers: LiveCaptureHandlers,
  ): Promise<LiveCaptureInfo> {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      throw new LiveCaptureError(
        "unsupported",
        "This browser cannot capture a microphone.",
      );
    }
    const Ctor = audioContextCtor();
    if (!Ctor || typeof window === "undefined") {
      throw new LiveCaptureError(
        "unsupported",
        "This browser cannot process live audio.",
      );
    }
    this.handlers = handlers;
    const constraints: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    };
    if (deviceId && deviceId !== "default") {
      constraints.deviceId = { exact: deviceId };
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: constraints,
      });
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        throw new LiveCaptureError(
          "permission_denied",
          "Microphone permission was denied. Allow it in the system or browser settings.",
        );
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        throw new LiveCaptureError(
          "no_device",
          "No microphone matched the selected device. Pick another input in Settings → Live.",
        );
      }
      throw new LiveCaptureError(
        "audio_failed",
        error instanceof Error
          ? error.message
          : "Could not open the microphone.",
      );
    }
    this.stream = stream;
    try {
      this.context = new Ctor();
      if (this.context.state === "suspended") {
        await this.context.resume();
      }
      this.source = this.context.createMediaStreamSource(stream);
      this.processor = this.context.createScriptProcessor(
        PROCESSOR_BUFFER,
        1,
        1,
      );
      this.sink = this.context.createGain();
      this.sink.gain.value = 0;
      this.source.connect(this.processor);
      this.processor.connect(this.sink);
      this.sink.connect(this.context.destination);
      this.ratio = this.context.sampleRate / TARGET_RATE;
      this.processor.onaudioprocess = (event) => this.processAudio(event);
    } catch (error) {
      this.stop();
      throw new LiveCaptureError(
        "audio_failed",
        error instanceof Error
          ? error.message
          : "Could not start the microphone.",
      );
    }
    const track = stream.getAudioTracks()[0] ?? null;
    const settings = track?.getSettings?.() ?? {};
    return {
      sampleRate: this.context.sampleRate,
      echoCancellation: settings.echoCancellation === true,
      deviceLabel: track?.label ?? "",
    };
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) {
      this.pending = new Float32Array(0);
    }
  }

  stop(): void {
    this.enabled = false;
    this.handlers = null;
    this.pending = new Float32Array(0);
    if (this.processor) {
      this.processor.onaudioprocess = null;
      try {
        this.processor.disconnect();
      } catch {
        // Already disconnected.
      }
      this.processor = null;
    }
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Already disconnected.
      }
      this.source = null;
    }
    if (this.sink) {
      try {
        this.sink.disconnect();
      } catch {
        // Already disconnected.
      }
      this.sink = null;
    }
    for (const track of this.stream?.getTracks() ?? []) {
      try {
        track.stop();
      } catch {
        // Already stopped.
      }
    }
    this.stream = null;
    if (this.context) {
      try {
        void this.context.close();
      } catch {
        // Already closed.
      }
      this.context = null;
    }
  }

  private processAudio(event: AudioProcessingEvent): void {
    const handlers = this.handlers;
    if (!(handlers && this.enabled)) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    if (input.length === 0) {
      return;
    }
    this.pending = appendFloat32(this.pending, input);
    const needed = Math.max(1, Math.floor(FRAME_SAMPLES * this.ratio));
    try {
      while (this.pending.length >= needed + 1) {
        const chunk = this.pending.subarray(0, needed + 1);
        const resampled = resampleLinear(chunk, FRAME_SAMPLES);
        const pcm = floatToInt16(resampled);
        handlers.onFrame(pcm.buffer as ArrayBuffer);
        this.pending = this.pending.slice(needed);
        const now = performance.now();
        if (now - this.lastLevelAt >= LEVEL_INTERVAL_MS) {
          this.lastLevelAt = now;
          handlers.onLevel(rms(resampled));
        }
      }
      // Guard against a stalled AudioContext growing the buffer without bound.
      if (this.pending.length > FRAME_SAMPLES * this.ratio * 8) {
        this.pending = this.pending.slice(this.pending.length - needed);
      }
    } catch (error) {
      handlers.onError(
        error instanceof Error
          ? error.message
          : "Live microphone processing failed.",
      );
    }
  }
}
