// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LiveMode = "turn" | "barge_in";

export type LiveState =
  | "idle"
  | "connecting"
  | "ready"
  | "listening"
  | "speech"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "interrupted"
  | "error";

export interface LiveCapabilities {
  audioCapture?: string;
  screenCapture?: string;
  stt?: string;
  tts?: string;
  onnxruntime?: string;
  vad?: string;
  endpoint?: string;
}

export interface LiveAssetStatus {
  role: string;
  name?: string;
  source?: string;
  repo: string;
  revision: string;
  filename: string;
  sha256?: string;
  sizeBytes?: number | null;
  license?: string;
  installed?: boolean;
  active?: boolean;
  envOverride?: string;
}

export interface LiveStatus {
  enabled: boolean;
  protocol: number;
  sampleRate: number;
  platform: string;
  capabilities: LiveCapabilities;
  assets?: LiveAssetStatus[];
}

export interface LiveServerEvent {
  type: string;
  [key: string]: unknown;
}

export interface LiveFramePayload {
  jpegBase64: string;
  width: number;
  height: number;
  timestampMs: number;
}

export interface LiveScreenStatus {
  active: boolean;
  paused: boolean;
  source?: string | null;
  error?: string | null;
}
