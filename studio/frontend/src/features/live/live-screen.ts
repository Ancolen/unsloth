// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import type { LiveFramePayload, LiveScreenStatus } from "./live-types";

export class LiveScreenUnsupportedError extends Error {
  constructor(
    message = "Screen sharing is only available in Unsloth Desktop on Linux.",
  ) {
    super(message);
    this.name = "LiveScreenUnsupportedError";
  }
}

async function invokeTauri<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  if (!isTauri) {
    throw new LiveScreenUnsupportedError();
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export function liveScreenSupported(): boolean {
  return isTauri;
}

export async function startLiveScreen(): Promise<LiveScreenStatus> {
  return invokeTauri<LiveScreenStatus>("live_screen_start");
}

export async function stopLiveScreen(): Promise<LiveScreenStatus> {
  return invokeTauri<LiveScreenStatus>("live_screen_stop");
}

export async function pauseLiveScreen(
  paused: boolean,
): Promise<LiveScreenStatus> {
  return invokeTauri<LiveScreenStatus>("live_screen_pause", { paused });
}

export async function latestLiveScreenFrame(): Promise<LiveFramePayload | null> {
  return invokeTauri<LiveFramePayload | null>("live_screen_latest");
}

export async function liveScreenStatus(): Promise<LiveScreenStatus | null> {
  try {
    return await invokeTauri<LiveScreenStatus>("live_screen_status");
  } catch {
    return null;
  }
}

export function frameToDataUrl(frame: LiveFramePayload): string {
  return `data:image/jpeg;base64,${frame.jpegBase64}`;
}

export function frameToFile(
  frame: LiveFramePayload,
  name = "screen.jpg",
): File {
  const binary = atob(frame.jpegBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], name, { type: "image/jpeg" });
}
