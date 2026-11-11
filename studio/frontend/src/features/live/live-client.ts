// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import type {
  LiveAssetStatus,
  LiveServerEvent,
  LiveStatus,
} from "./live-types";

export function getLiveToken(): string | null {
  return getAuthToken();
}

export async function fetchLiveStatus(): Promise<LiveStatus | null> {
  try {
    const response = await authFetch("/api/live/status");
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as LiveStatus;
    if (typeof payload !== "object" || payload === null) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

export async function downloadLiveAsset(
  role: string,
): Promise<LiveAssetStatus> {
  const timeout = (
    AbortSignal as unknown as {
      timeout?: (ms: number) => AbortSignal;
    }
  ).timeout?.(30_000);
  const response = await authFetch(
    `/api/live/assets/${encodeURIComponent(role)}/download`,
    { method: "POST", signal: timeout },
  );
  const payload = (await response.json().catch(() => null)) as
    | (LiveAssetStatus & { detail?: unknown })
    | null;
  if (!response.ok) {
    const detail =
      payload && typeof payload.detail === "string" ? payload.detail : "";
    throw new Error(
      detail || "The live detector model could not be downloaded.",
    );
  }
  if (!payload || typeof payload !== "object") {
    throw new Error("The live detector download returned an invalid response.");
  }
  return payload;
}

export function liveWebSocketUrl(token: string): string {
  const url = new URL(apiUrl("/api/live/ws"), window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("token", token);
  return url.toString();
}

export interface LiveSocketHandlers {
  onEvent: (event: LiveServerEvent) => void;
  onBinary: (data: ArrayBuffer) => void;
  onOpen: () => void;
  onClose: (code: number, reason: string) => void;
  onError: (message: string) => void;
}

export class LiveSocket {
  private socket: WebSocket | null = null;
  private closedByUs = false;

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  connect(token: string, handlers: LiveSocketHandlers): void {
    this.close();
    this.closedByUs = false;
    let socket: WebSocket;
    try {
      socket = new WebSocket(liveWebSocketUrl(token));
    } catch (error) {
      handlers.onError(
        error instanceof Error
          ? error.message
          : "Could not open the live socket.",
      );
      return;
    }
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket === socket) {
        handlers.onOpen();
      }
    };
    socket.onmessage = (event) => {
      if (this.socket !== socket) {
        return;
      }
      if (event.data instanceof ArrayBuffer) {
        handlers.onBinary(event.data);
        return;
      }
      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data) as LiveServerEvent;
          if (
            payload &&
            typeof payload === "object" &&
            typeof payload.type === "string"
          ) {
            handlers.onEvent(payload);
          }
        } catch {
          handlers.onError("The live backend sent an invalid message.");
        }
      }
    };
    socket.onerror = () => {
      if (this.socket === socket && !this.closedByUs) {
        handlers.onError("The live connection failed.");
      }
    };
    socket.onclose = (event) => {
      if (this.socket !== socket) {
        return;
      }
      this.socket = null;
      handlers.onClose(event.code, event.reason || "");
    };
  }

  sendJson(payload: Record<string, unknown>): boolean {
    if (!(this.isOpen && this.socket)) {
      return false;
    }
    try {
      this.socket.send(JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }

  sendBinary(payload: ArrayBuffer): boolean {
    if (!(this.isOpen && this.socket)) {
      return false;
    }
    try {
      this.socket.send(payload);
      return true;
    } catch {
      return false;
    }
  }

  close(): void {
    this.closedByUs = true;
    const socket = this.socket;
    this.socket = null;
    if (!socket) {
      return;
    }
    try {
      socket.close();
    } catch {
      // Already closed.
    }
  }
}
