// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Loader2,
  Mic,
  MicOff,
  Monitor,
  MonitorOff,
  Square,
  X,
} from "lucide-react";
import type { FC } from "react";
import { liveController } from "./live-controller";
import { liveScreenSupported } from "./live-screen";
import { useLiveStore } from "./live-store";
import type { LiveState } from "./live-types";

const BAR_COUNT = 28;

const STATE_KEYS: Record<LiveState, TranslationKey> = {
  idle: "live.states.idle",
  connecting: "live.states.connecting",
  ready: "live.states.ready",
  listening: "live.states.listening",
  speech: "live.states.speech",
  transcribing: "live.states.transcribing",
  thinking: "live.states.thinking",
  speaking: "live.states.speaking",
  interrupted: "live.states.interrupted",
  error: "live.states.error",
};

function stateDot(state: LiveState): string {
  if (state === "speech") {
    return "bg-destructive";
  }
  if (state === "speaking") {
    return "bg-primary";
  }
  if (state === "thinking" || state === "transcribing") {
    return "bg-amber-500";
  }
  if (state === "error") {
    return "bg-destructive";
  }
  if (state === "interrupted") {
    return "bg-orange-500";
  }
  return "bg-emerald-500";
}

export const LiveBar: FC = () => {
  const t = useT();
  const active = useLiveStore((state) => state.active);
  const state = useLiveStore((state) => state.state);
  const level = useLiveStore((state) => state.level);
  const muted = useLiveStore((state) => state.muted);
  const mode = useLiveStore((state) => state.mode);
  const transcript = useLiveStore((state) => state.transcript);
  const screenActive = useLiveStore((state) => state.screenActive);
  const screenStarting = useLiveStore((state) => state.screenStarting);
  const screenFrame = useLiveStore((state) => state.screenFrame);
  const error = useLiveStore((state) => state.error);
  const echoCancellation = useLiveStore((state) => state.echoCancellation);
  const screenSupported = liveScreenSupported();

  if (!active) {
    return null;
  }

  const bars = Array.from({ length: BAR_COUNT }, (_, index) => {
    const center =
      1 - Math.abs(index - (BAR_COUNT - 1) / 2) / ((BAR_COUNT - 1) / 2);
    const energy =
      state === "speech" || state === "speaking" ? level : level * 0.6;
    return Math.max(3, Math.round(3 + energy * 26 * (0.35 + 0.65 * center)));
  });

  return (
    <div className="unsloth-live-bar flex min-w-0 flex-1 flex-col gap-1.5 rounded-2xl border border-border/70 bg-background/95 px-3 py-2 shadow-sm">
      <div className="flex min-w-0 items-center gap-2.5">
        <span
          className={cn("size-2 shrink-0 rounded-full", stateDot(state))}
          aria-hidden="true"
        />
        <span className="shrink-0 text-xs font-medium text-foreground">
          {t(STATE_KEYS[state])}
        </span>
        <div
          className="flex h-8 min-w-0 flex-1 items-center justify-center gap-[2px] overflow-hidden"
          aria-hidden="true"
        >
          {bars.map((height, index) => (
            <span
              // biome-ignore lint/suspicious/noArrayIndexKey: the meter has a fixed bar count and every bar is stateless.
              key={`live-bar-${index}`}
              className="w-[2px] shrink-0 rounded-full bg-primary/70 transition-[height] duration-100"
              style={{ height: `${height}px` }}
            />
          ))}
        </div>
        {screenActive && screenFrame ? (
          <img
            src={screenFrame}
            alt={t("live.screenPreview")}
            className="h-8 w-14 shrink-0 rounded-md border border-border/60 object-cover"
          />
        ) : null}
        <div className="flex shrink-0 items-center gap-1">
          <TooltipIconButton
            tooltip={
              mode === "barge_in" ? t("live.mode.bargeIn") : t("live.mode.turn")
            }
            aria-label={
              mode === "barge_in" ? t("live.mode.bargeIn") : t("live.mode.turn")
            }
            type="button"
            className={cn(
              "size-8 rounded-full text-xs",
              mode === "barge_in" &&
                "bg-primary/10 text-primary hover:bg-primary/15",
            )}
            onClick={() =>
              liveController.setMode(mode === "barge_in" ? "turn" : "barge_in")
            }
          >
            <span className="px-1 text-[10px] font-semibold uppercase tracking-wide">
              {mode === "barge_in" ? "DUO" : "TURN"}
            </span>
          </TooltipIconButton>
          {screenSupported ? (
            <TooltipIconButton
              tooltip={screenActive ? t("live.screenOff") : t("live.screenOn")}
              aria-label={
                screenActive ? t("live.screenOff") : t("live.screenOn")
              }
              type="button"
              className={cn(
                "size-8 rounded-full",
                screenActive &&
                  "bg-primary/10 text-primary hover:bg-primary/15",
              )}
              disabled={screenStarting}
              onClick={() => void liveController.toggleScreen()}
            >
              {screenStarting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : screenActive ? (
                <MonitorOff className="size-4" />
              ) : (
                <Monitor className="size-4" />
              )}
            </TooltipIconButton>
          ) : null}
          <TooltipIconButton
            tooltip={muted ? t("live.unmute") : t("live.mute")}
            aria-label={muted ? t("live.unmute") : t("live.mute")}
            type="button"
            className={cn("size-8 rounded-full", muted && "text-destructive")}
            onClick={() => liveController.setMuted(!muted)}
          >
            {muted ? <MicOff className="size-4" /> : <Mic className="size-4" />}
          </TooltipIconButton>
          <TooltipIconButton
            tooltip={t("live.stop")}
            aria-label={t("live.stop")}
            type="button"
            className="size-8 rounded-full text-destructive"
            onClick={() => liveController.stop()}
          >
            <Square className="size-4" strokeWidth={2.2} />
          </TooltipIconButton>
        </div>
      </div>
      <div className="flex min-w-0 items-center gap-2 pl-4">
        <p
          className="min-w-0 flex-1 truncate text-xs text-muted-foreground"
          dir="auto"
        >
          {transcript || t("live.hint")}
        </p>
        {mode === "barge_in" && !echoCancellation ? (
          <span className="shrink-0 text-[10px] text-amber-600 dark:text-amber-400">
            {t("live.echoWarning")}
          </span>
        ) : null}
      </div>
      {error ? (
        <div className="flex items-start gap-2 rounded-xl bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
          <span className="min-w-0 flex-1" dir="auto">
            {error}
          </span>
          <button
            type="button"
            className="shrink-0 rounded-full p-0.5 hover:bg-destructive/15"
            aria-label={t("live.dismissError")}
            onClick={() => useLiveStore.getState().setError(null)}
          >
            <X className="size-3.5" />
          </button>
        </div>
      ) : null}
    </div>
  );
};
