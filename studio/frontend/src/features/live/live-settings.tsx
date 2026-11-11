// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { FC } from "react";
import { liveController } from "./live-controller";
import { useLiveStore } from "./live-store";

const LANGUAGES = ["tr", "en", "auto"];

/**
 * Compact Live settings block.  It is rendered inside Settings → Voice so the
 * same persisted store drives the composer and the live bar.
 */
export const LiveSettings: FC = () => {
  const t = useT();
  const mode = useLiveStore((state) => state.mode);
  const language = useLiveStore((state) => state.language);
  const attachScreen = useLiveStore((state) => state.attachScreen);
  const interruptWhileThinking = useLiveStore(
    (state) => state.interruptWhileThinking,
  );
  const echoCancellation = useLiveStore((state) => state.echoCancellation);
  const enabled = useLiveStore((state) => state.enabled);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium">{t("live.settings.title")}</span>
        <span className="text-xs text-muted-foreground">
          {enabled ? t("live.settings.enabled") : t("live.settings.disabled")}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className={cn(
            "rounded-full border px-3 py-1.5 text-xs",
            mode === "turn"
              ? "border-primary bg-primary/10 text-primary"
              : "border-border",
          )}
          onClick={() => liveController.setMode("turn")}
        >
          {t("live.mode.turn")}
        </button>
        <button
          type="button"
          className={cn(
            "rounded-full border px-3 py-1.5 text-xs",
            mode === "barge_in"
              ? "border-primary bg-primary/10 text-primary"
              : "border-border",
          )}
          onClick={() => liveController.setMode("barge_in")}
        >
          {t("live.mode.bargeIn")}
        </button>
      </div>

      <label className="flex items-center justify-between gap-4 text-sm">
        <span>{t("live.settings.language")}</span>
        <select
          className="rounded-lg border border-border bg-background px-2 py-1 text-sm"
          value={language}
          onChange={(event) => {
            useLiveStore.getState().setLanguage(event.target.value);
            liveController.setLanguage(event.target.value);
          }}
        >
          {LANGUAGES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center justify-between gap-4 text-sm">
        <span>{t("live.settings.attachScreen")}</span>
        <input
          type="checkbox"
          checked={attachScreen}
          onChange={(event) =>
            useLiveStore.getState().setAttachScreen(event.target.checked)
          }
        />
      </label>

      <label className="flex items-center justify-between gap-4 text-sm">
        <span>{t("live.settings.interruptWhileThinking")}</span>
        <input
          type="checkbox"
          checked={interruptWhileThinking}
          onChange={(event) =>
            useLiveStore
              .getState()
              .setInterruptWhileThinking(event.target.checked)
          }
        />
      </label>

      {mode === "barge_in" && !echoCancellation ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("live.echoWarning")}
        </p>
      ) : null}
    </div>
  );
};
