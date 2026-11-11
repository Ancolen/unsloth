// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { type FC, useEffect } from "react";
import { LiveAssetSettings } from "./live-asset-settings";
import { liveController } from "./live-controller";
import { LiveEndpointSettings } from "./live-endpoint-settings";
import { liveScreenSupported } from "./live-screen";
import { useLiveStore } from "./live-store";

const LANGUAGES = [
  { value: "tr", label: "live.settings.languageTr" },
  { value: "en", label: "live.settings.languageEn" },
  { value: "auto", label: "live.settings.languageAuto" },
] as const;

/**
 * Compact Live settings block.  It is rendered inside Settings → Voice so the
 * same persisted store drives the composer and the live bar.
 */
export const LiveSettings: FC = () => {
  const t = useT();
  const mode = useLiveStore((state) => state.mode);
  const language = useLiveStore((state) => state.language);
  const attachScreen = useLiveStore((state) => state.attachScreen);
  const allowCloudScreen = useLiveStore((state) => state.allowCloudScreen);
  const interruptWhileThinking = useLiveStore(
    (state) => state.interruptWhileThinking,
  );
  const echoCancellation = useLiveStore((state) => state.echoCancellation);
  const enabled = useLiveStore((state) => state.enabled);
  const status = useLiveStore((state) => state.status);
  const ttsEngine = useVoiceSettingsStore((state) => state.ttsEngine);
  const screenSupported = liveScreenSupported();
  const fakeEngine =
    status?.capabilities.stt === "fake" || status?.capabilities.tts === "fake";
  const systemVoiceInactive =
    ttsEngine === "system" && status?.capabilities.tts === "studio";
  const detectorLabel = (value: string | undefined): string => {
    if (!status) {
      return t("live.settings.checking");
    }
    if (value === "silero") {
      return t("live.settings.vadSilero");
    }
    if (value === "energy") {
      return t("live.settings.vadEnergy");
    }
    if (value === "smart_turn") {
      return t("live.settings.endpointSmartTurn");
    }
    if (value === "silence") {
      return t("live.settings.endpointSilence");
    }
    return value || t("live.settings.unknown");
  };

  useEffect(() => {
    void liveController.ensureStatus();
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted-foreground">
          {enabled ? t("live.settings.enabled") : t("live.settings.disabled")}
        </span>
      </div>

      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <span className="text-muted-foreground">
          {t("live.settings.sttEngine")}
        </span>
        <span>
          {status
            ? status.capabilities.stt || t("live.settings.unknown")
            : t("live.settings.checking")}
        </span>
        <span className="text-muted-foreground">
          {t("live.settings.ttsEngine")}
        </span>
        <span>
          {status
            ? status.capabilities.tts || t("live.settings.unknown")
            : t("live.settings.checking")}
        </span>
        <span className="text-muted-foreground">
          {t("live.settings.screenBackend")}
        </span>
        <span>
          {screenSupported
            ? status?.capabilities.screenCapture || t("live.settings.unknown")
            : t("live.settings.screenDesktopOnly")}
        </span>
        <span className="text-muted-foreground">{t("live.settings.vad")}</span>
        <span>{detectorLabel(status?.capabilities.vad)}</span>
        <span className="text-muted-foreground">
          {t("live.settings.endpoint")}
        </span>
        <span>{detectorLabel(status?.capabilities.endpoint)}</span>
      </div>

      <LiveAssetSettings />
      <LiveEndpointSettings />

      {fakeEngine ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("live.settings.fakeEngineWarning")}
        </p>
      ) : null}
      {systemVoiceInactive ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("live.settings.systemTtsWarning")}
        </p>
      ) : null}

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
          {LANGUAGES.map((language) => (
            <option key={language.value} value={language.value}>
              {t(language.label)}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center justify-between gap-4 text-sm">
        <span>{t("live.settings.attachScreen")}</span>
        <input
          type="checkbox"
          checked={attachScreen}
          disabled={!screenSupported}
          onChange={(event) =>
            useLiveStore.getState().setAttachScreen(event.target.checked)
          }
        />
      </label>
      {screenSupported ? null : (
        <p className="text-xs text-muted-foreground">
          {t("live.settings.screenDesktopOnly")}
        </p>
      )}
      {attachScreen ? (
        <>
          <p className="text-xs text-amber-600 dark:text-amber-400">
            {t("live.settings.attachScreenCloud")}
          </p>
          <label className="flex items-center justify-between gap-4 text-sm">
            <span>{t("live.settings.allowCloudScreen")}</span>
            <input
              type="checkbox"
              checked={allowCloudScreen}
              onChange={(event) =>
                useLiveStore
                  .getState()
                  .setAllowCloudScreen(event.target.checked)
              }
            />
          </label>
        </>
      ) : null}
      <p className="text-xs text-muted-foreground">
        {t("live.settings.reuseHint")}
      </p>

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
