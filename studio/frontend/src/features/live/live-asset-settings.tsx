// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { useState } from "react";
import { downloadLiveAsset } from "./live-client";
import { liveController } from "./live-controller";
import { useLiveStore } from "./live-store";
import type { LiveAssetStatus } from "./live-types";

/**
 * Pinned VAD/endpoint models.  They are optional: downloading one enables the
 * ONNX path (when onnxruntime is present) instead of the energy/silence
 * fallback, and a failed download leaves the fallback untouched.
 */
export const LiveAssetSettings = () => {
  const t = useT();
  const status = useLiveStore((state) => state.status);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const onnxMissing = status?.capabilities.onnxruntime === "missing";
  const assets = status?.assets ?? [];

  if (!status) {
    return null;
  }

  const stateLabel = (asset: LiveAssetStatus): string => {
    if (asset.active) {
      return t("live.settings.assetActive");
    }
    if (asset.installed) {
      return onnxMissing
        ? t("live.settings.assetRuntimeMissing")
        : t("live.settings.assetInstalled");
    }
    return t("live.settings.assetNotInstalled");
  };

  const install = async (role: string): Promise<void> => {
    setError(null);
    setDownloading(role);
    try {
      await downloadLiveAsset(role);
      await liveController.ensureStatus(true);
    } catch (downloadError) {
      setError(
        downloadError instanceof Error && downloadError.message
          ? downloadError.message
          : t("live.settings.assetDownloadFailed"),
      );
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border/60 p-3">
      <span className="text-xs font-medium">
        {t("live.settings.detectorModels")}
      </span>
      {assets.map((asset) => (
        <div
          key={asset.role}
          className="flex items-center justify-between gap-3 text-xs"
        >
          <div className="min-w-0">
            <div className="truncate">{asset.name || asset.role}</div>
            <div className="text-muted-foreground">
              {stateLabel(asset)}
              {asset.sizeBytes
                ? ` · ${(asset.sizeBytes / (1024 * 1024)).toFixed(1)} MB`
                : ""}
            </div>
          </div>
          {asset.active ? null : (
            <button
              type="button"
              className="shrink-0 rounded-full border border-border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-60"
              disabled={downloading !== null}
              onClick={() => void install(asset.role)}
            >
              {downloading === asset.role
                ? t("live.settings.assetDownloading")
                : t("live.settings.assetDownload")}
            </button>
          )}
        </div>
      ))}
      {onnxMissing ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("live.settings.onnxRuntimeMissing")}
        </p>
      ) : null}
      {error ? <p className="text-xs text-destructive">{error}</p> : null}
    </div>
  );
};
