// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { useLiveStore } from "./live-store";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(value)));
}

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (next: number) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: NumberFieldProps) {
  return (
    <label className="flex items-center justify-between gap-4 text-sm">
      <span>{label}</span>
      <input
        type="number"
        className="w-24 rounded-lg border border-border bg-background px-2 py-1 text-sm"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) =>
          onChange(clamp(Number(event.target.value) || min, min, max))
        }
      />
    </label>
  );
}

/**
 * Endpoint tuning is deliberately tucked behind a disclosure: the defaults fit
 * most speakers, and a bad value mainly makes the assistant cut in earlier or
 * later.  Values are sent in the next hello, so they apply on the next start.
 */
export const LiveEndpointSettings = () => {
  const t = useT();
  const fast = useLiveStore((state) => state.endpointFastMs);
  const extended = useLiveStore((state) => state.endpointExtendedMs);
  const maximum = useLiveStore((state) => state.maxUtteranceMs);
  const store = useLiveStore.getState();

  return (
    <details className="rounded-xl border border-border/60 p-3 text-sm">
      <summary className="cursor-pointer text-xs font-medium">
        {t("live.settings.advancedEndpointing")}
      </summary>
      <div className="mt-3 flex flex-col gap-2">
        <NumberField
          label={t("live.settings.endpointFastMs")}
          value={fast}
          min={100}
          max={2000}
          step={50}
          onChange={store.setEndpointFastMs}
        />
        <NumberField
          label={t("live.settings.endpointExtendedMs")}
          value={extended}
          min={200}
          max={5000}
          step={50}
          onChange={store.setEndpointExtendedMs}
        />
        <NumberField
          label={t("live.settings.maxUtteranceMs")}
          value={maximum}
          min={1000}
          max={120000}
          step={1000}
          onChange={store.setMaxUtteranceMs}
        />
        <p className="text-xs text-muted-foreground">
          {t("live.settings.endpointHint")}
        </p>
      </div>
    </details>
  );
};
