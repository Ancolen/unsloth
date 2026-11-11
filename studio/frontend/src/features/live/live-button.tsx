// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { useChatActive } from "@/features/chat";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Square, Waves } from "lucide-react";
import { type FC, useEffect } from "react";
import { liveController } from "./live-controller";
import { useLiveStore } from "./live-store";

/** Composer action beside the microphone: one click opens the live voice bar. */
export const LiveButton: FC<{ disabled?: boolean }> = ({ disabled }) => {
  const t = useT();
  const active = useLiveStore((state) => state.active);
  const enabled = useLiveStore((state) => state.enabled);
  const state = useLiveStore((state) => state.state);
  const chatActive = useChatActive();

  useEffect(() => {
    void liveController.ensureStatus();
  }, []);

  if (!(chatActive && enabled)) {
    return null;
  }

  const busy = active && state === "connecting";
  const label = active ? t("live.stop") : t("live.start");

  return (
    <TooltipIconButton
      tooltip={label}
      aria-label={label}
      aria-pressed={active}
      side="bottom"
      type="button"
      className={cn(
        "size-9 rounded-full text-foreground",
        active && "bg-primary/10 text-primary hover:bg-primary/15",
      )}
      disabled={disabled || busy}
      onClick={() => {
        if (active) {
          liveController.stop();
        } else {
          void liveController.start();
        }
      }}
    >
      {active ? (
        <Square className="unsloth-live-icon size-4" strokeWidth={2.2} />
      ) : (
        <Waves className="unsloth-live-icon size-5" strokeWidth={1.8} />
      )}
    </TooltipIconButton>
  );
};
