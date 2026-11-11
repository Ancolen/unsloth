// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatActive } from "@/features/chat";
import { toast } from "@/lib/toast";
import { useAui, useAuiState } from "@assistant-ui/react";
import { type FC, useEffect, useRef } from "react";
import { liveController } from "./live-controller";
import { extractAssistantSpeechText } from "./live-tts";

/**
 * Mounted once inside the chat thread.  It is the only place that touches the
 * assistant-ui runtime for Live Voice: it registers the send/cancel/draft
 * bridge and observes the current assistant message for sentence-level TTS.
 */
export const LiveRuntimeBridge: FC = () => {
  const aui = useAui();
  const chatActive = useChatActive();
  const assistantId = useAuiState(({ thread }) => {
    for (let index = thread.messages.length - 1; index >= 0; index -= 1) {
      const message = thread.messages[index];
      if (message?.role === "assistant") {
        return message.id;
      }
    }
    return null;
  });
  const assistantText = useAuiState(({ thread }) => {
    for (let index = thread.messages.length - 1; index >= 0; index -= 1) {
      const message = thread.messages[index];
      if (message?.role === "assistant") {
        return extractAssistantSpeechText(message.content);
      }
    }
    return "";
  });
  const running = useAuiState(({ thread }) => thread.isRunning);
  const assistantIdRef = useRef<string | null>(assistantId);
  assistantIdRef.current = assistantId;

  useEffect(() => {
    if (!chatActive) {
      return;
    }
    return liveController.registerBridge({
      sendUserMessage: async (text, attachment) => {
        const composer = aui.composer();
        if (attachment) {
          try {
            await composer.addAttachment(attachment);
          } catch (error) {
            toast.error("The current model cannot accept the screen image.", {
              description:
                error instanceof Error
                  ? error.message
                  : "Choose a vision-capable model, or send the text only.",
            });
          }
        }
        composer.setText(text);
        composer.send();
      },
      cancelRun: () => {
        try {
          aui.thread().cancelRun();
        } catch {
          // No run to cancel.
        }
      },
      saveDraft: () => {
        const composer = aui.composer();
        const draft = composer.getState().text;
        if (draft) {
          composer.setText("");
        }
        return draft;
      },
      restoreDraft: (draft) => {
        if (!draft) {
          return;
        }
        const composer = aui.composer();
        const current = composer.getState().text;
        composer.setText(current ? `${current}\n${draft}` : draft);
      },
      isRunActive: () => aui.thread().getState().isRunning,
      currentAssistantMessageId: () => assistantIdRef.current,
    });
  }, [aui, chatActive]);

  useEffect(() => {
    if (!chatActive) {
      return;
    }
    liveController.observeAssistant(assistantId, assistantText, running);
  }, [assistantId, assistantText, running, chatActive]);

  return null;
};
