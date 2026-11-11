// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { markdownToSpeechText } from "../chat/utils/speech-text.ts";
import { splitIntoSentences } from "../chat/voice/speech-text.ts";

/** Pull only the visible assistant text; reasoning and tool-call parts are not spoken. */
export function extractAssistantSpeechText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") {
      continue;
    }
    const value = part as { type?: unknown; text?: unknown };
    if (value.type === "text" && typeof value.text === "string") {
      parts.push(value.text);
    }
  }
  return parts.join("\n");
}

/**
 * Incremental sentence stream for a growing assistant message.
 *
 * The trailing sentence is withheld until the run finishes: it may still grow,
 * and synthesizing it twice is worse than a slightly later last clip.  A reset
 * in the underlying text (edit, regenerate) starts the stream over.
 */
export class LiveSentenceStream {
  private sent: string[] = [];

  reset(): void {
    this.sent = [];
  }

  update(text: string, final: boolean): string[] {
    const clean = markdownToSpeechText(text).trim();
    if (!clean) {
      return [];
    }
    const sentences = splitIntoSentences(clean)
      .map((sentence) => sentence.trim())
      .filter((sentence) => /[\p{L}\p{N}]/u.test(sentence));
    if (sentences.length === 0) {
      return [];
    }

    // A regenerate/edit can replace an already-spoken prefix.  Drop the old
    // prefix and speak only the new tail; the controller cancels pending audio
    // when the message identity changes.
    let common = 0;
    while (
      common < this.sent.length &&
      common < sentences.length &&
      this.sent[common] === sentences[common]
    ) {
      common += 1;
    }
    if (common < this.sent.length) {
      this.sent = this.sent.slice(0, common);
    }
    const completeCount = final
      ? sentences.length
      : Math.max(0, sentences.length - 1);
    if (completeCount <= this.sent.length) {
      return [];
    }
    const next = sentences.slice(this.sent.length, completeCount);
    this.sent = sentences.slice(0, completeCount);
    return next;
  }
}
