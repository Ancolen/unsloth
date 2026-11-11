// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { markdownToSpeechText } from "../chat/utils/speech-text.ts";
import { splitIntoSentences } from "../chat/voice/speech-text.ts";

/**
 * Drop fenced code blocks before the shared markdown-to-speech conversion.
 * Read-aloud intentionally speaks code, but a voice conversation should not read
 * a whole `print(...)` dump; an unclosed fence is dropped to the end, which is
 * also the right behavior while the reply is still streaming.
 */
function stripFencedCode(markdown: string): string {
  const kept: string[] = [];
  let fence: string | null = null;
  for (const line of markdown.split("\n")) {
    const opening = /^\s*(`{3,}|~{3,})/.exec(line);
    if (fence !== null) {
      if (
        opening &&
        opening[1][0] === fence[0] &&
        opening[1].length >= fence.length
      ) {
        fence = null;
      }
      continue;
    }
    if (opening) {
      fence = opening[1];
      continue;
    }
    kept.push(line);
  }
  return kept.join("\n");
}

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
    const clean = markdownToSpeechText(stripFencedCode(text)).trim();
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
