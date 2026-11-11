// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  LiveSentenceStream,
  extractAssistantSpeechText,
} from "../src/features/live/live-tts.ts";

test("extractAssistantSpeechText keeps visible text and drops reasoning", () => {
  const content = [
    { type: "reasoning", text: "internal" },
    { type: "text", text: "Merhaba." },
    { type: "tool-call", text: "ignored" },
    { type: "text", text: "Nasılsın?" },
  ];
  assert.equal(extractAssistantSpeechText(content), "Merhaba.\nNasılsın?");
  assert.equal(extractAssistantSpeechText("plain"), "plain");
  assert.equal(extractAssistantSpeechText(null), "");
});

test("LiveSentenceStream withholds the trailing sentence until the run ends", () => {
  const stream = new LiveSentenceStream();
  assert.deepEqual(stream.update("Merhaba. Nasılsın", false), ["Merhaba."]);
  assert.deepEqual(stream.update("Merhaba. Nasılsın?", true), ["Nasılsın?"]);
  assert.deepEqual(stream.update("Merhaba. Nasılsın?", true), []);
});

test("LiveSentenceStream restarts when the spoken prefix changes", () => {
  const stream = new LiveSentenceStream();
  assert.deepEqual(stream.update("Birinci. İkinci.", false), ["Birinci."]);
  assert.deepEqual(stream.update("Yeni. Cevap.", true), ["Yeni.", "Cevap."]);
});
