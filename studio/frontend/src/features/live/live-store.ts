// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { LiveMode, LiveState, LiveStatus } from "./live-types";

interface LiveSettings {
  mode: LiveMode;
  language: string;
  sttModel: string;
  ttsModel: string;
  voice: string;
  attachScreen: boolean;
  interruptWhileThinking: boolean;
}

interface LiveRuntime {
  enabled: boolean;
  active: boolean;
  state: LiveState;
  error: string | null;
  level: number;
  transcript: string;
  partial: string;
  muted: boolean;
  screenActive: boolean;
  screenStarting: boolean;
  screenPaused: boolean;
  screenFrame: string | null;
  modelsReady: boolean;
  echoCancellation: boolean;
  status: LiveStatus | null;
}

export interface LiveStore extends LiveRuntime, LiveSettings {
  setEnabled: (enabled: boolean) => void;
  setStatus: (status: LiveStatus | null) => void;
  setActive: (active: boolean) => void;
  setState: (state: LiveState) => void;
  setError: (error: string | null) => void;
  setLevel: (level: number) => void;
  setTranscript: (transcript: string) => void;
  setPartial: (partial: string) => void;
  setMuted: (muted: boolean) => void;
  setScreenActive: (active: boolean) => void;
  setScreenStarting: (starting: boolean) => void;
  setScreenPaused: (paused: boolean) => void;
  setScreenFrame: (frame: string | null) => void;
  setModelsReady: (ready: boolean) => void;
  setEchoCancellation: (supported: boolean) => void;
  setMode: (mode: LiveMode) => void;
  setLanguage: (language: string) => void;
  setSttModel: (model: string) => void;
  setTtsModel: (model: string) => void;
  setVoice: (voice: string) => void;
  setAttachScreen: (attach: boolean) => void;
  setInterruptWhileThinking: (interrupt: boolean) => void;
  resetRuntime: () => void;
}

const initialRuntime: LiveRuntime = {
  enabled: true,
  active: false,
  state: "idle",
  error: null,
  level: 0,
  transcript: "",
  partial: "",
  muted: false,
  screenActive: false,
  screenStarting: false,
  screenPaused: false,
  screenFrame: null,
  modelsReady: false,
  echoCancellation: false,
  status: null,
};

const initialSettings: LiveSettings = {
  mode: "turn",
  language: "tr",
  sttModel: "",
  ttsModel: "",
  voice: "",
  attachScreen: false,
  interruptWhileThinking: false,
};

export const useLiveStore = create<LiveStore>()(
  persist(
    (set) => ({
      ...initialRuntime,
      ...initialSettings,
      setEnabled: (enabled) => set({ enabled }),
      setStatus: (status) =>
        set((state) => ({
          status,
          enabled: status?.enabled ?? state.enabled,
        })),
      setActive: (active) => set({ active }),
      setState: (state) => set({ state }),
      setError: (error) => set({ error }),
      setLevel: (level) => set({ level }),
      setTranscript: (transcript) => set({ transcript }),
      setPartial: (partial) => set({ partial }),
      setMuted: (muted) => set({ muted }),
      setScreenActive: (screenActive) => set({ screenActive }),
      setScreenStarting: (screenStarting) => set({ screenStarting }),
      setScreenPaused: (screenPaused) => set({ screenPaused }),
      setScreenFrame: (screenFrame) => set({ screenFrame }),
      setModelsReady: (modelsReady) => set({ modelsReady }),
      setEchoCancellation: (echoCancellation) => set({ echoCancellation }),
      setMode: (mode) => set({ mode }),
      setLanguage: (language) => set({ language }),
      setSttModel: (sttModel) => set({ sttModel }),
      setTtsModel: (ttsModel) => set({ ttsModel }),
      setVoice: (voice) => set({ voice }),
      setAttachScreen: (attachScreen) => set({ attachScreen }),
      setInterruptWhileThinking: (interruptWhileThinking) =>
        set({ interruptWhileThinking }),
      resetRuntime: () =>
        set((state) => ({
          ...initialRuntime,
          enabled: state.enabled,
          status: state.status,
        })),
    }),
    {
      name: "unsloth-live-voice",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        mode: state.mode,
        language: state.language,
        sttModel: state.sttModel,
        ttsModel: state.ttsModel,
        voice: state.voice,
        attachScreen: state.attachScreen,
        interruptWhileThinking: state.interruptWhileThinking,
      }),
    },
  ),
);
