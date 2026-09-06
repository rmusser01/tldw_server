import React from "react"
import "@testing-library/jest-dom/vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { AssistantVoiceCard } from "../AssistantVoiceCard"

it("offers Stop during preparation and generation without claiming speech is ready", () => {
  const stop = vi.fn()
  const props = {
    connected: true,
    state: "thinking" as const,
    speechAvailable: true,
    isListening: false,
    isVoiceActive: true,
    voiceReady: false,
    isPreparing: true,
    heardText: "",
    lastCommittedText: "",
    activeToolStatus: "",
    pendingApprovalSummary: null,
    warning: null,
    recoveryMode: "none" as const,
    manualModeRequired: false,
    canSendNow: false,
    textOnlyDueToTtsFailure: false,
    sessionAutoResume: true,
    sessionBargeIn: false,
    onToggleListening: stop,
    onSendNow: vi.fn(),
    onSessionAutoResumeChange: vi.fn(),
    onSessionBargeInChange: vi.fn(),
    onKeepListening: vi.fn(),
    onResetTurn: vi.fn(),
    onWaitOnRecovery: vi.fn(),
    onCopyLastCommandToComposer: vi.fn(),
    onJumpToApproval: vi.fn(),
    onReconnectPersonaSession: vi.fn(),
    resolvedDefaults: {
      sttLanguage: "en",
      sttModel: "whisper",
      ttsProvider: "kokoro",
      ttsVoice: "af_heart",
      confirmationMode: "destructive_only" as const,
      wakeBehavior: "one_shot" as const,
      voiceChatTriggerPhrases: [],
      autoResume: true,
      bargeIn: false,
      autoCommitEnabled: true,
      vadThreshold: 0.5,
      minSilenceMs: 250,
      turnStopSecs: 0.2,
      minUtteranceSecs: 0.4
    }
  }
  const view = render(<AssistantVoiceCard {...props} />)
  expect(screen.getByText(/Preparing the selected speech/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Stop voice" }))
  expect(stop).toHaveBeenCalledOnce()
  view.rerender(
    <AssistantVoiceCard
      {...props}
      isPreparing={false}
      isVoiceActive={false}
      state="idle"
    />
  )
  expect(
    screen.getByText(/Start checks the selected speech/)
  ).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Start listening" })).toBeEnabled()
  view.rerender(<AssistantVoiceCard {...props} isPreparing={false} voiceReady={true} autoCommitEnabled={false} />)
  expect(screen.getByText("Server speech transcription ready. Use Send now to commit manually.")).toBeInTheDocument()
  expect(screen.getByRole("checkbox", { name: "Auto-commit (session only)" })).toBeEnabled()
})
