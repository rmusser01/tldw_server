import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { AssistantVoiceCard } from "../AssistantVoiceCard"
import { LiveSessionPanel } from "../LiveSessionPanel"

const createResolvedDefaults = (): React.ComponentProps<
  typeof AssistantVoiceCard
>["resolvedDefaults"] => ({
    sttLanguage: "en-US",
    sttModel: "whisper-1",
    ttsProvider: "openai",
    ttsVoice: "alloy",
    confirmationMode: "destructive_only",
    wakeBehavior: "one_shot",
    voiceChatTriggerPhrases: ["hey helper"],
    autoResume: true,
    bargeIn: false,
    autoCommitEnabled: true,
    vadThreshold: 0.5,
    minSilenceMs: 250,
    turnStopSecs: 0.2,
    minUtteranceSecs: 0.4
  })

const defaultVoiceCardProps = (): React.ComponentProps<typeof AssistantVoiceCard> => ({
  resolvedDefaults: createResolvedDefaults(),
  connected: true,
  state: "idle",
  speechAvailable: true,
  isListening: false,
  heardText: "",
  lastCommittedText: "",
  activeToolStatus: "",
  pendingApprovalSummary: null as string | null,
  warning: null as string | null,
  recoveryMode: "none" as const,
  manualModeRequired: false,
  canSendNow: false,
  textOnlyDueToTtsFailure: false,
  sessionAutoResume: true,
  sessionBargeIn: false,
  wakeArmed: false,
  wakeDetectorState: "idle",
  wakeWarning: null,
  wakeTriggerPhrases: ["hey helper"],
  sessionWakeBehavior: "one_shot",
  autoCommitEnabled: true,
  vadPreset: "balanced",
  vadThreshold: 0.5,
  minSilenceMs: 250,
  turnStopSecs: 0.2,
  minUtteranceSecs: 0.4,
  onToggleListening: vi.fn(),
  onSendNow: vi.fn(),
  onSessionAutoResumeChange: vi.fn(),
  onSessionBargeInChange: vi.fn(),
  onToggleWakeArmed: vi.fn(),
  onSessionWakeBehaviorChange: vi.fn(),
  onAutoCommitEnabledChange: vi.fn(),
  onVadPresetChange: vi.fn(),
  onVadThresholdChange: vi.fn(),
  onMinSilenceMsChange: vi.fn(),
  onTurnStopSecsChange: vi.fn(),
  onMinUtteranceSecsChange: vi.fn(),
  onKeepListening: vi.fn(),
  onResetTurn: vi.fn(),
  onWaitOnRecovery: vi.fn(),
  onCopyLastCommandToComposer: vi.fn(),
  onJumpToApproval: vi.fn(),
  onReconnectPersonaSession: vi.fn()
})

describe("LiveSessionPanel", () => {
  it("renders the assistant voice card before status panels", () => {
    render(
      <LiveSessionPanel
        controls={<div>controls</div>}
        assistantVoice={<div data-testid="assistant-voice-slot">voice card</div>}
        error={<div>errors</div>}
        pendingPlan={<div>plan</div>}
        transcript={<div>transcript</div>}
        composer={<div>composer</div>}
      />
    )

    expect(screen.getByTestId("assistant-voice-slot")).toBeInTheDocument()
    expect(screen.getByText("errors")).toBeInTheDocument()
  })
})

describe("AssistantVoiceCard", () => {
  it("renders resolved defaults, session toggles, and warning state", () => {
    const props = defaultVoiceCardProps()
    props.state = "speaking"
    props.heardText = "hey helper open notes"
    props.lastCommittedText = "open notes"
    props.warning = "Live TTS unavailable for this session. Continuing in text-only mode."
    props.textOnlyDueToTtsFailure = true
    props.manualModeRequired = true
    props.canSendNow = true
    props.sessionAutoResume = false
    props.sessionBargeIn = true

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByText("Assistant Voice")).toBeInTheDocument()
    expect(screen.getByTestId("live-voice-trigger-phrases")).toHaveTextContent(
      "hey helper"
    )
    expect(screen.getByTestId("live-voice-warning")).toHaveTextContent(
      "Continuing in text-only mode"
    )
    expect(screen.getByTestId("live-voice-heard-text")).toHaveTextContent(
      "hey helper open notes"
    )
    expect(screen.getByTestId("live-voice-last-commit")).toHaveTextContent("open notes")

    fireEvent.click(screen.getByTestId("live-voice-start-stop"))
    expect(props.onToggleListening).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-send-now"))
    expect(props.onSendNow).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-auto-resume"))
    expect(props.onSessionAutoResumeChange).toHaveBeenCalledWith(true)

    fireEvent.click(screen.getByTestId("live-voice-barge-in"))
    expect(props.onSessionBargeInChange).toHaveBeenCalledWith(false)
  })

  it("renders the turn detection section with auto-commit and presets", () => {
    const props = defaultVoiceCardProps()

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByText("Turn detection")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-auto-commit")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-preset-conservative")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-preset-balanced")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-preset-fast")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("live-vad-auto-commit"))
    expect(props.onAutoCommitEnabledChange).toHaveBeenCalledWith(false)

    fireEvent.click(screen.getByTestId("live-vad-preset-fast"))
    expect(props.onVadPresetChange).toHaveBeenCalledWith("fast")
  })

  it("renders wake phrase controls and current saved wake phrases", () => {
    const props = defaultVoiceCardProps()

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByTestId("live-wake-toggle")).toHaveTextContent(
      /listen for wake phrase/i
    )
    expect(screen.getByTestId("live-wake-phrases")).toHaveTextContent("hey helper")

    fireEvent.click(screen.getByTestId("live-wake-toggle"))
    expect(props.onToggleWakeArmed).toHaveBeenCalledTimes(1)
  })

  it("does not arm wake from resolved fallback trigger phrases when session wake props are omitted", () => {
    const props = defaultVoiceCardProps()
    props.resolvedDefaults = {
      ...props.resolvedDefaults,
      voiceChatTriggerPhrases: ["fallback wake"],
      wakeBehavior: "continuous"
    }
    props.wakeTriggerPhrases = undefined
    props.sessionWakeBehavior = undefined

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByTestId("live-wake-phrases")).toHaveTextContent(
      /add a trigger phrase/i
    )
    expect(screen.getByTestId("live-wake-toggle")).toBeDisabled()
    expect(screen.getByTestId("live-wake-behavior")).toHaveTextContent("Continuous")
  })

  it("blocks wake toggle display when no saved wake phrases are configured", () => {
    const props = defaultVoiceCardProps()
    props.wakeTriggerPhrases = []

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByTestId("live-wake-toggle")).toBeDisabled()
    expect(screen.getByText(/add a trigger phrase/i)).toBeInTheDocument()
  })

  it("shows the advanced drawer and current runtime values", () => {
    const props = defaultVoiceCardProps()
    props.vadThreshold = 0.61
    props.minSilenceMs = 640
    props.turnStopSecs = 0.48
    props.minUtteranceSecs = 0.82

    render(<AssistantVoiceCard {...props} />)

    fireEvent.click(screen.getByTestId("live-vad-advanced-toggle"))

    expect(screen.getByText("Speech threshold")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-threshold")).toHaveDisplayValue("0.61")
    expect(screen.getByTestId("live-vad-min-silence-ms")).toHaveDisplayValue("640")
    expect(screen.getByTestId("live-vad-turn-stop-secs")).toHaveDisplayValue("0.48")
    expect(screen.getByTestId("live-vad-min-utterance-secs")).toHaveDisplayValue("0.82")
  })

  it("disables turn detection tuning while disconnected or when manual mode is required", () => {
    const disconnectedProps = defaultVoiceCardProps()
    disconnectedProps.connected = false
    disconnectedProps.canSendNow = true

    const { rerender } = render(<AssistantVoiceCard {...disconnectedProps} />)

    expect(screen.getByTestId("live-voice-send-now")).toBeDisabled()
    expect(screen.getByTestId("live-voice-start-stop")).toBeDisabled()
    expect(screen.getByTestId("live-voice-auto-resume")).toBeDisabled()
    expect(screen.getByTestId("live-voice-barge-in")).toBeDisabled()
    expect(screen.getByTestId("live-vad-auto-commit")).toBeDisabled()
    fireEvent.click(screen.getByTestId("live-vad-advanced-toggle"))
    expect(screen.getByText("Connect to tune live turn detection for this session.")).toBeInTheDocument()

    const manualModeProps = defaultVoiceCardProps()
    manualModeProps.manualModeRequired = true

    rerender(<AssistantVoiceCard {...manualModeProps} />)

    expect(screen.getByTestId("live-vad-auto-commit")).toBeDisabled()
    expect(screen.getByTestId("live-vad-threshold")).toBeDisabled()
    expect(
      screen.getByText("Turn detection tuning will apply once server auto-commit is available again.")
    ).toBeInTheDocument()
  })

  it("shows custom as the active preset when advanced values diverge", () => {
    const props = defaultVoiceCardProps()
    props.vadPreset = "custom"

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByTestId("live-vad-preset-custom")).toBeInTheDocument()
    expect(screen.getByTestId("live-vad-preset-custom")).toHaveAttribute(
      "data-active",
      "true"
    )
  })

  it("renders compact persona visual-state feedback when provided", () => {
    const props = defaultVoiceCardProps()
    props.visualStateFeedback = {
      state: "tool_running",
      source: "override",
      reason: "persona_visuals.trigger_state",
      fallbackReason: "idle fallback"
    }

    render(<AssistantVoiceCard {...props} />)

    expect(screen.getByTestId("live-visual-state-feedback")).toHaveTextContent(
      "tool_running"
    )
    expect(screen.getByTestId("live-visual-state-feedback")).toHaveTextContent(
      "override"
    )
    expect(screen.getByTestId("live-visual-state-feedback")).toHaveTextContent(
      "persona_visuals.trigger_state"
    )
    expect(screen.getByTestId("live-visual-state-feedback")).toHaveTextContent(
      "idle fallback"
    )
  })

  it("renders listening recovery copy and actions", () => {
    const onKeepListening = vi.fn()
    const onResetTurn = vi.fn()
    const onReconnectPersonaSession = vi.fn()

    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="listening"
        speechAvailable
        isListening
        heardText="hey helper search my notes"
        lastCommittedText=""
        activeToolStatus=""
        pendingApprovalSummary={null}
        warning={null}
        recoveryMode="listening_stuck"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={onKeepListening}
        onResetTurn={onResetTurn}
        onWaitOnRecovery={vi.fn()}
        onCopyLastCommandToComposer={vi.fn()}
        onJumpToApproval={vi.fn()}
        onReconnectPersonaSession={onReconnectPersonaSession}
      />
    )

    expect(screen.getByText("Voice turn needs attention")).toBeInTheDocument()
    expect(
      screen.getByText("I heard speech, but this turn has not been committed yet.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("live-voice-recovery-keep-listening"))
    expect(onKeepListening).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-recovery-reset-turn"))
    expect(onResetTurn).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-recovery-reconnect"))
    expect(onReconnectPersonaSession).toHaveBeenCalledTimes(1)
  })

  it("renders thinking recovery copy and actions", () => {
    const onWaitOnRecovery = vi.fn()
    const onCopyLastCommandToComposer = vi.fn()
    const onResetTurn = vi.fn()
    const onReconnectPersonaSession = vi.fn()

    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="thinking"
        speechAvailable
        isListening={false}
        heardText="hey helper search my notes"
        lastCommittedText="search my notes"
        activeToolStatus=""
        pendingApprovalSummary={null}
        warning={null}
        recoveryMode="thinking_stuck"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow={false}
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={vi.fn()}
        onResetTurn={onResetTurn}
        onWaitOnRecovery={onWaitOnRecovery}
        onCopyLastCommandToComposer={onCopyLastCommandToComposer}
        onJumpToApproval={vi.fn()}
        onReconnectPersonaSession={onReconnectPersonaSession}
      />
    )

    expect(screen.getByText("Assistant response is delayed")).toBeInTheDocument()
    expect(
      screen.getByText("This voice turn was sent, but the assistant has not responded yet.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("live-voice-recovery-wait"))
    expect(onWaitOnRecovery).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-recovery-copy-command"))
    expect(onCopyLastCommandToComposer).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-recovery-reset-turn"))
    expect(onResetTurn).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("live-voice-recovery-reconnect"))
    expect(onReconnectPersonaSession).toHaveBeenCalledTimes(1)
  })

  it("renders the current action line while thinking with active tool status", () => {
    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="thinking"
        speechAvailable
        isListening={false}
        heardText=""
        lastCommittedText="search my notes"
        activeToolStatus="Running search_notes: Looking through your notes"
        pendingApprovalSummary={null}
        warning={null}
        recoveryMode="none"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow={false}
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={vi.fn()}
        onResetTurn={vi.fn()}
        onWaitOnRecovery={vi.fn()}
        onCopyLastCommandToComposer={vi.fn()}
        onJumpToApproval={vi.fn()}
        onReconnectPersonaSession={vi.fn()}
      />
    )

    expect(screen.getByText("Current action")).toBeInTheDocument()
    expect(
      screen.getByText("Running search_notes: Looking through your notes")
    ).toBeInTheDocument()
  })

  it("hides the current action line when active tool status is empty", () => {
    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="thinking"
        speechAvailable
        isListening={false}
        heardText=""
        lastCommittedText="search my notes"
        activeToolStatus=""
        pendingApprovalSummary={null}
        warning={null}
        recoveryMode="none"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow={false}
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={vi.fn()}
        onResetTurn={vi.fn()}
        onWaitOnRecovery={vi.fn()}
        onCopyLastCommandToComposer={vi.fn()}
        onJumpToApproval={vi.fn()}
        onReconnectPersonaSession={vi.fn()}
      />
    )

    expect(screen.queryByText("Current action")).not.toBeInTheDocument()
  })

  it("renders approval summary text in the current action block", () => {
    const onJumpToApproval = vi.fn()

    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="thinking"
        speechAvailable
        isListening={false}
        heardText=""
        lastCommittedText="search my notes"
        activeToolStatus=""
        pendingApprovalSummary="Waiting for approval: search_notes (+1 more)"
        warning={null}
        recoveryMode="none"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow={false}
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={vi.fn()}
        onResetTurn={vi.fn()}
        onWaitOnRecovery={vi.fn()}
        onCopyLastCommandToComposer={vi.fn()}
        onJumpToApproval={onJumpToApproval}
        onReconnectPersonaSession={vi.fn()}
      />
    )

    expect(screen.getByText("Current action")).toBeInTheDocument()
    expect(
      screen.getByText("Waiting for approval: search_notes (+1 more)")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("live-voice-jump-to-approval"))
    expect(onJumpToApproval).toHaveBeenCalledTimes(1)
  })

  it("prefers approval summary over active tool status", () => {
    render(
      <AssistantVoiceCard
        resolvedDefaults={createResolvedDefaults()}
        state="thinking"
        speechAvailable
        isListening={false}
        heardText=""
        lastCommittedText="search my notes"
        activeToolStatus="Running search_notes: Looking through your notes"
        pendingApprovalSummary="Waiting for approval: search_notes"
        warning={null}
        recoveryMode="none"
        textOnlyDueToTtsFailure={false}
        manualModeRequired={false}
        canSendNow={false}
        sessionAutoResume
        sessionBargeIn={false}
        onToggleListening={vi.fn()}
        onSendNow={vi.fn()}
        onSessionAutoResumeChange={vi.fn()}
        onSessionBargeInChange={vi.fn()}
        onKeepListening={vi.fn()}
        onResetTurn={vi.fn()}
        onWaitOnRecovery={vi.fn()}
        onCopyLastCommandToComposer={vi.fn()}
        onJumpToApproval={vi.fn()}
        onReconnectPersonaSession={vi.fn()}
      />
    )

    expect(screen.getByText("Waiting for approval: search_notes")).toBeInTheDocument()
    expect(
      screen.queryByText("Running search_notes: Looking through your notes")
    ).not.toBeInTheDocument()
  })
})
