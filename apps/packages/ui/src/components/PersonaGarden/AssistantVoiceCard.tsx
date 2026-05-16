import React from "react"
import { Button, Checkbox, Select, Tag, Typography } from "antd"

import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import {
  PersonaTurnDetectionControls,
  type PersonaTurnDetectionPreset
} from "@/components/PersonaGarden/PersonaTurnDetectionControls"
import type { WakeDetectorState } from "@/hooks/personaWakeDetector"
import type {
  PersonaWakeBehavior,
  ResolvedPersonaVoiceDefaults
} from "@/hooks/useResolvedPersonaVoiceDefaults"
import type {
  PersonaLiveVoiceRecoveryMode,
  PersonaLiveVoiceState
} from "@/hooks/usePersonaLiveVoiceController"
import type { PersonaVisualStateId } from "@/types/persona-visuals"

export type PersonaVisualStateFeedback = {
  state: PersonaVisualStateId
  source:
    | "live"
    | "default"
    | "override"
    | "authored_trigger"
    | "fallback"
    | "error_recovery"
  reason?: string | null
  fallbackReason?: string | null
}

type AssistantVoiceCardProps = {
  resolvedDefaults: ResolvedPersonaVoiceDefaults
  connected?: boolean
  state: PersonaLiveVoiceState
  speechAvailable: boolean
  isListening: boolean
  heardText: string
  lastCommittedText: string
  activeToolStatus: string
  pendingApprovalSummary: string | null
  warning: string | null
  recoveryMode: PersonaLiveVoiceRecoveryMode
  manualModeRequired: boolean
  canSendNow: boolean
  textOnlyDueToTtsFailure: boolean
  showSaveCurrentSettingsAsDefaults?: boolean
  savingCurrentSettingsAsDefaults?: boolean
  sessionAutoResume: boolean
  sessionBargeIn: boolean
  wakeArmed?: boolean
  wakeDetectorState?: WakeDetectorState
  wakeWarning?: string | null
  wakeTriggerPhrases?: string[]
  sessionWakeBehavior?: PersonaWakeBehavior
  autoCommitEnabled?: boolean
  vadPreset?: PersonaTurnDetectionPreset
  vadThreshold?: number
  minSilenceMs?: number
  turnStopSecs?: number
  minUtteranceSecs?: number
  visualStateFeedback?: PersonaVisualStateFeedback
  onToggleListening: () => void
  onSendNow: () => void
  onSessionAutoResumeChange: (next: boolean) => void
  onSessionBargeInChange: (next: boolean) => void
  onToggleWakeArmed?: () => void
  onSessionWakeBehaviorChange?: (next: PersonaWakeBehavior) => void
  onAutoCommitEnabledChange?: (next: boolean) => void
  onVadPresetChange?: (next: Exclude<PersonaTurnDetectionPreset, "custom">) => void
  onVadThresholdChange?: (next: number) => void
  onMinSilenceMsChange?: (next: number) => void
  onTurnStopSecsChange?: (next: number) => void
  onMinUtteranceSecsChange?: (next: number) => void
  onKeepListening: () => void
  onResetTurn: () => void
  onWaitOnRecovery: () => void
  onCopyLastCommandToComposer: () => void
  onJumpToApproval: () => void
  onSaveCurrentSettingsAsDefaults?: () => void
  onReconnectPersonaSession: () => void
}

const formatTriggerPhrases = (phrases: string[]): string =>
  phrases.length ? phrases.join(", ") : "No trigger phrases configured"

export const AssistantVoiceCard: React.FC<AssistantVoiceCardProps> = ({
  resolvedDefaults,
  connected = true,
  state,
  speechAvailable,
  isListening,
  heardText,
  lastCommittedText,
  activeToolStatus,
  pendingApprovalSummary,
  warning,
  recoveryMode,
  manualModeRequired,
  canSendNow,
  textOnlyDueToTtsFailure,
  showSaveCurrentSettingsAsDefaults = false,
  savingCurrentSettingsAsDefaults = false,
  sessionAutoResume,
  sessionBargeIn,
  wakeArmed = false,
  wakeDetectorState = "idle",
  wakeWarning = null,
  wakeTriggerPhrases,
  sessionWakeBehavior,
  autoCommitEnabled = true,
  vadPreset = "balanced",
  vadThreshold = 0.5,
  minSilenceMs = 250,
  turnStopSecs = 0.2,
  minUtteranceSecs = 0.4,
  visualStateFeedback,
  onToggleListening,
  onSendNow,
  onSessionAutoResumeChange,
  onSessionBargeInChange,
  onToggleWakeArmed = () => undefined,
  onSessionWakeBehaviorChange = () => undefined,
  onAutoCommitEnabledChange = () => undefined,
  onVadPresetChange = () => undefined,
  onVadThresholdChange = () => undefined,
  onMinSilenceMsChange = () => undefined,
  onTurnStopSecsChange = () => undefined,
  onMinUtteranceSecsChange = () => undefined,
  onKeepListening,
  onResetTurn,
  onWaitOnRecovery,
  onCopyLastCommandToComposer,
  onJumpToApproval,
  onSaveCurrentSettingsAsDefaults = () => undefined,
  onReconnectPersonaSession
}) => {
  const sessionControlsDisabled = !connected
  const effectiveWakeTriggerPhrases = wakeTriggerPhrases ?? []
  const effectiveSessionWakeBehavior =
    sessionWakeBehavior ?? resolvedDefaults.wakeBehavior
  const normalizedWakeTriggerPhrases = React.useMemo(
    () =>
      (effectiveWakeTriggerPhrases || [])
        .map((phrase) => String(phrase || "").trim())
        .filter(Boolean),
    [effectiveWakeTriggerPhrases]
  )
  const turnDetectionDisabled = sessionControlsDisabled || manualModeRequired
  const turnDetectionHelperText = !connected
    ? "Connect to tune live turn detection for this session."
    : manualModeRequired
      ? "Turn detection tuning will apply once server auto-commit is available again."
      : !autoCommitEnabled
        ? "Auto-commit is off for this live session. Use Send now to commit heard speech."
        : "Controls when speech auto-commits in this live session only."

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <Typography.Text strong>Assistant Voice</Typography.Text>
          <Typography.Text type="secondary" className="mt-1 block text-xs">
            Saved defaults live under Profiles. The toggles here only affect this live
            session.
          </Typography.Text>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Tag color={textOnlyDueToTtsFailure ? "orange" : "blue"}>{state}</Tag>
          <Button
            data-testid="live-voice-send-now"
            size="small"
            disabled={sessionControlsDisabled || !canSendNow}
            onClick={onSendNow}
          >
            Send now
          </Button>
          <Button
            data-testid="live-voice-start-stop"
            size="small"
            type={isListening ? "default" : "primary"}
            disabled={sessionControlsDisabled || !speechAvailable}
            onClick={onToggleListening}
          >
            {isListening ? "Stop listening" : "Start listening"}
          </Button>
        </div>
      </div>

      <div className="mt-3 grid gap-2 text-xs text-text sm:grid-cols-2">
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <div className="text-text-muted">Trigger phrases</div>
          <div data-testid="live-voice-trigger-phrases" className="mt-1">
            {formatTriggerPhrases(resolvedDefaults.voiceChatTriggerPhrases)}
          </div>
        </div>
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <div className="text-text-muted">STT</div>
          <div className="mt-1">{`${resolvedDefaults.sttLanguage} · ${resolvedDefaults.sttModel || "default model"}`}</div>
        </div>
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <div className="text-text-muted">TTS</div>
          <div className="mt-1">{`${resolvedDefaults.ttsProvider} · ${resolvedDefaults.ttsVoice || "default voice"}`}</div>
        </div>
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <div className="text-text-muted">Live status</div>
          <div className="mt-1">
            {!speechAvailable
              ? "Server speech transcription is unavailable for this connection."
              : manualModeRequired
                ? "Server speech transcription is ready, but VAD auto-commit is unavailable. Use Send now."
                : "Server speech transcription ready with VAD auto-commit."}
          </div>
        </div>
      </div>

      {visualStateFeedback ? (
        <div
          data-testid="live-visual-state-feedback"
          className="mt-3 rounded border border-border bg-surface2 px-2 py-1.5 text-xs text-text"
        >
          <div className="text-text-muted">Visual state</div>
          <div className="mt-1 flex flex-wrap gap-2">
            <span>{visualStateFeedback.state}</span>
            <span>{visualStateFeedback.source}</span>
            {visualStateFeedback.reason ? (
              <span>{visualStateFeedback.reason}</span>
            ) : null}
            {visualStateFeedback.fallbackReason ? (
              <span>{visualStateFeedback.fallbackReason}</span>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs">
        <Checkbox
          data-testid="live-voice-auto-resume"
          checked={sessionAutoResume}
          disabled={sessionControlsDisabled}
          onChange={(event) => onSessionAutoResumeChange(event.target.checked)}
        >
          Auto-resume (session only)
        </Checkbox>
        <Checkbox
          data-testid="live-voice-barge-in"
          checked={sessionBargeIn}
          disabled={sessionControlsDisabled}
          onChange={(event) => onSessionBargeInChange(event.target.checked)}
        >
          Barge-in (session only)
        </Checkbox>
      </div>

      <div className="mt-3 rounded-md border border-border bg-surface2 p-3 text-xs text-text">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <Typography.Text strong>Wake phrase</Typography.Text>
            <Typography.Text type="secondary" className="mt-1 block text-xs">
              Active only while this Persona Live surface is open.
            </Typography.Text>
          </div>
          <Button
            data-testid="live-wake-toggle"
            size="small"
            type={wakeArmed ? "default" : "primary"}
            disabled={
              sessionControlsDisabled || normalizedWakeTriggerPhrases.length === 0
            }
            onClick={onToggleWakeArmed}
          >
            {wakeArmed ? "Stop wake listening" : "Listen for wake phrase"}
          </Button>
        </div>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          <div>
            <div className="text-text-muted">Saved wake phrases</div>
            <div data-testid="live-wake-phrases">
              {normalizedWakeTriggerPhrases.length > 0
                ? normalizedWakeTriggerPhrases.join(", ")
                : "Add a trigger phrase in voice defaults."}
            </div>
          </div>
          <div>
            <div className="text-text-muted">Wake state</div>
            <div data-testid="live-wake-state">
              {wakeArmed ? wakeDetectorState : "idle"}
            </div>
          </div>
          <div>
            <div className="text-text-muted">After wake</div>
            <Select
              data-testid="live-wake-behavior"
              size="small"
              value={effectiveSessionWakeBehavior}
              disabled={sessionControlsDisabled}
              onChange={onSessionWakeBehaviorChange}
              options={[
                { value: "one_shot", label: "One turn" },
                { value: "continuous", label: "Continuous" },
                { value: "push_to_talk_after_wake", label: "Push to talk" }
              ]}
            />
          </div>
        </div>
        {wakeWarning ? (
          <DesignSystemAlert
            className="mt-2"
            variant="warning"
            title={wakeWarning}
          />
        ) : null}
      </div>

      <PersonaTurnDetectionControls
        title="Turn detection"
        helperText={turnDetectionHelperText}
        testIdPrefix="live-vad"
        autoCommitLabel="Auto-commit (session only)"
        currentPreset={vadPreset}
        values={{
          autoCommitEnabled,
          vadThreshold,
          minSilenceMs,
          turnStopSecs,
          minUtteranceSecs
        }}
        disabled={turnDetectionDisabled}
        advancedInputsDisabled={turnDetectionDisabled || !autoCommitEnabled}
        className="mt-3 rounded-md border border-border bg-surface2 p-3 text-xs text-text"
        advancedFooterText="Changes apply immediately and may affect the current live turn."
        onAutoCommitEnabledChange={onAutoCommitEnabledChange}
        onPresetChange={onVadPresetChange}
        onVadThresholdChange={onVadThresholdChange}
        onMinSilenceMsChange={onMinSilenceMsChange}
        onTurnStopSecsChange={onTurnStopSecsChange}
        onMinUtteranceSecsChange={onMinUtteranceSecsChange}
      />
      {showSaveCurrentSettingsAsDefaults ? (
        <div className="mt-2 flex justify-end">
          <Button
            size="small"
            loading={savingCurrentSettingsAsDefaults}
            onClick={onSaveCurrentSettingsAsDefaults}
          >
            Save current settings as defaults
          </Button>
        </div>
      ) : null}

      {warning ? (
        <div
          data-testid="live-voice-warning"
          className="mt-3 rounded-md border border-warning/40 bg-warning/10 p-2 text-xs text-warning"
        >
          {warning}
        </div>
      ) : null}

      {String(pendingApprovalSummary || "").trim() ? (
        <div
          data-testid="live-voice-current-action"
          className="mt-3 rounded-md border border-border bg-surface2 p-2 text-xs text-text"
        >
          <div className="text-text-muted">Current action</div>
          <div className="mt-1 whitespace-pre-wrap">{pendingApprovalSummary}</div>
          <div className="mt-2">
            <Button
              data-testid="live-voice-jump-to-approval"
              size="small"
              onClick={onJumpToApproval}
            >
              Jump to approval
            </Button>
          </div>
        </div>
      ) : state === "thinking" && String(activeToolStatus || "").trim() ? (
        <div
          data-testid="live-voice-current-action"
          className="mt-3 rounded-md border border-border bg-surface2 p-2 text-xs text-text"
        >
          <div className="text-text-muted">Current action</div>
          <div className="mt-1 whitespace-pre-wrap">{activeToolStatus}</div>
        </div>
      ) : null}

      {recoveryMode === "listening_stuck" ? (
        <div
          data-testid="live-voice-recovery-panel"
          className="mt-3 rounded-md border border-warning/40 bg-warning/10 p-3 text-xs"
        >
          <Typography.Text strong>Voice turn needs attention</Typography.Text>
          <div className="mt-1 text-text">
            I heard speech, but this turn has not been committed yet.
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              data-testid="live-voice-recovery-send-now"
              size="small"
              type="primary"
              disabled={!canSendNow}
              onClick={onSendNow}
            >
              Send now
            </Button>
            <Button
              data-testid="live-voice-recovery-keep-listening"
              size="small"
              onClick={onKeepListening}
            >
              Keep listening
            </Button>
            <Button
              data-testid="live-voice-recovery-reset-turn"
              size="small"
              onClick={onResetTurn}
            >
              Reset turn
            </Button>
            <Button
              data-testid="live-voice-recovery-reconnect"
              size="small"
              onClick={onReconnectPersonaSession}
            >
              Reconnect Persona session
            </Button>
          </div>
        </div>
      ) : null}

      {recoveryMode === "thinking_stuck" ? (
        <div
          data-testid="live-voice-recovery-panel"
          className="mt-3 rounded-md border border-warning/40 bg-warning/10 p-3 text-xs"
        >
          <Typography.Text strong>Assistant response is delayed</Typography.Text>
          <div className="mt-1 text-text">
            This voice turn was sent, but the assistant has not responded yet.
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              data-testid="live-voice-recovery-wait"
              size="small"
              onClick={onWaitOnRecovery}
            >
              Wait
            </Button>
            <Button
              data-testid="live-voice-recovery-copy-command"
              size="small"
              disabled={!String(lastCommittedText || "").trim()}
              onClick={onCopyLastCommandToComposer}
            >
              Copy last command to composer
            </Button>
            <Button
              data-testid="live-voice-recovery-reset-turn"
              size="small"
              onClick={onResetTurn}
            >
              Reset turn
            </Button>
            <Button
              data-testid="live-voice-recovery-reconnect"
              size="small"
              onClick={onReconnectPersonaSession}
            >
              Reconnect Persona session
            </Button>
          </div>
        </div>
      ) : null}

      {heardText ? (
        <div className="mt-3 rounded border border-border bg-surface2 px-2 py-1.5 text-xs">
          <div className="text-text-muted">Last heard</div>
          <div data-testid="live-voice-heard-text" className="mt-1 whitespace-pre-wrap">
            {heardText}
          </div>
        </div>
      ) : null}

      {lastCommittedText ? (
        <div className="mt-2 rounded border border-border bg-surface2 px-2 py-1.5 text-xs">
          <div className="text-text-muted">Last sent to Persona</div>
          <div data-testid="live-voice-last-commit" className="mt-1 whitespace-pre-wrap">
            {lastCommittedText}
          </div>
        </div>
      ) : null}
    </div>
  )
}
