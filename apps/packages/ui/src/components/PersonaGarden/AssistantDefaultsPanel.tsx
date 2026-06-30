import React from "react"
import { useTranslation } from "react-i18next"

import type { PersonaVoiceAnalytics } from "@/components/PersonaGarden/CommandAnalyticsSummary"
import { PersonaTurnDetectionFeedbackCard } from "@/components/PersonaGarden/PersonaTurnDetectionFeedbackCard"
import {
  PersonaTurnDetectionControls,
  PERSONA_TURN_DETECTION_PRESETS,
  derivePersonaTurnDetectionPreset
} from "@/components/PersonaGarden/PersonaTurnDetectionControls"
import {
  PERSONA_TURN_DETECTION_BALANCED_DEFAULTS,
  type PersonaConfirmationMode,
  type PersonaWakeBehavior,
  type PersonaVoiceDefaults,
  useResolvedPersonaVoiceDefaults
} from "@/hooks/useResolvedPersonaVoiceDefaults"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"

type AssistantDefaultsPanelProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  isActive?: boolean
  analytics?: PersonaVoiceAnalytics | null
  analyticsLoading?: boolean
  handoffFocusRequest?: {
    section: "assistant_defaults" | "confirmation_mode"
    token: number
  } | null
  onSetupHandoffFocusConsumed?: (token: number) => void
  onSaved?: (voiceDefaults: PersonaVoiceDefaults) => void
}

type PersonaProfileResponse = {
  id?: string
  voice_defaults?: PersonaVoiceDefaults | null
}

type AssistantDefaultsFormState = {
  sttLanguage: string
  sttModel: string
  ttsProvider: string
  ttsVoice: string
  confirmationMode: PersonaConfirmationMode
  wakeBehavior: PersonaWakeBehavior
  triggerPhrasesText: string
  autoResume: boolean | null
  bargeIn: boolean | null
  autoCommitEnabled: boolean
  vadThreshold: number
  minSilenceMs: number
  turnStopSecs: number
  minUtteranceSecs: number
}

const DEFAULT_FORM_STATE: AssistantDefaultsFormState = {
  sttLanguage: "",
  sttModel: "",
  ttsProvider: "",
  ttsVoice: "",
  confirmationMode: "destructive_only",
  wakeBehavior: "one_shot",
  triggerPhrasesText: "",
  autoResume: null,
  bargeIn: null,
  autoCommitEnabled: PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.autoCommitEnabled,
  vadThreshold: PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.vadThreshold,
  minSilenceMs: PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minSilenceMs,
  turnStopSecs: PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.turnStopSecs,
  minUtteranceSecs: PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minUtteranceSecs
}

const normalizeText = (value: string | null | undefined): string => String(value || "").trim()

const normalizePhrases = (value: string | null | undefined): string[] => {
  const seen = new Set<string>()
  const next: string[] = []
  const chunks = String(value || "")
    .split(/\r?\n/g)
    .map((item) => item.trim())
  for (const chunk of chunks) {
    if (!chunk || seen.has(chunk)) continue
    seen.add(chunk)
    next.push(chunk)
  }
  return next
}

const buildFormState = (
  voiceDefaults?: PersonaVoiceDefaults | null
): AssistantDefaultsFormState => ({
  sttLanguage: normalizeText(voiceDefaults?.stt_language),
  sttModel: normalizeText(voiceDefaults?.stt_model),
  ttsProvider: normalizeText(voiceDefaults?.tts_provider),
  ttsVoice: normalizeText(voiceDefaults?.tts_voice),
  confirmationMode: voiceDefaults?.confirmation_mode || "destructive_only",
  wakeBehavior: voiceDefaults?.wake_behavior || "one_shot",
  triggerPhrasesText: (voiceDefaults?.voice_chat_trigger_phrases || []).join("\n"),
  autoResume:
    typeof voiceDefaults?.auto_resume === "boolean"
      ? voiceDefaults.auto_resume
      : null,
  bargeIn:
    typeof voiceDefaults?.barge_in === "boolean" ? voiceDefaults.barge_in : null,
  autoCommitEnabled:
    typeof voiceDefaults?.auto_commit_enabled === "boolean"
      ? voiceDefaults.auto_commit_enabled
      : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.autoCommitEnabled,
  vadThreshold:
    typeof voiceDefaults?.vad_threshold === "number"
      ? voiceDefaults.vad_threshold
      : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.vadThreshold,
  minSilenceMs:
    typeof voiceDefaults?.min_silence_ms === "number"
      ? voiceDefaults.min_silence_ms
      : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minSilenceMs,
  turnStopSecs:
    typeof voiceDefaults?.turn_stop_secs === "number"
      ? voiceDefaults.turn_stop_secs
      : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.turnStopSecs,
  minUtteranceSecs:
    typeof voiceDefaults?.min_utterance_secs === "number"
      ? voiceDefaults.min_utterance_secs
      : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minUtteranceSecs
})

const buildPayload = (
  formState: AssistantDefaultsFormState
): PersonaVoiceDefaults => ({
  stt_language: normalizeText(formState.sttLanguage) || null,
  stt_model: normalizeText(formState.sttModel) || null,
  tts_provider: normalizeText(formState.ttsProvider) || null,
  tts_voice: normalizeText(formState.ttsVoice) || null,
  confirmation_mode: formState.confirmationMode,
  wake_behavior: formState.wakeBehavior,
  voice_chat_trigger_phrases: normalizePhrases(formState.triggerPhrasesText),
  auto_resume: formState.autoResume,
  barge_in: formState.bargeIn,
  auto_commit_enabled: formState.autoCommitEnabled,
  vad_threshold: formState.vadThreshold,
  min_silence_ms: formState.minSilenceMs,
  turn_stop_secs: formState.turnStopSecs,
  min_utterance_secs: formState.minUtteranceSecs
})

const formatBool = (value: boolean): string => (value ? "On" : "Off")
const booleanSelectValue = (value: boolean | null): string => {
  if (value === true) return "true"
  if (value === false) return "false"
  return "inherit"
}

const parseBooleanSelectValue = (value: string): boolean | null => {
  if (value === "true") return true
  if (value === "false") return false
  return null
}

export const AssistantDefaultsPanel: React.FC<AssistantDefaultsPanelProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  isActive = false,
  analytics = null,
  analyticsLoading = false,
  handoffFocusRequest = null,
  onSetupHandoffFocusConsumed,
  onSaved
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [success, setSuccess] = React.useState<string | null>(null)
  const [formState, setFormState] =
    React.useState<AssistantDefaultsFormState>(DEFAULT_FORM_STATE)
  const sttLanguageInputRef = React.useRef<HTMLInputElement | null>(null)
  const confirmationModeRef = React.useRef<HTMLSelectElement | null>(null)
  const lastHandledHandoffTokenRef = React.useRef<number | null>(null)

  React.useEffect(() => {
    let cancelled = false

    const load = async () => {
      if (!isActive || !selectedPersonaId) {
        setLoading(false)
        setError(null)
        setSuccess(null)
        setFormState(DEFAULT_FORM_STATE)
        return
      }

      setLoading(true)
      setError(null)
      setSuccess(null)
      setFormState(DEFAULT_FORM_STATE)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(`/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}`),
          { method: "GET" }
        )
        if (!response.ok) {
          throw new Error(
            response.error ||
              t("sidepanel:personaGarden.profile.assistantDefaultsLoadError", {
                defaultValue: "Failed to load assistant defaults."
              })
          )
        }
        const payload = (await response.json()) as PersonaProfileResponse
        if (!cancelled) {
          setFormState(buildFormState(payload.voice_defaults))
        }
      } catch (loadError) {
        if (!cancelled) {
          setFormState(DEFAULT_FORM_STATE)
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("sidepanel:personaGarden.profile.assistantDefaultsLoadError", {
                  defaultValue: "Failed to load assistant defaults."
                })
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [isActive, selectedPersonaId])

  React.useEffect(() => {
    if (!isActive || !selectedPersonaId || !handoffFocusRequest || loading) return
    if (lastHandledHandoffTokenRef.current === handoffFocusRequest.token) return

    const focusTarget =
      handoffFocusRequest.section === "confirmation_mode"
        ? confirmationModeRef.current
        : sttLanguageInputRef.current

    if (!focusTarget || focusTarget.disabled) return

    focusTarget.scrollIntoView?.({ block: "nearest", behavior: "smooth" })
    focusTarget.focus()
    lastHandledHandoffTokenRef.current = handoffFocusRequest.token
    onSetupHandoffFocusConsumed?.(handoffFocusRequest.token)
  }, [
    handoffFocusRequest,
    isActive,
    loading,
    onSetupHandoffFocusConsumed,
    selectedPersonaId
  ])

  const resolvedDefaults = useResolvedPersonaVoiceDefaults(buildPayload(formState))
  const savedVadPreset = React.useMemo(
    () =>
      derivePersonaTurnDetectionPreset({
        autoCommitEnabled: formState.autoCommitEnabled,
        vadThreshold: formState.vadThreshold,
        minSilenceMs: formState.minSilenceMs,
        turnStopSecs: formState.turnStopSecs,
        minUtteranceSecs: formState.minUtteranceSecs
      }),
    [
      formState.autoCommitEnabled,
      formState.minSilenceMs,
      formState.minUtteranceSecs,
      formState.turnStopSecs,
      formState.vadThreshold
    ]
  )

  const updateField = React.useCallback(
    (
      field: keyof AssistantDefaultsFormState,
      value: string | boolean | number | null
    ) => {
      setSuccess(null)
      setFormState((current) => ({
        ...current,
        [field]: value
      }))
    },
    []
  )

  const handleSave = React.useCallback(async () => {
    if (!selectedPersonaId) return
    setSaving(true)
    setError(null)
    setSuccess(null)

    try {
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(`/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}`),
        {
          method: "PATCH",
          body: {
            voice_defaults: buildPayload(formState)
          }
        }
      )
      if (!response.ok) {
        throw new Error(
          response.error ||
            t("sidepanel:personaGarden.profile.assistantDefaultsSaveError", {
              defaultValue: "Failed to save assistant defaults."
            })
        )
      }
      const payload = (await response.json()) as PersonaProfileResponse
      const nextFormState = buildFormState(payload.voice_defaults)
      setFormState(nextFormState)
      onSaved?.(payload.voice_defaults || buildPayload(formState))
      setSuccess(
        t("sidepanel:personaGarden.profile.assistantDefaultsSaved", {
          defaultValue: "Assistant defaults saved."
        })
      )
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("sidepanel:personaGarden.profile.assistantDefaultsSaveError", {
              defaultValue: "Failed to save assistant defaults."
            })
      )
    } finally {
      setSaving(false)
    }
  }, [formState, onSaved, selectedPersonaId, t])

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:personaGarden.profile.assistantDefaultsHeading", {
              defaultValue: "Assistant Defaults"
            })}
          </div>
          <p className="mt-2 text-xs text-text-muted">
            {t("sidepanel:personaGarden.profile.assistantDefaultsDescription", {
              defaultValue:
                "Persona defaults stay separate from browser-wide fallback settings. The preview below shows the effective values after local fallback is applied."
            })}
          </p>
        </div>
        {selectedPersonaName ? (
          <div className="text-right text-[11px] text-text-muted">
            {selectedPersonaName}
          </div>
        ) : null}
      </div>

      {!selectedPersonaId ? (
        <p className="mt-3 text-xs text-text-muted">
          {t("sidepanel:personaGarden.profile.assistantDefaultsNoPersona", {
            defaultValue: "Select a persona to manage assistant defaults."
          })}
        </p>
      ) : null}

      {error ? (
        <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          {error}
        </div>
      ) : null}
      {success ? (
        <div className="mt-3 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-100">
          {success}
        </div>
      ) : null}

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-text">
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.sttLanguage", {
              defaultValue: "STT language"
            })}
          </span>
          <input
            id="persona-assistant-defaults-stt-language"
            ref={sttLanguageInputRef}
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.sttLanguage}
            onChange={(event) => updateField("sttLanguage", event.target.value)}
            disabled={!selectedPersonaId || loading || saving}
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-text">
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.sttModel", {
              defaultValue: "STT model"
            })}
          </span>
          <input
            id="persona-assistant-defaults-stt-model"
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.sttModel}
            onChange={(event) => updateField("sttModel", event.target.value)}
            disabled={!selectedPersonaId || loading || saving}
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-text">
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.ttsProvider", {
              defaultValue: "TTS provider"
            })}
          </span>
          <select
            id="persona-assistant-defaults-tts-provider"
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.ttsProvider}
            onChange={(event) => updateField("ttsProvider", event.target.value)}
            disabled={!selectedPersonaId || loading || saving}
          >
            <option value="">
              {t("sidepanel:personaGarden.profile.assistantDefaults.browserFallback", {
                defaultValue: "Use browser fallback"
              })}
            </option>
            <option value="browser">browser</option>
            <option value="tldw">tldw</option>
            <option value="openai">openai</option>
            <option value="elevenlabs">elevenlabs</option>
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm text-text">
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.ttsVoice", {
              defaultValue: "TTS voice"
            })}
          </span>
          <input
            id="persona-assistant-defaults-tts-voice"
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.ttsVoice}
            onChange={(event) => updateField("ttsVoice", event.target.value)}
            disabled={!selectedPersonaId || loading || saving}
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-text">
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.confirmationMode", {
              defaultValue: "Confirmation mode"
            })}
          </span>
          <select
            id="persona-assistant-defaults-confirmation-mode"
            ref={confirmationModeRef}
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.confirmationMode}
            onChange={(event) =>
              updateField(
                "confirmationMode",
                event.target.value as PersonaConfirmationMode
              )
            }
            disabled={!selectedPersonaId || loading || saving}
          >
            <option value="always">always</option>
            <option value="destructive_only">destructive_only</option>
            <option value="never">never</option>
          </select>
        </label>

        <label
          htmlFor="persona-assistant-defaults-wake-behavior"
          className="flex flex-col gap-1 text-sm text-text"
        >
          <span>
            {t("sidepanel:personaGarden.profile.assistantDefaults.wakeBehavior", {
              defaultValue: "Wake behavior"
            })}
          </span>
          <select
            id="persona-assistant-defaults-wake-behavior"
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
            value={formState.wakeBehavior}
            onChange={(event) =>
              updateField("wakeBehavior", event.target.value as PersonaWakeBehavior)
            }
            disabled={!selectedPersonaId || loading || saving}
          >
            <option value="one_shot">One turn after wake</option>
            <option value="continuous">Continuous after wake</option>
            <option value="push_to_talk_after_wake">
              Push to talk after wake
            </option>
          </select>
        </label>

        <div className="flex flex-col gap-3 rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text">
          <label className="flex flex-col gap-1">
            <span>
              {t("sidepanel:personaGarden.profile.assistantDefaults.autoResume", {
                defaultValue: "Auto-resume"
              })}
            </span>
            <select
              value={booleanSelectValue(formState.autoResume)}
              onChange={(event) =>
                updateField("autoResume", parseBooleanSelectValue(event.target.value))
              }
              disabled={!selectedPersonaId || loading || saving}
            >
              <option value="inherit">
                {t("sidepanel:personaGarden.profile.assistantDefaults.inherit", {
                  defaultValue: "Use browser fallback"
                })}
              </option>
              <option value="true">On</option>
              <option value="false">Off</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span>
              {t("sidepanel:personaGarden.profile.assistantDefaults.bargeIn", {
                defaultValue: "Barge-in"
              })}
            </span>
            <select
              value={booleanSelectValue(formState.bargeIn)}
              onChange={(event) =>
                updateField("bargeIn", parseBooleanSelectValue(event.target.value))
              }
              disabled={!selectedPersonaId || loading || saving}
            >
              <option value="inherit">
                {t("sidepanel:personaGarden.profile.assistantDefaults.inherit", {
                  defaultValue: "Use browser fallback"
                })}
              </option>
              <option value="true">On</option>
              <option value="false">Off</option>
            </select>
          </label>
        </div>
      </div>

      <label className="mt-3 flex flex-col gap-1 text-sm text-text">
        <span>
          {t("sidepanel:personaGarden.profile.assistantDefaults.triggerPhrases", {
            defaultValue: "Trigger phrases"
          })}
        </span>
        <textarea
          id="persona-assistant-defaults-trigger-phrases"
          className="min-h-24 rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
          value={formState.triggerPhrasesText}
          onChange={(event) => updateField("triggerPhrasesText", event.target.value)}
          disabled={!selectedPersonaId || loading || saving}
          placeholder={t(
            "sidepanel:personaGarden.profile.assistantDefaults.triggerPhrasesPlaceholder",
            {
              defaultValue: "One phrase per line"
            }
          )}
        />
      </label>

      <PersonaTurnDetectionControls
        title="Turn detection defaults"
        helperText="Saved for future live sessions. Existing live sessions keep their current turn-detection settings until reconnect."
        testIdPrefix="assistant-defaults-vad"
        autoCommitLabel="Auto-commit (saved default)"
        currentPreset={savedVadPreset}
        values={{
          autoCommitEnabled: formState.autoCommitEnabled,
          vadThreshold: formState.vadThreshold,
          minSilenceMs: formState.minSilenceMs,
          turnStopSecs: formState.turnStopSecs,
          minUtteranceSecs: formState.minUtteranceSecs
        }}
        disabled={!selectedPersonaId || loading || saving}
        advancedInputsDisabled={
          !selectedPersonaId || loading || saving || !formState.autoCommitEnabled
        }
        className="mt-3 rounded-md border border-border bg-surface2 p-3 text-xs text-text"
        advancedFooterText="These saved values apply to future live sessions. Reconnect a running session to pick them up."
        onAutoCommitEnabledChange={(next) => updateField("autoCommitEnabled", next)}
        onPresetChange={(preset) => {
          const next = PERSONA_TURN_DETECTION_PRESETS[preset]
          setSuccess(null)
          setFormState((current) => ({
            ...current,
            autoCommitEnabled: next.autoCommitEnabled,
            vadThreshold: next.vadThreshold,
            minSilenceMs: next.minSilenceMs,
            turnStopSecs: next.turnStopSecs,
            minUtteranceSecs: next.minUtteranceSecs
          }))
        }}
        onVadThresholdChange={(next) => updateField("vadThreshold", next)}
        onMinSilenceMsChange={(next) => updateField("minSilenceMs", next)}
        onTurnStopSecsChange={(next) => updateField("turnStopSecs", next)}
        onMinUtteranceSecsChange={(next) => updateField("minUtteranceSecs", next)}
      />

      <PersonaTurnDetectionFeedbackCard
        analytics={analytics}
        loading={analyticsLoading}
      />

      <div className="mt-3 rounded-md border border-border/80 bg-surface2 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          {t("sidepanel:personaGarden.profile.assistantDefaults.effectivePreview", {
            defaultValue: "Effective Preview"
          })}
        </div>
        <dl className="mt-2 grid gap-2 text-sm text-text md:grid-cols-2">
          <div>
            <dt className="text-xs text-text-muted">STT language</dt>
            <dd>{resolvedDefaults.sttLanguage || "unset"}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">STT model</dt>
            <dd>{resolvedDefaults.sttModel || "unset"}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">TTS provider</dt>
            <dd>{resolvedDefaults.ttsProvider || "unset"}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">TTS voice</dt>
            <dd>{resolvedDefaults.ttsVoice || "unset"}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Confirmation mode</dt>
            <dd>{resolvedDefaults.confirmationMode}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Wake behavior</dt>
            <dd>{resolvedDefaults.wakeBehavior}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Auto-resume</dt>
            <dd>{formatBool(resolvedDefaults.autoResume)}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Barge-in</dt>
            <dd>{formatBool(resolvedDefaults.bargeIn)}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Trigger phrases</dt>
            <dd>
              {resolvedDefaults.voiceChatTriggerPhrases.length > 0
                ? resolvedDefaults.voiceChatTriggerPhrases.join(", ")
                : "None"}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Auto-commit</dt>
            <dd>{formatBool(resolvedDefaults.autoCommitEnabled)}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Speech threshold</dt>
            <dd>{resolvedDefaults.vadThreshold}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Silence before commit</dt>
            <dd>{resolvedDefaults.minSilenceMs} ms</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Minimum utterance</dt>
            <dd>{resolvedDefaults.minUtteranceSecs} s</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Turn tail</dt>
            <dd>{resolvedDefaults.turnStopSecs} s</dd>
          </div>
        </dl>
      </div>

      <div className="mt-3 flex justify-end">
        <button
          type="button"
          className="rounded-md border border-primary bg-primary/10 px-3 py-2 text-sm font-medium text-primary disabled:cursor-not-allowed disabled:opacity-60"
          disabled={!selectedPersonaId || loading || saving}
          onClick={() => {
            void handleSave()
          }}
        >
          {saving
            ? t("common:saving", { defaultValue: "Saving..." })
            : t("sidepanel:personaGarden.profile.assistantDefaults.save", {
                defaultValue: "Save assistant defaults"
              })}
        </button>
      </div>
    </div>
  )
}
