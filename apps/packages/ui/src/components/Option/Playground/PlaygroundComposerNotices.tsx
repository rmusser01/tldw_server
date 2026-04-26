import React from "react"
import {
  Input,
  Select,
  Button
} from "antd"
import { Link, useNavigate } from "react-router-dom"
import { FirstRunBanner } from "@/components/PersonaGarden/FirstRunBanner"
import { useFirstRunCheck } from "@/hooks/useFirstRunCheck"
import { ModelRecommendationsPanel } from "./ModelRecommendationsPanel"
import type { ModelRecommendationAction } from "./model-recommendations"
import { toText } from "./hooks/utils"
import type { TFunction } from "i18next"

const CHAT_NUDGE_DISMISSED_KEY = "assistant_nudge_dismissed_chat"

const readChatNudgeDismissedState = (): boolean => {
  try {
    return localStorage.getItem(CHAT_NUDGE_DISMISSED_KEY) === "true"
  } catch (error) {
    console.warn("Failed to read assistant chat nudge dismissal state", error)
    return false
  }
}

const persistChatNudgeDismissedState = (): void => {
  try {
    localStorage.setItem(CHAT_NUDGE_DISMISSED_KEY, "true")
  } catch (error) {
    console.warn("Failed to persist assistant chat nudge dismissal state", error)
  }
}

export type PlaygroundComposerNoticesProps = {
  modeAnnouncement: string | null
  characterPendingApply: boolean
  selectedCharacterGreeting: string | null
  selectedCharacterName: string | null
  compareModeActive: boolean
  compareSelectedModels: string[]
  compareSelectedModelLabels: string[]
  compareNeedsMoreModels: boolean
  compareSharedContextLabels: string[]
  compareInteroperabilityNotices: Array<{
    id: string
    text: string
    tone?: string
  }>
  noticesExpanded: boolean
  setNoticesExpanded: (expanded: boolean) => void
  contextDeltaLabels: string[]
  contextConflictWarnings: Array<{
    id: string
    text: string
    onAction?: () => void
    actionLabel?: string
  }>
  visibleModelRecommendations: any[]
  sessionInsightsTotalTokens: number
  jsonMode: boolean
  isConnectionReady: boolean
  connectionUxState: string
  isProMode: boolean
  selectedModel: string | null | undefined
  systemPrompt: string | null | undefined
  selectedCharacter: any
  ragPinnedResultsLength: number
  startupTemplateDraftName: string
  setStartupTemplateDraftName: (name: string) => void
  startupTemplates: Array<{ id: string; name: string }>
  handleSaveStartupTemplate: () => void
  handleOpenStartupTemplatePreview: (id: string) => void
  setOpenModelSettings: (open: boolean) => void
  setOpenActorSettings: (open: boolean) => void
  setMessageValue: (
    value: string,
    options?: { collapseLarge?: boolean }
  ) => void
  textAreaFocus: () => void
  openModelApiSelector: () => void
  openSessionInsightsModal: () => void
  handleModelRecommendationAction: (action: ModelRecommendationAction) => void
  dismissModelRecommendation: (id: string) => void
  getModelRecommendationActionLabel: (action: ModelRecommendationAction) => string
  wrapComposerProfile: (id: string, element: React.ReactElement) => React.ReactElement
  t: TFunction
}

/**
 * Thin wrapper around the chat-surface nudge banner dismiss state.
 * Keeps localStorage reads inside a hook so the memo boundary on the
 * outer component does not break.
 */
function ChatFirstRunNudge() {
  const { shouldShowSetup, resumeStep, loading } = useFirstRunCheck()
  const navigate = useNavigate()
  const [dismissed, setDismissed] = React.useState(readChatNudgeDismissedState)

  const handleDismiss = React.useCallback(() => {
    persistChatNudgeDismissedState()
    setDismissed(true)
  }, [])

  const handleNavigate = React.useCallback(() => {
    navigate("/persona")
  }, [navigate])

  if (loading || dismissed || (!shouldShowSetup && !resumeStep)) {
    return null
  }

  return (
    <div className="mt-1">
      <FirstRunBanner
        variant={resumeStep ? "resume" : "nudge"}
        resumeStep={resumeStep}
        onResume={handleNavigate}
        onDismiss={handleDismiss}
      />
    </div>
  )
}

export const PlaygroundComposerNotices = React.memo(function PlaygroundComposerNotices(
  props: PlaygroundComposerNoticesProps
) {
  const {
    modeAnnouncement,
    characterPendingApply,
    selectedCharacterGreeting,
    selectedCharacterName,
    compareModeActive,
    compareSelectedModels,
    compareSelectedModelLabels,
    compareNeedsMoreModels,
    compareSharedContextLabels,
    compareInteroperabilityNotices,
    noticesExpanded,
    setNoticesExpanded,
    contextDeltaLabels,
    contextConflictWarnings,
    visibleModelRecommendations,
    sessionInsightsTotalTokens,
    jsonMode,
    isConnectionReady,
    connectionUxState,
    isProMode,
    selectedModel,
    systemPrompt,
    selectedCharacter,
    ragPinnedResultsLength,
    startupTemplateDraftName,
    setStartupTemplateDraftName,
    startupTemplates,
    handleSaveStartupTemplate,
    handleOpenStartupTemplatePreview,
    setOpenModelSettings,
    setOpenActorSettings,
    setMessageValue,
    textAreaFocus,
    openModelApiSelector,
    openSessionInsightsModal,
    handleModelRecommendationAction,
    dismissModelRecommendation,
    getModelRecommendationActionLabel,
    wrapComposerProfile,
    t
  } = props

  return (
    <>
      <ChatFirstRunNudge />
      {modeAnnouncement && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 rounded-md border border-primary/30 bg-primary/10 px-2 py-1 text-xs text-primaryStrong"
        >
          {modeAnnouncement}
        </div>
      )}
      {characterPendingApply && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 flex flex-wrap items-center justify-between gap-2 rounded-md border border-primary/30 bg-primary/10 px-2 py-2 text-xs text-primaryStrong"
        >
          <span>
            {t(
              "playground:composer.characterPendingNotice",
              "Character updates will apply on your next turn."
            )}
          </span>
          <div className="flex items-center gap-2">
            {selectedCharacterGreeting && (
              <button
                type="button"
                onClick={() => {
                  setMessageValue(selectedCharacterGreeting, {
                    collapseLarge: true
                  })
                  textAreaFocus()
                }}
                className="rounded border border-primary/30 bg-surface px-2 py-0.5 text-[11px] font-medium text-primaryStrong hover:bg-primary/10"
              >
                {t(
                  "playground:composer.characterUseGreeting",
                  "Use greeting"
                )}
              </button>
            )}
            <button
              type="button"
              onClick={() => setOpenActorSettings(true)}
              className="rounded border border-primary/30 bg-surface px-2 py-0.5 text-[11px] font-medium text-primaryStrong hover:bg-primary/10"
            >
              {t(
                "playground:composer.characterReview",
                "Review character"
              )}
            </button>
          </div>
        </div>
      )}
      {compareModeActive && (
        <div
          role="status"
          aria-live="polite"
          data-testid="compare-activation-contract"
          className="mt-1 space-y-2 rounded-md border border-primary/30 bg-primary/10 px-2 py-2 text-xs text-primaryStrong"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-wide">
              {t(
                "playground:composer.compareActivationTitle",
                "Compare contract"
              )}
            </span>
            <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[10px] font-medium text-primaryStrong">
              {toText(
                t(
                  "playground:composer.compareActivationCount",
                  "{{count}} models",
                  {
                    count: compareSelectedModels.length
                  } as any
                )
              )}
            </span>
          </div>
          <p>
            {t(
              "playground:composer.compareActivationBody",
              "Next send fans out the same prompt and shared context to each selected model. Compare mode stays active until you turn it off."
            )}
          </p>
          <div className="space-y-1">
            <p className="text-[11px] font-medium text-primaryStrong">
              {t(
                "playground:composer.compareActivationModels",
                "Selected models"
              )}
            </p>
            <div className="flex flex-wrap gap-1">
              {compareSelectedModelLabels.length > 0 ? (
                compareSelectedModelLabels.map((label, index) => (
                  <span
                    key={`${label}-${index}`}
                    className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[10px] text-primaryStrong"
                  >
                    {label}
                  </span>
                ))
              ) : (
                <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[10px] text-primaryStrong">
                  {t(
                    "playground:compare.noModelsSelected",
                    "No models selected"
                  )}
                </span>
              )}
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-[11px] font-medium text-primaryStrong">
              {t(
                "playground:composer.compareActivationSharedContext",
                "Shared context"
              )}
            </p>
            <div className="flex flex-wrap gap-1">
              {compareSharedContextLabels.length > 0 ? (
                compareSharedContextLabels.map((label, index) => (
                  <span
                    key={`${label}-${index}`}
                    className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[10px] text-primaryStrong"
                  >
                    {label}
                  </span>
                ))
              ) : (
                <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5 text-[10px] text-primaryStrong">
                  {t(
                    "playground:composer.compareActivationNoSharedContext",
                    "No additional shared context modifiers are active."
                  )}
                </span>
              )}
            </div>
          </div>
          {compareInteroperabilityNotices.length > 0 && (
            <div className="space-y-1" data-testid="compare-interoperability-notices">
              <p className="text-[11px] font-medium text-primaryStrong">
                {t(
                  "playground:composer.compareActivationInteroperability",
                  "Interoperability notes"
                )}
              </p>
              <div className="space-y-1">
                {(noticesExpanded
                  ? compareInteroperabilityNotices
                  : compareInteroperabilityNotices.slice(0, 2)
                ).map((notice) => (
                  <div
                    key={notice.id}
                    className={`rounded border px-2 py-1 text-[11px] ${
                      notice.tone === "warning"
                        ? "border-warn/40 bg-warn/10 text-warn"
                        : "border-primary/30 bg-surface text-primaryStrong"
                    }`}
                  >
                    {notice.text}
                  </div>
                ))}
                {compareInteroperabilityNotices.length > 2 && (
                  <button
                    type="button"
                    onClick={() => setNoticesExpanded(!noticesExpanded)}
                    className="text-[10px] text-primary underline"
                  >
                    {noticesExpanded
                      ? t(
                          "playground:compareNoticesCollapse",
                          "Show fewer"
                        )
                      : t(
                          "playground:compareNoticesExpand",
                          "{{count}} more notes",
                          { count: compareInteroperabilityNotices.length - 2 }
                        )}
                  </button>
                )}
              </div>
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span
              className={
                compareNeedsMoreModels
                  ? "text-warn"
                  : "text-primaryStrong"
              }
            >
              {compareNeedsMoreModels
                ? t(
                    "playground:composer.compareActivationNeedsMoreModels",
                    "Add at least one more model before sending in Compare mode."
                  )
                : t(
                    "playground:composer.compareActivationPersistence",
                    "These selections persist for next turns until Compare mode is disabled."
                  )}
            </span>
            <button
              type="button"
              onClick={() => setOpenModelSettings(true)}
              className={`rounded border px-2 py-0.5 text-[11px] font-medium ${
                compareNeedsMoreModels
                  ? "border-warn/40 bg-surface text-warn hover:bg-warn/10"
                  : "border-primary/30 bg-surface text-primaryStrong hover:bg-primary/10"
              }`}
            >
              {compareNeedsMoreModels
                ? t("playground:compare.addModels", "Add models")
                : t(
                    "playground:composer.compareActivationReviewModels",
                    "Review models"
                  )}
            </button>
          </div>
        </div>
      )}
      {contextDeltaLabels.length > 0 && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 flex flex-wrap items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1"
        >
          <span className="text-[11px] font-medium text-text-muted">
            {t(
              "playground:composer.delta.title",
              "Changed since last send:"
            )}
          </span>
          {contextDeltaLabels.map((delta) => (
            <span
              key={delta}
              className="rounded-full border border-border bg-surface px-2 py-0.5 text-[10px] text-text-muted"
            >
              {delta}
            </span>
          ))}
        </div>
      )}
      {contextConflictWarnings.length > 0 && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 space-y-1 rounded-md border border-warn/40 bg-warn/10 px-2 py-2"
        >
          {contextConflictWarnings.map((warning) => (
            <div
              key={warning.id}
              className="flex items-start justify-between gap-2 text-xs text-warn"
            >
              <span>{warning.text}</span>
              {warning.onAction ? (
                <button
                  type="button"
                  onClick={warning.onAction}
                  className="shrink-0 rounded px-1 py-0.5 text-[11px] font-medium text-warn underline hover:bg-warn/10"
                >
                  {warning.actionLabel || t("common:review", "Review")}
                </button>
              ) : null}
            </div>
          ))}
        </div>
      )}
      {wrapComposerProfile(
        "model-recommendations",
        <ModelRecommendationsPanel
          t={t}
          recommendations={visibleModelRecommendations}
          showOpenInsights={sessionInsightsTotalTokens > 0}
          onOpenInsights={openSessionInsightsModal}
          onRunAction={handleModelRecommendationAction}
          onDismiss={dismissModelRecommendation}
          getActionLabel={getModelRecommendationActionLabel}
        />
      )}
      {jsonMode && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 flex flex-wrap items-center justify-between gap-2 rounded-md border border-primary/30 bg-primary/10 px-2 py-2 text-xs text-primaryStrong"
        >
          <span>
            {t(
              "playground:composer.jsonModeHint",
              "JSON output is active (developer). Responses will be formatted as JSON objects."
            )}
          </span>
          <button
            type="button"
            onClick={() => setOpenModelSettings(true)}
            className="rounded border border-primary/30 bg-surface px-2 py-0.5 text-[11px] font-medium text-primaryStrong hover:bg-primary/10"
          >
            {t(
              "playground:composer.jsonModeConfigure",
              "Configure"
            )}
          </button>
        </div>
      )}
      {!isConnectionReady && (
        <div
          role="status"
          aria-live="polite"
          className="mt-1 flex flex-wrap items-center justify-between gap-2 rounded-md border border-text-muted/30 bg-surface2/50 px-2 py-2 text-xs text-text-muted"
        >
          <span>
            {t(
              "playground:composer.disconnectedNotice",
              "You're offline \u2014 connect to a tldw server to start chatting."
            )}
          </span>
          <Link
            to="/settings/tldw"
            className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text"
          >
            {t(
              "playground:composer.disconnectedOpenSettings",
              "Open settings"
            )}
          </Link>
        </div>
      )}
      {isConnectionReady &&
        connectionUxState === "connected_degraded" && (
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2 rounded-md border border-warn/40 bg-warn/10 px-2 py-2 text-xs text-warn">
            <span>
              {t(
                "playground:composer.providerDegraded",
                "Provider connectivity is degraded. Responses may be slower or fail intermittently."
              )}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={openModelApiSelector}
                className="rounded border border-warn/40 bg-surface px-2 py-0.5 text-[11px] font-medium text-warn hover:bg-warn/10"
              >
                {t(
                  "playground:composer.providerDegradedSwitchModel",
                  "Switch model"
                )}
              </button>
              <Link
                to="/settings/health"
                className="text-[11px] font-medium text-warn underline hover:text-warn"
              >
                {t(
                  "settings:healthSummary.diagnostics",
                  "Health & diagnostics"
                )}
              </Link>
            </div>
          </div>
        )}
      {isProMode && (
        <div
          data-testid="startup-template-controls"
          className="mt-2 rounded-md border border-border/60 bg-surface2/70 px-2 py-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              {t(
                "playground:composer.startupTemplatesLabel",
                "Startup templates"
              )}
            </span>
            <Input
              size="small"
              value={startupTemplateDraftName}
              onChange={(event) =>
                setStartupTemplateDraftName(event.target.value)
              }
              placeholder={t(
                "playground:composer.startupTemplatesNamePlaceholder",
                "Template name"
              )}
              className="min-w-[180px] max-w-[260px]"
            />
            <Button
              size="small"
              onClick={handleSaveStartupTemplate}
              disabled={
                !selectedModel &&
                String(systemPrompt || "").trim().length === 0 &&
                !selectedCharacter &&
                ragPinnedResultsLength === 0
              }
            >
              {t(
                "playground:composer.startupTemplatesSave",
                "Save current"
              )}
            </Button>
            <Select
              size="small"
              placeholder={t(
                "playground:composer.startupTemplatesLaunch",
                "Launch saved template"
              )}
              options={startupTemplates.map((template) => ({
                value: template.id,
                label: template.name
              }))}
              onChange={handleOpenStartupTemplatePreview}
              className="min-w-[220px]"
              data-testid="startup-template-launch-select"
            />
          </div>
          {startupTemplates.length === 0 && (
            <p className="mt-1 text-xs text-text-muted">
              {t(
                "playground:composer.startupTemplatesHint",
                "Save your current model, prompt, character, and pinned-source setup to reuse it before first send."
              )}
            </p>
          )}
        </div>
      )}
    </>
  )
})
