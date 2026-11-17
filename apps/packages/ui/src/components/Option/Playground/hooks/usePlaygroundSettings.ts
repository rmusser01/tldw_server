import React from "react"
import { buildCompareInteroperabilityNotices } from "../compare-interoperability"
import { toText, CONTEXT_FOOTPRINT_THRESHOLD_PERCENT } from "./utils"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundSettingsDeps {
  selectedCharacterName: string | null
  selectedSystemPrompt: string
  selectedQuickPrompt: string | null
  systemPrompt: string | undefined
  ragPinnedResultsLength: number
  webSearch: boolean
  jsonMode: boolean
  compareModeActive: boolean
  compareNeedsMoreModels: boolean
  compareCapabilityIncompatibilities: string[]
  voiceChatEnabled: boolean
  showTokenBudgetWarning: boolean
  tokenBudgetWarningText: string
  summaryCheckpointSuggestion: {
    shouldSuggest: boolean
    reason?: string
  }
  messagesLength: number
  showNonMessageContextWarning: boolean
  nonMessageContextPercent: number | null
  largestContextContributor: { id: string } | null
  openKnowledgePanel: (tab: string) => void
  openContextWindowModal: () => void
  openModelApiSelector: () => void
  setOpenModelSettings: (value: boolean) => void
  setModeLauncherOpen: (value: boolean) => void
  setOpenActorSettings: (value: boolean) => void
  trimLargestContextContributor: () => void
  insertSummaryCheckpointPrompt: () => void
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaygroundSettings(deps: UsePlaygroundSettingsDeps) {
  const {
    selectedCharacterName,
    selectedSystemPrompt,
    systemPrompt,
    ragPinnedResultsLength,
    webSearch,
    jsonMode,
    compareModeActive,
    compareNeedsMoreModels,
    compareCapabilityIncompatibilities,
    voiceChatEnabled,
    showTokenBudgetWarning,
    tokenBudgetWarningText,
    summaryCheckpointSuggestion,
    messagesLength,
    showNonMessageContextWarning,
    nonMessageContextPercent,
    largestContextContributor,
    openKnowledgePanel,
    openContextWindowModal,
    openModelApiSelector,
    setOpenModelSettings,
    setModeLauncherOpen,
    setOpenActorSettings,
    trimLargestContextContributor,
    insertSummaryCheckpointPrompt,
    t
  } = deps

  const compareHasPromptContext = React.useMemo(
    () =>
      Boolean(selectedSystemPrompt) ||
      String(systemPrompt || "").trim().length > 0,
    [selectedSystemPrompt, systemPrompt]
  )

  const compareSharedContextLabels = React.useMemo(() => {
    const labels: string[] = []
    if (selectedCharacterName) {
      labels.push(
        String(
          t(
            "playground:composer.compareSharedCharacter",
            "Character: {{name}}",
            { name: selectedCharacterName } as any
          )
        )
      )
    }
    if (compareHasPromptContext) {
      labels.push(
        String(
          t(
            "playground:composer.compareSharedPrompt",
            "Prompt steering enabled"
          )
        )
      )
    }
    if (ragPinnedResultsLength > 0) {
      labels.push(
        String(
          t(
            "playground:composer.compareSharedPinned",
            "{{count}} pinned sources",
            { count: ragPinnedResultsLength } as any
          )
        )
      )
    }
    if (webSearch) {
      labels.push(
        String(t("playground:composer.compareSharedWebSearch", "Web search on"))
      )
    }
    if (jsonMode) {
      labels.push(
        String(t("playground:composer.compareSharedJson", "JSON mode on"))
      )
    }
    return labels
  }, [
    compareHasPromptContext,
    jsonMode,
    ragPinnedResultsLength,
    selectedCharacterName,
    t,
    webSearch
  ])

  const compareInteroperabilityNotices = React.useMemo(
    () =>
      buildCompareInteroperabilityNotices({
        t,
        characterName: selectedCharacterName,
        pinnedSourceCount: ragPinnedResultsLength,
        webSearch,
        hasPromptContext: compareHasPromptContext,
        jsonMode,
        voiceChatEnabled
      }),
    [
      compareHasPromptContext,
      jsonMode,
      ragPinnedResultsLength,
      selectedCharacterName,
      t,
      voiceChatEnabled,
      webSearch
    ]
  )

  const contextConflictWarnings = React.useMemo(() => {
    const warnings: Array<{
      id: string
      text: string
      actionLabel?: string
      onAction?: () => void
    }> = []

    const hasCustomPrompt =
      Boolean(selectedSystemPrompt) ||
      String(systemPrompt || "").trim().length > 0

    if (selectedCharacterName && ragPinnedResultsLength > 0) {
      warnings.push({
        id: "character-rag",
        text: t(
          "playground:composer.conflict.characterRagSelected",
          "Character and pinned sources are selected. Review this combination; saved character chats may refuse it."
        ),
        actionLabel: t(
          "playground:composer.conflict.reviewContext",
          "Review context"
        ),
        onAction: () => openKnowledgePanel("search")
      })
    }

    if (selectedCharacterName && hasCustomPrompt) {
      warnings.push({
        id: "character-prompt",
        text: t(
          "playground:composer.conflict.characterPromptSelected",
          "Character and custom system prompt are selected. Review this combination; saved character chats may refuse it."
        ),
        actionLabel: t(
          "playground:composer.conflict.reviewModes",
          "Review modes"
        ),
        onAction: () => setModeLauncherOpen(true)
      })
    }

    if (compareModeActive && voiceChatEnabled) {
      warnings.push({
        id: "compare-voice",
        text: t(
          "playground:composer.conflict.compareVoice",
          "Compare mode with voice can reduce output parity across models."
        ),
        actionLabel: t(
          "playground:composer.conflict.adjustModes",
          "Adjust modes"
        ),
        onAction: () => setModeLauncherOpen(true)
      })
    }

    if (compareNeedsMoreModels) {
      warnings.push({
        id: "compare-min-models",
        text: t(
          "playground:composer.validationCompareMinModelsInline",
          "Select at least two models for Compare mode."
        ),
        actionLabel: t(
          "playground:composer.conflict.reviewModels",
          "Review models"
        ),
        onAction: () => setOpenModelSettings(true)
      })
    }

    if (compareModeActive && compareCapabilityIncompatibilities.length > 0) {
      warnings.push({
        id: "compare-capability",
        text: toText(
          t(
            "playground:composer.conflict.compareCapabilities",
            "Compare models have incompatible capabilities: {{details}}. Outputs may not be directly comparable.",
            {
              details: compareCapabilityIncompatibilities.join(", ")
            } as any
          )
        ),
        actionLabel: t(
          "playground:composer.conflict.reviewModels",
          "Review models"
        ),
        onAction: openModelApiSelector
      })
    }

    if (showTokenBudgetWarning && tokenBudgetWarningText) {
      warnings.push({
        id: "token-budget",
        text: tokenBudgetWarningText,
        actionLabel: t(
          "playground:composer.conflict.adjustBudget",
          "Adjust budget"
        ),
        onAction: () => openContextWindowModal()
      })
    }
    if (summaryCheckpointSuggestion.shouldSuggest && messagesLength >= 2) {
      warnings.push({
        id: "summary-checkpoint",
        text:
          summaryCheckpointSuggestion.reason === "token-budget"
            ? t(
                "playground:composer.conflict.summaryCheckpointBudget",
                "Consider creating a checkpoint summary before your next turn to reduce truncation risk."
              )
            : t(
                "playground:composer.conflict.summaryCheckpointVolume",
                "This thread is getting long. A checkpoint summary can preserve key decisions before context is trimmed."
              ),
        actionLabel: t(
          "playground:composer.conflict.createCheckpointSummary",
          "Create checkpoint summary"
        ),
        onAction: insertSummaryCheckpointPrompt
      })
    }
    if (showNonMessageContextWarning) {
      warnings.push({
        id: "context-footprint",
        text: toText(
          t(
            "playground:composer.conflict.contextFootprint",
            "Non-message context is using {{percent}}% of the context window. Trim character/prompt/source context before sending.",
            {
              percent: Math.round(nonMessageContextPercent || 0)
            } as any
          )
        ),
        actionLabel: largestContextContributor
          ? t("playground:composer.conflict.trimLargest", "Trim largest")
          : t("playground:composer.conflict.reviewContext", "Review context"),
        onAction: largestContextContributor
          ? trimLargestContextContributor
          : () => openContextWindowModal()
      })
    }

    return warnings
  }, [
    compareCapabilityIncompatibilities,
    compareModeActive,
    compareNeedsMoreModels,
    largestContextContributor,
    insertSummaryCheckpointPrompt,
    messagesLength,
    nonMessageContextPercent,
    openKnowledgePanel,
    trimLargestContextContributor,
    ragPinnedResultsLength,
    selectedCharacterName,
    selectedSystemPrompt,
    openModelApiSelector,
    setOpenModelSettings,
    setModeLauncherOpen,
    showNonMessageContextWarning,
    summaryCheckpointSuggestion.reason,
    summaryCheckpointSuggestion.shouldSuggest,
    showTokenBudgetWarning,
    systemPrompt,
    t,
    tokenBudgetWarningText,
    openContextWindowModal,
    voiceChatEnabled
  ])

  return {
    compareHasPromptContext,
    compareSharedContextLabels,
    compareInteroperabilityNotices,
    contextConflictWarnings
  }
}

export type UsePlaygroundSettingsReturn = ReturnType<
  typeof usePlaygroundSettings
>
