import React from "react"
import type { ComposerContextItem } from "../ComposerToolbar"
import type { KnowledgeTab } from "@/components/Knowledge"
import { toText } from "./utils"

export type UsePlaygroundContextItemsDeps = {
  selectedModel: string | null | undefined
  modelSummaryLabel: string
  isSessionDegraded: boolean
  connectionStatusLabel: string
  compareModeActive: boolean
  compareSelectedModels: string[]
  currentPreset: { key: string; label: string } | null | undefined
  selectedCharacterName: string | null
  characterPendingApply: boolean
  contextToolsOpen: boolean
  ragPinnedResultsLength: number
  webSearch: boolean
  sessionUsageTotalTokens: number
  sessionUsageLabel: string
  selectedSystemPrompt: string | null | undefined
  selectedQuickPrompt: string | null | undefined
  systemPrompt: string | null | undefined
  promptSummaryLabel: string
  jsonMode: boolean
  showTokenBudgetWarning: boolean
  tokenBudgetRiskLevel: string
  tokenBudgetRiskLabel: string
  projectedBudgetUtilizationPercent: number | null | undefined
  nonMessageContextPercent: number | null | undefined
  showNonMessageContextWarning: boolean
  temporaryChat: boolean
  openModelApiSelector: () => void
  focusConnectionCard: () => void
  setOpenModelSettings: (open: boolean) => void
  setOpenActorSettings: (open: boolean) => void
  setContextToolsOpen: (open: boolean) => void
  handleToggleWebSearch: () => void
  openKnowledgePanel: (tab: KnowledgeTab) => void
  openContextWindowModal: () => void
  openSessionInsightsModal: () => void
  updateChatModelSetting: (key: string, value: any) => void
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

export function usePlaygroundContextItems(
  deps: UsePlaygroundContextItemsDeps
): ComposerContextItem[] {
  const {
    selectedModel,
    modelSummaryLabel,
    isSessionDegraded,
    connectionStatusLabel,
    compareModeActive,
    compareSelectedModels,
    currentPreset,
    selectedCharacterName,
    characterPendingApply,
    contextToolsOpen,
    ragPinnedResultsLength,
    webSearch,
    sessionUsageTotalTokens,
    sessionUsageLabel,
    selectedSystemPrompt,
    selectedQuickPrompt,
    systemPrompt,
    promptSummaryLabel,
    jsonMode,
    showTokenBudgetWarning,
    tokenBudgetRiskLevel,
    tokenBudgetRiskLabel,
    projectedBudgetUtilizationPercent,
    nonMessageContextPercent,
    showNonMessageContextWarning,
    temporaryChat,
    openModelApiSelector,
    focusConnectionCard,
    setOpenModelSettings,
    setOpenActorSettings,
    setContextToolsOpen,
    handleToggleWebSearch,
    openKnowledgePanel,
    openContextWindowModal,
    openSessionInsightsModal,
    updateChatModelSetting,
    t
  } = deps

  return React.useMemo<ComposerContextItem[]>(() => {
    const items: ComposerContextItem[] = []
    items.push({
      id: "model",
      label: t("playground:composer.context.model", "Model"),
      value: selectedModel ? modelSummaryLabel : t("common:none", "None"),
      tone: selectedModel ? "active" : "warning",
      onClick: openModelApiSelector
    })
    if (isSessionDegraded) {
      items.push({
        id: "sessionStatus",
        label: t("playground:composer.context.sessionStatus", "Session status"),
        value: connectionStatusLabel,
        tone: "warning",
        onClick: focusConnectionCard
      })
    }

    if (compareModeActive) {
      items.push({
        id: "compare",
        label: t("playground:composer.context.compare", "Compare"),
        value:
          compareSelectedModels.length > 0
            ? String(
                t("playground:composer.context.compareCount", {
                  defaultValue: "{{count}} models",
                  count: compareSelectedModels.length
                } as any)
              )
            : String(t("playground:composer.context.compareOn", "On")),
        tone: "active",
        onClick: () => setOpenModelSettings(true)
      })
    }

    if (currentPreset && currentPreset.key !== "custom") {
      items.push({
        id: "preset",
        label: toText(t("playground:composer.context.preset", "Preset")),
        value: toText(
          t(
            `playground:presets.${currentPreset.key}.label`,
            currentPreset.label
          )
        ),
        tone: "active",
        onClick: () => setOpenModelSettings(true)
      })
    }

    if (selectedCharacterName) {
      items.push({
        id: "character",
        label: toText(t("playground:composer.context.character", "Character")),
        value: characterPendingApply
          ? toText(
              t(
                "playground:composer.context.characterNextTurn",
                "{{name}} (next turn)",
                { name: selectedCharacterName } as any
              )
            )
          : selectedCharacterName,
        tone: "active",
        onClick: () => setOpenActorSettings(true)
      })
    }

    if (contextToolsOpen) {
      items.push({
        id: "knowledge",
        label: toText(t("playground:composer.context.knowledge", "Knowledge")),
        value: toText(t("common:open", "Open")),
        tone: "active",
        onClick: () => setContextToolsOpen(false)
      })
    }

    if (ragPinnedResultsLength > 0) {
      items.push({
        id: "ragPinned",
        label: toText(t("playground:composer.context.pinnedSources", "Pinned")),
        value: toText(
          t("playground:composer.context.pinnedCount", {
            defaultValue: "{{count}} sources",
            count: ragPinnedResultsLength
          } as any)
        ),
        tone: "active",
        onClick: () => openKnowledgePanel("search")
      })
    }

    if (webSearch) {
      items.push({
        id: "webSearch",
        label: toText(t("playground:composer.context.webSearch", "Web search")),
        value: toText(t("common:on", "On")),
        tone: "active",
        onClick: handleToggleWebSearch
      })
    }
    if (sessionUsageTotalTokens > 0) {
      items.push({
        id: "sessionUsage",
        label: toText(t("playground:composer.context.session", "Session")),
        value: sessionUsageLabel,
        tone: "neutral",
        onClick: openSessionInsightsModal
      })
    }
    if (
      selectedSystemPrompt ||
      selectedQuickPrompt ||
      String(systemPrompt || "").trim().length > 0
    ) {
      items.push({
        id: "prompt",
        label: toText(t("playground:composer.context.prompt", "Prompt")),
        value: promptSummaryLabel,
        tone: "active"
      })
    }

    if (jsonMode) {
      items.push({
        id: "json",
        label: toText(t("playground:composer.context.json", "JSON mode")),
        value: toText(
          t(
            "playground:composer.context.jsonShort",
            "Object responses"
          )
        ),
        tone: "active",
        onClick: () => updateChatModelSetting("jsonMode", undefined)
      })
    }

    if (showTokenBudgetWarning) {
      items.push({
        id: "budget",
        label: toText(t("playground:composer.context.budget", "Budget")),
        value: `${tokenBudgetRiskLabel}${
          projectedBudgetUtilizationPercent != null
            ? ` • ${Math.round(projectedBudgetUtilizationPercent)}%`
            : ""
        }`,
        tone: "warning",
        onClick: openContextWindowModal
      })
    }
    if (tokenBudgetRiskLevel !== "unknown" && !showTokenBudgetWarning) {
      items.push({
        id: "truncationRisk",
        label: toText(t("playground:composer.context.truncationRisk", "Truncation")),
        value: tokenBudgetRiskLabel,
        tone:
          tokenBudgetRiskLevel === "high" || tokenBudgetRiskLevel === "critical"
            ? "warning"
            : "neutral",
        onClick: openContextWindowModal
      })
    }
    if (nonMessageContextPercent != null) {
      items.push({
        id: "contextMix",
        label: toText(t("playground:composer.context.contextMix", "Context mix")),
        value: toText(
          t(
            "playground:composer.context.nonMessageShare",
            "{{percent}}% non-message",
            {
              percent: Math.max(0, Math.round(nonMessageContextPercent))
            } as any
          )
        ),
        tone: showNonMessageContextWarning ? "warning" : "neutral",
        onClick: openContextWindowModal
      })
    }

    if (temporaryChat) {
      items.push({
        id: "temporary",
        label: toText(t("playground:composer.context.temporary", "Temporary")),
        value: toText(t("playground:composer.context.notSaved", "Not saved")),
        tone: "warning"
      })
    }

    return items
  }, [
    compareModeActive,
    compareSelectedModels.length,
    connectionStatusLabel,
    contextToolsOpen,
    currentPreset,
    jsonMode,
    focusConnectionCard,
    handleToggleWebSearch,
    isSessionDegraded,
    modelSummaryLabel,
    openModelApiSelector,
    openKnowledgePanel,
    openContextWindowModal,
    nonMessageContextPercent,
    characterPendingApply,
    promptSummaryLabel,
    tokenBudgetRiskLevel,
    tokenBudgetRiskLabel,
    ragPinnedResultsLength,
    selectedCharacterName,
    selectedModel,
    selectedQuickPrompt,
    selectedSystemPrompt,
    sessionUsageLabel,
    sessionUsageTotalTokens,
    openSessionInsightsModal,
    setContextToolsOpen,
    setOpenModelSettings,
    setOpenActorSettings,
    showTokenBudgetWarning,
    projectedBudgetUtilizationPercent,
    showNonMessageContextWarning,
    systemPrompt,
    t,
    temporaryChat,
    updateChatModelSetting
  ])
}
