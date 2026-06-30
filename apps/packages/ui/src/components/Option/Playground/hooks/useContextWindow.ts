import React from "react"
import {
  aggregateSessionUsage,
  projectTokenBudget,
  resolveTokenBudgetRisk
} from "../usage-metrics"
import {
  evaluateSummaryCheckpointSuggestion
} from "../conversation-summary-checkpoint"
import {
  buildModelRecommendations,
  type ModelRecommendationAction
} from "../model-recommendations"
import { buildSessionInsights } from "../session-insights"
import { formatCost } from "@/utils/model-pricing"
import {
  estimateTokensFromText,
  collectStringSegments,
  CONTEXT_FOOTPRINT_THRESHOLD_PERCENT
} from "./utils"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseContextWindowDeps {
  /** Token counts from useComposerTokens */
  draftTokenCount: number
  conversationTokenCount: number
  /** Model context limit */
  resolvedMaxContext: number | undefined
  modelContextLength: number | undefined
  /** numCtx from model settings store */
  numCtx: number | undefined
  updateChatModelSetting: (key: string, value: any) => void
  /** Content for token estimation */
  selectedCharacter: any | null
  systemPrompt: string | undefined | null
  selectedQuickPrompt: string | undefined | null
  selectedSystemPrompt: string | undefined | null
  ragPinnedResults: any[]
  messages: any[]
  /** For session usage */
  selectedModel: string | undefined | null
  resolvedProviderKey: string | undefined
  /** For model recommendations */
  deferredComposerInput: string
  modelCapabilities: any
  webSearch: boolean
  jsonMode: boolean
  hasImageAttachment: boolean
  measureComposerPerf: <T>(label: string, fn: () => T) => T
  /** i18n */
  t: (key: string, ...args: any[]) => string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useContextWindow(deps: UseContextWindowDeps) {
  const {
    draftTokenCount,
    conversationTokenCount,
    resolvedMaxContext,
    modelContextLength,
    numCtx,
    updateChatModelSetting,
    selectedCharacter,
    systemPrompt,
    selectedQuickPrompt,
    selectedSystemPrompt,
    ragPinnedResults,
    messages,
    selectedModel,
    resolvedProviderKey,
    deferredComposerInput,
    modelCapabilities,
    webSearch,
    jsonMode,
    hasImageAttachment,
    measureComposerPerf,
    t
  } = deps

  // --- Context window modal state ---
  const [contextWindowModalOpen, setContextWindowModalOpen] = React.useState(false)
  const [contextWindowDraftValue, setContextWindowDraftValue] = React.useState<
    number | undefined
  >(undefined)
  const [sessionInsightsOpen, setSessionInsightsOpen] = React.useState(false)
  const [dismissedRecommendationIds, setDismissedRecommendationIds] =
    React.useState<string[]>([])

  // --- Session usage ---
  const sessionUsageSummary = React.useMemo(
    () => aggregateSessionUsage(messages as any[], selectedModel, resolvedProviderKey),
    [messages, resolvedProviderKey, selectedModel]
  )

  const sessionUsageLabel = React.useMemo(() => {
    const tokenPart = t("playground:tokens.total", "tokens")
    const base = `${sessionUsageSummary.totalTokens.toLocaleString()} ${tokenPart}`
    if (sessionUsageSummary.estimatedCostUsd == null) return base
    return `${base} (${formatCost(sessionUsageSummary.estimatedCostUsd)})`
  }, [sessionUsageSummary.estimatedCostUsd, sessionUsageSummary.totalTokens, t])

  const sessionInsights = React.useMemo(
    () => buildSessionInsights(messages as any[]),
    [messages]
  )

  // --- Token budget projection ---
  const projectedBudget = React.useMemo(
    () =>
      projectTokenBudget({
        conversationTokens: conversationTokenCount,
        draftTokens: draftTokenCount,
        maxTokens: resolvedMaxContext
      }),
    [conversationTokenCount, draftTokenCount, resolvedMaxContext]
  )

  const tokenBudgetRisk = React.useMemo(
    () => resolveTokenBudgetRisk(projectedBudget),
    [projectedBudget]
  )

  const tokenBudgetRiskLabel = React.useMemo(() => {
    if (tokenBudgetRisk.level === "critical")
      return t("playground:tokens.riskCritical", "Critical risk")
    if (tokenBudgetRisk.level === "high")
      return t("playground:tokens.riskHigh", "High risk")
    if (tokenBudgetRisk.level === "medium")
      return t("playground:tokens.riskMedium", "Medium risk")
    if (tokenBudgetRisk.level === "low")
      return t("playground:tokens.riskLow", "Low risk")
    return t("playground:tokens.riskUnknown", "Unknown")
  }, [t, tokenBudgetRisk.level])

  const showTokenBudgetWarning =
    projectedBudget.isOverLimit || projectedBudget.isNearLimit

  const tokenBudgetWarningText = React.useMemo(() => {
    if (!showTokenBudgetWarning) return null
    if (projectedBudget.isOverLimit) {
      return t(
        "playground:tokens.preSendOverLimit",
        "Projected send exceeds the model context window. Consider trimming prompt/context before sending."
      )
    }
    return t(
      "playground:tokens.preSendNearLimit",
      "Projected send is near the context window limit."
    )
  }, [projectedBudget.isOverLimit, showTokenBudgetWarning, t])

  // --- Per-contributor token estimates ---
  const characterContextTokenEstimate = React.useMemo(() => {
    if (!selectedCharacter) return 0
    const segments: string[] = []
    collectStringSegments(selectedCharacter.name, segments)
    collectStringSegments(selectedCharacter.title, segments)
    collectStringSegments(selectedCharacter.system_prompt, segments)
    collectStringSegments(selectedCharacter.greeting, segments)
    collectStringSegments(selectedCharacter.extensions, segments)
    const unique = Array.from(new Set(segments))
    if (unique.length === 0) return 0
    return unique.reduce(
      (total, segment) => total + estimateTokensFromText(segment),
      0
    )
  }, [selectedCharacter])

  const systemPromptTokenEstimate = React.useMemo(() => {
    const promptSegments = [
      String(systemPrompt || "")
    ]
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)
    if (promptSegments.length === 0) return 0
    return promptSegments.reduce(
      (total, segment) => total + estimateTokensFromText(segment),
      0
    )
  }, [systemPrompt])

  const pinnedSourceTokenEstimate = React.useMemo(() => {
    if (!Array.isArray(ragPinnedResults) || ragPinnedResults.length === 0) return 0
    return ragPinnedResults.reduce((total, result) => {
      const snippet = typeof result?.snippet === "string" ? result.snippet : ""
      const title = typeof result?.title === "string" ? result.title : ""
      const sourceLine = typeof result?.source === "string" ? result.source : ""
      const payload = [title, snippet, sourceLine].filter(Boolean).join("\n")
      return total + estimateTokensFromText(payload)
    }, 0)
  }, [ragPinnedResults])

  const historyTokenEstimate = React.useMemo(() => {
    if (!Array.isArray(messages) || messages.length === 0) return 0
    return messages.reduce((total, entry) => {
      const text = typeof entry?.message === "string" ? entry.message : ""
      return total + estimateTokensFromText(text)
    }, 0)
  }, [messages])

  // --- Summary checkpoint ---
  const summaryCheckpointSuggestion = React.useMemo(
    () =>
      evaluateSummaryCheckpointSuggestion({
        messageCount: messages.length,
        projectedBudget
      }),
    [messages.length, projectedBudget]
  )

  // --- Model recommendations ---
  const modelRecommendations = React.useMemo(
    () =>
      measureComposerPerf("derive:model-recommendations", () =>
        buildModelRecommendations({
          draftText: deferredComposerInput,
          selectedModel,
          modelCapabilities,
          webSearch,
          jsonMode,
          hasImageAttachment,
          tokenBudgetRiskLevel: tokenBudgetRisk.level,
          sessionInsights
        })
      ),
    [
      deferredComposerInput,
      hasImageAttachment,
      jsonMode,
      measureComposerPerf,
      modelCapabilities,
      selectedModel,
      sessionInsights,
      tokenBudgetRisk.level,
      webSearch
    ]
  )

  const visibleModelRecommendations = React.useMemo(
    () =>
      modelRecommendations.filter(
        (rec) => !dismissedRecommendationIds.includes(rec.id)
      ),
    [dismissedRecommendationIds, modelRecommendations]
  )

  React.useEffect(() => {
    if (dismissedRecommendationIds.length === 0) return

    const availableIds = new Set(modelRecommendations.map((r) => r.id))
    const next = dismissedRecommendationIds.filter((id) => availableIds.has(id))
    if (next.length === dismissedRecommendationIds.length) return

    setDismissedRecommendationIds(next)
  }, [dismissedRecommendationIds, modelRecommendations])

  // --- Context footprint rows ---
  const contextFootprintRows = React.useMemo(
    () => [
      {
        id: "character",
        label: t("playground:tokens.breakdown.character", "Character + world book"),
        tokens: characterContextTokenEstimate
      },
      {
        id: "prompt",
        label: t("playground:tokens.breakdown.prompt", "System/prompt steering"),
        tokens: systemPromptTokenEstimate
      },
      {
        id: "pinned",
        label: t("playground:tokens.breakdown.pinned", "Pinned sources"),
        tokens: pinnedSourceTokenEstimate
      },
      {
        id: "history",
        label: t("playground:tokens.breakdown.history", "Chat history"),
        tokens: historyTokenEstimate
      },
      {
        id: "draft",
        label: t("playground:tokens.breakdown.draft", "Current draft"),
        tokens: draftTokenCount
      }
    ],
    [
      characterContextTokenEstimate,
      draftTokenCount,
      historyTokenEstimate,
      pinnedSourceTokenEstimate,
      systemPromptTokenEstimate,
      t
    ]
  )

  const nonMessageContextTokenEstimate = React.useMemo(
    () =>
      characterContextTokenEstimate +
      systemPromptTokenEstimate +
      pinnedSourceTokenEstimate,
    [characterContextTokenEstimate, pinnedSourceTokenEstimate, systemPromptTokenEstimate]
  )

  const nonMessageContextPercent = React.useMemo(() => {
    if (
      typeof resolvedMaxContext !== "number" ||
      !Number.isFinite(resolvedMaxContext) ||
      resolvedMaxContext <= 0
    )
      return null
    return (nonMessageContextTokenEstimate / resolvedMaxContext) * 100
  }, [nonMessageContextTokenEstimate, resolvedMaxContext])

  const showNonMessageContextWarning =
    typeof nonMessageContextPercent === "number" &&
    nonMessageContextPercent > CONTEXT_FOOTPRINT_THRESHOLD_PERCENT

  const largestContextContributor = React.useMemo(
    () =>
      contextFootprintRows
        .filter((entry) => entry.tokens > 0)
        .sort((left, right) => right.tokens - left.tokens)[0],
    [contextFootprintRows]
  )

  // --- Context window formatting ---
  const contextWindowFormatter = React.useMemo(() => new Intl.NumberFormat(), [])
  const formatContextWindowValue = React.useCallback(
    (value: number | null | undefined) => {
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
        return t("common:unknown", "Unknown")
      }
      return contextWindowFormatter.format(Math.round(value))
    },
    [contextWindowFormatter, t]
  )

  const isContextWindowOverrideActive =
    typeof numCtx === "number" && Number.isFinite(numCtx) && numCtx > 0
  const requestedContextWindowOverride = isContextWindowOverrideActive
    ? Math.round(numCtx!)
    : null
  const isContextWindowOverrideClamped =
    typeof requestedContextWindowOverride === "number" &&
    typeof modelContextLength === "number" &&
    modelContextLength > 0 &&
    requestedContextWindowOverride > modelContextLength

  // --- Context window modal handlers ---
  const openContextWindowModal = React.useCallback(() => {
    const startingValue =
      typeof numCtx === "number" && Number.isFinite(numCtx) && numCtx > 0
        ? Math.round(numCtx)
        : typeof resolvedMaxContext === "number" &&
            Number.isFinite(resolvedMaxContext) &&
            resolvedMaxContext > 0
          ? Math.round(resolvedMaxContext)
          : undefined
    setContextWindowDraftValue(startingValue)
    setContextWindowModalOpen(true)
  }, [numCtx, resolvedMaxContext])

  const saveContextWindowSetting = React.useCallback(() => {
    if (
      typeof contextWindowDraftValue === "number" &&
      Number.isFinite(contextWindowDraftValue) &&
      contextWindowDraftValue > 0
    ) {
      updateChatModelSetting("numCtx", Math.round(contextWindowDraftValue))
    } else {
      updateChatModelSetting("numCtx", undefined)
    }
    setContextWindowModalOpen(false)
  }, [contextWindowDraftValue, updateChatModelSetting])

  const resetContextWindowSetting = React.useCallback(() => {
    updateChatModelSetting("numCtx", undefined)
    if (
      typeof modelContextLength === "number" &&
      Number.isFinite(modelContextLength) &&
      modelContextLength > 0
    ) {
      setContextWindowDraftValue(Math.round(modelContextLength))
      return
    }
    setContextWindowDraftValue(undefined)
  }, [modelContextLength, updateChatModelSetting])

  const openSessionInsightsModal = React.useCallback(() => {
    setSessionInsightsOpen(true)
  }, [])

  const dismissModelRecommendation = React.useCallback((id: string) => {
    setDismissedRecommendationIds((prev) =>
      prev.includes(id) ? prev : [...prev, id]
    )
  }, [])

  return {
    // Modal state
    contextWindowModalOpen,
    setContextWindowModalOpen,
    contextWindowDraftValue,
    setContextWindowDraftValue,
    sessionInsightsOpen,
    setSessionInsightsOpen,
    // Session usage
    sessionUsageSummary,
    sessionUsageLabel,
    sessionInsights,
    // Budget
    projectedBudget,
    tokenBudgetRisk,
    tokenBudgetRiskLabel,
    showTokenBudgetWarning,
    tokenBudgetWarningText,
    // Per-contributor estimates
    characterContextTokenEstimate,
    systemPromptTokenEstimate,
    pinnedSourceTokenEstimate,
    historyTokenEstimate,
    // Checkpoint
    summaryCheckpointSuggestion,
    // Model recommendations
    modelRecommendations,
    visibleModelRecommendations,
    dismissModelRecommendation,
    // Footprint
    contextFootprintRows,
    nonMessageContextTokenEstimate,
    nonMessageContextPercent,
    showNonMessageContextWarning,
    largestContextContributor,
    // Window formatting
    formatContextWindowValue,
    isContextWindowOverrideActive,
    requestedContextWindowOverride,
    isContextWindowOverrideClamped,
    // Actions
    openContextWindowModal,
    saveContextWindowSetting,
    resetContextWindowSetting,
    openSessionInsightsModal
  }
}
