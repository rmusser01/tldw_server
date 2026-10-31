import React from "react"
import {
  buildCompareModelMetaById,
  compareModelsSupportCapability as compareModelsSupportCapabilityCheck,
  getCompareCapabilityIncompatibilities
} from "../compare-preflight"
import { buildCompareInteroperabilityNotices } from "../compare-interoperability"
import {
  computeResponseDiffPreview,
  type CompareResponseDiff
} from "../compare-response-diff"
import { toText } from "./utils"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseModelComparisonDeps {
  /** Full model list from the query (composerModels) */
  composerModels: any[] | undefined
  /** Currently selected single model */
  selectedModel: string | undefined | null
  setSelectedModel: (model: string) => void
  /** Compare feature flags from useMessageOption */
  compareFeatureEnabled: boolean
  compareFeatureReady?: boolean
  compareMode: boolean
  setCompareMode: (mode: boolean) => void
  compareSelectedModels: string[]
  setCompareSelectedModels: (models: string[]) => void
  compareMaxModels: number
  /** Character / context state needed for interop notices */
  selectedCharacterName: string | null
  ragPinnedResultsLength: number
  webSearch: boolean
  hasPromptContext: boolean
  jsonMode: boolean
  voiceChatEnabled: boolean
  /** i18n */
  t: (key: string, ...args: any[]) => string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useModelComparison(deps: UseModelComparisonDeps) {
  const {
    composerModels,
    selectedModel,
    setSelectedModel,
    compareFeatureEnabled,
    compareFeatureReady = true,
    compareMode,
    setCompareMode,
    compareSelectedModels,
    setCompareSelectedModels,
    compareMaxModels,
    selectedCharacterName,
    ragPinnedResultsLength,
    webSearch,
    hasPromptContext,
    jsonMode,
    voiceChatEnabled,
    t
  } = deps

  const compareModeActive = compareFeatureEnabled && compareMode

  // Ensure compare selection has a sensible default when enabling compare mode
  React.useEffect(() => {
    if (
      compareFeatureReady && compareFeatureEnabled &&
      compareMode &&
      compareSelectedModels.length === 0 &&
      selectedModel
    ) {
      setCompareSelectedModels([selectedModel])
    }
  }, [
    compareFeatureReady,
    compareFeatureEnabled,
    compareMode,
    compareSelectedModels.length,
    selectedModel,
    setCompareSelectedModels
  ])

  React.useEffect(() => {
    if (compareFeatureReady && !compareFeatureEnabled && compareMode) {
      setCompareMode(false)
    }
  }, [compareFeatureReady, compareFeatureEnabled, compareMode, setCompareMode])

  const compareModelMetaById = React.useMemo(
    () => buildCompareModelMetaById((composerModels as any[]) || []),
    [composerModels]
  )

  const availableCompareModels = React.useMemo(
    () =>
      ((composerModels as any[]) || [])
        .filter((model) => model?.model)
        .map((model) => ({
          model: String(model.model),
          nickname:
            typeof model.nickname === "string" ? model.nickname : undefined,
          provider:
            typeof model.provider === "string" ? model.provider : undefined
        })),
    [composerModels]
  )

  const compareModelLabelById = React.useMemo(
    () =>
      new Map(
        availableCompareModels.map((model) => [
          model.model,
          model.nickname || model.model
        ])
      ),
    [availableCompareModels]
  )

  const compareSelectedModelLabels = React.useMemo(
    () =>
      compareSelectedModels.map(
        (modelId) => compareModelLabelById.get(modelId) || modelId
      ),
    [compareModelLabelById, compareSelectedModels]
  )

  const compareNeedsMoreModels =
    compareModeActive && compareSelectedModels.length < 2

  const compareModelsSupportCapability = React.useCallback(
    (modelIds: string[], capability: string) =>
      compareModelsSupportCapabilityCheck(
        modelIds,
        capability,
        compareModelMetaById
      ),
    [compareModelMetaById]
  )

  const compareCapabilityIncompatibilities = React.useMemo(() => {
    if (!compareModeActive || compareSelectedModels.length < 2) return []
    return getCompareCapabilityIncompatibilities({
      modelIds: compareSelectedModels,
      modelMetaById: compareModelMetaById,
      labels: {
        vision: t(
          "playground:composer.compareIncompatVision",
          "Mixed vision support"
        ),
        tools: t(
          "playground:composer.compareIncompatTools",
          "Mixed tool support"
        ),
        streaming: t(
          "playground:composer.compareIncompatStreaming",
          "Mixed streaming behavior"
        ),
        context: t(
          "playground:composer.compareIncompatContext",
          "Large context-window differences"
        )
      }
    })
  }, [compareModeActive, compareModelMetaById, compareSelectedModels, t])

  const toggleCompareMode = React.useCallback(() => {
    if (!compareFeatureEnabled) return
    const next = !compareModeActive
    setCompareMode(next)
    if (next && compareSelectedModels.length === 0 && selectedModel) {
      setCompareSelectedModels([selectedModel])
    }
  }, [
    compareFeatureEnabled,
    compareModeActive,
    compareSelectedModels.length,
    selectedModel,
    setCompareMode,
    setCompareSelectedModels
  ])

  const handleAddCompareModel = React.useCallback(
    (modelId: string) => {
      if (!modelId) return
      if (compareSelectedModels.includes(modelId)) return
      if (compareSelectedModels.length >= compareMaxModels) return
      setCompareSelectedModels([...compareSelectedModels, modelId])
    },
    [compareMaxModels, compareSelectedModels, setCompareSelectedModels]
  )

  const handleRemoveCompareModel = React.useCallback(
    (modelId: string) => {
      if (!modelId) return
      setCompareSelectedModels(
        compareSelectedModels.filter((id) => id !== modelId)
      )
    },
    [compareSelectedModels, setCompareSelectedModels]
  )

  const sendLabel = React.useMemo(() => {
    if (compareNeedsMoreModels) {
      return t(
        "playground:composer.compareAddModelToSend",
        "Add one more model"
      )
    }
    if (compareModeActive && compareSelectedModels.length > 1) {
      return t(
        "playground:composer.compareSendToModels",
        "Send to {{count}} models",
        { count: compareSelectedModels.length }
      )
    }
    return t("common:send", "Send")
  }, [compareModeActive, compareNeedsMoreModels, compareSelectedModels.length, t])

  // Shared-context labels for the compare activation banner
  const compareHasPromptContext = hasPromptContext

  const compareSharedContextLabels = React.useMemo(() => {
    const labels: string[] = []
    if (selectedCharacterName) {
      labels.push(
        String(
          t("playground:composer.compareSharedCharacter", "Character: {{name}}", {
            name: selectedCharacterName
          } as any)
        )
      )
    }
    if (compareHasPromptContext) {
      labels.push(
        String(
          t("playground:composer.compareSharedPrompt", "Prompt steering enabled")
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

  return {
    compareModeActive,
    compareModelMetaById,
    availableCompareModels,
    compareModelLabelById,
    compareSelectedModelLabels,
    compareNeedsMoreModels,
    compareModelsSupportCapability,
    compareCapabilityIncompatibilities,
    toggleCompareMode,
    handleAddCompareModel,
    handleRemoveCompareModel,
    sendLabel,
    compareHasPromptContext,
    compareSharedContextLabels,
    compareInteroperabilityNotices
  }
}
