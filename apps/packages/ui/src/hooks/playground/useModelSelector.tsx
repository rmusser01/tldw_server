import React from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { Tooltip } from "antd"
import { Star } from "lucide-react"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { tldwModels } from "@/services/tldw"
import { useStoreChatModelSettings } from "@/store/model"
import {
  LOCAL_PROVIDERS,
  filterModelsForScope,
  getCanonicalModelKey,
  getModelId,
  getModelProvider,
  modelMatchesSearch,
  sortModelsForSelector,
  type ModelListScope,
  type ModelSelectorDescriptor,
  type ModelSortMode,
  type ModelUsageStats
} from "./modelSelectorUtils"

export type { ModelListScope, ModelSortMode } from "./modelSelectorUtils"

export type UseModelSelectorParams = {
  composerModels: any[] | undefined
  selectedModel: string | null
  setSelectedModel: (model: string) => void
  navigate: (path: string) => void
}

export function useModelSelector({
  composerModels,
  selectedModel,
  setSelectedModel,
  navigate
}: UseModelSelectorParams) {
  const { t } = useTranslation(["playground", "common"])
  const apiProvider = useStoreChatModelSettings((state) => state.apiProvider)

  const [modelDropdownOpen, setModelDropdownOpen] = React.useState(false)
  const [modelSearchQuery, setModelSearchQuery] = React.useState("")
  const [favoriteModels, setFavoriteModels, favoriteModelsMeta] = useStorage<string[]>(
    "favoriteChatModels",
    []
  )
  const [modelSortMode, setModelSortMode] = useStorage<ModelSortMode>(
    "modelSelectSortMode",
    "provider"
  )
  const [storedModelListScope, setModelListScope] = useStorage<ModelListScope>(
    "modelSelectListScope",
    "configured"
  )
  const [modelUsageByKey, setModelUsageByKey] = useStorage<Record<string, ModelUsageStats>>(
    "chatModelUsageByProviderModel",
    {}
  )

  const modelListScope: ModelListScope =
    storedModelListScope === "catalog" ? "catalog" : "configured"
  const selectedProviderHint =
    typeof apiProvider === "string" && apiProvider.trim()
      ? apiProvider.trim().toLowerCase()
      : null

  const normalizedModelUsageByKey = React.useMemo<Record<string, ModelUsageStats>>(() => {
    if (!modelUsageByKey || typeof modelUsageByKey !== "object") return {}
    const normalized: Record<string, ModelUsageStats> = {}
    for (const [key, value] of Object.entries(modelUsageByKey)) {
      if (!value || typeof value !== "object") continue
      const selectedCount = Number((value as ModelUsageStats).selectedCount || 0)
      const lastSelectedAt = Number((value as ModelUsageStats).lastSelectedAt || 0)
      if (selectedCount > 0 || lastSelectedAt > 0) {
        normalized[key] = { selectedCount, lastSelectedAt }
      }
    }
    return normalized
  }, [modelUsageByKey])

  const selectedModelMeta = React.useMemo(() => {
    if (!selectedModel) return null
    const models = (composerModels as any[]) || []
    const selected = selectedModel.trim()
    const selectedLower = selected.toLowerCase()
    const canonicalMatch = models.find(
      (model) => getCanonicalModelKey(model).toLowerCase() === selectedLower
    )
    if (canonicalMatch) return canonicalMatch

    const providerMatch = selectedProviderHint
      ? models.find((model) => {
          const modelId = getModelId(model)
          return modelId === selected && getModelProvider(model) === selectedProviderHint
        })
      : null
    if (providerMatch) return providerMatch

    return (
      models.find((model) => {
        const modelId = getModelId(model)
        return modelId === selected || String(model.model || "") === selected
      }) || null
    )
  }, [composerModels, selectedModel, selectedProviderHint])

  const selectedModelKey = React.useMemo(() => {
    if (selectedModelMeta) return getCanonicalModelKey(selectedModelMeta)
    if (!selectedModel) return null
    return getCanonicalModelKey(selectedProviderHint || "custom", selectedModel)
  }, [selectedModel, selectedModelMeta, selectedProviderHint])

  const modelContextLength = React.useMemo(() => {
    const candidates = [
      selectedModelMeta?.context_length,
      selectedModelMeta?.contextLength,
      (selectedModelMeta as any)?.context_window,
      selectedModelMeta?.details?.context_length,
      (selectedModelMeta as any)?.details?.contextLength,
      (selectedModelMeta as any)?.details?.context_window,
      (selectedModelMeta as any)?.max_input_tokens,
      (selectedModelMeta as any)?.details?.max_input_tokens,
      (selectedModelMeta as any)?.max_context_tokens,
      (selectedModelMeta as any)?.details?.max_context_tokens
    ]
    for (const candidate of candidates) {
      if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) {
        return candidate
      }
    }
    return null
  }, [selectedModelMeta])

  const modelCapabilities = React.useMemo(() => {
    const caps =
      selectedModelMeta?.details?.capabilities ??
      (selectedModelMeta as any)?.capabilities
    return Array.isArray(caps) ? caps.map((cap) => String(cap).toLowerCase()) : []
  }, [selectedModelMeta])

  const numCtx = useStoreChatModelSettings((state) => state.numCtx)
  const requestedNumCtx = React.useMemo(() => {
    if (typeof numCtx !== "number" || !Number.isFinite(numCtx) || numCtx <= 0) {
      return null
    }
    return numCtx
  }, [numCtx])

  const resolvedMaxContext = React.useMemo(() => {
    if (typeof requestedNumCtx === "number") {
      if (typeof modelContextLength === "number" && modelContextLength > 0) {
        return Math.min(requestedNumCtx, modelContextLength)
      }
      return requestedNumCtx
    }
    if (typeof modelContextLength === "number" && modelContextLength > 0) {
      return modelContextLength
    }
    return null
  }, [modelContextLength, requestedNumCtx])

  const resolvedProviderKey = React.useMemo(() => {
    if (selectedModelMeta) return getModelProvider(selectedModelMeta)
    return selectedProviderHint || "custom"
  }, [selectedModelMeta, selectedProviderHint])

  const providerLabel = React.useMemo(
    () => tldwModels.getProviderDisplayName(resolvedProviderKey || "custom"),
    [resolvedProviderKey]
  )

  const modelSummaryLabel = React.useMemo(() => {
    if (!selectedModel) {
      return t(
        "playground:composer.modelPlaceholder",
        "API / model"
      )
    }
    return (
      selectedModelMeta?.nickname ||
      getModelId(selectedModelMeta as ModelSelectorDescriptor) ||
      selectedModel
    )
  }, [selectedModel, selectedModelMeta, t])

  const apiModelLabel = React.useMemo(() => {
    if (!selectedModel) {
      return t(
        "playground:composer.selectModel",
        "Select a model"
      )
    }
    return `${providerLabel} / ${modelSummaryLabel}`
  }, [modelSummaryLabel, providerLabel, selectedModel, t])

  const modelSelectorWarning = !selectedModel

  const favoriteModelSet = React.useMemo(
    () => new Set((favoriteModels || []).map((value) => String(value))),
    [favoriteModels]
  )

  const isFavoriteModel = React.useCallback(
    (model: ModelSelectorDescriptor) => {
      const key = getCanonicalModelKey(model)
      const id = getModelId(model)
      return favoriteModelSet.has(key) || favoriteModelSet.has(id)
    },
    [favoriteModelSet]
  )

  const toggleFavoriteModel = React.useCallback(
    (modelOrKey: ModelSelectorDescriptor | string) => {
      const canonicalKey =
        typeof modelOrKey === "string" ? modelOrKey : getCanonicalModelKey(modelOrKey)
      const legacyModelId =
        typeof modelOrKey === "string" ? modelOrKey : getModelId(modelOrKey)
      void setFavoriteModels((prev) => {
        const list = Array.isArray(prev) ? prev.map(String) : []
        const next = new Set(list)
        if (next.has(canonicalKey) || next.has(legacyModelId)) {
          next.delete(canonicalKey)
          next.delete(legacyModelId)
        } else {
          next.add(canonicalKey)
        }
        return Array.from(next)
      })
      setModelDropdownOpen(true)
    },
    [setFavoriteModels]
  )

  const recordModelUsage = React.useCallback(
    (model: ModelSelectorDescriptor) => {
      const key = getCanonicalModelKey(model)
      void setModelUsageByKey((prev) => {
        const previous =
          prev && typeof prev === "object" ? (prev as Record<string, ModelUsageStats>) : {}
        const current = previous[key]
        return {
          ...previous,
          [key]: {
            selectedCount: Number(current?.selectedCount || 0) + 1,
            lastSelectedAt: Date.now()
          }
        }
      })
    },
    [setModelUsageByKey]
  )

  const filteredModels = React.useMemo(() => {
    const list = (composerModels as any[]) || []
    const scoped = filterModelsForScope(list, modelListScope)
    const q = modelSearchQuery.trim().toLowerCase()
    const searched = q
      ? scoped.filter((model) => modelMatchesSearch(model, q, getProviderDisplayName))
      : scoped
    return sortModelsForSelector(searched, {
      selectedModel,
      selectedProvider: resolvedProviderKey,
      favoriteKeys: favoriteModelSet,
      usageByKey: normalizedModelUsageByKey,
      sortMode: modelSortMode
    })
  }, [
    composerModels,
    favoriteModelSet,
    modelListScope,
    modelSearchQuery,
    modelSortMode,
    normalizedModelUsageByKey,
    resolvedProviderKey,
    selectedModel
  ])

  const modelDropdownMenuItems = React.useMemo(() => {
    const models = filteredModels || []
    const allModels = (composerModels as any[]) || []

    if (allModels.length === 0) {
      return [
        {
          key: "no-models",
          disabled: true,
          label: (
            <div className="px-1 py-1 text-xs text-text-muted">
              {t(
                "playground:composer.noModelsAvailable",
                "No models available. Connect your server in Settings."
              )}
            </div>
          )
        },
        {
          type: "divider" as const,
          key: "no-models-divider"
        },
        {
          key: "open-model-settings",
          label: t(
            "playground:composer.openModelSettings",
            "Open model settings"
          ),
          onClick: () => navigate("/settings/tldw")
        }
      ]
    }

    if (models.length === 0) {
      return [
        {
          key: "no-matches",
          disabled: true,
          label: (
            <div className="px-1 py-1 text-xs text-text-muted">
              {t(
                "playground:composer.noModelsMatch",
                "No models match your search."
              )}
            </div>
          )
        }
      ]
    }

    const toGroupKey = (providerRaw: string) =>
      providerRaw === "chrome"
        ? "default"
        : LOCAL_PROVIDERS.has(providerRaw)
          ? "custom"
          : providerRaw

    const byLabel = (a: any, b: any) => {
      const aProvider = getProviderDisplayName(getModelProvider(a))
      const bProvider = getProviderDisplayName(getModelProvider(b))
      const aLabel = `${aProvider} ${a.nickname || getModelId(a)}`.toLowerCase()
      const bLabel = `${bProvider} ${b.nickname || getModelId(b)}`.toLowerCase()
      return aLabel.localeCompare(bLabel)
    }

    const firstFavoriteKey = favoriteModels?.length
      ? models.find(isFavoriteModel)
      : null
    const firstFavoriteModelKey = firstFavoriteKey
      ? getCanonicalModelKey(firstFavoriteKey)
      : null

    const normalizePositiveNumber = (value: unknown): number | undefined => {
      if (typeof value === "number" && Number.isFinite(value) && value > 0) {
        return value
      }
      if (typeof value === "string") {
        const parsed = Number.parseFloat(value)
        if (Number.isFinite(parsed) && parsed > 0) {
          return parsed
        }
      }
      return undefined
    }

    const resolvePriceHint = (model: any): string | null => {
      const directHints = [
        model?.details?.price_hint,
        model?.details?.pricing_hint,
        model?.price_hint,
        model?.pricing_hint
      ]
      const direct = directHints.find(
        (value) => typeof value === "string" && value.trim().length > 0
      )
      if (typeof direct === "string") {
        return direct.trim()
      }

      const pricing = model?.details?.pricing
      if (!pricing || typeof pricing !== "object") {
        return null
      }

      const input = normalizePositiveNumber(
        (pricing as any).input_per_million ??
          (pricing as any).prompt_per_million ??
          (pricing as any).input
      )
      const output = normalizePositiveNumber(
        (pricing as any).output_per_million ??
          (pricing as any).completion_per_million ??
          (pricing as any).output
      )

      const formatUsd = (value: number) =>
        `$${value >= 1 ? value.toFixed(2) : value.toPrecision(2)}`

      if (typeof input === "number" && typeof output === "number") {
        return `${formatUsd(input)}/${formatUsd(output)}`
      }
      if (typeof input === "number") {
        return `${formatUsd(input)} in`
      }
      if (typeof output === "number") {
        return `${formatUsd(output)} out`
      }
      return null
    }

    const getModelDescription = (
      model: any,
      capabilities: string[],
      contextLength: number | undefined,
      priceHint: string | null
    ) => {
      const parts: string[] = []
      const providerDisplay = getProviderDisplayName(getModelProvider(model))
      parts.push(`${providerDisplay} model.`)
      if (capabilities.includes("vision") || model.supportsVision) {
        parts.push("Can analyze images.")
      }
      if (capabilities.includes("tools") || model.supportsTools) {
        parts.push("Supports tool use and function calling.")
      }
      if (capabilities.includes("streaming") || model.supportsStreaming) {
        parts.push("Supports streaming output.")
      }
      if (typeof contextLength === "number") {
        if (contextLength > 100000) {
          parts.push(`Long context (${Math.round(contextLength / 1000)}k tokens).`)
        } else if (contextLength > 0) {
          parts.push(`Context: ${Math.round(contextLength / 1000)}k tokens.`)
        }
      }
      if (capabilities.includes("fast") || model.fast) {
        parts.push("Optimized for speed.")
      }
      if (priceHint) {
        parts.push(`Estimated price: ${priceHint}.`)
      }
      return parts.join(" ")
    }

    const buildItem = (model: any) => {
      const providerRaw = getModelProvider(model)
      const modelId = getModelId(model)
      const modelKey = getCanonicalModelKey(model)
      const modelLabel = model.nickname || modelId
      const isFavorite = isFavoriteModel(model)
      const isRecommended = firstFavoriteModelKey === modelKey
      const favoriteTitle = isFavorite
        ? t("playground:composer.favoriteRemove", "Remove from favorites")
        : t("playground:composer.favoriteAdd", "Add to favorites")

      const rawCapabilities = model.details?.capabilities || model.capabilities || []
      const capabilities = Array.isArray(rawCapabilities)
        ? rawCapabilities.map((cap) => String(cap).toLowerCase())
        : []
      const contextLength = normalizePositiveNumber(
        model.context_length ??
          model.contextLength ??
          model.context_window ??
          model.details?.context_length ??
          model.details?.contextLength ??
          model.details?.context_window
      )
      const priceHint = resolvePriceHint(model)
      const capabilityBadges: string[] = []
      if (capabilities.includes("vision") || model.supportsVision) {
        capabilityBadges.push("Vision")
      }
      if (capabilities.includes("tools") || model.supportsTools) {
        capabilityBadges.push("Tools")
      }
      if (capabilities.includes("streaming") || model.supportsStreaming) {
        capabilityBadges.push("Streaming")
      }
      if (typeof contextLength === "number" && contextLength > 0) {
        capabilityBadges.push(`${Math.max(1, Math.round(contextLength / 1000))}k ctx`)
      }
      if (capabilities.includes("fast") || model.fast) {
        capabilityBadges.push("Fast")
      }
      if (priceHint) {
        capabilityBadges.push(priceHint)
      }

      const modelDescription = getModelDescription(
        model,
        capabilities,
        contextLength,
        priceHint
      )

      return {
        key: modelKey,
        label: (
          <Tooltip
            title={modelDescription}
            placement="right"
            mouseEnterDelay={0.5}
            styles={{ root: { maxWidth: 280 } }}
          >
            <div className="flex items-center gap-2 text-sm">
              <ProviderIcons provider={providerRaw} className="h-3 w-3 text-text-subtle" />
              <span className="truncate flex-1">{modelLabel}</span>
              {isRecommended && (
                <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                  {t("playground:composer.recommended", "Recommended")}
                </span>
              )}
              {capabilityBadges.slice(0, 5).map(cap => (
                <span key={cap} className="rounded bg-surface2 px-1 py-0.5 text-[9px] text-text-muted">
                  {cap}
                </span>
              ))}
              <button
                type="button"
                className="rounded p-0.5 text-text-subtle transition hover:bg-surface2"
                onMouseDown={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                }}
                onClick={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                  toggleFavoriteModel(model)
                }}
                aria-label={favoriteTitle}
                title={favoriteTitle}
              >
                <Star
                  className={`h-3.5 w-3.5 ${
                    isFavorite ? "fill-warn text-warn" : "text-text-subtle"
                  }`}
                />
              </button>
            </div>
          </Tooltip>
        ),
        onClick: () => {
          recordModelUsage(model)
          setSelectedModel(modelKey)
        }
      }
    }

    if (modelSortMode === "az") {
      return models.slice().sort(byLabel).map(buildItem)
    }

    if (modelSortMode === "favorites") {
      const favorites = models.filter(isFavoriteModel)
      const others = models.filter((model) => !isFavoriteModel(model))
      const items: any[] = []
      if (favorites.length > 0) {
        items.push({
          type: "group" as const,
          key: "favorites",
          label: t("playground:composer.favorites", "Favorites"),
          children: favorites.slice().sort(byLabel).map(buildItem)
        })
      }
      items.push(...others.slice().sort(byLabel).map(buildItem))
      return items
    }

    const promotedGroups: any[] = []
    const promotedModelKeys = new Set<string>()
    const pushPromotedGroup = (
      key: string,
      label: string,
      groupModels: ModelSelectorDescriptor[]
    ) => {
      const uniqueModels = groupModels.filter((model) => {
        const modelKey = getCanonicalModelKey(model)
        if (promotedModelKeys.has(modelKey)) return false
        promotedModelKeys.add(modelKey)
        return true
      })
      if (uniqueModels.length === 0) return
      promotedGroups.push({
        type: "group" as const,
        key,
        label,
        children: uniqueModels.map(buildItem)
      })
    }

    if (selectedModelKey) {
      pushPromotedGroup(
        "current-model",
        t("playground:composer.currentModel", "Current"),
        models.filter((model) => getCanonicalModelKey(model) === selectedModelKey)
      )
    }

    pushPromotedGroup(
      "recent-models",
      t("playground:composer.recentModels", "Recent"),
      models.filter((model) => {
        const usage = normalizedModelUsageByKey[getCanonicalModelKey(model)]
        return Number(usage?.selectedCount || 0) > 0
      })
    )

    pushPromotedGroup(
      "favorite-models",
      t("playground:composer.favorites", "Favorites"),
      models.filter(isFavoriteModel)
    )

    const groups = new Map<string, any[]>()
    for (const model of models) {
      const modelKey = getCanonicalModelKey(model)
      if (promotedModelKeys.has(modelKey)) continue
      const providerRaw = getModelProvider(model)
      const groupKey = toGroupKey(providerRaw)
      if (!groups.has(groupKey)) groups.set(groupKey, [])
      groups.get(groupKey)!.push(buildItem(model))
    }

    const entries = Array.from(groups.entries())
    if (modelSortMode === "localFirst") {
      entries.sort(([aKey], [bKey]) => {
        const aLocal = LOCAL_PROVIDERS.has(aKey) || aKey === "default"
        const bLocal = LOCAL_PROVIDERS.has(bKey) || bKey === "default"
        if (aLocal !== bLocal) return aLocal ? -1 : 1
        return aKey.localeCompare(bKey)
      })
    }

    return [
      ...promotedGroups,
      ...entries.map(([key, children]) => ({
        type: "group" as const,
        key: `group-${key}`,
        label: (
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-text-subtle">
            <ProviderIcons provider={key} className="h-3 w-3" />
            <span>{getProviderDisplayName(key)}</span>
          </div>
        ),
        children
      }))
    ]
  }, [
    composerModels,
    favoriteModels,
    favoriteModelSet,
    filteredModels,
    isFavoriteModel,
    modelSortMode,
    navigate,
    normalizedModelUsageByKey,
    recordModelUsage,
    selectedModelKey,
    setSelectedModel,
    t,
    toggleFavoriteModel
  ])

  const isSmallModel =
    modelCapabilities.includes("fast") ||
    (typeof modelContextLength === "number" && modelContextLength <= 8192)

  return {
    modelDropdownOpen,
    setModelDropdownOpen,
    modelSearchQuery,
    setModelSearchQuery,
    modelSortMode,
    setModelSortMode,
    modelListScope,
    setModelListScope,
    selectedModelMeta,
    selectedModelKey,
    modelContextLength,
    modelCapabilities,
    resolvedMaxContext,
    resolvedProviderKey,
    providerLabel,
    modelSummaryLabel,
    apiModelLabel,
    modelSelectorWarning,
    favoriteModels,
    favoriteModelsIsLoading: favoriteModelsMeta.isLoading,
    favoriteModelSet,
    toggleFavoriteModel,
    filteredModels,
    modelDropdownMenuItems,
    isSmallModel
  }
}
