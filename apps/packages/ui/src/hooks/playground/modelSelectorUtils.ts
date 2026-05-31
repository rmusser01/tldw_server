export type ModelSortMode = "favorites" | "az" | "provider" | "localFirst"

export type ModelListScope = "configured" | "catalog"

export type ModelUsageStats = {
  selectedCount: number
  lastSelectedAt: number
}

export type ModelSelectorDescriptor = Record<string, any>

export const LOCAL_PROVIDERS = new Set([
  "lmstudio",
  "llamafile",
  "ollama",
  "ollama2",
  "llama",
  "llamacpp",
  "vllm",
  "custom",
  "local",
  "tldw",
  "chrome",
  "mlx"
])

const nonEmptyString = (value: unknown): string | null => {
  if (typeof value !== "string" && typeof value !== "number") return null
  const trimmed = String(value).trim()
  return trimmed.length > 0 ? trimmed : null
}

const stripInternalModelPrefix = (value: string | null): string | null => {
  if (!value) return null
  const stripped = value.replace(/^tldw:/i, "").trim()
  return stripped.length > 0 ? stripped : null
}

const normalizeProviderName = (value: unknown): string => {
  const normalized = nonEmptyString(value)?.toLowerCase() || ""
  if (!normalized) return "other"
  if (normalized === "llama.cpp") return "llamacpp"
  if (normalized === "local-llm") return "local"
  return normalized
}

const boolFromPaths = (
  model: ModelSelectorDescriptor,
  keys: string[]
): boolean | null => {
  for (const key of keys) {
    const value = model?.[key] ?? model?.details?.[key] ?? model?.metadata?.[key]
    if (typeof value === "boolean") return value
  }
  return null
}

const lowercaseField = (
  model: ModelSelectorDescriptor,
  keys: string[]
): string | null => {
  for (const key of keys) {
    const value = model?.[key] ?? model?.details?.[key] ?? model?.metadata?.[key]
    const normalized = nonEmptyString(value)?.toLowerCase()
    if (normalized) return normalized
  }
  return null
}

export const getModelId = (model: ModelSelectorDescriptor | null | undefined): string => {
  if (!model) return ""
  const modelField = nonEmptyString(model.model)
  const idField = nonEmptyString(model.id)
  const nameField = nonEmptyString(model.name)

  if (modelField?.toLowerCase().startsWith("tldw:")) {
    return (
      stripInternalModelPrefix(idField) ||
      stripInternalModelPrefix(modelField) ||
      stripInternalModelPrefix(nameField) ||
      ""
    )
  }

  return (
    stripInternalModelPrefix(modelField) ||
    stripInternalModelPrefix(idField) ||
    stripInternalModelPrefix(nameField) ||
    ""
  )
}

export const getModelProvider = (
  model: ModelSelectorDescriptor | null | undefined
): string => {
  if (!model) return "other"
  return normalizeProviderName(
    model.provider ??
      model.provider_key ??
      model.providerKey ??
      model.api_provider ??
      model.apiProvider ??
      model.details?.provider ??
      model.details?.provider_key ??
      model.metadata?.provider
  )
}

export function getCanonicalModelKey(
  modelOrProvider: ModelSelectorDescriptor | string | null | undefined,
  modelId?: unknown
): string {
  if (typeof modelOrProvider === "string") {
    const provider = normalizeProviderName(modelOrProvider)
    const id = nonEmptyString(modelId) || ""
    return id ? `${provider}:${id}` : provider
  }

  const model = modelOrProvider
  const id = getModelId(model)
  const provider = getModelProvider(model)
  return id ? `${provider}:${id}` : provider
}

export const getFavoriteModelKeyCandidates = (
  model: ModelSelectorDescriptor | null | undefined
): string[] => {
  if (!model) return []
  const id = getModelId(model)
  const rawModel = nonEmptyString(model.model)
  return Array.from(
    new Set(
      [
        getCanonicalModelKey(model),
        id,
        id ? `tldw:${id}` : null,
        rawModel
      ].filter((value): value is string => Boolean(value))
    )
  )
}

export const hasFavoriteModelKey = (
  model: ModelSelectorDescriptor | null | undefined,
  favoriteKeys: Set<string>
): boolean =>
  getFavoriteModelKeyCandidates(model).some((key) => favoriteKeys.has(key))

export const isCatalogOnlyModel = (
  model: ModelSelectorDescriptor | null | undefined
): boolean => {
  if (!model) return false

  const explicitCatalog = boolFromPaths(model, [
    "catalog_only",
    "catalogOnly",
    "is_catalog_only",
    "isCatalogOnly"
  ])
  if (explicitCatalog === true) return true

  const scope = lowercaseField(model, ["scope", "source", "availability"])
  return scope === "catalog" || scope === "catalog_only" || scope === "reference"
}

export const isConfiguredUsableModel = (
  model: ModelSelectorDescriptor | null | undefined
): boolean => {
  if (!model || !getModelId(model)) return false
  if (isCatalogOnlyModel(model)) return false

  const deprecated = boolFromPaths(model, ["deprecated", "is_deprecated", "isDeprecated"])
  if (deprecated === true) return false

  const explicitUsabilityFields = [
    "is_configured",
    "isConfigured",
    "configured",
    "provider_is_configured",
    "providerIsConfigured",
    "provider_configured",
    "providerConfigured",
    "usable",
    "is_usable",
    "isUsable",
    "available",
    "is_available",
    "isAvailable",
    "enabled",
    "active"
  ]
  for (const key of explicitUsabilityFields) {
    const value = boolFromPaths(model, [key])
    if (value === false) return false
  }

  const apiKeyRequired = boolFromPaths(model, [
    "api_key_required",
    "apiKeyRequired",
    "requires_api_key",
    "requiresApiKey"
  ])
  const apiKeyConfigured = boolFromPaths(model, [
    "api_key_configured",
    "apiKeyConfigured",
    "has_api_key",
    "hasApiKey"
  ])
  if (apiKeyRequired === true && apiKeyConfigured === false) return false

  const status = lowercaseField(model, ["status", "state"])
  if (
    status &&
    [
      "catalog_only",
      "disabled",
      "inactive",
      "unavailable",
      "unconfigured",
      "not_configured",
      "not_available",
      "deprecated"
    ].includes(status)
  ) {
    return false
  }

  return true
}

export const filterModelsForScope = (
  models: ModelSelectorDescriptor[] | null | undefined,
  scope: ModelListScope
): ModelSelectorDescriptor[] => {
  const list = Array.isArray(models) ? models : []
  if (scope === "catalog") {
    return list.filter((model) => Boolean(getModelId(model)))
  }
  return list.filter(isConfiguredUsableModel)
}

export const modelMatchesSearch = (
  model: ModelSelectorDescriptor,
  query: string,
  getProviderLabel?: (provider: string) => string
): boolean => {
  const q = query.trim().toLowerCase()
  if (!q) return true
  const providerRaw = getModelProvider(model)
  const providerLabel = getProviderLabel?.(providerRaw)?.toLowerCase() || ""
  const fields = [
    providerRaw,
    providerLabel,
    nonEmptyString(model.nickname),
    getModelId(model),
    nonEmptyString(model.name)
  ]
  return fields.some((value) => String(value || "").toLowerCase().includes(q))
}

const modelLabel = (model: ModelSelectorDescriptor): string => {
  const provider = getModelProvider(model)
  return `${provider} ${model.nickname || getModelId(model)}`.toLowerCase()
}

const selectedMatchesModel = (
  model: ModelSelectorDescriptor,
  selectedModel: string | null | undefined,
  selectedProvider?: string | null
): boolean => {
  const selected = nonEmptyString(selectedModel)
  if (!selected) return false
  const key = getCanonicalModelKey(model)
  if (selected.toLowerCase() === key.toLowerCase()) return true
  if (selected !== getModelId(model) && selected !== nonEmptyString(model.model)) {
    return false
  }
  const provider = normalizeProviderName(selectedProvider)
  return provider === "other" || provider === getModelProvider(model)
}

export const sortModelsForSelector = (
  models: ModelSelectorDescriptor[] | null | undefined,
  options: {
    selectedModel?: string | null
    selectedProvider?: string | null
    favoriteKeys?: Set<string>
    usageByKey?: Record<string, ModelUsageStats>
    sortMode?: ModelSortMode
  } = {}
): ModelSelectorDescriptor[] => {
  const list = Array.isArray(models) ? models.slice() : []
  const favoriteKeys = options.favoriteKeys || new Set<string>()
  const usageByKey = options.usageByKey || {}
  const sortMode = options.sortMode || "provider"

  const compareUsage = (a: ModelSelectorDescriptor, b: ModelSelectorDescriptor) => {
    const aUsage = usageByKey[getCanonicalModelKey(a)]
    const bUsage = usageByKey[getCanonicalModelKey(b)]
    const aCount = Number(aUsage?.selectedCount || 0)
    const bCount = Number(bUsage?.selectedCount || 0)
    if (aCount !== bCount) return bCount - aCount
    const aRecent = Number(aUsage?.lastSelectedAt || 0)
    const bRecent = Number(bUsage?.lastSelectedAt || 0)
    if (aRecent !== bRecent) return bRecent - aRecent
    return modelLabel(a).localeCompare(modelLabel(b))
  }

  const compareBase = (a: ModelSelectorDescriptor, b: ModelSelectorDescriptor) => {
    const aSelected = selectedMatchesModel(a, options.selectedModel, options.selectedProvider)
    const bSelected = selectedMatchesModel(b, options.selectedModel, options.selectedProvider)
    if (aSelected !== bSelected) return aSelected ? -1 : 1

    if (sortMode === "az") {
      return modelLabel(a).localeCompare(modelLabel(b))
    }

    const usageComparison = compareUsage(a, b)
    const aHasUsage = Boolean(usageByKey[getCanonicalModelKey(a)]?.selectedCount)
    const bHasUsage = Boolean(usageByKey[getCanonicalModelKey(b)]?.selectedCount)
    if (aHasUsage || bHasUsage) {
      if (aHasUsage !== bHasUsage) return aHasUsage ? -1 : 1
      if (usageComparison !== 0) return usageComparison
    }

    const aFavorite = hasFavoriteModelKey(a, favoriteKeys)
    const bFavorite = hasFavoriteModelKey(b, favoriteKeys)
    if (aFavorite !== bFavorite) return aFavorite ? -1 : 1

    if (sortMode === "localFirst") {
      const aLocal = LOCAL_PROVIDERS.has(getModelProvider(a))
      const bLocal = LOCAL_PROVIDERS.has(getModelProvider(b))
      if (aLocal !== bLocal) return aLocal ? -1 : 1
    }

    return modelLabel(a).localeCompare(modelLabel(b))
  }

  return list.sort(compareBase)
}
