export const formatModelsLastRefreshedTime = (timestamp: number): string => {
  const date = new Date(timestamp)
  const hours = date.getHours().toString().padStart(2, "0")
  const minutes = date.getMinutes().toString().padStart(2, "0")
  return `${hours}:${minutes}`
}

export type ModelDisplayEntry = {
  id: string
  provider: string
  nickname?: string | null
  configured?: boolean
  usable?: boolean
  selected?: boolean
}

export type ModelDisplayOption = {
  value: string
  label: string
  model?: ModelDisplayEntry
}

export type ConfiguredFirstModelSections = {
  configuredFirst: ModelDisplayEntry[]
  fullCatalog: ModelDisplayEntry[]
}

export type ProviderReadinessSummary = {
  totalProviders: number
  configuredProviders: number
  usableProviders: number
  unavailableProviders: number
  selectedModelIds: string[]
  hasConfiguredUsableProvider: boolean
}

const getReadinessRank = (model: ModelDisplayEntry): number => {
  if (model.selected) return 0
  if (model.configured === true && model.usable === true) return 1
  if (model.configured === true) return 2
  if (model.usable === true) return 3
  return 4
}

const getModelLabel = (model: ModelDisplayEntry): string =>
  (model.nickname || model.id).trim()

export const compareConfiguredFirst = (
  a: ModelDisplayEntry,
  b: ModelDisplayEntry
): number => {
  const readiness = getReadinessRank(a) - getReadinessRank(b)
  if (readiness !== 0) return readiness

  const provider = a.provider.localeCompare(b.provider)
  if (provider !== 0) return provider

  return getModelLabel(a).localeCompare(getModelLabel(b))
}

export const sortModelsConfiguredFirst = (
  models: ModelDisplayEntry[]
): ModelDisplayEntry[] => models.slice().sort(compareConfiguredFirst)

export const buildConfiguredFirstModelOptions = (
  models: ModelDisplayEntry[],
  options: { autoLabel: string }
): ModelDisplayOption[] => [
  { value: "auto", label: options.autoLabel },
  ...sortModelsConfiguredFirst(models).map((model) => ({
    value: model.id,
    label: `${model.provider} - ${getModelLabel(model)}`,
    model
  }))
]

export const buildConfiguredFirstModelSections = (
  models: ModelDisplayEntry[]
): ConfiguredFirstModelSections => {
  const fullCatalog = sortModelsConfiguredFirst(models)
  return {
    configuredFirst: fullCatalog.filter(
      (model) =>
        model.selected === true ||
        model.configured === true ||
        model.usable === true
    ),
    fullCatalog
  }
}

export const summarizeProviderReadiness = (
  models: ModelDisplayEntry[]
): ProviderReadinessSummary => {
  const providers = new Map<
    string,
    { configured: boolean; usable: boolean; selectedModelIds: string[] }
  >()

  for (const model of models) {
    const provider = model.provider.trim()
    if (!provider) continue
    const current = providers.get(provider) ?? {
      configured: false,
      usable: false,
      selectedModelIds: []
    }
    current.configured = current.configured || model.configured === true
    current.usable = current.usable || model.usable === true
    if (model.selected) current.selectedModelIds.push(model.id)
    providers.set(provider, current)
  }

  const providerStates = Array.from(providers.values())
  const configuredProviders = providerStates.filter(
    (provider) => provider.configured
  ).length
  const usableProviders = providerStates.filter((provider) => provider.usable)
    .length

  return {
    totalProviders: providers.size,
    configuredProviders,
    usableProviders,
    unavailableProviders: providerStates.filter((provider) => !provider.usable)
      .length,
    selectedModelIds: providerStates.flatMap(
      (provider) => provider.selectedModelIds
    ),
    hasConfiguredUsableProvider: providerStates.some(
      (provider) => provider.configured && provider.usable
    )
  }
}
