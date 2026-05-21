import {
  isAutoModelId,
  parseProviderQualifiedModelSelection
} from "./resolve-api-provider"

export type ModelDescriptor = {
  id?: unknown
  model?: unknown
  name?: unknown
  provider?: unknown
  provider_key?: unknown
  providerKey?: unknown
  api_provider?: unknown
  apiProvider?: unknown
  is_configured?: unknown
  isConfigured?: unknown
  configured?: unknown
  provider_is_configured?: unknown
  providerIsConfigured?: unknown
  provider_configured?: unknown
  providerConfigured?: unknown
  catalog_only?: unknown
  catalogOnly?: unknown
  is_catalog_only?: unknown
  isCatalogOnly?: unknown
  details?: {
    provider?: unknown
    provider_key?: unknown
    is_configured?: unknown
    isConfigured?: unknown
    configured?: unknown
    provider_is_configured?: unknown
    providerIsConfigured?: unknown
    provider_configured?: unknown
    providerConfigured?: unknown
    catalog_only?: unknown
    catalogOnly?: unknown
    is_catalog_only?: unknown
    isCatalogOnly?: unknown
  }
  metadata?: {
    provider?: unknown
    is_configured?: unknown
    isConfigured?: unknown
    configured?: unknown
    provider_is_configured?: unknown
    providerIsConfigured?: unknown
    provider_configured?: unknown
    providerConfigured?: unknown
    catalog_only?: unknown
    catalogOnly?: unknown
    is_catalog_only?: unknown
    isCatalogOnly?: unknown
  }
}

export const AUTO_CHAT_MODEL_ID = "auto"
export const CHARACTER_CHAT_MODEL_SETTINGS_PATH = "/settings/model"

export type CharacterChatReadinessMissingRequirement =
  | "server-connection"
  | "selected-character"
  | "chat-model"
  | "chat-send"

export type CharacterChatReadinessReason =
  | "server-unavailable"
  | "missing-character"
  | "models-loading"
  | "no-selected-model"
  | "no-models-available"
  | "selected-model-missing"
  | "provider-unconfigured"
  | "model-unavailable"
  | "selected-model-unavailable"
  | "send-disabled"

export type CharacterChatReadinessAction =
  | "open-server-settings"
  | "choose-character"
  | "open-model-settings"
  | "retry"

export type ChatModelUsabilityStatus =
  | "loading"
  | "no_server"
  | "no_selection"
  | "no_models"
  | "selected_missing"
  | "provider_unconfigured"
  | "model_unavailable"
  | "degraded"
  | "ready"

export type ChatModelUsability = {
  status: ChatModelUsabilityStatus
  canSend: boolean
  selectedModelId: string | null
  providerQualifiedModelId: string | null
  matchedModelId: string | null
  matchedProvider: string | null
  recommendedAction: CharacterChatReadinessAction | null
  detailReason: string | null
}

export type ChatModelUsabilityInput = {
  isServerConnected?: boolean
  selectedModel?: string | null
  availableModels?: ModelDescriptor[] | null
  modelsLoading?: boolean
  allowDegradedSend?: boolean
  serverDegraded?: boolean
}

export type CharacterChatReadiness =
  | {
      status: "ready"
      canStart: true
      missingRequirement: null
      recommendedAction: null
      reason: null
    }
  | {
      status: "blocked"
      canStart: false
      missingRequirement: CharacterChatReadinessMissingRequirement
      recommendedAction: CharacterChatReadinessAction
      reason: CharacterChatReadinessReason
    }

type CharacterChatReadinessInput = {
  isServerConnected?: boolean
  selectedCharacter?: unknown
  selectedModel?: string | null
  availableModels?: ModelDescriptor[] | null
  modelsLoading?: boolean
  allowDegradedSend?: boolean
  serverDegraded?: boolean
  isSendBlocked?: boolean
}

type BuildAvailableChatModelIdsOptions = {
  requireConfiguredFlags?: boolean
}

type TranslationFn = (key: string, fallbackOrOptions?: any) => any

type CharacterChatReadinessCopyContext = {
  characterName?: string | null
}

type CharacterChatReadinessCopy = {
  title: string
  description: string
  actionLabel: string
}

const translateReadinessCopy = (
  t: TranslationFn,
  key: string,
  fallbackOrOptions: string | { defaultValue?: string; [key: string]: unknown }
): string => {
  const value = t(key, fallbackOrOptions)
  if (typeof value === "string") return value

  const fallback =
    typeof fallbackOrOptions === "string"
      ? fallbackOrOptions
      : fallbackOrOptions.defaultValue
  return String(fallback ?? key)
}

export function normalizeChatModelId(value: string | null | undefined): string {
  const trimmed = String(value ?? "").trim()
  return trimmed.replace(/^tldw:/i, "")
}

const PROVIDER_ALIASES: Record<string, string> = {
  customopenai: "custom-openai-api",
  custom_openai_api: "custom-openai-api",
  custom_openai_api2: "custom-openai-api-2",
  customopenai2: "custom-openai-api-2",
  "custom-openai-api2": "custom-openai-api-2",
  custom_openai_api_2: "custom-openai-api-2",
  local: "local-llm",
  local_llm: "local-llm",
  localllm: "local-llm",
  "llama-cpp": "llama.cpp",
  llama_cpp: "llama.cpp",
  llamacpp: "llama.cpp"
}

function normalizeProviderValue(value: unknown): string | null {
  const provider = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "")
  if (!provider || provider === "unknown") return null
  return PROVIDER_ALIASES[provider] ?? provider
}

const KNOWN_CHAT_PROVIDER_KEYS = new Set<string>([
  "openai",
  "anthropic",
  "cohere",
  "groq",
  "qwen",
  "openrouter",
  "deepseek",
  "mistral",
  "google",
  "gemini",
  "huggingface",
  "moonshot",
  "zai",
  "llama.cpp",
  "kobold",
  "ooba",
  "tabbyapi",
  "vllm",
  "local-llm",
  "ollama",
  "aphrodite",
  "mlx",
  "custom-openai-api",
  "custom-openai-api-2",
  "custom",
  "together",
  "xai",
  "siliconflow",
  "volcengine",
  "tencentcloud",
  "alibabacloud",
  "fireworks",
  "novita",
  "chutes",
  "bedrock"
])

function normalizeKnownChatProviderKey(value: unknown): string | null {
  const provider = normalizeProviderValue(value)
  if (!provider || !KNOWN_CHAT_PROVIDER_KEYS.has(provider)) return null
  return provider
}

function normalizeAvailableModelId(model: ModelDescriptor): string {
  const modelValue = String(model?.model ?? "").trim()
  const idValue = String(model?.id ?? "").trim()
  const nameValue = String(model?.name ?? "").trim()

  if (modelValue.toLowerCase().startsWith("tldw:") && idValue) {
    return normalizeChatModelId(idValue)
  }

  return normalizeChatModelId(modelValue || idValue || nameValue)
}

function normalizeProviderKey(model: ModelDescriptor): string | null {
  return normalizeProviderValue(
    model?.provider ??
      model?.provider_key ??
      model?.providerKey ??
      model?.api_provider ??
      model?.apiProvider ??
      model?.details?.provider ??
      model?.details?.provider_key ??
      model?.metadata?.provider ??
      ""
  )
}

const toBooleanFlag = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (normalized === "true") return true
    if (normalized === "false") return false
  }
  return null
}

const CATALOG_ONLY_FLAG_KEYS = [
  "catalog_only",
  "catalogOnly",
  "is_catalog_only",
  "isCatalogOnly"
] as const

const CONFIGURED_FLAG_KEYS = [
  "is_configured",
  "isConfigured",
  "configured",
  "provider_is_configured",
  "providerIsConfigured",
  "provider_configured",
  "providerConfigured"
] as const

function getChatModelDescriptorRecords(
  model: ModelDescriptor
): Record<string, unknown>[] {
  return [model, model.details, model.metadata].filter(
    (record): record is Record<string, unknown> =>
      Boolean(record && typeof record === "object")
  )
}

function readCatalogOnlyFlagFromRecords(
  records: Record<string, unknown>[]
): boolean | null {
  let hasCatalogFlag = false
  for (const record of records) {
    for (const key of CATALOG_ONLY_FLAG_KEYS) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue
      const flag = toBooleanFlag(record[key])
      if (flag === true) return true
      if (flag === false) hasCatalogFlag = true
    }
  }

  return hasCatalogFlag ? false : null
}

function readConfiguredFlagFromRecords(
  records: Record<string, unknown>[]
): boolean | null {
  let hasConfiguredFlag = false
  for (const record of records) {
    for (const key of CONFIGURED_FLAG_KEYS) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue
      const flag = toBooleanFlag(record[key])
      if (flag == null) continue
      hasConfiguredFlag = true
      if (flag === false) return false
    }
  }

  return hasConfiguredFlag ? true : null
}

function isUsableChatModelDescriptor(
  model: ModelDescriptor,
  options: BuildAvailableChatModelIdsOptions = {}
): boolean {
  const records = getChatModelDescriptorRecords(model)
  const catalogOnly = readCatalogOnlyFlagFromRecords(records)
  if (catalogOnly === true) return false

  let hasConfiguredFlag = false
  for (const record of records) {
    for (const key of CONFIGURED_FLAG_KEYS) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue
      const flag = toBooleanFlag(record[key])
      if (flag == null) continue
      hasConfiguredFlag = true
      if (flag === false) return false
    }
  }

  if (options.requireConfiguredFlags && !hasConfiguredFlag) {
    return false
  }

  return true
}

function normalizeProviderQualifiedChatModelId(
  value: string | null | undefined
): string | null {
  const parsed = parseProviderQualifiedModelSelection(value)
  if (!parsed.isProviderQualified || !parsed.provider) {
    const normalized = normalizeChatModelId(value)
    const separatorIndex = normalized.indexOf(":")
    if (separatorIndex <= 0 || separatorIndex === normalized.length - 1) {
      return null
    }

    const provider = normalizeKnownChatProviderKey(
      normalized.slice(0, separatorIndex)
    )
    const modelId = normalizeChatModelId(
      normalized.slice(separatorIndex + 1).trim()
    )
    if (!provider || !modelId) {
      return null
    }
    return `${provider}:${modelId}`
  }

  const modelId = normalizeChatModelId(parsed.modelId)
  return modelId ? `${parsed.provider}:${modelId}` : null
}

function normalizeBaseChatModelId(value: string | null | undefined): string {
  const normalized = normalizeChatModelId(value)
  const providerQualified = normalizeProviderQualifiedChatModelId(normalized)
  const separatorIndex = providerQualified?.indexOf(":") ?? -1
  if (
    !providerQualified ||
    separatorIndex <= 0 ||
    separatorIndex === providerQualified.length - 1
  ) {
    return normalized
  }

  return normalizeChatModelId(providerQualified.slice(separatorIndex + 1).trim())
}

function providerFromQualifiedChatModelId(
  value: string | null | undefined
): string | null {
  const qualified = normalizeProviderQualifiedChatModelId(value)
  const separatorIndex = qualified?.indexOf(":") ?? -1
  if (!qualified || separatorIndex <= 0) return null
  return qualified.slice(0, separatorIndex)
}

type ChatModelDescriptorMatch = {
  descriptor: ModelDescriptor
  modelId: string
  baseModelId: string
  provider: string | null
  providerQualifiedModelId: string | null
}

function describeChatModelDescriptor(
  descriptor: ModelDescriptor
): ChatModelDescriptorMatch | null {
  const modelId = normalizeAvailableModelId(descriptor)
  if (!modelId) return null

  const provider =
    normalizeProviderKey(descriptor) ??
    providerFromQualifiedChatModelId(modelId)
  const baseModelId = normalizeBaseChatModelId(modelId)
  const providerQualifiedModelId =
    provider && baseModelId ? `${provider}:${baseModelId}` : null

  return {
    descriptor,
    modelId,
    baseModelId,
    provider,
    providerQualifiedModelId
  }
}

function selectedModelCandidateIds(
  selectedModel: string | null | undefined
): Set<string> {
  const normalized = normalizeChatModelId(selectedModel)
  const providerQualified = normalizeProviderQualifiedChatModelId(selectedModel)
  const baseModelId = normalizeBaseChatModelId(selectedModel)

  return new Set(
    [normalized, providerQualified, baseModelId].filter(Boolean) as string[]
  )
}

function findMatchingChatModelDescriptor(
  selectedModel: string,
  availableModels: ModelDescriptor[]
): ChatModelDescriptorMatch | null {
  const descriptors = availableModels
    .map(describeChatModelDescriptor)
    .filter((descriptor): descriptor is ChatModelDescriptorMatch =>
      Boolean(descriptor)
    )
  const selectedProviderQualified =
    normalizeProviderQualifiedChatModelId(selectedModel)
  const selectedProvider = providerFromQualifiedChatModelId(selectedModel)
  const selectedBaseModelId = normalizeBaseChatModelId(selectedModel)
  const selectedCandidates = selectedModelCandidateIds(selectedModel)

  const exactModelIdMatches = descriptors.filter(
    (descriptor) => descriptor.modelId === selectedModel
  )
  if (exactModelIdMatches.length > 0) {
    return (
      exactModelIdMatches.find((descriptor) =>
        isUsableChatModelDescriptor(descriptor.descriptor)
      ) ??
      exactModelIdMatches[0] ??
      null
    )
  }

  if (selectedProvider && selectedProviderQualified) {
    const exactProviderMatches = descriptors.filter(
      (descriptor) =>
        descriptor.provider === selectedProvider &&
        descriptor.baseModelId === selectedBaseModelId
    )
    if (exactProviderMatches.length > 0) {
      return (
        exactProviderMatches.find((descriptor) =>
          isUsableChatModelDescriptor(descriptor.descriptor)
        ) ??
        exactProviderMatches[0] ??
        null
      )
    }

    const providerSpecificConflict = descriptors.some(
      (descriptor) =>
        descriptor.baseModelId === selectedBaseModelId &&
        Boolean(descriptor.provider)
    )
    if (providerSpecificConflict) return null

    return (
      descriptors.find(
        (descriptor) =>
          descriptor.baseModelId === selectedBaseModelId && !descriptor.provider
      ) ?? null
    )
  }

  const matchingDescriptors = descriptors.filter((descriptor) =>
      [descriptor.modelId, descriptor.baseModelId, descriptor.providerQualifiedModelId]
        .filter(Boolean)
        .some((candidate) => selectedCandidates.has(candidate as string))
  )
  return (
    matchingDescriptors.find((descriptor) =>
      isUsableChatModelDescriptor(descriptor.descriptor)
    ) ??
    matchingDescriptors[0] ??
    null
  )
}

const buildChatModelUsabilityResult = (
  status: ChatModelUsabilityStatus,
  options: Partial<Omit<ChatModelUsability, "status">> = {}
): ChatModelUsability => ({
  status,
  canSend: status === "ready" || status === "degraded",
  selectedModelId: null,
  providerQualifiedModelId: null,
  matchedModelId: null,
  matchedProvider: null,
  recommendedAction: null,
  detailReason: null,
  ...options
})

export function buildChatModelUsability({
  isServerConnected = true,
  selectedModel,
  availableModels,
  modelsLoading = false,
  allowDegradedSend = false,
  serverDegraded = false
}: ChatModelUsabilityInput): ChatModelUsability {
  const normalizedSelectedModel = normalizeChatModelId(selectedModel)
  const selectedModelId = normalizedSelectedModel || null
  const providerQualifiedModelId =
    normalizeProviderQualifiedChatModelId(selectedModel) ?? null

  if (!isServerConnected) {
    return buildChatModelUsabilityResult("no_server", {
      canSend: false,
      selectedModelId,
      providerQualifiedModelId,
      recommendedAction: "open-server-settings"
    })
  }

  if (!selectedModelId) {
    return buildChatModelUsabilityResult("no_selection", {
      canSend: false,
      recommendedAction: "open-model-settings"
    })
  }

  if (modelsLoading || !Array.isArray(availableModels)) {
    return buildChatModelUsabilityResult("loading", {
      canSend: false,
      selectedModelId,
      providerQualifiedModelId,
      recommendedAction: "retry"
    })
  }

  const callableModels = availableModels.filter((model) =>
    isUsableChatModelDescriptor(model)
  )

  if (
    isAutoModelId(selectedModelId) ||
    isAutoModelId(normalizeBaseChatModelId(selectedModelId))
  ) {
    if (callableModels.length === 0) {
      return buildChatModelUsabilityResult("no_models", {
        canSend: false,
        selectedModelId,
        providerQualifiedModelId,
        recommendedAction: "open-model-settings"
      })
    }

    if (serverDegraded) {
      return buildChatModelUsabilityResult("degraded", {
        canSend: allowDegradedSend,
        selectedModelId,
        providerQualifiedModelId,
        recommendedAction: allowDegradedSend ? null : "retry",
        detailReason: "server-degraded"
      })
    }

    return buildChatModelUsabilityResult("ready", {
      selectedModelId,
      providerQualifiedModelId
    })
  }

  const matchedDescriptor = findMatchingChatModelDescriptor(
    selectedModelId,
    availableModels
  )

  if (!matchedDescriptor) {
    return buildChatModelUsabilityResult(
      callableModels.length > 0 ? "selected_missing" : "no_models",
      {
        canSend: false,
        selectedModelId,
        providerQualifiedModelId,
        recommendedAction: "open-model-settings"
      }
    )
  }

  const records = getChatModelDescriptorRecords(matchedDescriptor.descriptor)
  const configured = readConfiguredFlagFromRecords(records)
  const catalogOnly = readCatalogOnlyFlagFromRecords(records)
  const matchedFields = {
    selectedModelId,
    providerQualifiedModelId,
    matchedModelId: matchedDescriptor.baseModelId || matchedDescriptor.modelId,
    matchedProvider: matchedDescriptor.provider
  }

  if (configured === false) {
    return buildChatModelUsabilityResult("provider_unconfigured", {
      ...matchedFields,
      canSend: false,
      recommendedAction: "open-model-settings",
      detailReason: "provider-unconfigured"
    })
  }

  if (
    catalogOnly === true ||
    !isUsableChatModelDescriptor(matchedDescriptor.descriptor)
  ) {
    return buildChatModelUsabilityResult("model_unavailable", {
      ...matchedFields,
      canSend: false,
      recommendedAction: "open-model-settings",
      detailReason: catalogOnly === true ? "catalog-only" : "model-unavailable"
    })
  }

  if (serverDegraded) {
    return allowDegradedSend
      ? buildChatModelUsabilityResult("degraded", {
          ...matchedFields,
          canSend: true,
          detailReason: "server-degraded"
        })
      : buildChatModelUsabilityResult("degraded", {
          ...matchedFields,
          canSend: false,
          recommendedAction: "retry",
          detailReason: "server-degraded"
        })
  }

  return buildChatModelUsabilityResult("ready", matchedFields)
}

export function buildAvailableChatModelIds(
  models: ModelDescriptor[] | null | undefined,
  options: BuildAvailableChatModelIdsOptions = {}
): Set<string> {
  const ids = new Set<string>()
  for (const model of models || []) {
    if (!isUsableChatModelDescriptor(model, options)) {
      continue
    }

    const modelId = normalizeAvailableModelId(model)
    if (modelId) {
      ids.add(modelId)
      const providerQualifiedModel = parseProviderQualifiedModelSelection(modelId)
      const baseModelId = providerQualifiedModel.isProviderQualified
        ? normalizeChatModelId(providerQualifiedModel.modelId)
        : modelId
      if (baseModelId && baseModelId !== modelId) {
        ids.add(baseModelId)
      }

      const provider = normalizeProviderKey(model) ?? providerQualifiedModel.provider
      if (provider) {
        ids.add(`${provider}:${baseModelId || modelId}`)
      }
    }
  }
  return ids
}

export function findUnavailableChatModel(
  selectedModelIds: string[],
  availableModelIds: Set<string>
): string | null {
  if (availableModelIds.size === 0) {
    return null
  }

  for (const selectedModelId of selectedModelIds) {
    const normalized = normalizeChatModelId(selectedModelId)
    const providerQualified = normalizeProviderQualifiedChatModelId(
      selectedModelId
    )
    const baseModelId = normalizeBaseChatModelId(selectedModelId)
    if (isAutoModelId(normalized) || isAutoModelId(baseModelId)) {
      continue
    }

    const candidates = new Set(
      [normalized, providerQualified, baseModelId].filter(Boolean) as string[]
    )
    const hasAvailableCandidate = [...candidates].some((candidate) =>
      availableModelIds.has(candidate)
    )

    if (normalized && !hasAvailableCandidate) {
      return normalized
    }
  }

  return null
}

const blockedCharacterChatReadiness = (
  missingRequirement: CharacterChatReadinessMissingRequirement,
  reason: CharacterChatReadinessReason,
  recommendedAction: CharacterChatReadinessAction
): CharacterChatReadiness => ({
  status: "blocked",
  canStart: false,
  missingRequirement,
  recommendedAction,
  reason
})

export function buildCharacterChatReadiness({
  isServerConnected = true,
  selectedCharacter,
  selectedModel,
  availableModels,
  modelsLoading = false,
  allowDegradedSend = false,
  serverDegraded = false,
  isSendBlocked = false
}: CharacterChatReadinessInput): CharacterChatReadiness {
  if (!isServerConnected) {
    return blockedCharacterChatReadiness(
      "server-connection",
      "server-unavailable",
      "open-server-settings"
    )
  }

  if (!selectedCharacter) {
    return blockedCharacterChatReadiness(
      "selected-character",
      "missing-character",
      "choose-character"
    )
  }

  const modelUsability = buildChatModelUsability({
    isServerConnected: true,
    selectedModel,
    availableModels,
    modelsLoading,
    allowDegradedSend,
    serverDegraded
  })

  switch (modelUsability.status) {
    case "loading":
      return blockedCharacterChatReadiness(
        "chat-model",
        "models-loading",
        "retry"
      )
    case "no_server":
      return blockedCharacterChatReadiness(
        "server-connection",
        "server-unavailable",
        "open-server-settings"
      )
    case "no_selection":
      return blockedCharacterChatReadiness(
        "chat-model",
        "no-selected-model",
        "open-model-settings"
      )
    case "no_models":
      return blockedCharacterChatReadiness(
        "chat-model",
        "no-models-available",
        "open-model-settings"
      )
    case "selected_missing":
      return blockedCharacterChatReadiness(
        "chat-model",
        "selected-model-missing",
        "open-model-settings"
      )
    case "provider_unconfigured":
      return blockedCharacterChatReadiness(
        "chat-model",
        "provider-unconfigured",
        "open-model-settings"
      )
    case "model_unavailable":
      return blockedCharacterChatReadiness(
        "chat-model",
        "model-unavailable",
        "open-model-settings"
      )
    case "degraded":
      if (!modelUsability.canSend) {
        return blockedCharacterChatReadiness(
          "chat-send",
          "send-disabled",
          "retry"
        )
      }
      break
    case "ready":
      break
  }

  if (isSendBlocked) {
    return blockedCharacterChatReadiness(
      "chat-send",
      "send-disabled",
      "retry"
    )
  }

  return {
    status: "ready",
    canStart: true,
    missingRequirement: null,
    recommendedAction: null,
    reason: null
  }
}

export function getCharacterChatReadinessCopy(
  readiness: CharacterChatReadiness,
  t: TranslationFn,
  context: CharacterChatReadinessCopyContext = {}
): CharacterChatReadinessCopy {
  const characterName = context.characterName?.trim()

  if (readiness.missingRequirement === "server-connection") {
    return {
      title: translateReadinessCopy(
        t,
        "characterChatReadiness.server.title",
        "Connect to tldw_server before starting character chat"
      ),
      description: translateReadinessCopy(
        t,
        "characterChatReadiness.server.description",
        "Character chat needs a reachable tldw_server before it can load chats, characters, and model settings."
      ),
      actionLabel: translateReadinessCopy(
        t,
        "characterChatReadiness.server.action",
        "Open server settings"
      )
    }
  }

  if (readiness.missingRequirement === "selected-character") {
    return {
      title: translateReadinessCopy(
        t,
        "characterChatReadiness.character.title",
        "Choose a character to start character chat"
      ),
      description: translateReadinessCopy(
        t,
        "characterChatReadiness.character.description",
        "Character chat starts with the character, then checks model readiness before the first message."
      ),
      actionLabel: translateReadinessCopy(
        t,
        "characterChatReadiness.character.action",
        "Choose character"
      )
    }
  }

  if (readiness.missingRequirement === "chat-model") {
    const inContextDescription = characterName
      ? {
          defaultValue:
            "Your character selection and draft are kept. Configure a chat model, then return here to continue with {{characterName}}.",
          characterName
        }
      : "Your character selection and draft are kept. Configure a chat model, then return to start the character chat."
    const modelActionLabel = translateReadinessCopy(
      t,
      "characterChatReadiness.model.action",
      "Open model settings"
    )

    if (readiness.reason === "models-loading") {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.loadingTitle",
          "Checking chat model readiness"
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.loadingDescription",
          characterName
            ? {
                defaultValue:
                  "Your character selection and draft are kept while tldw_server checks which chat models can be used with {{characterName}}.",
                characterName
              }
            : "Your character selection and draft are kept while tldw_server checks which chat models are available."
        ),
        actionLabel: translateReadinessCopy(
          t,
          "characterChatReadiness.model.retryAction",
          "Try again"
        )
      }
    }

    if (readiness.reason === "provider-unconfigured") {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.providerUnconfiguredTitle",
          characterName
            ? {
                defaultValue:
                  "Configure the selected model provider before chatting as {{characterName}}",
                characterName
              }
            : "Configure the selected model provider before starting character chat"
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.providerUnconfiguredDescription",
          characterName
            ? {
                defaultValue:
                  "Your character selection and draft are kept. Configure the provider for the selected model, then return here to continue with {{characterName}}.",
                characterName
              }
            : "Your character selection and draft are kept. Configure the provider for the selected model, then return to start the character chat."
        ),
        actionLabel: modelActionLabel
      }
    }

    if (
      readiness.reason === "selected-model-missing" ||
      readiness.reason === "selected-model-unavailable"
    ) {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.selectedMissingTitle",
          characterName
            ? {
                defaultValue:
                  "Choose an available chat model before chatting as {{characterName}}",
                characterName
              }
            : "Choose an available chat model before starting character chat"
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.selectedMissingDescription",
          characterName
            ? {
                defaultValue:
                  "Your character selection and draft are kept. Choose a model from the available chat catalog, then return here to continue with {{characterName}}.",
                characterName
              }
            : "Your character selection and draft are kept. Choose a model from the available chat catalog, then return to start the character chat."
        ),
        actionLabel: modelActionLabel
      }
    }

    if (readiness.reason === "model-unavailable") {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.unavailableTitle",
          "The selected chat model is not callable right now"
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.unavailableDescription",
          characterName
            ? {
                defaultValue:
                  "Your character selection and draft are kept. Choose or configure a callable chat model, then return here to continue with {{characterName}}.",
                characterName
              }
            : "Your character selection and draft are kept. Choose or configure a callable chat model, then return to start the character chat."
        ),
        actionLabel: modelActionLabel
      }
    }

    if (readiness.reason === "no-models-available") {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.noneAvailableTitle",
          characterName
            ? {
                defaultValue:
                  "Configure a chat model before chatting as {{characterName}}",
                characterName
              }
            : "Configure a chat model before starting character chat"
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.noneAvailableDescription",
          inContextDescription
        ),
        actionLabel: modelActionLabel
      }
    }

    if (characterName) {
      return {
        title: translateReadinessCopy(
          t,
          "characterChatReadiness.model.titleWithCharacter",
          {
            defaultValue:
              "Choose a chat model before chatting as {{characterName}}",
            characterName
          }
        ),
        description: translateReadinessCopy(
          t,
          "characterChatReadiness.model.descriptionWithCharacter",
          {
            defaultValue:
              "Your character selection and draft are kept. Configure a chat model, then return here to continue with {{characterName}}.",
            characterName
          }
        ),
        actionLabel: translateReadinessCopy(
          t,
          "characterChatReadiness.model.action",
          "Open model settings"
        )
      }
    }

    return {
      title: translateReadinessCopy(
        t,
        "characterChatReadiness.model.title",
        "Choose a chat model before starting character chat"
      ),
      description: translateReadinessCopy(
        t,
        "characterChatReadiness.model.description",
        "Saved characters are still available. Configure a chat model, then return to start the character chat."
      ),
      actionLabel: translateReadinessCopy(
        t,
        "characterChatReadiness.model.action",
        "Open model settings"
      )
    }
  }

  if (readiness.missingRequirement === "chat-send") {
    return {
      title: translateReadinessCopy(
        t,
        "characterChatReadiness.send.title",
        "Character chat is preparing"
      ),
      description: translateReadinessCopy(
        t,
        "characterChatReadiness.send.description",
        "Wait for the current message or setup step to finish before sending."
      ),
      actionLabel: translateReadinessCopy(
        t,
        "characterChatReadiness.send.action",
        "Try again"
      )
    }
  }

  return {
    title: translateReadinessCopy(
      t,
      "characterChatReadiness.ready.title",
      "Character chat is ready"
    ),
    description: translateReadinessCopy(
      t,
      "characterChatReadiness.ready.description",
      "The selected character and chat model are ready for the first message."
    ),
    actionLabel: translateReadinessCopy(
      t,
      "characterChatReadiness.ready.action",
      "Start chat"
    )
  }
}
