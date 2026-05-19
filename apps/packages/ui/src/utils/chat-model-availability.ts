import {
  isAutoModelId,
  parseProviderQualifiedModelSelection
} from "./resolve-api-provider"

type ModelDescriptor = {
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
  | "no-selected-model"
  | "no-models-available"
  | "selected-model-unavailable"
  | "send-disabled"

export type CharacterChatReadinessAction =
  | "open-server-settings"
  | "choose-character"
  | "open-model-settings"
  | "retry"

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
  const provider = String(
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
    .trim()
    .toLowerCase()

  return provider && provider !== "unknown" ? provider : null
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

function readBooleanFlagFromRecords(
  records: Record<string, unknown>[],
  keys: readonly string[]
): boolean | null {
  for (const record of records) {
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue
      const flag = toBooleanFlag(record[key])
      if (flag != null) return flag
    }
  }

  return null
}

function isUsableChatModelDescriptor(
  model: ModelDescriptor,
  options: BuildAvailableChatModelIdsOptions = {}
): boolean {
  const records = getChatModelDescriptorRecords(model)
  const catalogOnly = readBooleanFlagFromRecords(
    records,
    CATALOG_ONLY_FLAG_KEYS
  )
  if (catalogOnly === true) return false

  let hasConfiguredFlag = catalogOnly != null
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

    const provider = normalized.slice(0, separatorIndex).trim().toLowerCase()
    const modelId = normalizeChatModelId(
      normalized.slice(separatorIndex + 1).trim()
    )
    if (!provider || provider === "http" || provider === "https" || !modelId) {
      return null
    }
    return `${provider}:${modelId}`
  }

  const modelId = normalizeChatModelId(parsed.modelId)
  return modelId ? `${parsed.provider}:${modelId}` : null
}

function normalizeBaseChatModelId(value: string | null | undefined): string {
  const normalized = normalizeChatModelId(value)
  const separatorIndex = normalized.indexOf(":")
  if (separatorIndex <= 0 || separatorIndex === normalized.length - 1) {
    return normalized
  }

  const provider = normalized.slice(0, separatorIndex).trim().toLowerCase()
  if (!provider || provider === "http" || provider === "https") {
    return normalized
  }

  return normalizeChatModelId(normalized.slice(separatorIndex + 1).trim())
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

  const normalizedSelectedModel = normalizeChatModelId(selectedModel)
  if (!normalizedSelectedModel) {
    return blockedCharacterChatReadiness(
      "chat-model",
      "no-selected-model",
      "open-model-settings"
    )
  }

  if (Array.isArray(availableModels)) {
    const availableModelIds = buildAvailableChatModelIds(availableModels, {
      requireConfiguredFlags: true
    })
    if (availableModelIds.size === 0) {
      return blockedCharacterChatReadiness(
        "chat-model",
        "no-models-available",
        "open-model-settings"
      )
    }

    const unavailableModel = findUnavailableChatModel(
      [normalizedSelectedModel],
      availableModelIds
    )
    if (unavailableModel) {
      return blockedCharacterChatReadiness(
        "chat-model",
        "selected-model-unavailable",
        "open-model-settings"
      )
    }
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
              "Saved characters are still available. Configure a chat model, then return here to continue with {{characterName}}.",
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
