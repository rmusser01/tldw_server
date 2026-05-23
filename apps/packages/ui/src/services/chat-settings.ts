import { createSafeStorage } from "@/utils/safe-storage"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatLinkedResearchPath } from "@/components/Option/Playground/research-run-status"
import { normalizeConversationContextIdList } from "@/services/conversation-context/conversationContextSettings"
import {
  CHAT_SETTINGS_SCHEMA_VERSION,
  ChatAssistantOverlay,
  ChatSettingsRecord,
  CharacterMemoryEntry,
  ConversationContextSettings,
  DeepResearchAttachment
} from "@/types/chat-session-settings"

const storage = createSafeStorage()
const MAX_CHAT_SETTINGS_BYTES = 200_000
const MAX_DEEP_RESEARCH_ATTACHMENT_CLAIMS = 5
const MAX_DEEP_RESEARCH_ATTACHMENT_UNRESOLVED_QUESTIONS = 5
const MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY = 3
const DEEP_RESEARCH_ATTACHMENT_ALLOWED_KEYS = new Set([
  "run_id",
  "query",
  "question",
  "outline",
  "key_claims",
  "unresolved_questions",
  "verification_summary",
  "source_trust_summary",
  "research_url",
  "attached_at",
  "updatedAt"
])
const ASSISTANT_OVERLAY_ALLOWED_KEYS = new Set([
  "kind",
  "id",
  "name",
  "avatar_url",
  "system_prompt_snapshot",
  "updatedAt"
])
const ASSISTANT_OVERLAY_ALLOWED_KINDS = new Set(["character", "persona"])
const MAX_ASSISTANT_OVERLAY_TEXT_CHARS = 20_000
const CHAT_SETTINGS_OPTIONAL_KEYS = [
  "autoSummaryEnabled",
  "autoSummaryThresholdMessages",
  "autoSummaryWindowMessages",
  "pinnedMessageIds",
  "greetingSelectionId",
  "greetingsVersion",
  "greetingsChecksum",
  "useCharacterDefault",
  "greetingEnabled",
  "greetingScope",
  "presetScope",
  "memoryScope",
  "directedCharacterId",
  "chatPresetOverrideId",
  "authorNote",
  "authorNotePosition",
  "characterMemoryById",
  "chatGenerationOverride",
  "conversationContext",
  "chat_dictionary_ids",
  "summary",
  "imageEventSyncMode",
  "assistantOverlay"
] as const

export const getChatSettingsStorageKey = (chatKey: string) =>
  `chatSettings:${chatKey}`

export const resolveChatSettingsKey = (params: {
  historyId: string | null
  serverChatId: string | null
}): string => {
  const { historyId, serverChatId } = params
  if (serverChatId) return `server:${serverChatId}`
  if (historyId) return `local:${historyId}`
  return "scratch"
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const hasOwn = (record: Record<string, unknown>, key: string): boolean =>
  Object.prototype.hasOwnProperty.call(record, key)

const asNonEmptyString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const asBoundedNonEmptyString = (
  value: unknown,
  maxLength = MAX_ASSISTANT_OVERLAY_TEXT_CHARS
): string | null => {
  const text = asNonEmptyString(value)
  if (!text || text.length > maxLength) return null
  return text
}

const asIsoString = (value: unknown): string | null => {
  const text = asNonEmptyString(value)
  if (!text) return null
  const parsed = Date.parse(text)
  return Number.isNaN(parsed) ? null : text
}

const asOptionalBoundedString = (
  value: unknown,
  maxLength = MAX_ASSISTANT_OVERLAY_TEXT_CHARS
): string | null | undefined => {
  if (value === undefined) return undefined
  if (value === null) return null
  if (typeof value !== "string") return undefined
  if (value.length > maxLength) return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const asNonNegativeInteger = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.trunc(value)
  }
  return null
}

const copyKnownChatSettings = (
  raw: Record<string, unknown>
): Partial<ChatSettingsRecord> => {
  const next: Partial<ChatSettingsRecord> = {}
  for (const key of CHAT_SETTINGS_OPTIONAL_KEYS) {
    if (Object.prototype.hasOwnProperty.call(raw, key)) {
      ;(next as Record<string, unknown>)[key] = raw[key]
    }
  }
  return next
}

const normalizeConversationContextMirrors = (
  settings: ChatSettingsRecord,
  raw: Record<string, unknown>
): void => {
  const rawContext = isRecord(raw.conversationContext)
    ? raw.conversationContext
    : null
  const context: ConversationContextSettings =
    isRecord(settings.conversationContext)
      ? { ...settings.conversationContext }
      : {}
  const hasNestedDictionaryIds =
    rawContext !== null && hasOwn(rawContext, "chat_dictionary_ids")
  const hasLegacyDictionaryIds = hasOwn(raw, "chat_dictionary_ids")

  if (rawContext && hasOwn(rawContext, "world_book_ids")) {
    context.world_book_ids = normalizeConversationContextIdList(
      rawContext.world_book_ids
    )
    settings.conversationContext = context
  }

  if (hasNestedDictionaryIds || hasLegacyDictionaryIds) {
    const source = hasNestedDictionaryIds
      ? rawContext?.chat_dictionary_ids
      : raw.chat_dictionary_ids
    const dictionaryIds = normalizeConversationContextIdList(source)
    context.chat_dictionary_ids = dictionaryIds
    settings.conversationContext = context
    settings.chat_dictionary_ids = dictionaryIds
  }
}

const sanitizeDeepResearchAttachment = (
  value: unknown
): DeepResearchAttachment | null => {
  if (!isRecord(value)) return null
  const keys = Object.keys(value)
  if (keys.some((key) => !DEEP_RESEARCH_ATTACHMENT_ALLOWED_KEYS.has(key))) {
    return null
  }

  const runId = asNonEmptyString(value.run_id)
  const query = asNonEmptyString(value.query)
  const question = asNonEmptyString(value.question)
  const attachedAt = asIsoString(value.attached_at)
  const updatedAt = asIsoString(value.updatedAt)

  if (!runId || !query || !question || !attachedAt || !updatedAt) {
    return null
  }

  const outline = Array.isArray(value.outline)
    ? value.outline
        .map((entry) =>
          isRecord(entry) ? asNonEmptyString(entry.title) : null
        )
        .filter((title): title is string => title !== null)
        .map((title) => ({ title }))
    : []

  const keyClaims = Array.isArray(value.key_claims)
    ? value.key_claims
        .map((entry) => (isRecord(entry) ? asNonEmptyString(entry.text) : null))
        .filter((text): text is string => text !== null)
        .slice(0, MAX_DEEP_RESEARCH_ATTACHMENT_CLAIMS)
        .map((text) => ({ text }))
    : []

  const unresolvedQuestions = Array.isArray(value.unresolved_questions)
    ? value.unresolved_questions
        .map((entry) => asNonEmptyString(entry))
        .filter((entry): entry is string => entry !== null)
        .slice(0, MAX_DEEP_RESEARCH_ATTACHMENT_UNRESOLVED_QUESTIONS)
    : []

  const unsupportedClaimCount = isRecord(value.verification_summary)
    ? asNonNegativeInteger(value.verification_summary.unsupported_claim_count)
    : null
  const highTrustCount = isRecord(value.source_trust_summary)
    ? asNonNegativeInteger(value.source_trust_summary.high_trust_count)
    : null

  return {
    run_id: runId,
    query,
    question,
    outline,
    key_claims: keyClaims,
    unresolved_questions: unresolvedQuestions,
    verification_summary:
      unsupportedClaimCount === null
        ? undefined
        : { unsupported_claim_count: unsupportedClaimCount },
    source_trust_summary:
      highTrustCount === null
        ? undefined
        : { high_trust_count: highTrustCount },
    research_url: buildChatLinkedResearchPath(runId),
    attached_at: attachedAt,
    updatedAt
  }
}

const sanitizeAssistantOverlay = (
  value: unknown
): ChatAssistantOverlay | null => {
  if (!isRecord(value)) return null
  const keys = Object.keys(value)
  if (keys.some((key) => !ASSISTANT_OVERLAY_ALLOWED_KEYS.has(key))) {
    return null
  }

  const kind =
    typeof value.kind === "string" ? value.kind.trim().toLowerCase() : null
  const id = asBoundedNonEmptyString(value.id)
  const name = asBoundedNonEmptyString(value.name)
  const updatedAt = asIsoString(value.updatedAt)

  if (
    !kind ||
    !ASSISTANT_OVERLAY_ALLOWED_KINDS.has(kind) ||
    !id ||
    !name ||
    !updatedAt
  ) {
    return null
  }

  const avatarUrl = asOptionalBoundedString(value.avatar_url)
  if (value.avatar_url !== undefined && avatarUrl === undefined) {
    return null
  }

  const systemPromptSnapshot = asOptionalBoundedString(
    value.system_prompt_snapshot
  )
  if (
    value.system_prompt_snapshot !== undefined &&
    systemPromptSnapshot === undefined
  ) {
    return null
  }

  return {
    kind: kind as ChatAssistantOverlay["kind"],
    id,
    name,
    avatar_url: avatarUrl,
    system_prompt_snapshot: systemPromptSnapshot,
    updatedAt
  }
}

const sanitizeDeepResearchAttachmentHistory = (
  value: unknown,
  excludedRunIds?: Iterable<string | null | undefined>
): DeepResearchAttachment[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const excluded = new Set(
    Array.from(excludedRunIds ?? []).filter(
      (runId): runId is string => typeof runId === "string" && runId.length > 0
    )
  )

  const byRunId = new Map<string, DeepResearchAttachment>()
  for (const rawEntry of value) {
    const entry = sanitizeDeepResearchAttachment(rawEntry)
    if (!entry) continue
    if (excluded.has(entry.run_id)) continue
    const existing = byRunId.get(entry.run_id)
    if (!existing || toEpoch(entry.updatedAt) > toEpoch(existing.updatedAt)) {
      byRunId.set(entry.run_id, entry)
    }
  }

  return Array.from(byRunId.values())
    .sort((left, right) => toEpoch(right.updatedAt) - toEpoch(left.updatedAt))
    .slice(0, MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY)
}

const enforceChatSettingsSize = (
  settings: ChatSettingsRecord
): ChatSettingsRecord | null => {
  try {
    const encoded = new TextEncoder().encode(JSON.stringify(settings))
    if (encoded.byteLength > MAX_CHAT_SETTINGS_BYTES) {
      return null
    }
    return settings
  } catch {
    return null
  }
}

const coerceSettings = (raw: any): ChatSettingsRecord | null => {
  if (!isRecord(raw)) return null
  const schemaVersion =
    typeof raw.schemaVersion === "number"
      ? raw.schemaVersion
      : CHAT_SETTINGS_SCHEMA_VERSION
  const updatedAt =
    typeof raw.updatedAt === "string"
      ? raw.updatedAt
      : new Date().toISOString()
  const next: ChatSettingsRecord = {
    ...copyKnownChatSettings(raw),
    schemaVersion,
    updatedAt
  }
  normalizeConversationContextMirrors(next, raw)
  if (Object.prototype.hasOwnProperty.call(raw, "assistantOverlay")) {
    if (raw.assistantOverlay === null) {
      next.assistantOverlay = null
    } else {
      const sanitizedOverlay = sanitizeAssistantOverlay(raw.assistantOverlay)
      if (sanitizedOverlay) {
        next.assistantOverlay = sanitizedOverlay
      } else {
        delete next.assistantOverlay
      }
    }
  }
  const hasAttachment = Object.prototype.hasOwnProperty.call(
    raw,
    "deepResearchAttachment"
  )
  const rawAttachment = hasAttachment ? raw.deepResearchAttachment : undefined
  const sanitizedAttachment =
    hasAttachment && rawAttachment !== null
      ? sanitizeDeepResearchAttachment(rawAttachment)
      : undefined

  if (hasAttachment) {
    if (rawAttachment === null) {
      next.deepResearchAttachment = null
    } else if (sanitizedAttachment) {
      next.deepResearchAttachment = sanitizedAttachment
    }
  }
  const hasPinnedAttachment = Object.prototype.hasOwnProperty.call(
    raw,
    "deepResearchPinnedAttachment"
  )
  const rawPinnedAttachment = hasPinnedAttachment
    ? raw.deepResearchPinnedAttachment
    : undefined
  const sanitizedPinnedAttachment =
    hasPinnedAttachment && rawPinnedAttachment !== null
      ? sanitizeDeepResearchAttachment(rawPinnedAttachment)
      : undefined

  if (hasPinnedAttachment) {
    if (rawPinnedAttachment === null) {
      next.deepResearchPinnedAttachment = null
    } else if (sanitizedPinnedAttachment) {
      next.deepResearchPinnedAttachment = sanitizedPinnedAttachment
    }
  }
  if (Object.prototype.hasOwnProperty.call(raw, "deepResearchAttachmentHistory")) {
    next.deepResearchAttachmentHistory = sanitizeDeepResearchAttachmentHistory(
      raw.deepResearchAttachmentHistory,
      [
        sanitizedAttachment?.run_id ?? next.deepResearchAttachment?.run_id ?? null,
        sanitizedPinnedAttachment?.run_id ??
          next.deepResearchPinnedAttachment?.run_id ??
          null
      ]
    )
  }
  return next
}

export const normalizeChatSettingsRecord = (
  raw: any
): ChatSettingsRecord | null => coerceSettings(raw)

const toEpoch = (value: string | undefined): number => {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? 0 : parsed
}

const normalizeComparableChatSettingsValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeComparableChatSettingsValue(entry))
  }
  if (!value || typeof value !== "object") {
    return value
  }

  const normalizedEntries = Object.entries(value as Record<string, unknown>)
    .filter(([key, entryValue]) => key !== "updatedAt" && entryValue !== undefined)
    .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
    .map(([key, entryValue]) => [
      key,
      normalizeComparableChatSettingsValue(entryValue)
    ])

  return Object.fromEntries(normalizedEntries)
}

export const areEquivalentChatSettings = (
  left: ChatSettingsRecord | null,
  right: ChatSettingsRecord | null
): boolean => {
  if (!left || !right) return left === right
  return (
    JSON.stringify(normalizeComparableChatSettingsValue(left)) ===
    JSON.stringify(normalizeComparableChatSettingsValue(right))
  )
}

const mergeEntry = (
  left: CharacterMemoryEntry | undefined,
  right: CharacterMemoryEntry | undefined
): CharacterMemoryEntry | undefined => {
  if (!left) return right
  if (!right) return left
  const leftTime = toEpoch(left.updatedAt)
  const rightTime = toEpoch(right.updatedAt)
  return rightTime >= leftTime ? right : left
}

export const mergeChatSettings = (
  local: ChatSettingsRecord | null,
  remote: ChatSettingsRecord | null
): ChatSettingsRecord | null => {
  if (!local) return remote
  if (!remote) return local

  const localTime = toEpoch(local.updatedAt)
  const remoteTime = toEpoch(remote.updatedAt)
  const base = localTime >= remoteTime ? local : remote
  const other = base === local ? remote : local

  const merged: ChatSettingsRecord = {
    ...other,
    ...base,
    updatedAt: base.updatedAt
  }

  const baseMemory = base.characterMemoryById
  const otherMemory = other.characterMemoryById
  if (baseMemory || otherMemory) {
    const mergedMap: Record<string, CharacterMemoryEntry> = {}
    const keys = new Set([
      ...Object.keys(baseMemory || {}),
      ...Object.keys(otherMemory || {})
    ])
    for (const key of keys) {
      mergedMap[key] = mergeEntry(otherMemory?.[key], baseMemory?.[key]) || {
        note: "",
        updatedAt: base.updatedAt
      }
    }
    merged.characterMemoryById = mergedMap
  }

  if (
    local.deepResearchAttachment &&
    remote.deepResearchAttachment &&
    local.deepResearchAttachment !== null &&
    remote.deepResearchAttachment !== null
  ) {
    const localAttachmentTime = toEpoch(local.deepResearchAttachment.updatedAt)
    const remoteAttachmentTime = toEpoch(remote.deepResearchAttachment.updatedAt)
    merged.deepResearchAttachment =
      remoteAttachmentTime >= localAttachmentTime
        ? remote.deepResearchAttachment
        : local.deepResearchAttachment
  }

  if (
    local.deepResearchPinnedAttachment &&
    remote.deepResearchPinnedAttachment &&
    local.deepResearchPinnedAttachment !== null &&
    remote.deepResearchPinnedAttachment !== null
  ) {
    const localPinnedTime = toEpoch(local.deepResearchPinnedAttachment.updatedAt)
    const remotePinnedTime = toEpoch(remote.deepResearchPinnedAttachment.updatedAt)
    merged.deepResearchPinnedAttachment =
      remotePinnedTime >= localPinnedTime
        ? remote.deepResearchPinnedAttachment
        : local.deepResearchPinnedAttachment
  } else if (remote.deepResearchPinnedAttachment !== undefined) {
    merged.deepResearchPinnedAttachment = remote.deepResearchPinnedAttachment
  } else if (local.deepResearchPinnedAttachment !== undefined) {
    merged.deepResearchPinnedAttachment = local.deepResearchPinnedAttachment
  }

  const hasAttachmentHistory =
    Array.isArray(local.deepResearchAttachmentHistory) ||
    Array.isArray(remote.deepResearchAttachmentHistory)
  if (hasAttachmentHistory) {
    const mergedHistory = sanitizeDeepResearchAttachmentHistory(
      [
        ...(local.deepResearchAttachmentHistory || []),
        ...(remote.deepResearchAttachmentHistory || [])
      ],
      [
        merged.deepResearchAttachment?.run_id ?? null,
        merged.deepResearchPinnedAttachment?.run_id ?? null
      ]
    )
    if (mergedHistory !== undefined) {
      merged.deepResearchAttachmentHistory = mergedHistory
    }
  }

  return merged
}

const buildPatchedChatSettingsInput = (
  existing: ChatSettingsRecord | null,
  patch: Partial<ChatSettingsRecord>
): Record<string, unknown> => {
  const next: Record<string, unknown> = {
    ...(existing || {
      schemaVersion: CHAT_SETTINGS_SCHEMA_VERSION,
      updatedAt: new Date().toISOString()
    }),
    ...patch,
    schemaVersion: CHAT_SETTINGS_SCHEMA_VERSION,
    updatedAt: new Date().toISOString()
  }

  if (Object.prototype.hasOwnProperty.call(patch, "assistantOverlay")) {
    if (patch.assistantOverlay === null) {
      next.assistantOverlay = null
    } else if (isRecord(patch.assistantOverlay) && isRecord(existing?.assistantOverlay)) {
      next.assistantOverlay = {
        ...existing?.assistantOverlay,
        ...patch.assistantOverlay
      }
    } else {
      next.assistantOverlay = patch.assistantOverlay as unknown
    }
  }

  return next
}

export const getChatSettingsForKey = async (
  chatKey: string
): Promise<ChatSettingsRecord | null> => {
  try {
    const key = getChatSettingsStorageKey(chatKey)
    const stored = await storage.get<ChatSettingsRecord | undefined>(key)
    return coerceSettings(stored)
  } catch (error) {
    console.error("Failed to load chat settings", error)
    return null
  }
}

export const saveChatSettingsForKey = async (
  chatKey: string,
  settings: ChatSettingsRecord
): Promise<boolean> => {
  try {
    const key = getChatSettingsStorageKey(chatKey)
    const payload = normalizeChatSettingsRecord({
      ...settings,
      schemaVersion: CHAT_SETTINGS_SCHEMA_VERSION,
      updatedAt: settings.updatedAt || new Date().toISOString()
    })
    if (!payload) {
      return false
    }
    const boundedPayload = enforceChatSettingsSize(payload)
    if (!boundedPayload) {
      return false
    }
    await storage.set(key, boundedPayload)
    return true
  } catch (error) {
    console.error("Failed to save chat settings", error)
    return false
  }
}

export const getChatSettingsForChat = async (params: {
  historyId: string | null
  serverChatId: string | null
}): Promise<ChatSettingsRecord | null> => {
  const chatKey = resolveChatSettingsKey(params)
  return await getChatSettingsForKey(chatKey)
}

export const saveChatSettingsForChat = async (params: {
  historyId: string | null
  serverChatId: string | null
  settings: ChatSettingsRecord
}): Promise<boolean> => {
  const chatKey = resolveChatSettingsKey(params)
  return await saveChatSettingsForKey(chatKey, params.settings)
}

export const syncChatSettingsForServerChat = async (params: {
  historyId: string | null
  serverChatId: string | null
  allowScratchFallback?: boolean
}): Promise<ChatSettingsRecord | null> => {
  const { historyId, serverChatId, allowScratchFallback = false } = params
  if (!serverChatId) return null

  const serverKey = resolveChatSettingsKey({ historyId: null, serverChatId })
  const localKey = resolveChatSettingsKey({ historyId, serverChatId: null })
  const scratchKey = resolveChatSettingsKey({ historyId: null, serverChatId: null })

  const localForServer = await getChatSettingsForKey(serverKey)
  const localFromHistory = historyId ? await getChatSettingsForKey(localKey) : null
  const localFromScratch =
    allowScratchFallback && !localForServer && !localFromHistory
      ? await getChatSettingsForKey(scratchKey)
      : null
  const usedScratchFallback = Boolean(localFromScratch)
  const localSettings = localForServer || localFromHistory || localFromScratch

  await tldwClient.initialize().catch(() => null)

  let remoteSettings: ChatSettingsRecord | null = null
  try {
    const res = await tldwClient.getChatSettings(serverChatId)
    remoteSettings = coerceSettings(res.settings)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error || "")
    if (!message.includes("404")) {
      console.warn("Failed to fetch server chat settings", error)
    }
  }

  if (!remoteSettings && localSettings) {
    try {
      const res = await tldwClient.updateChatSettings(serverChatId, localSettings)
      const synced = coerceSettings(res.settings) || localSettings
      await saveChatSettingsForKey(serverKey, synced)
      if (usedScratchFallback) {
        await storage.remove(getChatSettingsStorageKey(scratchKey))
      }
      return synced
    } catch (error) {
      console.warn("Failed to push local chat settings", error)
      return localSettings
    }
  }

  if (remoteSettings && !localSettings) {
    await saveChatSettingsForKey(serverKey, remoteSettings)
    return remoteSettings
  }

  const merged = mergeChatSettings(localSettings, remoteSettings)
  if (!merged) return null

  if (remoteSettings && areEquivalentChatSettings(merged, remoteSettings)) {
    await saveChatSettingsForKey(serverKey, merged)
    return merged
  }

  const mergedTime = toEpoch(merged.updatedAt)
  const remoteTime = toEpoch(remoteSettings?.updatedAt)

  if (mergedTime >= remoteTime) {
    try {
      await tldwClient.updateChatSettings(serverChatId, merged)
    } catch (error) {
      console.warn("Failed to reconcile chat settings to server", error)
    }
  }

  await saveChatSettingsForKey(serverKey, merged)
  if (usedScratchFallback) {
    await storage.remove(getChatSettingsStorageKey(scratchKey))
  }
  return merged
}

export const applyChatSettingsPatch = async (params: {
  historyId: string | null
  serverChatId: string | null
  patch: Partial<ChatSettingsRecord>
}): Promise<ChatSettingsRecord | null> => {
  const { historyId, serverChatId, patch } = params
  const chatKey = resolveChatSettingsKey({ historyId, serverChatId })
  const existing = await getChatSettingsForKey(chatKey)
  const patchedInput = buildPatchedChatSettingsInput(existing, patch)
  const normalized = normalizeChatSettingsRecord(patchedInput)
  const assistantOverlayPatched = Object.prototype.hasOwnProperty.call(
    patch,
    "assistantOverlay"
  )
  const next =
    normalized ||
    ({
      schemaVersion: CHAT_SETTINGS_SCHEMA_VERSION,
      updatedAt: new Date().toISOString()
    } as ChatSettingsRecord)

  if (
    normalized &&
    assistantOverlayPatched &&
    patch.assistantOverlay !== null &&
    existing?.assistantOverlay !== undefined &&
    normalized.assistantOverlay === undefined
  ) {
    next.assistantOverlay = existing.assistantOverlay
  }

  const saved = await saveChatSettingsForKey(chatKey, next)
  if (!saved) {
    return null
  }

  if (!serverChatId) {
    return next
  }

  await tldwClient.initialize().catch(() => null)
  try {
    const res = await tldwClient.updateChatSettings(serverChatId, next)
    const synced = normalizeChatSettingsRecord(res.settings) || next
    const serverKey = resolveChatSettingsKey({ historyId: null, serverChatId })
    await saveChatSettingsForKey(serverKey, synced)
    return synced
  } catch (error) {
    console.warn("Failed to update server chat settings", error)
    return next
  }
}
