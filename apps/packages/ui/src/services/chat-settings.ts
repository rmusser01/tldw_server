import { db } from "@/db/dexie/schema"
import Dexie from "dexie"
import { requirePersistentStorage } from "@/utils/persistent-storage"
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
import { toChatScopeParams, type ChatScope } from "@/types/chat-scope"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { chatDraftAddressKey, type ChatDraftAddress } from "@/utils/chat-draft-address"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { createLocalRegistryBucket, type LocalRegistryRecord } from "./settings/local-bucket"

const storage = createSafeStorage()
const localChatSettingsStorage = createSafeStorage({ area: "local" })
export const chatSettingsStorageForKey = (chatKey: string) =>
  chatKey.startsWith("local:") || isQualifiedSettingsKey(chatKey) ? localChatSettingsStorage : storage
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
  "generationOverrides",
  "turnTakingMode",
  "participantCharacterIds",
  "conversationContext",
  "chat_dictionary_ids",
  "summary",
  "imageEventSyncMode",
  "assistantOverlay"
] as const

export const getChatSettingsStorageKey = (chatKey: string) =>
  `chatSettings:${chatKey}`

export type ChatSettingsTarget = {
  historyId: string | null
  serverChatId: string | null
  requestScope?: ServicePromptRequestScope
  scope?: ChatScope
  draftAddress?: ChatDraftAddress | null
  signal?: AbortSignal
  validate_lease?: () => boolean
}
const isQualifiedSettingsKey = (key: string) => key.startsWith("server:owned:") || key.startsWith("draft:owned:")
const qualifiedSettingsAddress = (target: ChatSettingsTarget) => target.requestScope
  ? JSON.stringify([serverChatMirrorOwnerKey({ requestScope: target.requestScope }), toChatScopeParams(target.scope)])
  : null

export const resolveChatSettingsKey = (target: ChatSettingsTarget): string => {
  const owner = qualifiedSettingsAddress(target)
  if (target.serverChatId) return owner ? `server:owned:${owner}:${encodeURIComponent(target.serverChatId)}` : `unqualified:${target.serverChatId}`
  if (target.historyId) return `local:${target.historyId}`
  if (owner && target.draftAddress?.kind === "fresh") {
    return `draft:owned:${qualifiedSettingsAddress({ ...target, scope: target.draftAddress.scope })}:${chatDraftAddressKey(target.draftAddress)}`
  }
  return "scratch"
}

// These are private device projections. They neither grant native capability nor replay in the background.
const settingsCache = createLocalRegistryBucket<ChatSettingsRecord | null>({ prefix: "chatSettings:" })
type SettingsSave = {
  version: 1
  destination: string
  operation_id: string
  status: "pending" | "acknowledged" | "settled" | "rejected"
  intent: Record<string, unknown>
  patch: Record<string, unknown>
  serverSettings?: ChatSettingsRecord | null
}
const settingsSaves = createLocalRegistryBucket<SettingsSave>({ prefix: "tldw:chatSettingsSaves:v1:" })
const assertSettingsCurrent = (target: ChatSettingsTarget) => {
  if (target.signal?.aborted || target.validate_lease?.() === false) throw createServicePromptScopeChangedError()
  if ((target.serverChatId || !target.historyId) && (!target.requestScope || !target.validate_lease ||
      (!target.serverChatId && target.draftAddress?.kind !== "fresh"))) throw new Error("chat_settings_owner_unavailable")
}
const settingsRequestOptions = (target: ChatSettingsTarget) => ({
  requestScope: target.requestScope,
  ...(target.scope ? { scope: target.scope } : {}),
  ...(target.signal ? { signal: target.signal } : {})
})
const readQualifiedSettings = async (chatKey: string) => {
  const record = await settingsCache.get(chatKey, { failClosed: true })
  if (record && record.value !== null && (!isRecord(record.value) ||
      !Number.isInteger(record.value.schemaVersion) || record.value.schemaVersion < 1 ||
      typeof record.value.updatedAt !== "string" || !coerceSettings(record.value))) throw new Error("chat_settings_cache_invalid")
  return record
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
  const context: ConversationContextSettings = isRecord(
    settings.conversationContext
  )
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
    typeof raw.updatedAt === "string" ? raw.updatedAt : new Date().toISOString()
  const next: ChatSettingsRecord = {
    ...copyKnownChatSettings(raw),
    schemaVersion,
    updatedAt
  }
  if (hasOwn(raw, "generationOverrides") && raw.generationOverrides !== null && !isRecord(raw.generationOverrides))
    delete next.generationOverrides
  if (hasOwn(raw, "turnTakingMode") && raw.turnTakingMode !== "single" && raw.turnTakingMode !== "round_robin")
    delete next.turnTakingMode
  if (hasOwn(raw, "participantCharacterIds") && (!Array.isArray(raw.participantCharacterIds) ||
    !raw.participantCharacterIds.every((id: unknown) =>
      typeof id === "string" ? Boolean(id.trim()) : typeof id === "number" && Number.isSafeInteger(id) && id > 0)))
    delete next.participantCharacterIds
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
  if (
    Object.prototype.hasOwnProperty.call(raw, "deepResearchAttachmentHistory")
  ) {
    next.deepResearchAttachmentHistory = sanitizeDeepResearchAttachmentHistory(
      raw.deepResearchAttachmentHistory,
      [
        sanitizedAttachment?.run_id ??
          next.deepResearchAttachment?.run_id ??
          null,
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

const requireServerSettingsResponse = (
  serverChatId: string,
  response: unknown
): Record<string, unknown> => {
  if (!isRecord(response) || response.conversation_id !== serverChatId || !isRecord(response.settings)) {
    throw new Error("chat_settings_ack_invalid")
  }
  return response.settings
}

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
    .filter(
      ([key, entryValue]) => key !== "updatedAt" && entryValue !== undefined
    )
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
    const remoteAttachmentTime = toEpoch(
      remote.deepResearchAttachment.updatedAt
    )
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
    const localPinnedTime = toEpoch(
      local.deepResearchPinnedAttachment.updatedAt
    )
    const remotePinnedTime = toEpoch(
      remote.deepResearchPinnedAttachment.updatedAt
    )
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

  for (const key of ["conversationContext", "chatGenerationOverride", "generationOverrides"] as const) {
    const base = key === "chatGenerationOverride" && !isRecord(existing?.chatGenerationOverride)
      ? existing?.generationOverrides : existing?.[key]
    if (isRecord(patch[key]) && isRecord(base)) next[key] = { ...base, ...patch[key] }
  }

  if (Object.prototype.hasOwnProperty.call(patch, "assistantOverlay")) {
    if (patch.assistantOverlay === null) {
      next.assistantOverlay = null
    } else if (
      isRecord(patch.assistantOverlay) &&
      isRecord(existing?.assistantOverlay)
    ) {
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

type LocalSettingsGuard =
  import("@/db/dexie/types").HistoryInfo["local_settings_guard"]
const settingsGuardKey = (guard: LocalSettingsGuard) =>
  JSON.stringify(guard ?? null)
const validSettingsGuard = (guard: LocalSettingsGuard) =>
  guard === undefined ||
  (isRecord(guard) &&
    typeof guard.revision === "string" &&
    guard.revision.length > 0 &&
    (guard.initialized === undefined ||
      typeof guard.initialized === "boolean") &&
    Array.isArray(guard.pending) &&
    guard.pending.every(
      (token) => typeof token === "string" && token.length > 0
    ) &&
    new Set(guard.pending).size === guard.pending.length)

/** Strict fork read: the ordinary null-on-error reader cannot certify required state absent. */
const requirePlainForkSettings = (raw: unknown) => {
  if (raw === undefined) return
  if (
    !isRecord(raw) ||
    raw.schemaVersion !== CHAT_SETTINGS_SCHEMA_VERSION ||
    typeof raw.updatedAt !== "string"
  )
    throw new Error("fork_chat_settings_unavailable")
  for (const [key, value] of Object.entries(raw)) {
    if (["schemaVersion", "updatedAt", "pinnedMessageIds"].includes(key))
      continue
    if (
      key === "summary" &&
      ((value === null || value === undefined) ||
        (isRecord(value) &&
          Object.keys(value).every((k) =>
            ["enabled", "content", "sourceRange", "updatedAt"].includes(k)
          ) &&
          (value.enabled === undefined || value.enabled === false)))
    )
      continue
    const emptyArray =
      [
        "chat_dictionary_ids",
        "participantCharacterIds",
        "deepResearchAttachmentHistory"
      ].includes(key) &&
      Array.isArray(value) &&
      value.length === 0
    const emptyObject =
      ["characterMemoryById", "conversationContext"].includes(key) &&
      isRecord(value) &&
      Object.keys(value).length === 0
    const inactive = ["autoSummaryEnabled"].includes(key) && value === false
    const unset =
      [
        "assistantOverlay",
        "chatGenerationOverride",
        "generationOverrides",
        "directedCharacterId",
        "chatPresetOverrideId",
        "greetingSelectionId",
        "greetingsVersion",
        "greetingsChecksum",
        "authorNotePosition",
        "deepResearchAttachment",
        "deepResearchPinnedAttachment",
        "autoSummaryThresholdMessages",
        "autoSummaryWindowMessages"
      ].includes(key) && (value === null || value === undefined)
    if (
      emptyArray ||
      emptyObject ||
      inactive ||
      unset ||
      (key === "authorNote" && value === "") ||
      (key === "imageEventSyncMode" && value === "off")
    )
      continue
    throw new Error("unsupported_fork_chat_settings")
  }
}

/** The external read happens outside Dexie; the supplied check runs inside the owning transaction. */
export const withPlainLocalForkSettings = async <T>(
  historyId: string,
  operation: (
    validate: (history: import("@/db/dexie/types").HistoryInfo) => void
  ) => Promise<T>
): Promise<T> => {
  try {
    await ensureLocalChatSettingsBaseline(historyId)
  } catch {
    throw new Error("fork_chat_settings_unavailable")
  }
  for (let attempt = 0; attempt < 3; attempt++) {
    const history = await db.chatHistories.get(historyId)
    if (!history) throw new Error("missing_conversation")
    const guard = history.local_settings_guard
    if (!validSettingsGuard(guard) || guard?.pending.length)
      throw new Error("fork_chat_settings_unavailable")
    let raw: unknown
    try {
      requirePersistentStorage(localChatSettingsStorage)
      raw = await localChatSettingsStorage.get(
        getChatSettingsStorageKey(`local:${historyId}`)
      )
    } catch {
      throw new Error("fork_chat_settings_unavailable")
    }
    if (raw !== null) requirePlainForkSettings(raw)
    if (raw === undefined) throw new Error("fork_chat_settings_unavailable")
    try {
      return await operation((current) => {
        if (
          !validSettingsGuard(current.local_settings_guard) ||
          current.local_settings_guard?.pending.length
        )
          throw new Error("fork_chat_settings_unavailable")
        if (
          settingsGuardKey(current.local_settings_guard) !==
          settingsGuardKey(guard)
        )
          throw new Error("fork_chat_settings_changed")
      })
    } catch (error) {
      if (
        !(error instanceof Error) ||
        error.message !== "fork_chat_settings_changed"
      )
        throw error
      // A summary-only edit is not retained-context drift. Reread policy under the new guard.
    }
  }
  throw new Error("fork_chat_settings_changed")
}

/** Publish the pending guard before external I/O. Dexie.waitFor keeps the owning lock alive. */
const localSettingsWrite = async (
  historyId: string,
  operation: (initialized: boolean) => Promise<void>
) => {
  requirePersistentStorage(localChatSettingsStorage)
  const token = crypto.randomUUID()
  await db.transaction("rw", [db.chatHistories], async () => {
    const history = await db.chatHistories.get(historyId)
    if (!history) throw new Error("missing_conversation")
    if (!validSettingsGuard(history.local_settings_guard))
      throw new Error("fork_chat_settings_unavailable")
    await db.chatHistories.update(historyId, {
      local_settings_guard: {
        ...history.local_settings_guard,
        revision: token,
        pending: [...(history.local_settings_guard?.pending ?? []), token]
      }
    })
  })
  // A failed/uncertain external effect is never retried here. Its committed pending token remains.
  await db.transaction("rw", [db.chatHistories], async () => {
    const history = await db.chatHistories.get(historyId)
    const guard = history?.local_settings_guard
    if (!guard || !validSettingsGuard(guard) || !guard.pending.includes(token))
      throw new Error("fork_chat_settings_unavailable")
    requirePersistentStorage(localChatSettingsStorage)
    await Dexie.waitFor(operation(guard.initialized === true))
    requirePersistentStorage(localChatSettingsStorage)
    await db.chatHistories.update(historyId, {
      local_settings_guard: {
        initialized: true,
        revision: crypto.randomUUID(),
        pending: guard.pending.filter((value) => value !== token)
      }
    })
  })
}
const requireSettingsRecord = (raw: unknown) => {
  if (
    raw !== undefined &&
    raw !== null &&
    (!isRecord(raw) ||
      raw.schemaVersion !== CHAT_SETTINGS_SCHEMA_VERSION ||
      typeof raw.updatedAt !== "string")
  )
    throw new Error("fork_chat_settings_unavailable")
}
/** One local payload authority. null is the persisted, initialized empty baseline. */
const ensureLocalChatSettingsBaseline = async (historyId: string) => {
  requirePersistentStorage(localChatSettingsStorage)
  const history = await db.chatHistories.get(historyId)
  if (!history) throw new Error("missing_conversation")
  if (!validSettingsGuard(history.local_settings_guard))
    throw new Error("fork_chat_settings_unavailable")
  if (history.local_settings_guard?.initialized) return
  await localSettingsWrite(historyId, async (initialized) => {
    if (initialized) return
    const key = getChatSettingsStorageKey(`local:${historyId}`)
    // WebUI historically used local already. Always preserve an existing local baseline first.
    const local = await localChatSettingsStorage.get(key)
    requireSettingsRecord(local)
    if (local !== undefined) return
    requirePersistentStorage(storage)
    const legacy = await storage.get(key)
    requireSettingsRecord(legacy)
    await localChatSettingsStorage.set(key, legacy ?? null)
  })
}
const writeLocalChatSettings = async (
  historyId: string,
  key: string,
  payload: ChatSettingsRecord
) => {
  // A complete ordinary edit can initialize the local baseline even after failed migration.
  await localSettingsWrite(historyId, async () => {
    await localChatSettingsStorage.set(key, payload)
  })
}

const readChatSettingsForKey = async (chatKey: string): Promise<ChatSettingsRecord | null> => {
  if (isQualifiedSettingsKey(chatKey)) return coerceSettings((await readQualifiedSettings(chatKey))?.value)
  const key = getChatSettingsStorageKey(chatKey)
  if (chatKey.startsWith("local:")) await ensureLocalChatSettingsBaseline(chatKey.slice(6))
  const settingsStorage = chatSettingsStorageForKey(chatKey)
  requirePersistentStorage(settingsStorage)
  const stored = await settingsStorage.get<ChatSettingsRecord | undefined>(key)
  requireSettingsRecord(stored)
  return coerceSettings(stored)
}

export const getChatSettingsForKey = async (chatKey: string): Promise<ChatSettingsRecord | null> => {
  try {
    return await readChatSettingsForKey(chatKey)
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
    if (isQualifiedSettingsKey(chatKey)) throw new Error("chat_settings_owner_unavailable")
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
    if (chatKey.startsWith("local:"))
      await writeLocalChatSettings(chatKey.slice(6), key, boundedPayload)
    else {
      requirePersistentStorage(storage)
      await storage.set(key, boundedPayload)
    }
    return true
  } catch (error) {
    console.error("Failed to save chat settings", error)
    return false
  }
}

export const getChatSettingsForChat = async (params: ChatSettingsTarget): Promise<ChatSettingsRecord | null> => {
  return await getChatSettingsForKey(resolveChatSettingsKey(params))
}

export const saveChatSettingsForChat = async (params: ChatSettingsTarget & { settings: ChatSettingsRecord }): Promise<boolean> => {
  return await saveChatSettingsForKey(resolveChatSettingsKey(params), params.settings)
}

const saveAcknowledgedServerSettings = async (
  target: ChatSettingsTarget,
  settings: ChatSettingsRecord | null,
  expected: LocalRegistryRecord<ChatSettingsRecord | null> | null
) => {
  assertSettingsCurrent(target)
  const saved = await settingsCache.set(resolveChatSettingsKey(target), settings, Date.now(), expected, () => {
    assertSettingsCurrent(target)
    return true
  })
  assertSettingsCurrent(target)
  if (!saved) throw Object.assign(new Error("chat_settings_device_save_failed"), { serverChatId: target.serverChatId, serverSettings: settings })
  return settings
}

export const syncChatSettingsForServerChat = async (target: ChatSettingsTarget & {
  allowScratchFallback?: boolean
  sourceDraftAddress?: ChatDraftAddress | null
}): Promise<ChatSettingsRecord | null> => {
  if (!target.serverChatId) return null
  assertSettingsCurrent(target)
  const chatKey = resolveChatSettingsKey(target)
  const before = await readQualifiedSettings(chatKey)
  assertSettingsCurrent(target)
  await tldwClient.initialize()
  assertSettingsCurrent(target)
  const raw = requireServerSettingsResponse(target.serverChatId, await tldwClient.getChatSettings(target.serverChatId, settingsRequestOptions(target)))
  assertSettingsCurrent(target)
  const remote = Object.keys(raw).length ? coerceSettings(raw) : null
  if (Object.keys(raw).length && !remote) throw new Error("chat_settings_ack_invalid")
  await saveAcknowledgedServerSettings(target, remote, before)
  // Only an explicit first binding may copy its captured fresh draft. Bare-ID caches and local mirrors never become native authority.
  if (!remote && target.allowScratchFallback && target.sourceDraftAddress?.kind === "fresh") {
    const draftTarget = { ...target, historyId: null, serverChatId: null, draftAddress: target.sourceDraftAddress }
    const source = await readQualifiedSettings(resolveChatSettingsKey(draftTarget))
    assertSettingsCurrent(target)
    if (source?.value) return await applyChatSettingsPatch({ ...target, patch: source.value })
  }
  return remote
}

const settingsSubset = (value: Record<string, unknown>, shape: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.keys(shape).filter(key => (hasOwn(value, key) || (shape[key] === null && ["deepResearchAttachment", "deepResearchPinnedAttachment"].includes(key))) && !["schemaVersion", "updatedAt"].includes(key)).map(key => [key,
    isRecord(shape[key]) && isRecord(value[key]) ? settingsSubset(value[key], shape[key]) : hasOwn(value, key) ? value[key] : null
  ]))
const sameSettingsIntent = (a: Record<string, unknown>, b: Record<string, unknown>) =>
  JSON.stringify(normalizeComparableChatSettingsValue(a)) === JSON.stringify(normalizeComparableChatSettingsValue(b))
const requireSettingsSave = (record: LocalRegistryRecord<SettingsSave> | null, destination: string) => {
  if (!record) return null
  const value = record.value
  if (!isRecord(value) || value.version !== 1 || value.destination !== destination ||
      typeof value.operation_id !== "string" || !value.operation_id ||
      !["pending", "acknowledged", "settled", "rejected"].includes(value.status) || !isRecord(value.intent) || !isRecord(value.patch) ||
      (value.status !== "pending" && !(value.status === "rejected" && value.serverSettings === null) && !coerceSettings(value.serverSettings))) throw new Error("chat_settings_pending_invalid")
  return record
}

export type ChatSettingsRecovery = {
  destination: string
  operationId: string
  status: "pending" | "acknowledged"
  patch: Partial<ChatSettingsRecord>
}

/** Read the existing private receipt; never replay or infer a rollback. */
export const getChatSettingsRecovery = async (target: ChatSettingsTarget): Promise<ChatSettingsRecovery | null> => {
  const captured = { ...target, scope: target.scope ? { ...target.scope } : undefined }
  assertSettingsCurrent(captured)
  if (!captured.serverChatId) return null
  const destination = resolveChatSettingsKey(captured)
  const record = requireSettingsSave(await settingsSaves.get(destination, { failClosed: true }), destination)
  assertSettingsCurrent(captured)
  if (!record || (record.value.status !== "pending" && record.value.status !== "acknowledged")) return null
  return { destination, operationId: record.value.operation_id, status: record.value.status,
    patch: structuredClone(record.value.intent) as Partial<ChatSettingsRecord> }
}

const applyCapturedChatSettingsPatch = async (target: ChatSettingsTarget & {
  patch: Partial<ChatSettingsRecord>
  recoveryOperationId?: string
}): Promise<ChatSettingsRecord | null> => {
  assertSettingsCurrent(target)
  const chatKey = resolveChatSettingsKey(target)
  const cacheBefore = isQualifiedSettingsKey(chatKey) ? await readQualifiedSettings(chatKey) : null
  const local = cacheBefore?.value ?? (isQualifiedSettingsKey(chatKey) ? null : await readChatSettingsForKey(chatKey))
  assertSettingsCurrent(target)
  let existing = local
  let previous: LocalRegistryRecord<SettingsSave> | null = null
  if (target.serverChatId) {
    previous = requireSettingsSave(await settingsSaves.get(chatKey, { failClosed: true }), chatKey)
    assertSettingsCurrent(target)
    if (target.recoveryOperationId && (previous?.value.operation_id !== target.recoveryOperationId ||
      !["pending", "acknowledged"].includes(previous.value.status))) throw new Error("chat_settings_recovery_changed")
    await tldwClient.initialize()
    assertSettingsCurrent(target)
    const raw = requireServerSettingsResponse(target.serverChatId, await tldwClient.getChatSettings(target.serverChatId, settingsRequestOptions(target)))
    assertSettingsCurrent(target)
    existing = Object.keys(raw).length ? coerceSettings(raw) : null
    if (Object.keys(raw).length && !existing) throw new Error("chat_settings_ack_invalid")
  }
  const normalized = normalizeChatSettingsRecord(buildPatchedChatSettingsInput(existing, target.patch))
  if (!normalized) throw new Error("chat_settings_patch_invalid")
  if (hasOwn(target.patch, "assistantOverlay") && target.patch.assistantOverlay !== null &&
      existing?.assistantOverlay !== undefined && normalized.assistantOverlay === undefined) normalized.assistantOverlay = existing.assistantOverlay
  const next = enforceChatSettingsSize(normalized)
  if (!next) throw new Error("chat_settings_size_exceeded")
  if (!target.serverChatId) {
    if (!isQualifiedSettingsKey(chatKey)) {
      if (!await saveChatSettingsForKey(chatKey, next)) throw new Error("chat_settings_device_save_failed")
      assertSettingsCurrent(target)
      return next
    }
    assertSettingsCurrent(target)
    const saved = await settingsCache.set(chatKey, next, Date.now(), cacheBefore, () => { assertSettingsCurrent(target); return true })
    assertSettingsCurrent(target)
    if (!saved) throw new Error("chat_settings_device_save_failed")
    return next
  }
  const input = next as unknown as Record<string, unknown>
  const intent = settingsSubset(input, target.patch as Record<string, unknown>)
  if (!Object.keys(intent).length) throw new Error("chat_settings_patch_invalid")
  const patch: Record<string, unknown> = { schemaVersion: next.schemaVersion }
  for (const key of Object.keys(intent)) patch[key] = input[key]
  if (hasOwn(target.patch, "chat_dictionary_ids") || (isRecord(target.patch.conversationContext) && hasOwn(target.patch.conversationContext, "chat_dictionary_ids"))) {
    patch.chat_dictionary_ids = next.chat_dictionary_ids
    patch.conversationContext = next.conversationContext
  }
  const sameIntent = previous && sameSettingsIntent(previous.value.intent, intent)
  if (previous && ["pending", "acknowledged"].includes(previous.value.status) && !sameIntent) {
    throw Object.assign(new Error("chat_settings_pending_changed"), { pendingPatch: previous.value.intent })
  }
  const retry = sameIntent && previous && ["pending", "acknowledged"].includes(previous.value.status)
  const alreadyApplied = sameIntent && previous?.value.status !== "rejected" && existing && sameSettingsIntent(settingsSubset(existing as unknown as Record<string, unknown>, intent), intent)
  if (retry && previous.value.status === "acknowledged" && !alreadyApplied) throw new Error("chat_settings_changed_after_acknowledgement")
  let pending = previous
  if (!retry || !alreadyApplied) {
    const value: SettingsSave = { version: 1, destination: chatKey, operation_id: retry ? previous.value.operation_id : crypto.randomUUID(), status: "pending", intent, patch }
    const updatedAt = Date.now()
    if (!await settingsSaves.set(chatKey, value, updatedAt, previous, () => { assertSettingsCurrent(target); return true })) {
      assertSettingsCurrent(target)
      throw new Error("chat_settings_pending_changed")
    }
    pending = { value, updatedAt }
  }
  try {
  assertSettingsCurrent(target)
  let synced = existing
  if (!alreadyApplied) {
    let response: unknown
    try {
      response = await tldwClient.updateChatSettings(target.serverChatId, patch, settingsRequestOptions(target))
    } catch (cause) {
      assertSettingsCurrent(target)
      // This endpoint's 413/422 validation paths precede persistence or roll back
      // their transaction. Transport and server errors retain an unknown outcome.
      if (isRecord(cause) && [413, 422].includes(cause.status as number) && pending) {
        if (!await settingsSaves.set(chatKey, { ...pending.value, status: "rejected", serverSettings: existing }, Date.now(), pending, () => { assertSettingsCurrent(target); return true })) {
          assertSettingsCurrent(target)
          throw new Error("chat_settings_device_save_failed", { cause })
        }
        assertSettingsCurrent(target)
        throw Object.assign(new Error("chat_settings_server_rejected", { cause }), { serverSettings: existing, pendingPatch: intent })
      }
      throw Object.assign(new Error("chat_settings_server_unconfirmed", { cause }), { pendingPatch: intent })
    }
    assertSettingsCurrent(target)
    synced = coerceSettings(requireServerSettingsResponse(target.serverChatId, response))
    if (!synced) throw new Error("chat_settings_ack_invalid")
  }
  if (!synced || !pending) throw new Error("chat_settings_ack_invalid")
  if (!sameSettingsIntent(settingsSubset(synced as unknown as Record<string, unknown>, intent), intent)) {
    await settingsSaves.set(chatKey, { ...pending.value, status: "rejected", serverSettings: synced }, Date.now(), pending, () => { assertSettingsCurrent(target); return true })
    assertSettingsCurrent(target)
    throw Object.assign(new Error("chat_settings_ack_mismatch"), { serverChatId: target.serverChatId, serverSettings: synced, pendingPatch: intent })
  }
  const value: SettingsSave = { ...pending.value, status: "acknowledged", serverSettings: synced }
  const acknowledged: LocalRegistryRecord<SettingsSave> = { value, updatedAt: Date.now() }
  if (!await settingsSaves.set(chatKey, value, acknowledged.updatedAt, pending, () => { assertSettingsCurrent(target); return true })) {
    assertSettingsCurrent(target)
    throw Object.assign(new Error("chat_settings_device_save_failed"), { serverChatId: target.serverChatId, serverSettings: synced })
  }
  await saveAcknowledgedServerSettings(target, synced, cacheBefore)
  if (!await settingsSaves.set(chatKey, { ...value, status: "settled" }, Date.now(), acknowledged, () => { assertSettingsCurrent(target); return true })) {
    assertSettingsCurrent(target)
    throw Object.assign(new Error("chat_settings_device_save_failed"), { serverChatId: target.serverChatId, serverSettings: synced })
  }
  assertSettingsCurrent(target)
  return synced
  } catch (error) {
    if (error instanceof Error && pending && ["chat_settings_server_unconfirmed", "chat_settings_device_save_failed", "chat_settings_ack_invalid"].includes(error.message))
      Object.assign(error, { settingsOperationId: pending.value.operation_id })
    throw error
  }
}

export const applyChatSettingsPatch = async (target: ChatSettingsTarget & { patch: Partial<ChatSettingsRecord>; recoveryOperationId?: string }): Promise<ChatSettingsRecord | null> => {
  const captured = { ...target, patch: structuredClone(target.patch), scope: target.scope ? { ...target.scope } : undefined }
  assertSettingsCurrent(captured)
  if (!captured.serverChatId) return await applyCapturedChatSettingsPatch(captured)
  if (typeof navigator === "undefined" || !navigator.locks?.request) throw new Error("chat_settings_device_save_unavailable")
  // The existing platform lock serializes read/compose/PUT, not just its device receipt.
  return await navigator.locks.request(`tldw:chatSettingsSave:${resolveChatSettingsKey(captured)}`, () => applyCapturedChatSettingsPatch(captured))
}
