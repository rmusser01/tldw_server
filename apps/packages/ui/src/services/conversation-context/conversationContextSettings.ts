import type {
  ConversationContextSeed,
  ConversationContextSelection,
  ConversationContextSettingsPatch
} from "@/types/conversation-context"

type ResolveConversationContextSelectionParams = {
  settings?: Record<string, unknown> | null
  seed?: ConversationContextSeed | null
}

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const hasOwn = (record: Record<string, unknown>, key: string): boolean =>
  Object.prototype.hasOwnProperty.call(record, key)

const toPositiveInteger = (value: unknown): number | null => {
  if (typeof value === "boolean" || value === null || value === undefined) {
    return null
  }
  const numeric =
    typeof value === "number"
      ? value
      : typeof value === "string" && value.trim()
        ? Number(value)
        : NaN
  if (!Number.isInteger(numeric) || numeric <= 0) return null
  return numeric
}

export const normalizeConversationContextIdList = (
  value: unknown
): number[] => {
  if (!Array.isArray(value)) return []
  const ids: number[] = []
  const seen = new Set<number>()
  for (const item of value) {
    const id = toPositiveInteger(item)
    if (id === null || seen.has(id)) continue
    seen.add(id)
    ids.push(id)
  }
  return ids
}

const mergeIdLists = (...lists: unknown[]): number[] =>
  normalizeConversationContextIdList(lists.flatMap((list) => (
    Array.isArray(list) ? list : []
  )))

export const resolveConversationContextSelection = ({
  settings,
  seed
}: ResolveConversationContextSelectionParams): ConversationContextSelection => {
  const settingsRecord = toRecord(settings)
  const contextRecord = toRecord(settingsRecord?.conversationContext)

  const settingsWorldBookIds = normalizeConversationContextIdList(
    contextRecord?.world_book_ids
  )

  const settingsDictionaryIds =
    contextRecord && hasOwn(contextRecord, "chat_dictionary_ids")
      ? normalizeConversationContextIdList(contextRecord.chat_dictionary_ids)
      : normalizeConversationContextIdList(settingsRecord?.chat_dictionary_ids)

  return {
    chatId: seed?.chatId,
    characterId: seed?.characterId ?? null,
    worldBookIds: mergeIdLists(seed?.worldBookIds, settingsWorldBookIds),
    dictionaryIds: mergeIdLists(seed?.dictionaryIds, settingsDictionaryIds),
    workspaceId: seed?.workspaceId ?? null,
    providerId: seed?.providerId ?? null,
    modelId: seed?.modelId ?? null
  }
}

export const buildConversationContextSettingsPatch = ({
  worldBookIds,
  dictionaryIds
}: {
  worldBookIds: Array<number | string | null | undefined>
  dictionaryIds: Array<number | string | null | undefined>
}): ConversationContextSettingsPatch => {
  const normalizedWorldBookIds = normalizeConversationContextIdList(worldBookIds)
  const normalizedDictionaryIds =
    normalizeConversationContextIdList(dictionaryIds)

  return {
    conversationContext: {
      world_book_ids: normalizedWorldBookIds,
      chat_dictionary_ids: normalizedDictionaryIds
    },
    chat_dictionary_ids: normalizedDictionaryIds
  }
}
