export type PlaygroundRestoreDecisionInput = {
  hasPersistedSession: boolean
  persistedHistoryId: string | null
  persistedServerChatId: string | null
  initialHistoryId?: string | null
  initialServerChatId?: string | null
  initialMessagesLength?: number
  initialHistoryLength?: number
  currentHistoryId: string | null
  currentServerChatId: string | null
  currentMessagesLength: number
  currentHistoryLength: number
}

/**
 * Restore a persisted chat session when no conversation is currently loaded,
 * or when the loaded conversation differs from the last persisted chat session.
 */
export const shouldRestorePersistedPlaygroundSession = ({
  hasPersistedSession,
  persistedHistoryId,
  persistedServerChatId,
  initialHistoryId,
  initialServerChatId,
  initialMessagesLength,
  initialHistoryLength,
  currentHistoryId,
  currentServerChatId,
  currentMessagesLength,
  currentHistoryLength
}: PlaygroundRestoreDecisionInput): boolean => {
  if (!hasPersistedSession) return false

  const conversationChangedAfterMount =
    (initialHistoryId !== undefined &&
      initialHistoryId !== currentHistoryId) ||
    (initialServerChatId !== undefined &&
      initialServerChatId !== currentServerChatId) ||
    (initialMessagesLength !== undefined &&
      initialMessagesLength !== currentMessagesLength) ||
    (initialHistoryLength !== undefined &&
      initialHistoryLength !== currentHistoryLength)

  if (conversationChangedAfterMount) return false

  const hasCurrentConversation =
    currentMessagesLength > 0 ||
    currentHistoryLength > 0 ||
    Boolean(currentHistoryId) ||
    Boolean(currentServerChatId)

  if (!hasCurrentConversation) {
    return true
  }

  const normalizedPersistedHistoryId = persistedHistoryId ?? null
  const normalizedPersistedServerChatId = persistedServerChatId ?? null
  const normalizedCurrentHistoryId = currentHistoryId ?? null
  const normalizedCurrentServerChatId = currentServerChatId ?? null

  return (
    normalizedPersistedHistoryId !== normalizedCurrentHistoryId ||
    normalizedPersistedServerChatId !== normalizedCurrentServerChatId
  )
}
