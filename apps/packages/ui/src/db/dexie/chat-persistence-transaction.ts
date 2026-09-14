import { db } from "./schema"

export const rollbackScopedComparePersistence = async ({
  clusterId,
  historyId,
  removeHistory
}: {
  clusterId: string
  historyId: string
  removeHistory: boolean
}): Promise<void> => {
  await db.transaction("rw", [db.chatHistories, db.messages], async () => {
    if (removeHistory) {
      await db.messages.where("history_id").equals(historyId).delete()
      await db.chatHistories.delete(historyId)
      return
    }
    await db.messages
      .where("clusterId")
      .equals(clusterId)
      .and((message) => message.history_id === historyId)
      .delete()
  })
}

const shouldAbort = (
  signal?: AbortSignal,
  shouldAbortForScopeChange?: () => boolean
): boolean => signal?.aborted === true &&
  (shouldAbortForScopeChange?.() ?? true)

const throwIfAborted = (
  signal?: AbortSignal,
  shouldAbortForScopeChange?: () => boolean
): void => {
  if (!shouldAbort(signal, shouldAbortForScopeChange)) return
  const error = new Error("Request scope changed")
  error.name = "AbortError"
  throw error
}

export const runChatPersistenceTransaction = async <T>(
  signal: AbortSignal | undefined,
  operation: () => Promise<T>,
  shouldAbortForScopeChange?: () => boolean
): Promise<T> => {
  throwIfAborted(signal, shouldAbortForScopeChange)
  let activeTransaction: { abort: () => void } | undefined
  const abort = () => {
    if (!shouldAbort(signal, shouldAbortForScopeChange)) return
    try {
      activeTransaction?.abort()
    } catch {
      // A completed transaction needs no further rollback.
    }
  }
  signal?.addEventListener("abort", abort, { once: true })
  try {
    const result = await db.transaction(
      "rw",
      [db.chatHistories, db.messages, db.modelNickname, db.sessionFiles],
      async (transaction) => {
        activeTransaction = transaction
        throwIfAborted(signal, shouldAbortForScopeChange)
        const operationResult = await operation()
        throwIfAborted(signal, shouldAbortForScopeChange)
        return operationResult
      }
    )
    throwIfAborted(signal, shouldAbortForScopeChange)
    return result
  } finally {
    signal?.removeEventListener("abort", abort)
  }
}
