import { normalizeConversationId } from './notes-manager-utils'

/**
 * Chat-title lookups for the notes list (NL-04, #3101).
 *
 * Every visible row that came from a chat names that chat, whatever its id
 * looks like. UUID ids used to be skipped until their note was opened, so most
 * rows showed only "Linked chat". The lookups now run for the visible rows, a
 * few at a time.
 */

/** Title lookups in flight at once, so a 100-row page does not send 100 requests together. */
export const CONVERSATION_LABEL_BATCH_SIZE = 6

/** The distinct chat ids of the visible rows, plus the open note's chat. */
export const collectConversationIdsToResolve = (
  notes: ReadonlyArray<{ conversation_id?: unknown } | null | undefined>,
  openConversationId: unknown
): string[] => {
  const ids = new Set<string>()
  for (const note of notes) {
    const normalized = normalizeConversationId(note?.conversation_id)
    if (normalized) ids.add(normalized)
  }
  const open = normalizeConversationId(openConversationId)
  if (open) ids.add(open)
  return Array.from(ids)
}

/**
 * Runs `worker` over `items`, at most `batchSize` at a time, and settles every
 * call. Results keep the order of `items`, like `Promise.allSettled`.
 */
export const settleInBatches = async <T, R>(
  items: readonly T[],
  worker: (item: T) => Promise<R>,
  batchSize: number = CONVERSATION_LABEL_BATCH_SIZE
): Promise<PromiseSettledResult<R>[]> => {
  const size = Math.max(1, Math.floor(batchSize) || 1)
  const settled: PromiseSettledResult<R>[] = []
  for (let start = 0; start < items.length; start += size) {
    const batch = items.slice(start, start + size)
    settled.push(...(await Promise.allSettled(batch.map((item) => worker(item)))))
  }
  return settled
}
