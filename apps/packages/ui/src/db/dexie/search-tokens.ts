import { db } from "./schema"
import { buildSearchTokenRows, tokenizeSearchText } from "./search-tokenizer"
import type { Message, SearchTokenRow } from "./types"

/**
 * Token side-table for message-content search (WP-23 / TASK-13523).
 *
 * `searchTokens` maps every word of every message to its history, keyed by
 * `[token+history_id]`. A content query tokenizes, looks candidate histories
 * up through the `token` index, and only candidate message rows are then read
 * for the exact word-boundary check — the messages table is never
 * materialized in full.
 *
 * Maintenance contract:
 * - Missing tokens are a correctness bug (a message becomes unfindable), so
 *   every message write path indexes (addMessage, updateMessage, imports,
 *   mirror writes, history-selection admits, forks) and the v17 upgrade
 *   backfills pre-existing rows.
 * - Stale tokens (content edited away, messages deleted) only cost an extra
 *   candidate that exact verification then rejects, so delete paths skip
 *   token cleanup deliberately.
 */

/** Index one message's content. Idempotent per (token, history) pair. */
export async function indexMessageSearchTokens(
  message: Pick<Message, "history_id" | "content">
): Promise<void> {
  const rows = buildSearchTokenRows(message.history_id, message.content)
  if (rows.length === 0) return
  await db.searchTokens.bulkPut(rows)
}

/** Bulk variant for imports, forks and backfills. */
export async function indexMessagesSearchTokens(
  messages: Array<Pick<Message, "history_id" | "content">>
): Promise<void> {
  const rows: SearchTokenRow[] = []
  const pairs = new Set<string>()
  for (const message of messages) {
    for (const row of buildSearchTokenRows(message.history_id, message.content)) {
      const pair = `${row.token}|${row.history_id}`
      if (pairs.has(pair)) continue
      pairs.add(pair)
      rows.push(row)
    }
  }
  if (rows.length === 0) return
  await db.searchTokens.bulkPut(rows)
}

/**
 * Histories whose messages contain at least one word of the (already
 * normalized) query — a superset of the exact word-boundary matches. Returns
 * null when the query has no word runs (punctuation-only): the token index
 * cannot prefilter such a query.
 */
export async function searchTokenHistoryIds(normalizedQuery: string): Promise<Set<string> | null> {
  const tokens = tokenizeSearchText(normalizedQuery)
  if (tokens.length === 0) return null
  const keys = (await db.searchTokens.where("token").anyOf(tokens).primaryKeys()) as Array<
    [string, string]
  >
  const historyIds = new Set<string>()
  for (const key of keys) historyIds.add(key[1])
  return historyIds
}
