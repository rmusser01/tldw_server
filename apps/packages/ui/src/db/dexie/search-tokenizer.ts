/**
 * Pure tokenizer shared by the `searchTokens` side-table write path and the
 * v17 schema backfill (WP-23 / TASK-13523). Its only import is type-only
 * (erased at runtime): the module must stay free of RUNTIME dependencies so
 * schema.ts can import it statically inside its upgrade handler — a dynamic
 * import there would await a non-Dexie promise at the start of the
 * version-change transaction.
 *
 * Word runs are ASCII `\w+` sequences, matching the `\b`-anchored regex the
 * content search uses for exact verification. Keeping both on the same
 * definition guarantees the token index is always a superset of the exact
 * word-boundary matches, so candidate prefiltering never drops a real match.
 */

import type { SearchTokenRow } from "./types"

const WORD_RUN = /[a-z0-9_]+/g

/** Unique lowercase word runs of `text`, in first-seen order. */
export function tokenizeSearchText(text: string): string[] {
  if (!text) return []
  const words = text.toLowerCase().match(WORD_RUN)
  if (!words) return []
  return [...new Set(words)]
}

/** Rows describing every distinct word of `content` under `historyId`. */
export function buildSearchTokenRows(
  historyId: string,
  content: string | undefined | null
): SearchTokenRow[] {
  if (!historyId || !content) return []
  return tokenizeSearchText(content).map((token) => ({ token, history_id: historyId }))
}
