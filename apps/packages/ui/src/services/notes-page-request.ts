import type { ClientPath, ClientPathRuntimeWithQuery } from "@/services/tldw/openapi-guard"

const NOTES_LIST_PATH = "/api/v1/notes/" satisfies ClientPath
const NOTES_SEARCH_PATH = "/api/v1/notes/search/" satisfies ClientPath
/** GET /api/v1/notes/search/ caps `limit` at 100. */
const NOTES_PAGE_LIMIT_MAX = 100

export type NotesPageSort = {
  sortBy: "last_modified" | "created_at" | "title"
  sortOrder: "asc" | "desc"
}

type NotesPageRequestArgs = {
  query: string
  tokens: string[]
  page: number
  pageSize: number
  /** Applies to the plain list only: search results come in relevance order. */
  sort: NotesPageSort | null
}

/**
 * The GET path for one page of notes, on the notes API's own contract (#3149).
 *
 * Both notes endpoints page by `limit`/`offset` and ignore `page`/`results_per_page`.
 * A text query or tags go to `/api/v1/notes/search/`, which matches each tag exactly,
 * ignoring case, and with `tokens_match=all` keeps notes that have every tag.
 */
export const buildNotesPageRequestPath = ({
  query,
  tokens,
  page,
  pageSize,
  sort
}: NotesPageRequestArgs): ClientPathRuntimeWithQuery => {
  const limit = Math.min(Math.max(1, Math.floor(pageSize) || 1), NOTES_PAGE_LIMIT_MAX)
  const offset = (Math.max(1, Math.floor(page) || 1) - 1) * limit
  const params = new URLSearchParams()
  const text = query.trim()
  const tags = Array.from(new Set(tokens.map((token) => token.trim()).filter(Boolean)))
  const searching = text.length > 0 || tags.length > 0
  if (text) params.set("query", text)
  for (const tag of tags) params.append("tokens", tag)
  if (tags.length > 0) params.set("tokens_match", "all")
  params.set("limit", String(limit))
  params.set("offset", String(offset))
  params.set("include_keywords", "true")
  if (!searching && sort) {
    params.set("sort_by", sort.sortBy)
    params.set("sort_order", sort.sortOrder)
  }
  return `${searching ? NOTES_SEARCH_PATH : NOTES_LIST_PATH}?${params.toString()}`
}

type NotesPageResponse = {
  total?: unknown
  offset?: unknown
  pagination?: { total?: unknown; offset?: unknown; has_more?: unknown }
}

const nonNegative = (value: unknown): number | null => {
  const number = value == null ? Number.NaN : Number(value)
  return Number.isFinite(number) && number >= 0 ? number : null
}

/**
 * The notes API's total: `pagination.total`, then the top-level `total`.
 *
 * A null total means the server could not count. The rows seen so far
 * (`offset` + this page's `rowCount`) are then a lower bound, plus one when
 * `has_more` says a row follows, so the next page stays reachable.
 */
export const readNotesTotal = (response: unknown, rowCount: number): number => {
  const record = (response ?? {}) as NotesPageResponse
  const total = nonNegative(record.pagination?.total ?? record.total)
  if (total != null) return total
  const offset = nonNegative(record.pagination?.offset ?? record.offset) ?? 0
  return offset + rowCount + (record.pagination?.has_more === true ? 1 : 0)
}
