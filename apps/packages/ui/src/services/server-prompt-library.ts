/**
 * Read-only client for the user's server prompt library (`/api/v1/prompts`).
 *
 * Shapes mirror the backend schemas in
 * `tldw_Server_API/app/api/v1/schemas/prompt_schemas.py`:
 * - GET /api/v1/prompts          -> PaginatedPromptsResponse of PromptBriefResponse
 * - GET /api/v1/prompts/{id}     -> PromptResponse
 *
 * The list endpoint is page based (`page`, `per_page` <= 100) and returns no
 * prompt text, so callers list briefs here and fetch the full prompt when the
 * user picks one.
 */
import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

/** One item of GET /api/v1/prompts (PromptBriefResponse). */
export type ServerPromptBrief = {
  id: number
  uuid: string
  name: string
  author?: string | null
  last_modified?: string
  usage_count?: number
  last_used_at?: string | null
}

/** Canonical page metadata (schemas/pagination.py PagePaginationMeta). */
export type ServerPromptPagination = {
  mode?: "page"
  page: number
  per_page: number
  total?: number | null
  total_pages?: number | null
  has_more: boolean
}

/** GET /api/v1/prompts response (PaginatedPromptsResponse). */
export type ServerPromptListPage = {
  items: ServerPromptBrief[]
  total_pages: number
  current_page: number
  total_items: number
  pagination?: ServerPromptPagination
}

/** GET /api/v1/prompts/{prompt_identifier} response (PromptResponse). */
export type ServerPromptDetail = ServerPromptBrief & {
  details?: string | null
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_format?: "legacy" | "structured"
  keywords?: string[]
  version?: number
  deleted?: boolean
}

export type ServerPromptLibrary = {
  prompts: ServerPromptBrief[]
  /** Server-reported total, never smaller than the prompts actually listed. */
  totalItems: number
  /** True only when the page ceiling stopped paging before the server did. */
  truncated: boolean
}

/** The list endpoint rejects `per_page` above 100. */
export const SERVER_PROMPT_LIBRARY_PAGE_SIZE = 100
/** Runaway guard only: 100 pages x 100 prompts. */
export const SERVER_PROMPT_LIBRARY_MAX_PAGES = 100

/**
 * True when the failure only means "no server is connected" (bgRequest's
 * "tldw server not configured"), not an outage worth surfacing.
 */
export const isServerPromptLibraryUnconfiguredError = (error: unknown): boolean =>
  error instanceof Error &&
  error.message.toLowerCase().includes("server not configured")

export const buildServerPromptListPath = (page: number, perPage: number) => {
  const query = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
    // Stable ordering so prompts edited mid-listing do not shift between pages.
    sort_by: "id",
    sort_order: "asc"
  })
  return appendPathQuery(toAllowedPath("/api/v1/prompts"), `?${query}`)
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const toFiniteNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null

const normalizeBrief = (value: unknown): ServerPromptBrief | null => {
  if (!isRecord(value)) return null
  const id = toFiniteNumber(value.id)
  const name = typeof value.name === "string" ? value.name.trim() : ""
  if (id === null || !name) return null
  return {
    id,
    uuid: typeof value.uuid === "string" ? value.uuid : "",
    name,
    author: typeof value.author === "string" ? value.author : null,
    last_modified:
      typeof value.last_modified === "string" ? value.last_modified : undefined,
    usage_count: toFiniteNumber(value.usage_count) ?? undefined,
    last_used_at:
      typeof value.last_used_at === "string" ? value.last_used_at : null
  }
}

const resolveHasMore = (
  response: Record<string, unknown>,
  page: number
): boolean => {
  const pagination = isRecord(response.pagination) ? response.pagination : null
  if (pagination && typeof pagination.has_more === "boolean") {
    return pagination.has_more
  }
  const totalPages =
    toFiniteNumber(pagination?.total_pages) ??
    toFiniteNumber(response.total_pages)
  return totalPages !== null && page < totalPages
}

const resolveTotalItems = (response: Record<string, unknown>): number | null => {
  const pagination = isRecord(response.pagination) ? response.pagination : null
  return toFiniteNumber(response.total_items) ?? toFiniteNumber(pagination?.total)
}

/**
 * List every prompt in the server library by walking all pages.
 * Duplicates that appear on two pages (edits during paging) are dropped.
 */
export async function listAllServerPrompts(options?: {
  signal?: AbortSignal
  pageSize?: number
  maxPages?: number
}): Promise<ServerPromptLibrary> {
  const pageSize = options?.pageSize ?? SERVER_PROMPT_LIBRARY_PAGE_SIZE
  const maxPages = options?.maxPages ?? SERVER_PROMPT_LIBRARY_MAX_PAGES
  const byKey = new Map<string, ServerPromptBrief>()
  let totalItems = 0
  let truncated = false

  for (let page = 1; ; page += 1) {
    if (options?.signal?.aborted) {
      throw new DOMException("Server prompt listing aborted", "AbortError")
    }
    const response = await bgRequest<unknown>({
      path: buildServerPromptListPath(page, pageSize),
      method: "GET",
      ...(options?.signal ? { abortSignal: options.signal } : {})
    })
    if (!isRecord(response) || !Array.isArray(response.items)) {
      throw new Error("Unexpected response from the server prompt library")
    }
    for (const raw of response.items) {
      const brief = normalizeBrief(raw)
      if (!brief) continue
      byKey.set(brief.uuid || `id:${brief.id}`, brief)
    }
    totalItems = resolveTotalItems(response) ?? totalItems
    if (!resolveHasMore(response, page) || response.items.length === 0) break
    if (page >= maxPages) {
      truncated = true
      break
    }
  }

  const prompts = Array.from(byKey.values())
  return {
    prompts,
    totalItems: Math.max(totalItems, prompts.length),
    truncated
  }
}

/** Fetch one server prompt with its text (system_prompt / user_prompt). */
export async function getServerPromptDetail(
  id: number
): Promise<ServerPromptDetail> {
  const response = await bgRequest<unknown>({
    path: toAllowedPath(`/api/v1/prompts/${encodeURIComponent(String(id))}`),
    method: "GET"
  })
  const brief = normalizeBrief(response)
  if (!brief || !isRecord(response)) {
    throw new Error("Server prompt not found")
  }
  return {
    ...brief,
    details: typeof response.details === "string" ? response.details : null,
    system_prompt:
      typeof response.system_prompt === "string" ? response.system_prompt : null,
    user_prompt:
      typeof response.user_prompt === "string" ? response.user_prompt : null,
    prompt_format:
      response.prompt_format === "structured" ? "structured" : "legacy",
    keywords: Array.isArray(response.keywords)
      ? response.keywords.filter(
          (keyword): keyword is string => typeof keyword === "string"
        )
      : [],
    version: toFiniteNumber(response.version) ?? undefined,
    deleted: response.deleted === true
  }
}
