/**
 * A fake of the real server prompt library endpoints, for contract tests.
 *
 * It follows the backend rather than the client:
 * - GET /api/v1/prompts: endpoints/prompts.py list_all_prompts takes
 *   `page` (>= 1), `per_page` (1..100), `include_deleted`, `sort_by`,
 *   `sort_order`; Prompts_DB.list_prompts pages with LIMIT/OFFSET and
 *   `total_pages = ceil(total / per_page)` (0 when empty); the response is
 *   schemas/prompt_schemas.py PaginatedPromptsResponse of PromptBriefResponse
 *   items plus utils/pagination.py build_page_pagination_meta.
 * - GET /api/v1/prompts/{prompt_identifier}: PromptResponse.
 * Out-of-range query values answer 422, as FastAPI validation would.
 */

export type FakeServerPrompt = {
  id: number
  uuid: string
  name: string
  author?: string | null
  details?: string | null
  system_prompt?: string | null
  user_prompt?: string | null
  keywords?: string[]
}

export class FakePromptsApiError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message)
  }
}

const LAST_MODIFIED = "2026-09-30T12:00:00Z"

export const buildFakeServerPrompt = (
  id: number,
  overrides: Partial<FakeServerPrompt> = {}
): FakeServerPrompt => ({
  id,
  uuid: `00000000-0000-4000-8000-${String(id).padStart(12, "0")}`,
  name: `Server prompt ${String(id).padStart(3, "0")}`,
  author: "server",
  details: null,
  system_prompt: null,
  user_prompt: `User text ${id}`,
  keywords: [],
  ...overrides
})

/** PromptBriefResponse: exactly the fields the list endpoint serialises. */
const toBrief = (prompt: FakeServerPrompt) => ({
  id: prompt.id,
  uuid: prompt.uuid,
  name: prompt.name,
  author: prompt.author ?? null,
  last_modified: LAST_MODIFIED,
  usage_count: 0,
  last_used_at: null
})

/** PromptResponse for GET /api/v1/prompts/{prompt_identifier}. */
const toDetail = (prompt: FakeServerPrompt) => ({
  name: prompt.name,
  author: prompt.author ?? null,
  details: prompt.details ?? null,
  system_prompt: prompt.system_prompt ?? null,
  user_prompt: prompt.user_prompt ?? null,
  prompt_format: "legacy",
  prompt_schema_version: null,
  prompt_definition: null,
  id: prompt.id,
  uuid: prompt.uuid,
  last_modified: LAST_MODIFIED,
  version: 1,
  usage_count: 0,
  last_used_at: null,
  keywords: prompt.keywords ?? [],
  deleted: false
})

const readIntParam = (
  params: URLSearchParams,
  name: string,
  fallback: number,
  min: number,
  max: number
) => {
  const raw = params.get(name)
  if (raw === null) return fallback
  const value = Number(raw)
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new FakePromptsApiError(422, `Invalid ${name}: ${raw}`)
  }
  return value
}

const SORT_FIELDS = new Set([
  "last_modified",
  "name",
  "author",
  "id",
  "usage_count",
  "last_used_at"
])

export const createFakePromptsApi = (prompts: FakeServerPrompt[]) => {
  const listRequests: URLSearchParams[] = []
  const detailRequests: string[] = []

  const list = (params: URLSearchParams) => {
    listRequests.push(params)
    const page = readIntParam(params, "page", 1, 1, Number.MAX_SAFE_INTEGER)
    const perPage = readIntParam(params, "per_page", 10, 1, 100)
    const sortBy = params.get("sort_by") ?? "last_modified"
    const sortOrder = params.get("sort_order") ?? "desc"
    if (!SORT_FIELDS.has(sortBy) || !["asc", "desc"].includes(sortOrder)) {
      throw new FakePromptsApiError(400, "Unsupported sort")
    }
    const ordered = [...prompts].sort((left, right) => {
      const delta =
        sortBy === "name"
          ? left.name.localeCompare(right.name) || left.id - right.id
          : left.id - right.id
      return sortOrder === "asc" ? delta : -delta
    })
    const total = ordered.length
    const totalPages = total > 0 ? Math.ceil(total / perPage) : 0
    const items = ordered.slice((page - 1) * perPage, page * perPage)
    return {
      items: items.map(toBrief),
      total_pages: totalPages,
      current_page: page,
      total_items: total,
      pagination: {
        mode: "page",
        page,
        per_page: perPage,
        total,
        total_pages: totalPages,
        has_more: page < totalPages
      }
    }
  }

  /** Answer a bgRequest init the way the server would. */
  const handle = (request: { path?: unknown; method?: unknown }) => {
    const method = String(request?.method || "GET").toUpperCase()
    const [pathname, query = ""] = String(request?.path || "").split("?")
    if (method !== "GET") return undefined
    if (/^\/api\/v1\/prompts\/?$/.test(pathname)) {
      return list(new URLSearchParams(query))
    }
    const detailMatch = pathname.match(/^\/api\/v1\/prompts\/([^/]+)$/)
    if (detailMatch) {
      const identifier = decodeURIComponent(detailMatch[1])
      detailRequests.push(identifier)
      const prompt = prompts.find(
        (candidate) =>
          String(candidate.id) === identifier || candidate.uuid === identifier
      )
      if (!prompt) throw new FakePromptsApiError(404, "Prompt not found.")
      return toDetail(prompt)
    }
    return undefined
  }

  return { handle, listRequests, detailRequests }
}
