import { apiSend } from "@/services/api-send"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type PromptSearchField =
  | "name"
  | "author"
  | "details"
  | "system_prompt"
  | "user_prompt"
  | "keywords"

export type PromptSearchItem = {
  id: number
  uuid: string
  name: string
  author?: string | null
  details?: string | null
  system_prompt?: string | null
  user_prompt?: string | null
  last_modified?: string
  version?: number
  usage_count?: number
  last_used_at?: string | null
  keywords?: string[]
  deleted?: boolean
  relevance_score?: number | null
}

export type PromptSearchResponse = {
  items: PromptSearchItem[]
  total_matches: number
  page: number
  per_page: number
}

export type SearchPromptsParams = {
  searchQuery: string
  searchFields?: PromptSearchField[]
  page?: number
  resultsPerPage?: number
  includeDeleted?: boolean
}

export type PromptExportFormat = "csv" | "markdown"
export type PromptExportResponse = {
  message: string
  file_path?: string | null
  file_content_b64?: string | null
}

export type PromptCollection = {
  collection_id: number
  name: string
  description?: string | null
  prompt_ids: number[]
}

export type PromptCollectionListResponse = {
  collections: PromptCollection[]
}

export type PromptCollectionCreatePayload = {
  name: string
  description?: string | null
  prompt_ids?: number[]
}

export type PromptCollectionCreateResponse = {
  collection_id: number
}

export type PromptCollectionUpdatePayload = {
  name?: string
  description?: string | null
  prompt_ids?: number[]
}

export type StructuredPromptPreviewRequest = {
  prompt_format: "legacy" | "structured"
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_schema_version?: number | null
  prompt_definition?: Record<string, any> | null
  variables?: Record<string, any>
}

export type StructuredPromptPreviewResponse = {
  prompt_format: "legacy" | "structured"
  prompt_schema_version?: number | null
  assembled_messages: Array<{
    role: string
    content: string
  }>
  legacy_system_prompt: string
  legacy_user_prompt: string
}

export type PromptImprovementLimits = {
  max_request_bytes: number
  max_draft_chars: number
  max_candidate_chars: number
  max_raw_output_chars: number
  max_findings: number
  max_finding_text_chars: number
  max_provider_chars: number
  max_model_chars: number
  max_meta_prompt_version_chars: number
  max_warning_chars: number
  max_warnings: number
  max_protected_tokens: number
  max_protected_token_kind_chars: number
  max_protected_token_chars: number
  max_protected_token_occurrences: number
  max_protected_token_total_chars: number
}

export type PromptCapabilities = {
  availability: "available" | "unavailable"
  prompt_improvement_v1: {
    supported: boolean
    limits: PromptImprovementLimits | null
  }
  single_text_recipe_v2: {
    supported: boolean
  }
}

const unavailablePromptCapabilities = (): PromptCapabilities => ({
  availability: "unavailable",
  prompt_improvement_v1: { supported: false, limits: null },
  single_text_recipe_v2: { supported: false }
})

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const PROMPT_LIMIT_KEYS = [
  "max_request_bytes",
  "max_draft_chars",
  "max_candidate_chars",
  "max_raw_output_chars",
  "max_findings",
  "max_finding_text_chars",
  "max_provider_chars",
  "max_model_chars",
  "max_meta_prompt_version_chars",
  "max_warning_chars",
  "max_warnings",
  "max_protected_tokens",
  "max_protected_token_kind_chars",
  "max_protected_token_chars",
  "max_protected_token_occurrences",
  "max_protected_token_total_chars"
] as const satisfies readonly (keyof PromptImprovementLimits)[]

const parsePromptImprovementLimits = (
  value: unknown
): PromptImprovementLimits | null => {
  if (!isRecord(value)) return null
  if (
    !PROMPT_LIMIT_KEYS.every(
      (key) => Number.isInteger(value[key]) && Number(value[key]) > 0
    )
  ) {
    return null
  }
  return Object.fromEntries(
    PROMPT_LIMIT_KEYS.map((key) => [key, Number(value[key])])
  ) as PromptImprovementLimits
}

const parsePromptCapabilities = (value: unknown): PromptCapabilities | null => {
  if (!isRecord(value)) return null
  const improvement = value.prompt_improvement_v1
  const recipe = value.single_text_recipe_v2
  if (!isRecord(improvement) || !isRecord(recipe)) return null
  if (
    typeof improvement.supported !== "boolean" ||
    typeof recipe.supported !== "boolean"
  ) {
    return null
  }
  const limits = parsePromptImprovementLimits(improvement.limits)
  if (!limits) return null
  return {
    availability:
      improvement.supported || recipe.supported ? "available" : "unavailable",
    prompt_improvement_v1: {
      supported: improvement.supported,
      limits
    },
    single_text_recipe_v2: { supported: recipe.supported }
  }
}

export const buildPromptSearchQuery = ({
  searchQuery,
  searchFields = [],
  page = 1,
  resultsPerPage = 20,
  includeDeleted = false
}: SearchPromptsParams): string => {
  const qs = new URLSearchParams()
  qs.set("search_query", searchQuery)
  qs.set("page", String(page))
  qs.set("results_per_page", String(resultsPerPage))
  qs.set("include_deleted", includeDeleted ? "true" : "false")

  for (const field of searchFields) {
    qs.append("search_fields", field)
  }

  return `?${qs.toString()}`
}

export async function searchPromptsServer(
  params: SearchPromptsParams
): Promise<PromptSearchResponse> {
  const query = buildPromptSearchQuery(params)
  const response = await apiSend<PromptSearchResponse>({
    path: appendPathQuery(toAllowedPath("/api/v1/prompts/search"), query),
    method: "POST"
  })

  if (!response.ok) {
    throw new Error(response.error || "Failed to search prompts")
  }

  return (
    response.data || {
      items: [],
      total_matches: 0,
      page: params.page || 1,
      per_page: params.resultsPerPage || 20
    }
  )
}

export const buildPromptExportQuery = (format: PromptExportFormat): string => {
  const qs = new URLSearchParams()
  qs.set("export_format", format)
  return `?${qs.toString()}`
}

export async function exportPromptsServer(
  format: PromptExportFormat
): Promise<PromptExportResponse> {
  const response = await apiSend<PromptExportResponse>({
    path: appendPathQuery(
      toAllowedPath("/api/v1/prompts/export"),
      buildPromptExportQuery(format)
    ),
    method: "GET"
  })

  if (!response.ok) {
    throw new Error(response.error || "Failed to export prompts")
  }

  return (
    response.data || {
      message: ""
    }
  )
}

export async function listPromptCollectionsServer(): Promise<PromptCollection[]> {
  const response = await apiSend<PromptCollectionListResponse>({
    path: toAllowedPath("/api/v1/prompts/collections"),
    method: "GET"
  })

  if (!response.ok) {
    throw new Error(response.error || "Failed to load prompt collections")
  }

  return response.data?.collections || []
}

export async function createPromptCollectionServer(
  payload: PromptCollectionCreatePayload
): Promise<PromptCollectionCreateResponse> {
  const response = await apiSend<PromptCollectionCreateResponse>({
    path: toAllowedPath("/api/v1/prompts/collections/create"),
    method: "POST",
    body: payload
  })

  if (!response.ok || !response.data) {
    throw new Error(response.error || "Failed to create prompt collection")
  }

  return response.data
}

export async function updatePromptCollectionServer(
  collectionId: number,
  payload: PromptCollectionUpdatePayload
): Promise<PromptCollection> {
  const response = await apiSend<PromptCollection>({
    path: toAllowedPath(`/api/v1/prompts/collections/${collectionId}`),
    method: "PUT",
    body: payload
  })

  if (!response.ok || !response.data) {
    throw new Error(response.error || "Failed to update prompt collection")
  }

  return response.data
}

export async function previewStructuredPromptServer(
  payload: StructuredPromptPreviewRequest
): Promise<StructuredPromptPreviewResponse> {
  const response = await apiSend<StructuredPromptPreviewResponse>({
    path: toAllowedPath("/api/v1/prompts/preview"),
    method: "POST",
    body: payload
  })

  if (!response.ok || !response.data) {
    throw new Error(response.error || "Failed to preview structured prompt")
  }

  return response.data
}

export async function fetchPromptCapabilities(): Promise<PromptCapabilities> {
  try {
    const response = await apiSend<unknown>({
      path: toAllowedPath("/api/v1/prompts/capabilities"),
      method: "GET"
    })
    if (!response.ok) return unavailablePromptCapabilities()
    return parsePromptCapabilities(response.data) ?? unavailablePromptCapabilities()
  } catch {
    return unavailablePromptCapabilities()
  }
}
