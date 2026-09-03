import type { ApiSendResponse } from "@/services/api-send"
import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"
import { z } from "zod"

const FINGERPRINT = /^sha256:[0-9a-f]{64}$/
const STRONG_ETAG = /^"(sha256:[0-9a-f]{64})"$/
const MAX_LIST_ITEMS = 100
const MAX_CURSOR_LENGTH = 4096
export const NOTES_GRAPH_SEMANTIC_MAX_TOP_K = 50

const ERROR_MESSAGES = {
  notes_graph_active_run_conflict:
    "A matching suggestion run is already active.",
  notes_graph_admission_rate_limited:
    "Suggestion generation is rate limited; try again later.",
  notes_graph_capabilities_changed:
    "Suggestion capabilities changed; refresh and retry.",
  notes_graph_cursor_invalid:
    "The suggestion cursor is invalid or no longer applicable.",
  notes_graph_fingerprint_stale: "The note changed; refresh before retrying.",
  notes_graph_fts_not_ready:
    "Notes search is not ready for suggestion generation.",
  notes_graph_invalid_request: "The Notes graph suggestion request is invalid.",
  notes_graph_owner_active_run_conflict:
    "Another suggestion run is already active.",
  notes_graph_provider_call_policy_unsupported:
    "The selected provider cannot safely generate suggestions.",
  notes_graph_provider_disallowed:
    "The selected provider or model is not allowed.",
  notes_graph_projection_not_ready: "The Notes graph projection is not ready.",
  notes_graph_provider_model_disallowed:
    "The selected provider or model is unavailable.",
  notes_graph_provider_not_configured:
    "The selected provider is not configured.",
  notes_graph_provider_retry_policy_unsupported:
    "The selected provider retry policy is unsupported.",
  notes_graph_provider_unavailable: "The selected provider is unavailable.",
  notes_graph_source_too_large:
    "The selected note is too large for suggestion generation.",
  notes_graph_suggestion_conflict: "The suggestion changed; refresh and retry.",
  notes_graph_suggestion_idempotency_mismatch:
    "The idempotency key was reused for another request.",
  notes_graph_suggestion_not_found:
    "The requested Notes graph resource was not found.",
  notes_semantic_conversion_manual_link_exists:
    "A manual link already exists; refreshing the Notes graph.",
  notes_graph_suggestions_disabled: "Notes graph suggestions are disabled.",
  notes_graph_suggestions_unavailable:
    "Notes graph suggestions are temporarily unavailable.",
  notes_graph_suggestions_worker_unavailable:
    "The suggestion worker is unavailable.",
  notes_graph_sync_not_ready: "Notes Sync is not ready for this decision.",
  notes_graph_invalid_response:
    "The Notes graph server returned an invalid response.",
  notes_graph_offline: "Notes graph changes are unavailable while offline."
} as const

export type NotesGraphSuggestionErrorCode = keyof typeof ERROR_MESSAGES

const CAPABILITY_UNAVAILABLE_REASONS = [
  "notes_graph_fts_not_ready",
  "notes_graph_provider_call_policy_unsupported",
  "notes_graph_provider_disallowed",
  "notes_graph_provider_model_disallowed",
  "notes_graph_provider_not_configured",
  "notes_graph_provider_retry_policy_unsupported",
  "notes_graph_provider_unavailable",
  "notes_graph_suggestions_disabled",
  "notes_graph_suggestions_worker_unavailable"
] as const

const RUN_ERROR_CODES = [
  "notes_graph_admission_failed",
  "notes_graph_capabilities_changed_before_queue",
  "notes_graph_capabilities_changed_before_provider",
  "notes_graph_fingerprint_stale",
  "notes_graph_fts_not_ready",
  "notes_graph_job_missing",
  "notes_graph_publication_receipt_mismatch",
  "notes_graph_publication_receipt_missing",
  "notes_graph_publication_state_missing",
  "notes_graph_provider_retry_policy_unsupported",
  "notes_graph_provider_unavailable",
  "notes_graph_source_changed",
  "notes_graph_source_too_large",
  "notes_graph_suggestion_no_valid_items",
  "notes_graph_suggestion_suppression_limit",
  "notes_graph_target_changed"
] as const

const RUN_GUIDANCE_KEYS = [
  "configure_provider",
  "contact_administrator",
  "refresh_note",
  "retry_generation"
] as const

export class NotesGraphSuggestionClientError extends Error {
  readonly status: number
  readonly code: NotesGraphSuggestionErrorCode
  readonly retryAfterMs: number | null

  constructor(
    status: number,
    code: NotesGraphSuggestionErrorCode,
    message = ERROR_MESSAGES[code],
    retryAfterMs: number | null = null
  ) {
    super(message)
    this.name = "NotesGraphSuggestionClientError"
    this.status = status
    this.code = code
    this.retryAfterMs = retryAfterMs
  }
}

export type NotesGraphEdgeType =
  | "manual"
  | "wikilink"
  | "backlink"
  | "tag_membership"
  | "source_membership"
  | "semantic"

export type NotesSemanticExcerpt = {
  field: "title" | "content"
  start_code_point: number
  end_code_point: number
  text: string
}

export type NotesSemanticEdgeEvidence = {
  similarity: number
  qualitative_band: "low" | "moderate" | "high" | "very_high"
  source_note_id: string
  target_note_id: string
  source_content_version: number
  target_content_version: number
  generation_id: string
  semantic_index_revision: number
  configuration_revision: number
  normalization_version: string
  chunker_version: string
  provider_label: string
  model_label: string
  model_revision: string | null
  excerpt_pairs: Array<{
    source: NotesSemanticExcerpt
    target: NotesSemanticExcerpt
  }>
}

export type NotesSemanticGraphStatus = {
  available: boolean
  state:
    | "off"
    | "preparing"
    | "ready"
    | "updating"
    | "needs_attention"
    | "unavailable"
    | "focus_required"
  detail_reason: string | null
  generation_id: string | null
  semantic_index_revision: number | null
  configuration_revision: number | null
  active_notes: number
  indexed_notes: number
  dirty_notes: number
  excluded_notes: number
  failed_notes: number
  effective_top_k: number | null
  effective_threshold: number | null
  max_top_k: number
  max_admission_nodes: number
  max_admission_edges: number
  max_evidence_pairs: number
  max_excerpt_code_points: number
  max_edge_evidence_code_points: number
  max_response_evidence_bytes: number
  truncated_by: Array<
    | "semantic_candidates"
    | "semantic_nodes"
    | "semantic_edges"
    | "semantic_evidence_bytes"
  >
}

export type NotesGraphNode = {
  id: string
  type: "note" | "tag" | "source"
  label: string
  created_at: string | null
  deleted: boolean | null
  degree: number | null
  tag_count: number | null
  primary_source_id: string | null
}

export type NotesGraphEdge = {
  id: string
  source: string
  target: string
  type: NotesGraphEdgeType
  directed: boolean
  weight: number | null
  label: string | null
  evidence?: NotesSemanticEdgeEvidence
  evidence_omitted?: "response_byte_cap"
}

export type NotesGraphResponse = {
  nodes: NotesGraphNode[]
  edges: NotesGraphEdge[]
  truncated: boolean
  truncated_by: string[]
  has_more: boolean
  cursor: string | null
  limits: {
    max_nodes: number
    max_edges: number
    max_degree: number
  }
  radius_cap_applied: boolean
  active_note_count: number
  all_notes_note_cap: number
  all_notes_eligible: boolean
  suggestions_authorized?: boolean
  manual_link_authorized: boolean
  semantic_status?: NotesSemanticGraphStatus
}

export type FetchNotesGraphInput = {
  centerNoteId?: string
  datasetId?: string
  radius?: 1 | 2
  edgeTypes?: NotesGraphEdgeType[]
  maxNodes?: number
  maxEdges?: number
  maxDegree?: number
  cursor?: string
  semanticTopK?: number
  semanticThreshold?: number
}

export type CreateSemanticManualLinkInput = {
  sourceNoteId: string
  targetNoteId: string
  datasetId?: string
  generationId: string
  idempotencyKey: string
}

export type NotesManualLinkMutationResponse = {
  status: "created"
  edge: {
    edge_id: string
    from_note_id: string
    to_note_id: string
  }
}

export type NotesGraphSuggestionCapabilityLimits = {
  max_candidates: number
  max_relationships: number
  max_tags: number
  max_new_tags: number
  max_tag_catalog: number
  max_estimated_input_tokens: number
  max_output_tokens: number
  provider_timeout_seconds: number
  response_candidates: 1
}

export type NotesGraphCapabilityUnavailableReason =
  (typeof CAPABILITY_UNAVAILABLE_REASONS)[number]

export type NotesGraphSuggestionRunErrorCode = (typeof RUN_ERROR_CODES)[number]

export type NotesGraphSuggestionRunGuidanceKey =
  (typeof RUN_GUIDANCE_KEYS)[number]

export type NotesGraphSuggestionCapabilities = {
  provider: string
  model: string
  endpoint_origin_revision: string
  data_boundary: "local" | "remote" | "unknown"
  disclosure_external: boolean
  outbound_data_categories: NotesGraphOutboundDataCategory[]
  generation_available: boolean
  unavailable_reason: NotesGraphCapabilityUnavailableReason | null
  limits: NotesGraphSuggestionCapabilityLimits
  allowed_actions: NotesGraphSuggestionAction[]
  revision: string
  etag: string
}

export type NotesGraphSuggestionRun = {
  id: string
  provider: string
  model: string
  state: NotesGraphSuggestionRunState
  revision: number
  created_at: string
  started_at: string | null
  completed_at: string | null
  suggestion_count: number
  related_note_count: number
  tag_count: number
  invalid_item_count: number
  cancellation_available: boolean
  error_code: NotesGraphSuggestionRunErrorCode | null
  guidance_key: NotesGraphSuggestionRunGuidanceKey | null
}

export type NotesGraphSuggestionEvidence = {
  side: "source" | "target"
  note_id: string
  field: "title" | "content"
  start_offset: number
  end_offset: number
  text: string
}

export type NotesGraphSuggestion = {
  id: string
  run_id: string
  kind: "related_note" | "tag"
  state: NotesGraphSuggestionState
  revision: number
  source_note_id: string
  source_fingerprint: string
  target_note_id: string | null
  target_fingerprint: string | null
  target_title: string | null
  normalized_tag: string | null
  display_tag: string | null
  existing_tag: boolean
  match_strength: NotesGraphMatchStrength | null
  rationale: string | null
  evidence: NotesGraphSuggestionEvidence[]
  updated_at: string
}

export type NotesGraphSuggestionRunPage = {
  items: NotesGraphSuggestionRun[]
  next_cursor: string | null
}

export type NotesGraphSuggestionPage = {
  items: NotesGraphSuggestion[]
  next_cursor: string | null
  current_source_fingerprint: string
  rejection_set_revision: number
  rejection_count: number
}

export type NotesGraphSuggestionMutation = {
  resource_id: string
  state: NotesGraphSuggestionMutationState
  revision: number
  cleared_count: number | null
}

type NoteScope = {
  noteId: string
  datasetId?: string
}

export type GetNotesGraphSuggestionCapabilitiesInput = NoteScope & {
  provider?: string
  model?: string
}

export type CreateNotesGraphSuggestionCommandInput = NoteScope & {
  provider?: string
  model?: string
}

export type CreateNotesGraphSuggestionCommand =
  CreateNotesGraphSuggestionCommandInput & {
    idempotencyKey: string
  }

export type ListNotesGraphSuggestionRunsInput = NoteScope & {
  states?: NotesGraphSuggestionRunState[]
  limit?: number
  cursor?: string
}

export type GetNotesGraphSuggestionRunInput = NoteScope & {
  runId: string
}

export type ListNotesGraphSuggestionsInput = NoteScope & {
  states?: NotesGraphSuggestionState[]
  limit?: number
  cursor?: string
}

export type CancelNotesGraphSuggestionRunInput =
  GetNotesGraphSuggestionRunInput & {
    expectedRevision: number
    idempotencyKey: string
  }

export type DecideNotesGraphSuggestionInput = NoteScope & {
  suggestionId: string
  expectedRevision: number
  expectedSourceFingerprint: string
  expectedTargetFingerprint?: string | null
  idempotencyKey: string
}

export type ResetNotesGraphSuggestionRejectionsInput = NoteScope & {
  expectedRejectionRevision: number
  sourceFingerprint: string
  idempotencyKey: string
}

export type NotesGraphOutboundDataCategory =
  | "selected_note_title"
  | "selected_note_excerpts"
  | "candidate_note_titles"
  | "candidate_note_excerpts"
  | "existing_tag_labels"

export type NotesGraphSuggestionAction =
  | "generate"
  | "cancel"
  | "accept"
  | "reject"
  | "reset_rejections"

export type NotesGraphSuggestionRunState =
  | "admitting"
  | "queued"
  | "running"
  | "cancelling"
  | "publishing"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "stale"

export type NotesGraphSuggestionState =
  | "staged"
  | "pending"
  | "accepting"
  | "accepted"
  | "rejected"
  | "stale"

export type NotesGraphMatchStrength = "strong" | "possible"

export type NotesGraphSuggestionMutationState =
  | "cancelling"
  | "cancelled"
  | "succeeded"
  | "failed"
  | "stale"
  | "accepted"
  | "rejected"
  | "reset"
  | "completed"

const invalidResponse = (): NotesGraphSuggestionClientError =>
  new NotesGraphSuggestionClientError(502, "notes_graph_invalid_response")

const invalidRequest = (): NotesGraphSuggestionClientError =>
  new NotesGraphSuggestionClientError(422, "notes_graph_invalid_request")

const boundedTextSchema = (maximum: number) =>
  z.string().refine((value) => Array.from(value).length <= maximum)

const idSchema = z.string().min(1)
const inputIdSchema = z.string().trim().min(1)
const boundedInputTextSchema = (maximum: number) =>
  inputIdSchema.refine((value) => Array.from(value).length <= maximum)
const datasetIdSchema = boundedInputTextSchema(256)
const idempotencyKeySchema = boundedInputTextSchema(256)
const providerInputSchema = boundedInputTextSchema(128)
const modelInputSchema = boundedInputTextSchema(256)
const fingerprintSchema = z.string().regex(FINGERPRINT)
const cursorSchema = z.string().max(MAX_CURSOR_LENGTH)
const safeCountSchema = z.number().int().min(0)
const positiveRevisionSchema = z.number().int().min(1)
const nonnegativeRevisionSchema = z.number().int().min(0)

const edgeTypeSchema = z.enum([
  "manual",
  "wikilink",
  "backlink",
  "tag_membership",
  "source_membership",
  "semantic"
])
const runStateSchema = z.enum([
  "admitting",
  "queued",
  "running",
  "cancelling",
  "publishing",
  "succeeded",
  "failed",
  "cancelled",
  "stale"
])
const suggestionStateSchema = z.enum([
  "staged",
  "pending",
  "accepting",
  "accepted",
  "rejected",
  "stale"
])
const mutationStateSchema = z.enum([
  "cancelling",
  "cancelled",
  "succeeded",
  "failed",
  "stale",
  "accepted",
  "rejected",
  "reset",
  "completed"
])
const actionSchema = z.enum([
  "generate",
  "cancel",
  "accept",
  "reject",
  "reset_rejections"
])
const outboundCategorySchema = z.enum([
  "selected_note_title",
  "selected_note_excerpts",
  "candidate_note_titles",
  "candidate_note_excerpts",
  "existing_tag_labels"
])

const parseResponse = <T>(schema: z.ZodType<T>, value: unknown): T => {
  const parsed = schema.safeParse(value)
  if (!parsed.success) throw invalidResponse()
  return parsed.data
}

// The package compiles without strictNullChecks, which makes Zod object outputs
// appear optional. Runtime validation remains authoritative for these public types.
const parseResponseAs = <T>(schema: z.ZodType, value: unknown): T =>
  parseResponse(schema, value) as T

const parseInput = <T>(schema: z.ZodType<T>, value: unknown): T => {
  const parsed = schema.safeParse(value)
  if (!parsed.success) throw invalidRequest()
  return parsed.data
}

const normalizedId = (value: string): string => parseInput(inputIdSchema, value)

const pathId = (value: string): string =>
  encodeURIComponent(normalizedId(value))

const queryPath = (
  path: string,
  entries: Array<[string, string | number | undefined]>
): ReturnType<typeof toAllowedPath> => {
  const params = new URLSearchParams()
  entries.forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value))
  })
  return appendPathQuery(
    toAllowedPath(path),
    params.size ? `?${params.toString()}` : ""
  )
}

const retryAfter = (value: unknown): number | null => {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.min(Math.trunc(value), 3_600_000)
    : null
}

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const errorCode = (value: unknown): NotesGraphSuggestionErrorCode => {
  const details = record(record(value).details)
  const data = record(record(value).data)
  const detail = record(details.detail ?? data.detail)
  const candidate = detail.error_code ?? details.error_code ?? data.error_code
  return typeof candidate === "string" &&
    Object.prototype.hasOwnProperty.call(ERROR_MESSAGES, candidate)
    ? (candidate as NotesGraphSuggestionErrorCode)
    : "notes_graph_suggestions_unavailable"
}

const clientError = (value: unknown): NotesGraphSuggestionClientError => {
  if (value instanceof NotesGraphSuggestionClientError) return value
  const source = record(value)
  const status =
    typeof source.status === "number" &&
    Number.isInteger(source.status) &&
    source.status >= 0 &&
    source.status <= 599
      ? source.status
      : 503
  const code = errorCode(value)
  return new NotesGraphSuggestionClientError(
    status,
    code,
    ERROR_MESSAGES[code],
    retryAfter(source.retryAfterMs)
  )
}

const request = async <T>(
  init: Parameters<typeof bgRequest<T>>[0]
): Promise<T> => {
  try {
    return await bgRequest<T>(init)
  } catch (error) {
    throw clientError(error)
  }
}

const graphNodeSchema = z.strictObject({
  id: idSchema,
  type: z.enum(["note", "tag", "source"]),
  label: z.string(),
  created_at: z.string().nullable(),
  deleted: z.boolean().nullable(),
  degree: z.number().int().min(0).nullable(),
  tag_count: z.number().int().min(0).nullable(),
  primary_source_id: idSchema.nullable()
})

const semanticExcerptSchema = z
  .strictObject({
    field: z.enum(["title", "content"]),
    start_code_point: nonnegativeRevisionSchema,
    end_code_point: positiveRevisionSchema,
    text: boundedTextSchema(480)
  })
  .superRefine((value, context) => {
    if (
      value.end_code_point <= value.start_code_point ||
      Array.from(value.text).length !==
        value.end_code_point - value.start_code_point
    ) {
      context.addIssue({ code: "custom", message: "invalid excerpt offsets" })
    }
  })

const semanticEvidenceSchema = z.strictObject({
  similarity: z.number().finite().min(0).max(1),
  qualitative_band: z.enum(["low", "moderate", "high", "very_high"]),
  source_note_id: idSchema,
  target_note_id: idSchema,
  source_content_version: positiveRevisionSchema,
  target_content_version: positiveRevisionSchema,
  generation_id: idSchema,
  semantic_index_revision: nonnegativeRevisionSchema,
  configuration_revision: nonnegativeRevisionSchema,
  normalization_version: idSchema,
  chunker_version: idSchema,
  provider_label: idSchema,
  model_label: idSchema,
  model_revision: idSchema.nullable(),
  excerpt_pairs: z
    .array(
      z.strictObject({
        source: semanticExcerptSchema,
        target: semanticExcerptSchema
      })
    )
    .max(3)
})

const graphEdgeSchema = z
  .strictObject({
    id: idSchema,
    source: idSchema,
    target: idSchema,
    type: edgeTypeSchema,
    directed: z.boolean(),
    weight: z.number().finite().min(0).nullable(),
    label: z.string().nullable(),
    evidence: semanticEvidenceSchema.optional(),
    evidence_omitted: z.literal("response_byte_cap").optional()
  })
  .superRefine((value, context) => {
    if (value.type === "semantic") {
      if (value.directed) {
        context.addIssue({ code: "custom", message: "semantic edge directed" })
      }
      if (Boolean(value.evidence) === Boolean(value.evidence_omitted)) {
        context.addIssue({
          code: "custom",
          message: "semantic evidence invalid"
        })
      }
      if (
        value.evidence &&
        (value.evidence.source_note_id !== value.source ||
          value.evidence.target_note_id !== value.target)
      ) {
        context.addIssue({
          code: "custom",
          message: "semantic evidence mismatch"
        })
      }
    } else if (value.evidence || value.evidence_omitted) {
      context.addIssue({ code: "custom", message: "ordinary edge evidence" })
    }
  })

const semanticGraphStatusSchema = z.strictObject({
  available: z.boolean(),
  state: z.enum([
    "off",
    "preparing",
    "ready",
    "updating",
    "needs_attention",
    "unavailable",
    "focus_required"
  ]),
  detail_reason: z.string().nullable(),
  generation_id: idSchema.nullable(),
  semantic_index_revision: nonnegativeRevisionSchema.nullable(),
  configuration_revision: nonnegativeRevisionSchema.nullable(),
  active_notes: safeCountSchema,
  indexed_notes: safeCountSchema,
  dirty_notes: safeCountSchema,
  excluded_notes: safeCountSchema,
  failed_notes: safeCountSchema,
  effective_top_k: z
    .number()
    .int()
    .min(1)
    .max(NOTES_GRAPH_SEMANTIC_MAX_TOP_K)
    .nullable(),
  effective_threshold: z.number().finite().min(0).max(1).nullable(),
  max_top_k: z.number().int().min(1).max(NOTES_GRAPH_SEMANTIC_MAX_TOP_K),
  max_admission_nodes: z
    .number()
    .int()
    .min(0)
    .max(NOTES_GRAPH_SEMANTIC_MAX_TOP_K),
  max_admission_edges: z
    .number()
    .int()
    .min(0)
    .max(NOTES_GRAPH_SEMANTIC_MAX_TOP_K),
  max_evidence_pairs: z.number().int().min(0).max(3),
  max_excerpt_code_points: z.number().int().min(0).max(480),
  max_edge_evidence_code_points: z.number().int().min(0).max(2880),
  max_response_evidence_bytes: z.number().int().min(0).max(262144),
  truncated_by: z.array(
    z.enum([
      "semantic_candidates",
      "semantic_nodes",
      "semantic_edges",
      "semantic_evidence_bytes"
    ])
  )
})

const graphResponseSchema = z
  .strictObject({
    nodes: z.array(graphNodeSchema),
    edges: z.array(graphEdgeSchema),
    truncated: z.boolean(),
    truncated_by: z.array(z.string()),
    has_more: z.boolean(),
    cursor: cursorSchema.nullable(),
    limits: z.strictObject({
      max_nodes: z.number().int().min(1),
      max_edges: z.number().int().min(0),
      max_degree: z.number().int().min(1)
    }),
    radius_cap_applied: z.boolean(),
    active_note_count: safeCountSchema,
    all_notes_note_cap: z.number().int().min(1),
    all_notes_eligible: z.boolean(),
    suggestions_authorized: z.boolean().optional(),
    manual_link_authorized: z.boolean().optional().default(false),
    semantic_status: semanticGraphStatusSchema.optional()
  })
  .superRefine((value, context) => {
    if (value.nodes.length > value.limits.max_nodes) {
      context.addIssue({ code: "custom", message: "node limit exceeded" })
    }
    if (value.edges.length > value.limits.max_edges) {
      context.addIssue({ code: "custom", message: "edge limit exceeded" })
    }
  })

const graphInputSchema = z
  .strictObject({
    centerNoteId: inputIdSchema.optional(),
    datasetId: datasetIdSchema.optional(),
    radius: z
      .union([z.literal(1), z.literal(2)])
      .optional()
      .default(1),
    edgeTypes: z.array(edgeTypeSchema).optional().default([]),
    maxNodes: z.number().int().min(1).optional().default(120),
    maxEdges: z.number().int().min(0).optional().default(480),
    maxDegree: z.number().int().min(1).optional(),
    cursor: cursorSchema.optional(),
    semanticTopK: z
      .number()
      .int()
      .min(1)
      .max(NOTES_GRAPH_SEMANTIC_MAX_TOP_K)
      .optional(),
    semanticThreshold: z.number().finite().min(0).max(1).optional()
  })
  .superRefine((value, context) => {
    if (
      (value.semanticTopK !== undefined ||
        value.semanticThreshold !== undefined) &&
      !value.edgeTypes.includes("semantic")
    ) {
      context.addIssue({
        code: "custom",
        message: "semantic controls require semantic"
      })
    }
  })

const normalizeGraph = (value: unknown): NotesGraphResponse => {
  return parseResponseAs<NotesGraphResponse>(graphResponseSchema, value)
}

export const fetchNotesGraph = async (
  input: FetchNotesGraphInput
): Promise<NotesGraphResponse> => {
  const parsed = parseInput(graphInputSchema, input)
  const payload = await request<unknown>({
    path: queryPath("/api/v1/notes/graph", [
      ["center_note_id", parsed.centerNoteId],
      ["dataset_id", parsed.datasetId],
      ["radius", parsed.radius],
      [
        "edge_types",
        parsed.edgeTypes.length ? parsed.edgeTypes.join(",") : undefined
      ],
      ["max_nodes", parsed.maxNodes],
      ["max_edges", parsed.maxEdges],
      ["semantic_top_k", parsed.semanticTopK],
      ["semantic_threshold", parsed.semanticThreshold],
      ["max_degree", parsed.maxDegree],
      ["cursor", parsed.cursor]
    ]),
    method: "GET"
  })
  return normalizeGraph(payload)
}

const manualLinkMutationSchema = z.object({
  status: z.literal("created"),
  edge: z.object({
    edge_id: idSchema,
    from_note_id: idSchema,
    to_note_id: idSchema
  })
})

export const createSemanticManualLink = async (
  input: CreateSemanticManualLinkInput
): Promise<NotesManualLinkMutationResponse> => {
  const parsed = parseInput(
    z.strictObject({
      sourceNoteId: inputIdSchema,
      targetNoteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      generationId: boundedInputTextSchema(256),
      idempotencyKey: boundedInputTextSchema(128)
    }),
    input
  )
  return parseResponseAs<NotesManualLinkMutationResponse>(
    manualLinkMutationSchema,
    await request<unknown>({
      path: toAllowedPath(`/api/v1/notes/${pathId(parsed.sourceNoteId)}/links`),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": parsed.idempotencyKey
      },
      body: {
        to_note_id: parsed.targetNoteId,
        directed: false,
        weight: 1,
        dataset_id: parsed.datasetId,
        idempotency_key: parsed.idempotencyKey,
        semantic_conversion: { generation_id: parsed.generationId }
      }
    })
  )
}

const capabilitySchema = z.strictObject({
  provider: z.string(),
  model: z.string(),
  endpoint_origin_revision: fingerprintSchema,
  data_boundary: z.enum(["local", "remote", "unknown"]),
  disclosure_external: z.boolean(),
  outbound_data_categories: z.array(outboundCategorySchema),
  generation_available: z.boolean(),
  unavailable_reason: z.enum(CAPABILITY_UNAVAILABLE_REASONS).nullable(),
  limits: z.strictObject({
    max_candidates: z.number().int().min(1).max(30),
    max_relationships: z.number().int().min(1).max(5),
    max_tags: z.number().int().min(1).max(5),
    max_new_tags: z.number().int().min(1).max(2),
    max_tag_catalog: z.number().int().min(1).max(100),
    max_estimated_input_tokens: z.number().int().min(1).max(24_000),
    max_output_tokens: z.number().int().min(1).max(2_000),
    provider_timeout_seconds: z.number().int().min(1).max(120),
    response_candidates: z.literal(1)
  }),
  allowed_actions: z.array(actionSchema),
  revision: fingerprintSchema
})

const normalizeCapabilities = (
  payload: unknown,
  headers: Record<string, string> | undefined
): NotesGraphSuggestionCapabilities => {
  const capability = parseResponseAs<
    Omit<NotesGraphSuggestionCapabilities, "etag">
  >(capabilitySchema, payload)
  const rawEtag = new Headers(headers).get("etag") ?? ""
  const match = rawEtag.match(STRONG_ETAG)
  if (!match || match[1] !== capability.revision) throw invalidResponse()
  return {
    ...capability,
    etag: rawEtag
  }
}

export const getNotesGraphSuggestionCapabilities = async (
  input: GetNotesGraphSuggestionCapabilitiesInput
): Promise<NotesGraphSuggestionCapabilities> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      provider: providerInputSchema.optional(),
      model: modelInputSchema.optional()
    }),
    input
  )
  const response = await request<ApiSendResponse<unknown>>({
    path: queryPath(
      `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/capabilities`,
      [
        ["provider", parsed.provider],
        ["model", parsed.model],
        ["dataset_id", parsed.datasetId]
      ]
    ),
    method: "GET",
    returnResponse: true
  })
  if (!response?.ok || response.status < 200 || response.status >= 300) {
    throw clientError(response)
  }
  return normalizeCapabilities(response.data, response.headers)
}

const fallbackUuid = (): string => {
  const bytes = new Uint8Array(16)
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"))
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`
}

const newIdempotencyKey = (): string => {
  try {
    const value = globalThis.crypto?.randomUUID?.()
    if (value) return value
  } catch {
    // Use a process-local fallback only when randomUUID is unavailable.
  }
  return fallbackUuid()
}

export const createNotesGraphSuggestionCommand = (
  input: CreateNotesGraphSuggestionCommandInput
): CreateNotesGraphSuggestionCommand => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      provider: providerInputSchema.optional(),
      model: modelInputSchema.optional()
    }),
    input
  )
  return { ...parsed, idempotencyKey: newIdempotencyKey() }
}

const runSchema = z.strictObject({
  id: idSchema,
  provider: z.string(),
  model: z.string(),
  state: runStateSchema,
  revision: positiveRevisionSchema,
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  suggestion_count: safeCountSchema,
  related_note_count: safeCountSchema,
  tag_count: safeCountSchema,
  invalid_item_count: safeCountSchema,
  cancellation_available: z.boolean(),
  error_code: z.enum(RUN_ERROR_CODES).nullable(),
  guidance_key: z.enum(RUN_GUIDANCE_KEYS).nullable()
})

const normalizeRun = (value: unknown): NotesGraphSuggestionRun =>
  parseResponseAs<NotesGraphSuggestionRun>(runSchema, value)

const runCommandSchema = z.strictObject({
  noteId: inputIdSchema,
  datasetId: datasetIdSchema.optional(),
  provider: providerInputSchema.optional(),
  model: modelInputSchema.optional(),
  idempotencyKey: idempotencyKeySchema
})

const postRun = async (
  command: CreateNotesGraphSuggestionCommand,
  capability: NotesGraphSuggestionCapabilities
): Promise<NotesGraphSuggestionRun> => {
  const parsed = parseInput(runCommandSchema, command)
  const payload = await request<unknown>({
    path: queryPath(
      `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/runs`,
      [["dataset_id", parsed.datasetId]]
    ),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": parsed.idempotencyKey,
      "If-Match": capability.etag
    },
    body: {
      provider: parsed.provider ?? capability.provider,
      model: parsed.model ?? capability.model
    }
  })
  return normalizeRun(payload)
}

export const createNotesGraphSuggestionRun = async (
  command: CreateNotesGraphSuggestionCommand,
  capability: NotesGraphSuggestionCapabilities,
  options?: {
    canRetry?: () => boolean
    onCapabilitiesChanged?: (value: NotesGraphSuggestionCapabilities) => void
  }
): Promise<NotesGraphSuggestionRun> => {
  try {
    return await postRun(command, capability)
  } catch (error) {
    if (!isNotesGraphCapabilitiesChangedError(error)) throw error
    if (options?.canRetry?.() === false) throw error
    const refreshed = await getNotesGraphSuggestionCapabilities({
      noteId: command.noteId,
      datasetId: command.datasetId,
      provider: command.provider,
      model: command.model
    })
    if (options?.canRetry?.() === false) throw error
    options?.onCapabilitiesChanged?.(refreshed)
    if (options?.canRetry?.() === false) throw error
    return await postRun(command, refreshed)
  }
}

export const listNotesGraphSuggestionRuns = async (
  input: ListNotesGraphSuggestionRunsInput
): Promise<NotesGraphSuggestionRunPage> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      states: z.array(runStateSchema).optional(),
      limit: z.number().int().min(1).max(MAX_LIST_ITEMS).optional().default(20),
      cursor: cursorSchema.optional()
    }),
    input
  )
  const payload = parseResponseAs<NotesGraphSuggestionRunPage>(
    z.strictObject({
      items: z.array(runSchema).max(MAX_LIST_ITEMS),
      next_cursor: cursorSchema.nullable()
    }),
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/runs`,
        [
          ["state", parsed.states?.join(",")],
          ["limit", parsed.limit],
          ["cursor", parsed.cursor],
          ["dataset_id", parsed.datasetId]
        ]
      ),
      method: "GET"
    })
  )
  return payload
}

export const getNotesGraphSuggestionRun = async (
  input: GetNotesGraphSuggestionRunInput
): Promise<NotesGraphSuggestionRun> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      runId: inputIdSchema
    }),
    input
  )
  return normalizeRun(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/runs/${pathId(parsed.runId)}`,
        [["dataset_id", parsed.datasetId]]
      ),
      method: "GET"
    })
  )
}

const mutationSchema = z.strictObject({
  resource_id: idSchema,
  state: mutationStateSchema,
  revision: nonnegativeRevisionSchema,
  cleared_count: safeCountSchema.nullable()
})

const mutation = (value: unknown): NotesGraphSuggestionMutation =>
  parseResponseAs<NotesGraphSuggestionMutation>(mutationSchema, value)

export const cancelNotesGraphSuggestionRun = async (
  input: CancelNotesGraphSuggestionRunInput
): Promise<NotesGraphSuggestionMutation> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      runId: inputIdSchema,
      expectedRevision: positiveRevisionSchema,
      idempotencyKey: idempotencyKeySchema
    }),
    input
  )
  return mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/runs/${pathId(parsed.runId)}/cancel`,
        [["dataset_id", parsed.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": parsed.idempotencyKey
      },
      body: {
        expected_revision: parsed.expectedRevision
      }
    })
  )
}

const evidenceSchema = z
  .strictObject({
    side: z.enum(["source", "target"]),
    note_id: idSchema,
    field: z.enum(["title", "content"]),
    start_offset: safeCountSchema,
    end_offset: z.number().int().min(1),
    text: boundedTextSchema(480)
  })
  .refine((value) => value.end_offset > value.start_offset)

const suggestionSchema = z
  .strictObject({
    id: idSchema,
    run_id: idSchema,
    kind: z.enum(["related_note", "tag"]),
    state: suggestionStateSchema,
    revision: positiveRevisionSchema,
    source_note_id: idSchema,
    source_fingerprint: fingerprintSchema,
    target_note_id: idSchema.nullable(),
    target_fingerprint: fingerprintSchema.nullable(),
    target_title: z.string().nullable(),
    normalized_tag: z.string().nullable(),
    display_tag: z.string().nullable(),
    existing_tag: z.boolean(),
    match_strength: z.enum(["strong", "possible"]).nullable(),
    rationale: boundedTextSchema(240).nullable(),
    evidence: z.array(evidenceSchema).max(6),
    updated_at: z.string()
  })
  .superRefine((value, context) => {
    if (
      value.kind === "related_note" &&
      (value.target_note_id === null || value.target_fingerprint === null)
    ) {
      context.addIssue({
        code: "custom",
        message: "missing related note target"
      })
    }
    if (value.kind === "tag" && value.normalized_tag === null) {
      context.addIssue({ code: "custom", message: "missing normalized tag" })
    }
    if (value.kind === "tag" && value.target_title !== null) {
      context.addIssue({
        code: "custom",
        message: "tag target title must be null"
      })
    }
  })

const suggestionPageSchema = z.strictObject({
  items: z.array(suggestionSchema).max(MAX_LIST_ITEMS),
  next_cursor: cursorSchema.nullable(),
  current_source_fingerprint: fingerprintSchema,
  rejection_set_revision: nonnegativeRevisionSchema,
  rejection_count: safeCountSchema
})

export const listNotesGraphSuggestions = async (
  input: ListNotesGraphSuggestionsInput
): Promise<NotesGraphSuggestionPage> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      states: z.array(suggestionStateSchema).optional(),
      limit: z.number().int().min(1).max(MAX_LIST_ITEMS).optional().default(20),
      cursor: cursorSchema.optional()
    }),
    input
  )
  return parseResponseAs<NotesGraphSuggestionPage>(
    suggestionPageSchema,
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions`,
        [
          ["state", parsed.states?.join(",")],
          ["limit", parsed.limit],
          ["cursor", parsed.cursor],
          ["dataset_id", parsed.datasetId]
        ]
      ),
      method: "GET"
    })
  )
}

const decide = async (
  action: "accept" | "reject",
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      suggestionId: inputIdSchema,
      expectedRevision: positiveRevisionSchema,
      expectedSourceFingerprint: fingerprintSchema,
      expectedTargetFingerprint: fingerprintSchema.nullable().optional(),
      idempotencyKey: idempotencyKeySchema
    }),
    input
  )
  return mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/${pathId(parsed.suggestionId)}/${action}`,
        [["dataset_id", parsed.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": parsed.idempotencyKey
      },
      body: {
        expected_revision: parsed.expectedRevision,
        expected_source_fingerprint: parsed.expectedSourceFingerprint,
        expected_target_fingerprint: parsed.expectedTargetFingerprint ?? null
      }
    })
  )
}

export const acceptNotesGraphSuggestion = async (
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> => decide("accept", input)

export const rejectNotesGraphSuggestion = async (
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> => decide("reject", input)

export const resetNotesGraphSuggestionRejections = async (
  input: ResetNotesGraphSuggestionRejectionsInput
): Promise<NotesGraphSuggestionMutation> => {
  const parsed = parseInput(
    z.strictObject({
      noteId: inputIdSchema,
      datasetId: datasetIdSchema.optional(),
      expectedRejectionRevision: nonnegativeRevisionSchema,
      sourceFingerprint: fingerprintSchema,
      idempotencyKey: idempotencyKeySchema
    }),
    input
  )
  return mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(parsed.noteId)}/graph/suggestions/rejections/reset`,
        [["dataset_id", parsed.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": parsed.idempotencyKey
      },
      body: {
        expected_rejection_revision: parsed.expectedRejectionRevision,
        source_fingerprint: parsed.sourceFingerprint,
        confirm: true
      }
    })
  )
}

export const isNotesGraphCapabilitiesChangedError = (error: unknown): boolean =>
  error instanceof NotesGraphSuggestionClientError &&
  error.status === 412 &&
  error.code === "notes_graph_capabilities_changed"

export const createNotesGraphOfflineError =
  (): NotesGraphSuggestionClientError =>
    new NotesGraphSuggestionClientError(0, "notes_graph_offline")
