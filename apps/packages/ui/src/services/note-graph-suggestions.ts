import type { ApiSendResponse } from "@/services/api-send"
import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

const FINGERPRINT = /^sha256:[0-9a-f]{64}$/
const STRONG_ETAG = /^"(sha256:[0-9a-f]{64})"$/
const MAX_LIST_ITEMS = 100
const MAX_GRAPH_NODES = 2000
const MAX_GRAPH_EDGES = 8000
const MAX_CURSOR_LENGTH = 4096

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

const CAPABILITY_UNAVAILABLE_REASONS = new Set([
  "notes_graph_fts_not_ready",
  "notes_graph_provider_call_policy_unsupported",
  "notes_graph_provider_disallowed",
  "notes_graph_provider_model_disallowed",
  "notes_graph_provider_not_configured",
  "notes_graph_provider_retry_policy_unsupported",
  "notes_graph_provider_unavailable",
  "notes_graph_suggestions_disabled",
  "notes_graph_suggestions_worker_unavailable"
])

const RUN_ERROR_CODES = new Set([
  "notes_graph_capabilities_changed_before_provider",
  "notes_graph_fingerprint_stale",
  "notes_graph_fts_not_ready",
  "notes_graph_provider_retry_policy_unsupported",
  "notes_graph_provider_unavailable",
  "notes_graph_source_too_large",
  "notes_graph_suggestion_no_valid_items",
  "notes_graph_suggestion_suppression_limit"
])

const RUN_GUIDANCE_KEYS = new Set([
  "configure_provider",
  "contact_administrator",
  "refresh_note",
  "retry_generation"
])

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

export type NotesGraphNode = {
  id: string
  type: "note" | "tag" | "source"
  label: string
  created_at?: string | null
  deleted?: boolean | null
  degree?: number | null
  tag_count?: number | null
  primary_source_id?: string | null
}

export type NotesGraphEdge = {
  id: string
  source: string
  target: string
  type: NotesGraphEdgeType
  directed: boolean
  weight: number | null
  label: string | null
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

export type NotesGraphSuggestionCapabilities = {
  provider: string
  model: string
  endpoint_origin_revision: string
  data_boundary: "local" | "remote" | "unknown"
  disclosure_external: boolean
  outbound_data_categories: string[]
  generation_available: boolean
  unavailable_reason: string | null
  limits: NotesGraphSuggestionCapabilityLimits
  allowed_actions: string[]
  revision: string
  etag: string
}

export type NotesGraphSuggestionRun = {
  id: string
  provider: string
  model: string
  state: string
  revision: number
  created_at: string
  started_at: string | null
  completed_at: string | null
  suggestion_count: number
  related_note_count: number
  tag_count: number
  invalid_item_count: number
  cancellation_available: boolean
  error_code: string | null
  guidance_key: string | null
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
  state: string
  revision: number
  source_note_id: string
  source_fingerprint: string
  target_note_id: string | null
  target_fingerprint: string | null
  normalized_tag: string | null
  display_tag: string | null
  existing_tag: boolean
  match_strength: string | null
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
  state: string
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
  states?: string[]
  limit?: number
  cursor?: string
}

export type GetNotesGraphSuggestionRunInput = NoteScope & {
  runId: string
}

export type ListNotesGraphSuggestionsInput = NoteScope & {
  states?: string[]
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

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const stringValue = (value: unknown, maximum: number, fallback = ""): string =>
  (typeof value === "string" ? value : fallback).slice(0, maximum)

const optionalString = (value: unknown, maximum: number): string | null =>
  typeof value === "string" ? value.slice(0, maximum) : null

const allowlistedString = (
  value: unknown,
  allowed: ReadonlySet<string>
): string | null =>
  typeof value === "string" && allowed.has(value) ? value : null

const integer = (
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number
): number => {
  const parsed = typeof value === "number" ? value : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.min(maximum, Math.max(minimum, Math.trunc(parsed)))
}

const requiredFingerprint = (value: unknown): string => {
  if (typeof value !== "string" || !FINGERPRINT.test(value)) {
    throw invalidResponse()
  }
  return value
}

const invalidResponse = (): NotesGraphSuggestionClientError =>
  new NotesGraphSuggestionClientError(502, "notes_graph_invalid_response")

const normalizedId = (value: string): string => {
  const result = String(value ?? "").trim()
  if (!result || result.length > 512)
    throw new NotesGraphSuggestionClientError(
      422,
      "notes_graph_invalid_request"
    )
  return result
}

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
  const parsed = typeof value === "number" ? value : Number.NaN
  return Number.isFinite(parsed) && parsed >= 0
    ? Math.min(Math.trunc(parsed), 3_600_000)
    : null
}

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
  const status = integer(source.status, 0, 599, 503)
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

const normalizeNode = (value: unknown): NotesGraphNode | null => {
  const source = record(value)
  const type = source.type
  if (type !== "note" && type !== "tag" && type !== "source") return null
  const id = stringValue(source.id, 512)
  if (!id) return null
  return {
    id,
    type,
    label: stringValue(source.label, 512),
    created_at: optionalString(source.created_at, 64),
    deleted: typeof source.deleted === "boolean" ? source.deleted : null,
    degree:
      source.degree == null
        ? null
        : integer(source.degree, 0, MAX_GRAPH_EDGES, 0),
    tag_count:
      source.tag_count == null
        ? null
        : integer(source.tag_count, 0, MAX_GRAPH_NODES, 0),
    primary_source_id: optionalString(source.primary_source_id, 512)
  }
}

const EDGE_TYPES = new Set<NotesGraphEdgeType>([
  "manual",
  "wikilink",
  "backlink",
  "tag_membership",
  "source_membership"
])

const normalizeEdge = (value: unknown): NotesGraphEdge | null => {
  const source = record(value)
  if (!EDGE_TYPES.has(source.type as NotesGraphEdgeType)) return null
  const id = stringValue(source.id, 512)
  const from = stringValue(source.source, 512)
  const target = stringValue(source.target, 512)
  if (!id || !from || !target) return null
  const weight =
    typeof source.weight === "number" && Number.isFinite(source.weight)
      ? Math.max(0, source.weight)
      : null
  return {
    id,
    source: from,
    target,
    type: source.type as NotesGraphEdgeType,
    directed: source.directed === true,
    weight,
    label: optionalString(source.label, 256)
  }
}

const normalizeGraph = (value: unknown): NotesGraphResponse => {
  const source = record(value)
  const rawLimits = record(source.limits)
  const limits = {
    max_nodes: integer(rawLimits.max_nodes, 1, MAX_GRAPH_NODES, 1),
    max_edges: integer(rawLimits.max_edges, 0, MAX_GRAPH_EDGES, 0),
    max_degree: integer(rawLimits.max_degree, 1, MAX_GRAPH_NODES, 1)
  }
  return {
    nodes: (Array.isArray(source.nodes) ? source.nodes : [])
      .map(normalizeNode)
      .filter((item): item is NotesGraphNode => item !== null)
      .slice(0, limits.max_nodes),
    edges: (Array.isArray(source.edges) ? source.edges : [])
      .map(normalizeEdge)
      .filter((item): item is NotesGraphEdge => item !== null)
      .slice(0, limits.max_edges),
    truncated: source.truncated === true,
    truncated_by: (Array.isArray(source.truncated_by)
      ? source.truncated_by
      : []
    )
      .map((item) => stringValue(item, 64))
      .filter(Boolean)
      .slice(0, 16),
    has_more: source.has_more === true,
    cursor: optionalString(source.cursor, MAX_CURSOR_LENGTH),
    limits,
    radius_cap_applied: source.radius_cap_applied === true,
    active_note_count: integer(
      source.active_note_count,
      0,
      Number.MAX_SAFE_INTEGER,
      0
    ),
    all_notes_note_cap: integer(
      source.all_notes_note_cap,
      1,
      MAX_GRAPH_NODES,
      1
    ),
    all_notes_eligible: source.all_notes_eligible === true
  }
}

export const fetchNotesGraph = async (
  input: FetchNotesGraphInput
): Promise<NotesGraphResponse> => {
  const edgeTypes = (input.edgeTypes ?? []).filter((item) =>
    EDGE_TYPES.has(item)
  )
  const payload = await request<unknown>({
    path: queryPath("/api/v1/notes/graph", [
      ["center_note_id", input.centerNoteId],
      ["dataset_id", input.datasetId],
      ["radius", input.radius ?? 1],
      ["edge_types", edgeTypes.length ? edgeTypes.join(",") : undefined],
      ["max_nodes", integer(input.maxNodes, 1, MAX_GRAPH_NODES, 120)],
      ["max_edges", integer(input.maxEdges, 0, MAX_GRAPH_EDGES, 480)],
      [
        "max_degree",
        input.maxDegree == null
          ? undefined
          : integer(input.maxDegree, 1, MAX_GRAPH_NODES, 40)
      ],
      ["cursor", input.cursor]
    ]),
    method: "GET"
  })
  return normalizeGraph(payload)
}

const normalizeCapabilities = (
  payload: unknown,
  headers: Record<string, string> | undefined
): NotesGraphSuggestionCapabilities => {
  const source = record(payload)
  const revision = requiredFingerprint(source.revision)
  const rawEtag = new Headers(headers).get("etag") ?? ""
  const match = rawEtag.match(STRONG_ETAG)
  if (!match || match[1] !== revision) throw invalidResponse()
  const rawLimits = record(source.limits)
  const boundary = source.data_boundary
  if (boundary !== "local" && boundary !== "remote" && boundary !== "unknown") {
    throw invalidResponse()
  }
  return {
    provider: stringValue(source.provider, 128),
    model: stringValue(source.model, 256),
    endpoint_origin_revision: requiredFingerprint(
      source.endpoint_origin_revision
    ),
    data_boundary: boundary,
    disclosure_external: source.disclosure_external === true,
    outbound_data_categories: (Array.isArray(source.outbound_data_categories)
      ? source.outbound_data_categories
      : []
    )
      .map((item) => stringValue(item, 128))
      .filter(Boolean)
      .slice(0, 32),
    generation_available: source.generation_available === true,
    unavailable_reason: allowlistedString(
      source.unavailable_reason,
      CAPABILITY_UNAVAILABLE_REASONS
    ),
    limits: {
      max_candidates: integer(rawLimits.max_candidates, 1, 30, 1),
      max_relationships: integer(rawLimits.max_relationships, 1, 5, 1),
      max_tags: integer(rawLimits.max_tags, 1, 5, 1),
      max_new_tags: integer(rawLimits.max_new_tags, 1, 2, 1),
      max_tag_catalog: integer(rawLimits.max_tag_catalog, 1, 100, 1),
      max_estimated_input_tokens: integer(
        rawLimits.max_estimated_input_tokens,
        1,
        24000,
        1
      ),
      max_output_tokens: integer(rawLimits.max_output_tokens, 1, 2000, 1),
      provider_timeout_seconds: integer(
        rawLimits.provider_timeout_seconds,
        1,
        120,
        1
      ),
      response_candidates: 1
    },
    allowed_actions: (Array.isArray(source.allowed_actions)
      ? source.allowed_actions
      : []
    )
      .map((item) => stringValue(item, 64))
      .filter(Boolean)
      .slice(0, 16),
    revision,
    etag: rawEtag
  }
}

export const getNotesGraphSuggestionCapabilities = async (
  input: GetNotesGraphSuggestionCapabilitiesInput
): Promise<NotesGraphSuggestionCapabilities> => {
  const response = await request<ApiSendResponse<unknown>>({
    path: queryPath(
      `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/capabilities`,
      [
        ["provider", input.provider],
        ["model", input.model],
        ["dataset_id", input.datasetId]
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
): CreateNotesGraphSuggestionCommand => ({
  noteId: normalizedId(input.noteId),
  ...(input.datasetId ? { datasetId: normalizedId(input.datasetId) } : {}),
  ...(input.provider ? { provider: input.provider.trim().slice(0, 128) } : {}),
  ...(input.model ? { model: input.model.trim().slice(0, 256) } : {}),
  idempotencyKey: newIdempotencyKey()
})

const normalizeRun = (value: unknown): NotesGraphSuggestionRun => {
  const source = record(value)
  const id = stringValue(source.id, 512)
  if (!id) throw invalidResponse()
  return {
    id,
    provider: stringValue(source.provider, 128),
    model: stringValue(source.model, 256),
    state: stringValue(source.state, 32),
    revision: integer(source.revision, 1, Number.MAX_SAFE_INTEGER, 1),
    created_at: stringValue(source.created_at, 64),
    started_at: optionalString(source.started_at, 64),
    completed_at: optionalString(source.completed_at, 64),
    suggestion_count: integer(source.suggestion_count, 0, MAX_LIST_ITEMS, 0),
    related_note_count: integer(
      source.related_note_count,
      0,
      MAX_LIST_ITEMS,
      0
    ),
    tag_count: integer(source.tag_count, 0, MAX_LIST_ITEMS, 0),
    invalid_item_count: integer(source.invalid_item_count, 0, 1000, 0),
    cancellation_available: source.cancellation_available === true,
    error_code: allowlistedString(source.error_code, RUN_ERROR_CODES),
    guidance_key: allowlistedString(source.guidance_key, RUN_GUIDANCE_KEYS)
  }
}

const postRun = async (
  command: CreateNotesGraphSuggestionCommand,
  capability: NotesGraphSuggestionCapabilities
): Promise<NotesGraphSuggestionRun> => {
  const payload = await request<unknown>({
    path: queryPath(
      `/api/v1/notes/${pathId(command.noteId)}/graph/suggestions/runs`,
      [["dataset_id", command.datasetId]]
    ),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": normalizedId(command.idempotencyKey),
      "If-Match": capability.etag
    },
    body: {
      provider: command.provider ?? capability.provider,
      model: command.model ?? capability.model
    }
  })
  return normalizeRun(payload)
}

export const createNotesGraphSuggestionRun = async (
  command: CreateNotesGraphSuggestionCommand,
  capability: NotesGraphSuggestionCapabilities,
  options?: {
    onCapabilitiesChanged?: (value: NotesGraphSuggestionCapabilities) => void
  }
): Promise<NotesGraphSuggestionRun> => {
  try {
    return await postRun(command, capability)
  } catch (error) {
    if (!isNotesGraphCapabilitiesChangedError(error)) throw error
    const refreshed = await getNotesGraphSuggestionCapabilities({
      noteId: command.noteId,
      datasetId: command.datasetId,
      provider: command.provider,
      model: command.model
    })
    options?.onCapabilitiesChanged?.(refreshed)
    return await postRun(command, refreshed)
  }
}

export const listNotesGraphSuggestionRuns = async (
  input: ListNotesGraphSuggestionRunsInput
): Promise<NotesGraphSuggestionRunPage> => {
  const payload = record(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/runs`,
        [
          [
            "state",
            input.states
              ?.map((item) => item.trim())
              .filter(Boolean)
              .join(",")
          ],
          ["limit", integer(input.limit, 1, MAX_LIST_ITEMS, 20)],
          ["cursor", input.cursor?.slice(0, MAX_CURSOR_LENGTH)],
          ["dataset_id", input.datasetId]
        ]
      ),
      method: "GET"
    })
  )
  return {
    items: (Array.isArray(payload.items) ? payload.items : [])
      .slice(0, MAX_LIST_ITEMS)
      .map(normalizeRun),
    next_cursor: optionalString(payload.next_cursor, MAX_CURSOR_LENGTH)
  }
}

export const getNotesGraphSuggestionRun = async (
  input: GetNotesGraphSuggestionRunInput
): Promise<NotesGraphSuggestionRun> =>
  normalizeRun(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/runs/${pathId(input.runId)}`,
        [["dataset_id", input.datasetId]]
      ),
      method: "GET"
    })
  )

const mutation = (value: unknown): NotesGraphSuggestionMutation => {
  const source = record(value)
  return {
    resource_id: stringValue(source.resource_id, 512),
    state: stringValue(source.state, 32),
    revision: integer(source.revision, 0, Number.MAX_SAFE_INTEGER, 0),
    cleared_count:
      source.cleared_count == null
        ? null
        : integer(source.cleared_count, 0, Number.MAX_SAFE_INTEGER, 0)
  }
}

export const cancelNotesGraphSuggestionRun = async (
  input: CancelNotesGraphSuggestionRunInput
): Promise<NotesGraphSuggestionMutation> =>
  mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/runs/${pathId(input.runId)}/cancel`,
        [["dataset_id", input.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": normalizedId(input.idempotencyKey)
      },
      body: {
        expected_revision: integer(
          input.expectedRevision,
          1,
          Number.MAX_SAFE_INTEGER,
          1
        )
      }
    })
  )

const normalizeEvidence = (
  value: unknown
): NotesGraphSuggestionEvidence | null => {
  const source = record(value)
  const side = source.side
  const field = source.field
  if (
    (side !== "source" && side !== "target") ||
    (field !== "title" && field !== "content")
  ) {
    return null
  }
  const start = integer(source.start_offset, 0, Number.MAX_SAFE_INTEGER, 0)
  const end = integer(source.end_offset, 1, Number.MAX_SAFE_INTEGER, 1)
  if (end <= start) return null
  return {
    side,
    note_id: stringValue(source.note_id, 512),
    field,
    start_offset: start,
    end_offset: end,
    text: stringValue(source.text, 480)
  }
}

const normalizeSuggestion = (value: unknown): NotesGraphSuggestion | null => {
  const source = record(value)
  const kind = source.kind
  if (kind !== "related_note" && kind !== "tag") return null
  const id = stringValue(source.id, 512)
  if (!id) return null
  try {
    return {
      id,
      run_id: stringValue(source.run_id, 512),
      kind,
      state: stringValue(source.state, 32),
      revision: integer(source.revision, 1, Number.MAX_SAFE_INTEGER, 1),
      source_note_id: stringValue(source.source_note_id, 512),
      source_fingerprint: requiredFingerprint(source.source_fingerprint),
      target_note_id: optionalString(source.target_note_id, 512),
      target_fingerprint:
        source.target_fingerprint == null
          ? null
          : requiredFingerprint(source.target_fingerprint),
      normalized_tag: optionalString(source.normalized_tag, 120),
      display_tag: optionalString(source.display_tag, 120),
      existing_tag: source.existing_tag === true,
      match_strength: optionalString(source.match_strength, 32),
      rationale: optionalString(source.rationale, 240),
      evidence: (Array.isArray(source.evidence) ? source.evidence : [])
        .map(normalizeEvidence)
        .filter((item): item is NotesGraphSuggestionEvidence => item !== null)
        .slice(0, 4),
      updated_at: stringValue(source.updated_at, 64)
    }
  } catch {
    return null
  }
}

export const listNotesGraphSuggestions = async (
  input: ListNotesGraphSuggestionsInput
): Promise<NotesGraphSuggestionPage> => {
  const payload = record(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions`,
        [
          [
            "state",
            input.states
              ?.map((item) => item.trim())
              .filter(Boolean)
              .join(",")
          ],
          ["limit", integer(input.limit, 1, MAX_LIST_ITEMS, 20)],
          ["cursor", input.cursor?.slice(0, MAX_CURSOR_LENGTH)],
          ["dataset_id", input.datasetId]
        ]
      ),
      method: "GET"
    })
  )
  return {
    items: (Array.isArray(payload.items) ? payload.items : [])
      .map(normalizeSuggestion)
      .filter((item): item is NotesGraphSuggestion => item !== null)
      .slice(0, MAX_LIST_ITEMS),
    next_cursor: optionalString(payload.next_cursor, MAX_CURSOR_LENGTH),
    current_source_fingerprint: requiredFingerprint(
      payload.current_source_fingerprint
    ),
    rejection_set_revision: integer(
      payload.rejection_set_revision,
      0,
      Number.MAX_SAFE_INTEGER,
      0
    ),
    rejection_count: integer(
      payload.rejection_count,
      0,
      Number.MAX_SAFE_INTEGER,
      0
    )
  }
}

const decide = async (
  action: "accept" | "reject",
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> =>
  mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/${pathId(input.suggestionId)}/${action}`,
        [["dataset_id", input.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": normalizedId(input.idempotencyKey)
      },
      body: {
        expected_revision: integer(
          input.expectedRevision,
          1,
          Number.MAX_SAFE_INTEGER,
          1
        ),
        expected_source_fingerprint: requiredFingerprint(
          input.expectedSourceFingerprint
        ),
        expected_target_fingerprint:
          input.expectedTargetFingerprint == null
            ? null
            : requiredFingerprint(input.expectedTargetFingerprint)
      }
    })
  )

export const acceptNotesGraphSuggestion = async (
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> => decide("accept", input)

export const rejectNotesGraphSuggestion = async (
  input: DecideNotesGraphSuggestionInput
): Promise<NotesGraphSuggestionMutation> => decide("reject", input)

export const resetNotesGraphSuggestionRejections = async (
  input: ResetNotesGraphSuggestionRejectionsInput
): Promise<NotesGraphSuggestionMutation> =>
  mutation(
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/${pathId(input.noteId)}/graph/suggestions/rejections/reset`,
        [["dataset_id", input.datasetId]]
      ),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": normalizedId(input.idempotencyKey)
      },
      body: {
        expected_rejection_revision: integer(
          input.expectedRejectionRevision,
          0,
          Number.MAX_SAFE_INTEGER,
          0
        ),
        source_fingerprint: requiredFingerprint(input.sourceFingerprint),
        confirm: true
      }
    })
  )

export const isNotesGraphCapabilitiesChangedError = (error: unknown): boolean =>
  error instanceof NotesGraphSuggestionClientError &&
  error.status === 412 &&
  error.code === "notes_graph_capabilities_changed"

export const createNotesGraphOfflineError =
  (): NotesGraphSuggestionClientError =>
    new NotesGraphSuggestionClientError(0, "notes_graph_offline")
