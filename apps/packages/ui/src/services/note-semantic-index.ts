import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"
import { z } from "zod"

const ERROR_MESSAGES = {
  notes_semantic_active_generation_required:
    "An active semantic generation is required.",
  notes_semantic_backend_change_requires_delete:
    "Delete the existing semantic index before changing vector storage.",
  notes_semantic_capability_revision_conflict:
    "Semantic capabilities changed; refresh and retry.",
  notes_semantic_configuration_revision_conflict:
    "The semantic index changed; refresh and retry.",
  notes_semantic_dataset_authority_unavailable:
    "Semantic dataset authority is temporarily unavailable.",
  notes_semantic_dataset_not_found: "The semantic dataset was not found.",
  notes_semantic_idempotency_conflict:
    "The idempotency key was reused for another request.",
  notes_semantic_invalid_request: "The semantic index request is invalid.",
  notes_semantic_invalid_response:
    "The semantic index server returned an invalid response.",
  notes_semantic_jobs_unavailable:
    "Semantic indexing is temporarily unavailable.",
  notes_semantic_offline:
    "Semantic index changes are unavailable while offline.",
  notes_semantic_permission_denied:
    "You do not have permission to manage the semantic index.",
  notes_semantic_provider_unavailable:
    "Semantic indexing is temporarily unavailable.",
  notes_semantic_quota_exceeded:
    "The semantic indexing quota has been reached.",
  notes_semantic_run_not_found: "The requested semantic run was not found.",
  notes_semantic_run_revision_conflict:
    "The semantic run changed; refresh and retry.",
  notes_semantic_writer_conflict:
    "Another semantic index operation is already active."
} as const

export type NotesSemanticErrorCode = keyof typeof ERROR_MESSAGES

export class NotesSemanticClientError extends Error {
  readonly status: number
  readonly code: NotesSemanticErrorCode

  constructor(
    status: number,
    code: NotesSemanticErrorCode,
    message = ERROR_MESSAGES[code]
  ) {
    super(message)
    this.name = "NotesSemanticClientError"
    this.status = status
    this.code = code
  }
}

export const NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES = [
  "note_content_chunks",
  "note_title"
] as const

export type NotesSemanticOutboundDataCategory =
  (typeof NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES)[number]

export type NotesSemanticCapabilities = {
  active_note_count: number
  estimated_chunk_count: number
  estimated_run_count: number
  provider_label: string
  model: string
  endpoint_display: string
  execution_boundary: "external" | "local"
  storage_boundary: "external" | "local" | "unavailable"
  storage_label: string
  outbound_data_categories: NotesSemanticOutboundDataCategory[]
  capability_revision: string
  indexing_available: boolean
  unavailable_reason: string | null
  metric: "cosine"
  resolved_dimensions: number | null
  dimension_probe_required: boolean
  renewal_requires_delete: boolean
  manage_authorized: boolean
}

export type NotesSemanticRunStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled"
  | "quarantined"

export type NotesSemanticRun = {
  run_id: string
  mode: string
  status: NotesSemanticRunStatus
  revision: number
  indexed_notes: number
  excluded_notes: number
  failed_notes: number
  pending_notes: number
  published_chunks: number
  cleanup_complete: boolean
  error_code: string | null
  link: string
}

export type NotesSemanticIndexStatus = {
  state: "off" | "preparing" | "ready" | "updating" | "needs_attention"
  detail_reason: string | null
  desired_state: "enabled" | "disabled"
  configuration_revision: number
  semantic_index_revision: number
  active_generation_id: string | null
  active_generation_usable: boolean
  indexed_notes: number
  excluded_notes: number
  failed_notes: number
  pending_notes: number
  published_chunks: number
  cleanup_pending: boolean
  active_run: NotesSemanticRun | null
}

export type NotesSemanticMutation = {
  resource: NotesSemanticIndexStatus
  run: NotesSemanticRun
}

type DatasetScope = { datasetId?: string }
type IdempotentCommand = DatasetScope & { idempotencyKey: string }

export type EnableNotesSemanticIndexInput = IdempotentCommand & {
  expectedRevision: number
  capabilityRevision: string
}
export type DeleteNotesSemanticIndexInput = IdempotentCommand & {
  expectedRevision: number
}
export type CreateNotesSemanticRunInput = DeleteNotesSemanticIndexInput & {
  mode: "rebuild" | "retry_failed"
}
export type GetNotesSemanticRunInput = DatasetScope & { runId: string }
export type CancelNotesSemanticRunInput = DeleteNotesSemanticIndexInput & {
  runId: string
}

const nonnegative = z.number().int().min(0)
const nonempty = z.string().min(1)
const inputText = z.string().trim().min(1)
const datasetInput = inputText.refine(
  (value) => Array.from(value).length <= 256
)
const idempotencyInput = inputText.refine(
  (value) => new TextEncoder().encode(value).length <= 256
)
const runStatusSchema = z.enum([
  "queued",
  "processing",
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])
const outboundCategorySchema = z.enum(NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES)
const boundedIdentity = (maximum: number) =>
  z.string().trim().min(1).max(maximum)
const endpointDisplaySchema = boundedIdentity(512).superRefine(
  (value, context) => {
    try {
      const endpoint = new URL(value)
      const defaultPort = endpoint.protocol === "https:" ? "443" : "80"
      const matchesSanitizedOrigin =
        value === endpoint.origin ||
        value === `${endpoint.origin}:${defaultPort}`
      if (
        !["http:", "https:"].includes(endpoint.protocol) ||
        endpoint.username ||
        endpoint.password ||
        endpoint.pathname !== "/" ||
        endpoint.search ||
        endpoint.hash ||
        !matchesSanitizedOrigin
      ) {
        throw new Error("endpoint must be a sanitized HTTP origin")
      }
    } catch {
      context.addIssue({
        code: "custom",
        message: "sanitized endpoint origin required"
      })
    }
  }
)

const runSchema = z.strictObject({
  run_id: nonempty,
  mode: nonempty,
  status: runStatusSchema,
  revision: nonnegative,
  indexed_notes: nonnegative,
  excluded_notes: nonnegative,
  failed_notes: nonnegative,
  pending_notes: nonnegative,
  published_chunks: nonnegative,
  cleanup_complete: z.boolean(),
  error_code: nonempty.nullable(),
  link: nonempty
})

const statusSchema: z.ZodType<NotesSemanticIndexStatus> = z.strictObject({
  state: z.enum(["off", "preparing", "ready", "updating", "needs_attention"]),
  detail_reason: nonempty.nullable(),
  desired_state: z.enum(["enabled", "disabled"]),
  configuration_revision: nonnegative,
  semantic_index_revision: nonnegative,
  active_generation_id: nonempty.nullable(),
  active_generation_usable: z.boolean(),
  indexed_notes: nonnegative,
  excluded_notes: nonnegative,
  failed_notes: nonnegative,
  pending_notes: nonnegative,
  published_chunks: nonnegative,
  cleanup_pending: z.boolean(),
  active_run: runSchema.nullable()
})

const capabilitiesSchema: z.ZodType<NotesSemanticCapabilities> = z
  .strictObject({
    active_note_count: nonnegative,
    estimated_chunk_count: nonnegative,
    estimated_run_count: nonnegative,
    provider_label: boundedIdentity(128),
    model: boundedIdentity(256),
    endpoint_display: endpointDisplaySchema,
    execution_boundary: z.enum(["external", "local"]),
    storage_boundary: z.enum(["external", "local", "unavailable"]),
    storage_label: boundedIdentity(128),
    outbound_data_categories: z.array(outboundCategorySchema).max(2),
    capability_revision: nonempty,
    indexing_available: z.boolean(),
    unavailable_reason: nonempty.nullable(),
    metric: z.literal("cosine"),
    resolved_dimensions: z.number().int().min(1).nullable(),
    dimension_probe_required: z.boolean(),
    renewal_requires_delete: z.boolean(),
    manage_authorized: z.boolean()
  })
  .superRefine((capability, context) => {
    const outbound = new Set(capability.outbound_data_categories)
    const completeOutboundDisclosure =
      capability.outbound_data_categories.length ===
        NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES.length &&
      NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES.every((category) =>
        outbound.has(category)
      )
    if (!completeOutboundDisclosure) {
      context.addIssue({
        code: "custom",
        path: ["outbound_data_categories"],
        message: "complete outbound disclosure required"
      })
    }
    const invalidProbeDisclosure =
      capability.dimension_probe_required &&
      (capability.resolved_dimensions !== null ||
        !capability.indexing_available ||
        capability.unavailable_reason !== null)
    const missingAvailableDimensionDisclosure =
      capability.indexing_available &&
      capability.resolved_dimensions === null &&
      !capability.dimension_probe_required
    if (invalidProbeDisclosure || missingAvailableDimensionDisclosure) {
      context.addIssue({
        code: "custom",
        path: ["dimension_probe_required"],
        message: "dimension disclosure is contradictory"
      })
    }
    if (!capability.indexing_available) return
    if (
      capability.storage_boundary === "unavailable" ||
      capability.unavailable_reason !== null
    ) {
      context.addIssue({
        code: "custom",
        message: "available capability disclosure is contradictory"
      })
    }
  })

const mutationSchema: z.ZodType<NotesSemanticMutation> = z.strictObject({
  resource: statusSchema,
  run: runSchema
})

const invalidRequest = () =>
  new NotesSemanticClientError(422, "notes_semantic_invalid_request")
const invalidResponse = () =>
  new NotesSemanticClientError(502, "notes_semantic_invalid_response")

const parseInput = <T>(schema: z.ZodType<T>, value: unknown): T => {
  const parsed = schema.safeParse(value)
  if (!parsed.success) throw invalidRequest()
  return parsed.data
}

const parseResponse = <T>(schema: z.ZodType<T>, value: unknown): T => {
  const parsed = schema.safeParse(value)
  if (!parsed.success) throw invalidResponse()
  return parsed.data
}

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const clientError = (value: unknown): NotesSemanticClientError => {
  if (value instanceof NotesSemanticClientError) return value
  const source = record(value)
  const status =
    typeof source.status === "number" && Number.isInteger(source.status)
      ? source.status
      : 503
  if (status === 403) {
    return new NotesSemanticClientError(403, "notes_semantic_permission_denied")
  }
  const details = record(source.details)
  const data = record(source.data)
  const detail = record(details.detail ?? data.detail)
  const candidate = detail.error_code ?? details.error_code ?? data.error_code
  const code =
    typeof candidate === "string" &&
    Object.prototype.hasOwnProperty.call(ERROR_MESSAGES, candidate)
      ? (candidate as NotesSemanticErrorCode)
      : "notes_semantic_provider_unavailable"
  return new NotesSemanticClientError(status, code)
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

const queryPath = (path: string, datasetId?: string) => {
  const params = new URLSearchParams()
  if (datasetId) params.set("dataset_id", datasetId)
  return appendPathQuery(
    toAllowedPath(path),
    params.size ? `?${params.toString()}` : ""
  )
}

const datasetSchema = z.strictObject({ datasetId: datasetInput.optional() })

export const getNotesSemanticCapabilities = async (
  input: DatasetScope
): Promise<NotesSemanticCapabilities> => {
  const parsed = parseInput(datasetSchema, input)
  return parseResponse(
    capabilitiesSchema,
    await request<unknown>({
      path: queryPath(
        "/api/v1/notes/graph/semantic-index/capabilities",
        parsed.datasetId
      ),
      method: "GET"
    })
  )
}

export const getNotesSemanticStatus = async (
  input: DatasetScope
): Promise<NotesSemanticIndexStatus> => {
  const parsed = parseInput(datasetSchema, input)
  return parseResponse(
    statusSchema,
    await request<unknown>({
      path: queryPath("/api/v1/notes/graph/semantic-index", parsed.datasetId),
      method: "GET"
    })
  )
}

const commandHeaders = (idempotencyKey: string) => ({
  "Content-Type": "application/json",
  "Idempotency-Key": idempotencyKey
})

export const enableNotesSemanticIndex = async (
  input: EnableNotesSemanticIndexInput
): Promise<NotesSemanticMutation> => {
  const parsed = parseInput(
    z.strictObject({
      datasetId: datasetInput.optional(),
      expectedRevision: nonnegative,
      capabilityRevision: inputText,
      idempotencyKey: idempotencyInput
    }),
    input
  )
  return parseResponse(
    mutationSchema,
    await request<unknown>({
      path: queryPath("/api/v1/notes/graph/semantic-index", parsed.datasetId),
      method: "PUT",
      headers: commandHeaders(parsed.idempotencyKey),
      body: {
        expected_revision: parsed.expectedRevision,
        capability_revision: parsed.capabilityRevision
      }
    })
  )
}

export const deleteNotesSemanticIndex = async (
  input: DeleteNotesSemanticIndexInput
): Promise<NotesSemanticMutation> => {
  const parsed = parseInput(
    z.strictObject({
      datasetId: datasetInput.optional(),
      expectedRevision: nonnegative,
      idempotencyKey: idempotencyInput
    }),
    input
  )
  return parseResponse(
    mutationSchema,
    await request<unknown>({
      path: queryPath("/api/v1/notes/graph/semantic-index", parsed.datasetId),
      method: "DELETE",
      headers: commandHeaders(parsed.idempotencyKey),
      body: { expected_revision: parsed.expectedRevision }
    })
  )
}

export const createNotesSemanticRun = async (
  input: CreateNotesSemanticRunInput
): Promise<NotesSemanticRun> => {
  const parsed = parseInput(
    z.strictObject({
      datasetId: datasetInput.optional(),
      mode: z.enum(["rebuild", "retry_failed"]),
      expectedRevision: nonnegative,
      idempotencyKey: idempotencyInput
    }),
    input
  )
  return parseResponse(
    runSchema,
    await request<unknown>({
      path: queryPath(
        "/api/v1/notes/graph/semantic-index/runs",
        parsed.datasetId
      ),
      method: "POST",
      headers: commandHeaders(parsed.idempotencyKey),
      body: {
        mode: parsed.mode,
        expected_revision: parsed.expectedRevision
      }
    })
  )
}

const pathId = (value: string) =>
  encodeURIComponent(parseInput(inputText, value))

export const getNotesSemanticRun = async (
  input: GetNotesSemanticRunInput
): Promise<NotesSemanticRun> => {
  const parsed = parseInput(
    z.strictObject({ datasetId: datasetInput.optional(), runId: inputText }),
    input
  )
  return parseResponse(
    runSchema,
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/graph/semantic-index/runs/${pathId(parsed.runId)}`,
        parsed.datasetId
      ),
      method: "GET"
    })
  )
}

export const cancelNotesSemanticRun = async (
  input: CancelNotesSemanticRunInput
): Promise<NotesSemanticMutation> => {
  const parsed = parseInput(
    z.strictObject({
      datasetId: datasetInput.optional(),
      runId: inputText,
      expectedRevision: nonnegative,
      idempotencyKey: idempotencyInput
    }),
    input
  )
  return parseResponse(
    mutationSchema,
    await request<unknown>({
      path: queryPath(
        `/api/v1/notes/graph/semantic-index/runs/${pathId(parsed.runId)}/cancel`,
        parsed.datasetId
      ),
      method: "POST",
      headers: commandHeaders(parsed.idempotencyKey),
      body: { expected_revision: parsed.expectedRevision }
    })
  )
}

const fallbackUuid = (): string => {
  const bytes = new Uint8Array(16)
  globalThis.crypto?.getRandomValues?.(bytes)
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"))
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`
}

export const createNotesSemanticCommand = (): { idempotencyKey: string } => ({
  idempotencyKey: globalThis.crypto?.randomUUID?.() ?? fallbackUuid()
})

export const createNotesSemanticOfflineError = () =>
  new NotesSemanticClientError(0, "notes_semantic_offline")
