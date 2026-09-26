/**
 * API-first workspace state helpers.
 * These functions enable server hydration and optimistic mutation
 * with rollback on 409 conflicts.
 */

import { z } from "zod"
import type {
  WorkspaceApiResponse,
  WorkspaceArtifactApiResponse,
  WorkspaceNoteApiResponse,
  WorkspaceSourceApiResponse
} from "../services/tldw/domains/workspace-api"
import type {
  ArtifactExportTarget,
  ArtifactReviewStatus,
  ArtifactSourceLineage,
  ArtifactStatus,
  ArtifactType,
  GeneratedArtifact,
  TraceableArtifactExportRef,
  TraceableArtifactProducerLinks,
  TraceableArtifactProducerMetadata,
  TraceableArtifactRedaction,
  TraceableArtifactReviewMetadata,
  TraceableArtifactVersionMetadata,
  WorkspaceSource,
  WorkspaceSourceReviewUpdate,
  WorkspaceSourceType
} from "../types/workspace"

export type OwnedWorkspaceBundle = {
  workspace: WorkspaceApiResponse
  sources: WorkspaceSourceApiResponse[]
  artifacts: WorkspaceArtifactApiResponse[]
  notes: WorkspaceNoteApiResponse[]
}

/** Readers must be bound to the verified account/server request context. */
export type OwnedWorkspaceReader = {
  getWorkspace(id: string): Promise<WorkspaceApiResponse>
  getWorkspaceSources(id: string): Promise<WorkspaceSourceApiResponse[]>
  getWorkspaceArtifacts(id: string): Promise<WorkspaceArtifactApiResponse[]>
  getWorkspaceNotes(id: string): Promise<WorkspaceNoteApiResponse[]>
}

export class OwnedWorkspaceLoadError extends Error {
  constructor(
    public readonly reason: "invalid-response" | "unavailable",
    public readonly resource: keyof OwnedWorkspaceBundle
  ) {
    super(`Workspace ${resource} ${reason}`)
    this.name = "OwnedWorkspaceLoadError"
  }
}

const nonemptyId = z.string().refine((value) => value.trim().length > 0)
const versionSchema = z.number().int().positive()
const timestampSchema = z
  .string()
  .refine((value) => Number.isFinite(Date.parse(value)))
const recordSchema = z.record(z.string(), z.unknown())
const optionalRecord = recordSchema.nullish()
const memoryModeSchema = z.enum(["read_only", "read_write"])
const defaultStatusSchema = z.enum(["available", "unavailable", "none"])
const defaultSourceSchema = z.enum(["workspace", "none"])
const degradedReasonSchema = z.enum([
  "persona_deleted",
  "persona_unavailable",
  "persona_feature_disabled",
  "permission_denied",
  "invalid_default",
  "unsupported_assistant_kind"
])

/** Match the backend's effective-default status relationships in either casing. */
function validEffectiveDefault(value: {
  status: string
  assistantKind?: string | null
  assistantId?: string | null
  label?: string | null
  personaMemoryMode?: string | null
  degradedReason?: string | null
}): boolean {
  if (value.status === "available") {
    return (
      value.assistantKind != null &&
      value.assistantId != null &&
      value.degradedReason == null
    )
  }
  if (value.status === "unavailable") return value.degradedReason != null
  return [
    value.assistantKind,
    value.assistantId,
    value.label,
    value.personaMemoryMode,
    value.degradedReason
  ].every((field) => field == null)
}

const workspaceSchema = z
  .object({
    id: nonemptyId,
    name: z.string().nullable(),
    archived: z.boolean(),
    deleted: z.boolean(),
    workspace_profile: z.enum(["research", "project"]),
    study_materials_policy: z.enum(["general", "workspace"]),
    banner_title: z.string().nullable(),
    banner_subtitle: z.string().nullable(),
    banner_color: z.string().nullable(),
    audio_provider: z.string().nullable(),
    audio_model: z.string().nullable(),
    audio_voice: z.string().nullable(),
    audio_speed: z.number().finite().nullable(),
    created_at: timestampSchema,
    last_modified: timestampSchema,
    version: versionSchema,
    assistant_defaults: z
      .object({
        assistant_kind: z.literal("persona"),
        assistant_id: nonemptyId,
        persona_memory_mode: memoryModeSchema.optional(),
        voice: z.null().optional(),
        style: z.null().optional(),
        tool_policy_profile_id: z.null().optional()
      })
      .passthrough()
      .nullish(),
    assistantDefaults: z
      .object({
        assistantKind: z.literal("persona"),
        assistantId: nonemptyId,
        personaMemoryMode: memoryModeSchema,
        voice: z.null(),
        style: z.null(),
        toolPolicyProfileId: z.null()
      })
      .passthrough()
      .nullish(),
    effective_assistant_default: z
      .object({
        status: defaultStatusSchema,
        source: defaultSourceSchema,
        assistant_kind: z.literal("persona").nullish(),
        assistant_id: z.string().nullish(),
        label: z.string().nullish(),
        persona_memory_mode: memoryModeSchema.nullish(),
        degraded_reason: degradedReasonSchema.nullish()
      })
      .passthrough()
      .refine((value) =>
        validEffectiveDefault({
          status: value.status,
          assistantKind: value.assistant_kind,
          assistantId: value.assistant_id,
          label: value.label,
          personaMemoryMode: value.persona_memory_mode,
          degradedReason: value.degraded_reason
        })
      )
      .nullish(),
    effectiveAssistantDefault: z
      .object({
        status: defaultStatusSchema,
        source: defaultSourceSchema,
        assistantKind: z.literal("persona").nullable(),
        assistantId: z.string().nullable(),
        label: z.string().nullable(),
        personaMemoryMode: memoryModeSchema.nullable(),
        degradedReason: degradedReasonSchema.nullable()
      })
      .passthrough()
      .refine(validEffectiveDefault)
      .optional()
  })
  .passthrough()

const sourceSchema = z
  .object({
    id: nonemptyId,
    workspace_id: nonemptyId,
    media_id: z.number().int().positive(),
    title: z.string(),
    source_type: z.string(),
    url: z.string().nullable(),
    position: z.number().int(),
    selected: z.boolean(),
    added_at: timestampSchema,
    version: versionSchema,
    review_state: z.enum(["unset", "reviewed", "needs_review"]).optional(),
    review_state_updated_at: timestampSchema.nullish(),
    reviewed_at: timestampSchema.nullish(),
    reviewed_by_user_id: z.string().nullish()
  })
  .passthrough()

const artifactSchema = z
  .object({
    id: nonemptyId,
    workspace_id: nonemptyId,
    artifact_type: z.string(),
    title: z.string(),
    status: z.string(),
    content: z.string().nullable(),
    total_tokens: z.number().finite().nullable(),
    total_cost_usd: z.number().finite().nullable(),
    created_at: timestampSchema,
    completed_at: timestampSchema.nullable(),
    version: versionSchema,
    review_state: z.string().nullish(),
    content_type: z.string().nullish(),
    preview_text: z.string().nullish(),
    summary: z.string().nullish(),
    owner_scope: z.string().nullish(),
    owner_id: z.string().nullish(),
    project_id: z.string().nullish(),
    task_id: z.string().nullish(),
    source_collection_id: z.string().nullish(),
    root_artifact_id: z.string().nullish(),
    artifact_version_id: z.string().nullish(),
    previous_version_id: z.string().nullish(),
    schema_version: z.number().int().nullish(),
    producer_metadata: optionalRecord,
    review_metadata: optionalRecord,
    version_metadata: optionalRecord,
    redaction: optionalRecord,
    source_lineage: z.union([recordSchema, z.array(recordSchema)]).nullish(),
    export_refs: z.array(recordSchema).nullish()
  })
  .passthrough()

const noteSchema: z.ZodType<WorkspaceNoteApiResponse> = z
  .object({
    id: z.number().int().positive(),
    workspace_id: nonemptyId,
    title: z.string(),
    content: z.string(),
    keywords_json: z.string().refine((value) => {
      try {
        const keywords: unknown = JSON.parse(value)
        return (
          Array.isArray(keywords) &&
          keywords.every((keyword) => typeof keyword === "string")
        )
      } catch {
        return false
      }
    }),
    created_at: timestampSchema,
    last_modified: timestampSchema,
    version: versionSchema
  })
  .passthrough()

function validateOwnedCollection(
  rows: unknown,
  schema: z.ZodType<{ id: string | number; workspace_id: string }>,
  workspaceId: string,
  resource: "sources" | "artifacts" | "notes"
): void {
  const parsed = z.array(schema).safeParse(rows)
  if (!parsed.success)
    throw new OwnedWorkspaceLoadError("invalid-response", resource)
  const identities = new Set<string | number>()
  for (const row of parsed.data) {
    if (row.workspace_id !== workspaceId || identities.has(row.id)) {
      throw new OwnedWorkspaceLoadError("invalid-response", resource)
    }
    identities.add(row.id)
  }
}

export function validateOwnedWorkspaceNotes(
  rows: unknown,
  workspaceId: string
): asserts rows is WorkspaceNoteApiResponse[] {
  validateOwnedCollection(rows, noteSchema, workspaceId, "notes")
}

/** Reject promptly even when the underlying reader cannot abort its transport. */
function readUntilAborted<T>(
  signal: AbortSignal,
  read: () => Promise<T>
): Promise<T> {
  signal.throwIfAborted()
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason)
    signal.addEventListener("abort", onAbort, { once: true })
    const cleanup = () => signal.removeEventListener("abort", onAbort)
    try {
      read().then(
        (value) => {
          cleanup()
          if (signal.aborted) reject(signal.reason)
          else resolve(value)
        },
        (error) => {
          cleanup()
          reject(error)
        }
      )
    } catch (error) {
      cleanup()
      reject(error)
    }
  })
}

/** Validate metadata without masking malformed or unavailable canonical rows. */
export function validateOwnedWorkspaceRecord(
  workspace: unknown,
  id: string
): asserts workspace is WorkspaceApiResponse {
  if (
    !workspaceSchema.safeParse(workspace).success ||
    (workspace as WorkspaceApiResponse).id !== id
  ) {
    throw new OwnedWorkspaceLoadError("invalid-response", "workspace")
  }
  const metadata = workspace as WorkspaceApiResponse
  if (metadata.deleted) {
    throw new OwnedWorkspaceLoadError("unavailable", "workspace")
  }
}

/** Archived rows are readable for lifecycle review, never editable activations. */
export function validateOwnedWorkspaceMetadata(
  workspace: unknown,
  id: string
): asserts workspace is WorkspaceApiResponse {
  validateOwnedWorkspaceRecord(workspace, id)
  if (workspace.archived)
    throw new OwnedWorkspaceLoadError("unavailable", "workspace")
}

/** Load a complete authorized bundle without mutating server or browser state. */
export async function loadOwnedWorkspace(
  id: string,
  reader: OwnedWorkspaceReader,
  signal: AbortSignal
): Promise<OwnedWorkspaceBundle> {
  signal.throwIfAborted()
  if (!nonemptyId.safeParse(id).success) {
    throw new OwnedWorkspaceLoadError("invalid-response", "workspace")
  }
  const workspace = await readUntilAborted(signal, () =>
    reader.getWorkspace(id)
  )
  validateOwnedWorkspaceMetadata(workspace, id)
  const [sources, artifacts, notes] = await readUntilAborted(signal, () =>
    Promise.all([
      reader.getWorkspaceSources(id),
      reader.getWorkspaceArtifacts(id),
      reader.getWorkspaceNotes(id)
    ])
  )
  signal.throwIfAborted()
  validateOwnedCollection(sources, sourceSchema, id, "sources")
  validateOwnedCollection(artifacts, artifactSchema, id, "artifacts")
  validateOwnedWorkspaceNotes(notes, id)
  // Keep the original typed payload, including optional metadata and effective defaults.
  return { workspace, sources, artifacts, notes }
}

const workspaceSourceTypes = new Set<WorkspaceSourceType>([
  "pdf",
  "video",
  "audio",
  "website",
  "document",
  "text"
])

const artifactTypes = new Set<ArtifactType>([
  "summary",
  "audio_overview",
  "mindmap",
  "report",
  "compare_sources",
  "flashcards",
  "quiz",
  "timeline",
  "slides",
  "data_table"
])

const generationStatuses = new Set<ArtifactStatus>([
  "pending",
  "generating",
  "completed",
  "failed"
])

const reviewStatuses = new Set<ArtifactReviewStatus>([
  "draft",
  "reviewing",
  "accepted",
  "needs_revision",
  "rejected",
  "exported",
  "assigned",
  "archived"
])

const exportTargetAliases: Record<string, ArtifactExportTarget> = {
  md: "markdown",
  markdown: "markdown",
  docx: "docx",
  pdf: "pdf",
  ppt: "slides",
  pptx: "slides",
  presentation: "slides",
  slides: "slides",
  chatbook: "chatbook"
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  isRecord(value) ? value : undefined

const asString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : undefined

const asNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined

const asStringArray = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const strings = value
    .map((item) => asString(item))
    .filter((item): item is string => Boolean(item))
  return strings.length > 0 ? strings : undefined
}

const pickString = (
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined => {
  for (const key of keys) {
    const value = asString(record[key])
    if (value) return value
  }
  return undefined
}

const pickNumber = (
  record: Record<string, unknown>,
  ...keys: string[]
): number | undefined => {
  for (const key of keys) {
    const value = asNumber(record[key])
    if (value !== undefined) return value
  }
  return undefined
}

const pickValue = (
  record: Record<string, unknown>,
  ...keys: string[]
): unknown => {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) {
      return record[key]
    }
  }
  return undefined
}

const normalizeWorkspaceSourceType = (
  sourceType: string
): WorkspaceSourceType =>
  workspaceSourceTypes.has(sourceType as WorkspaceSourceType)
    ? (sourceType as WorkspaceSourceType)
    : "document"

const normalizeArtifactType = (artifactType: string): ArtifactType =>
  artifactTypes.has(artifactType as ArtifactType)
    ? (artifactType as ArtifactType)
    : "report"

const mapServerGenerationStatus = (
  artifact: WorkspaceArtifactApiResponse
): ArtifactStatus => {
  if (generationStatuses.has(artifact.status as ArtifactStatus)) {
    return artifact.status as ArtifactStatus
  }

  if (artifact.completed_at || (artifact.content?.trim().length ?? 0) > 0) {
    return "completed"
  }

  return "pending"
}

const mapServerReviewStatus = (
  status: string
): ArtifactReviewStatus | undefined =>
  reviewStatuses.has(status as ArtifactReviewStatus)
    ? (status as ArtifactReviewStatus)
    : undefined

const normalizeProducerLinks = (
  links: unknown
): TraceableArtifactProducerLinks | undefined => {
  const record = asRecord(links)
  if (!record) return undefined

  const normalized: TraceableArtifactProducerLinks = {}
  for (const [key, value] of Object.entries(record)) {
    const url = asString(value)
    if (url) normalized[key] = url
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined
}

const normalizeProducerMetadata = (
  metadata: unknown
): TraceableArtifactProducerMetadata | undefined => {
  const record = asRecord(metadata)
  if (!record) return undefined

  return {
    ...record,
    producerType: pickString(record, "producerType", "producer_type"),
    producerId: pickString(record, "producerId", "producer_id"),
    runId: pickString(record, "runId", "run_id"),
    sessionId: pickString(record, "sessionId", "session_id"),
    reviewId: pickString(record, "reviewId", "review_id"),
    taskId: pickString(record, "taskId", "task_id"),
    promptId: pickString(record, "promptId", "prompt_id"),
    templateId: pickString(record, "templateId", "template_id"),
    model: pickString(record, "model", "model_id", "modelId"),
    provider: pickString(record, "provider", "provider_id", "providerId"),
    completionReason: pickString(
      record,
      "completionReason",
      "completion_reason"
    ),
    links: normalizeProducerLinks(record.links)
  }
}

const normalizeReviewMetadata = (
  metadata: unknown
): TraceableArtifactReviewMetadata | undefined => {
  const record = asRecord(metadata)
  if (!record) return undefined

  return {
    ...record,
    reviewerId: pickString(record, "reviewerId", "reviewer_id"),
    decision: pickString(record, "decision", "review_state"),
    decidedAt: pickString(record, "decidedAt", "decided_at"),
    reason: pickString(record, "reason", "revision_reason", "rejection_reason")
  }
}

const normalizeVersionMetadata = (
  metadata: unknown
): TraceableArtifactVersionMetadata | undefined => {
  const record = asRecord(metadata)
  if (!record) return undefined

  return {
    ...record,
    revisionReason: pickString(record, "revisionReason", "revision_reason"),
    versionLabel: pickString(record, "versionLabel", "version_label"),
    comparedToVersionId: pickString(
      record,
      "comparedToVersionId",
      "compared_to_version_id"
    )
  }
}

const normalizeRedaction = (
  metadata: unknown
): TraceableArtifactRedaction | undefined => {
  const record = asRecord(metadata)
  if (!record) return undefined

  return {
    ...record,
    supportSafe:
      typeof pickValue(record, "supportSafe", "support_safe") === "boolean"
        ? (pickValue(record, "supportSafe", "support_safe") as boolean)
        : undefined,
    redacted:
      typeof pickValue(record, "redacted") === "boolean"
        ? (pickValue(record, "redacted") as boolean)
        : undefined,
    retentionClass: pickString(record, "retentionClass", "retention_class"),
    redactedFields: asStringArray(
      record.redactedFields ?? record.redacted_fields
    ),
    visibility: pickString(record, "visibility")
  }
}

const normalizeSourceLineage = (
  lineage: unknown
): ArtifactSourceLineage[] | undefined => {
  const entries = Array.isArray(lineage)
    ? lineage
    : isRecord(lineage)
      ? Array.isArray(lineage.sources)
        ? lineage.sources
        : Array.isArray(lineage.source_refs)
          ? lineage.source_refs
          : []
      : []

  const normalized = entries
    .map((entry, index): ArtifactSourceLineage | null => {
      const record = asRecord(entry)
      if (!record) {
        const sourceId = asString(entry)
        return sourceId ? { sourceId } : null
      }

      const citationSpans = Array.isArray(
        record.citationSpans ?? record.citation_spans
      )
        ? ((record.citationSpans ?? record.citation_spans) as unknown[])
        : undefined
      const sourceId =
        pickString(record, "sourceId", "source_id", "id") ||
        `source-${index + 1}`
      const title = pickString(record, "title", "label", "name")

      return {
        ...record,
        sourceId,
        sourceType: pickString(record, "sourceType", "source_type", "type"),
        mediaId: pickNumber(record, "mediaId", "media_id"),
        title,
        label: pickString(record, "label", "title", "name"),
        citationCount:
          pickNumber(record, "citationCount", "citation_count") ??
          (citationSpans && citationSpans.length > 0
            ? citationSpans.length
            : undefined),
        citationSpans,
        evidenceIds: asStringArray(record.evidenceIds ?? record.evidence_ids),
        coverageNotes: pickString(record, "coverageNotes", "coverage_notes")
      }
    })
    .filter((entry): entry is ArtifactSourceLineage => entry !== null)

  return normalized.length > 0 ? normalized : undefined
}

const normalizeExportFormat = (format: unknown): string | undefined => {
  const value = asString(format)?.toLowerCase()
  if (!value) return undefined
  return exportTargetAliases[value] || value
}

const normalizeExportRefs = (
  refs: unknown
): TraceableArtifactExportRef[] | undefined => {
  if (!Array.isArray(refs)) return undefined

  const normalized = refs
    .map((entry): TraceableArtifactExportRef | null => {
      const record = asRecord(entry)
      if (!record) return null

      const format = normalizeExportFormat(
        pickValue(record, "format", "target", "type")
      )
      if (!format) return null

      return {
        ...record,
        id: pickValue(record, "id", "export_id", "exportId") as
          | number
          | string
          | undefined,
        format,
        fileId: pickValue(record, "fileId", "file_id") as
          | number
          | string
          | undefined,
        jobId: pickValue(record, "jobId", "job_id") as
          | number
          | string
          | undefined,
        artifactVersionId: pickString(
          record,
          "artifactVersionId",
          "artifact_version_id"
        ),
        generatedAt: pickString(record, "generatedAt", "generated_at"),
        expiresAt: pickString(record, "expiresAt", "expires_at"),
        status: pickString(record, "status"),
        url: pickString(record, "url"),
        error: pickString(record, "error")
      }
    })
    .filter((entry): entry is TraceableArtifactExportRef => entry !== null)

  return normalized.length > 0 ? normalized : undefined
}

const normalizeExportTargets = (
  refs: TraceableArtifactExportRef[] | undefined
): ArtifactExportTarget[] | undefined => {
  const targets = new Set<ArtifactExportTarget>()
  refs?.forEach((ref) => {
    const target = exportTargetAliases[ref.format.toLowerCase()]
    if (target) targets.add(target)
  })
  return targets.size > 0 ? Array.from(targets) : undefined
}

export const mapServerSourceReviewFields = (
  source: WorkspaceSourceApiResponse
): Omit<WorkspaceSourceReviewUpdate, "id"> => ({
  reviewState: source.review_state ?? "unset",
  reviewStateUpdatedAt: source.review_state_updated_at
    ? new Date(source.review_state_updated_at)
    : undefined,
  reviewedAt: source.reviewed_at ? new Date(source.reviewed_at) : undefined,
  reviewedByUserId: source.reviewed_by_user_id || undefined
})

export const mapServerSourceToLocal = (
  source: WorkspaceSourceApiResponse
): WorkspaceSource => ({
  id: source.id,
  mediaId: source.media_id,
  title: source.title,
  type: normalizeWorkspaceSourceType(source.source_type),
  status: "ready",
  url: source.url || undefined,
  addedAt: new Date(source.added_at),
  ...mapServerSourceReviewFields(source)
})

export const mapServerArtifactToLocal = (
  artifact: WorkspaceArtifactApiResponse
): GeneratedArtifact => {
  const exportRefs = normalizeExportRefs(artifact.export_refs)

  return {
    id: artifact.id,
    type: normalizeArtifactType(artifact.artifact_type),
    title: artifact.title,
    status: mapServerGenerationStatus(artifact),
    reviewStatus: mapServerReviewStatus(
      artifact.review_state || artifact.status
    ),
    serverId: artifact.id,
    content: artifact.content || undefined,
    contentType: artifact.content_type || undefined,
    previewText: artifact.preview_text || undefined,
    summary: artifact.summary || undefined,
    totalTokens: artifact.total_tokens ?? undefined,
    totalCostUsd: artifact.total_cost_usd ?? undefined,
    ownerScope: artifact.owner_scope || undefined,
    ownerId: artifact.owner_id || undefined,
    projectId: artifact.project_id || undefined,
    taskId: artifact.task_id || undefined,
    sourceCollectionId: artifact.source_collection_id || undefined,
    rootArtifactId: artifact.root_artifact_id || undefined,
    artifactVersionId: artifact.artifact_version_id || undefined,
    previousVersionId: artifact.previous_version_id || undefined,
    schemaVersion: artifact.schema_version ?? undefined,
    version: artifact.version,
    producerMetadata: normalizeProducerMetadata(artifact.producer_metadata),
    sourceLineage: normalizeSourceLineage(artifact.source_lineage),
    reviewMetadata: normalizeReviewMetadata(artifact.review_metadata),
    versionMetadata: normalizeVersionMetadata(artifact.version_metadata),
    exportRefs,
    exportTargets: normalizeExportTargets(exportRefs),
    redaction: normalizeRedaction(artifact.redaction),
    createdAt: new Date(artifact.created_at),
    completedAt: artifact.completed_at
      ? new Date(artifact.completed_at)
      : undefined
  }
}

/**
 * Perform an optimistic workspace update.
 * On success, returns the server's updated state.
 * On 409 conflict, returns the server's current state (rollback).
 */
export async function optimisticWorkspaceUpdate(
  current: { id: string; name: string; version: number },
  updates: Record<string, unknown>,
  deps: { update: (id: string, body: any) => Promise<any> }
): Promise<{ name: string; version: number; [key: string]: unknown }> {
  try {
    const result = await deps.update(current.id, {
      ...updates,
      version: current.version
    })
    return result
  } catch (err: any) {
    if (err.status === 409 && err.body) {
      return err.body
    }
    throw err
  }
}
