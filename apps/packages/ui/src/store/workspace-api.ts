/**
 * API-first workspace state helpers.
 * These functions enable server hydration and optimistic mutation
 * with rollback on 409 conflicts.
 */

import type {
  WorkspaceArtifactApiResponse,
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
  WorkspaceSourceType
} from "../types/workspace"

export interface ServerWorkspaceState {
  id: string
  name: string | null
  sources?: WorkspaceSourceApiResponse[]
  artifacts?: WorkspaceArtifactApiResponse[]
  notes?: any[]
  version: number
  [key: string]: unknown
}

export interface LocalWorkspaceState {
  id: string
  name: string
  sources: WorkspaceSource[]
  artifacts: GeneratedArtifact[]
  notes: any[]
  version: number
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
  typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined

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

const normalizeRedaction = (metadata: unknown): TraceableArtifactRedaction | undefined => {
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
    redactedFields: asStringArray(record.redactedFields ?? record.redacted_fields),
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
        fileId: pickValue(record, "fileId", "file_id") as number | string | undefined,
        jobId: pickValue(record, "jobId", "job_id") as number | string | undefined,
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

const mapServerSourceToLocal = (
  source: WorkspaceSourceApiResponse
): WorkspaceSource => ({
  id: source.id,
  mediaId: source.media_id,
  title: source.title,
  type: normalizeWorkspaceSourceType(source.source_type),
  status: "ready",
  url: source.url || undefined,
  addedAt: new Date(source.added_at)
})

const mapServerArtifactToLocal = (
  artifact: WorkspaceArtifactApiResponse
): GeneratedArtifact => {
  const exportRefs = normalizeExportRefs(artifact.export_refs)

  return {
    id: artifact.id,
    type: normalizeArtifactType(artifact.artifact_type),
    title: artifact.title,
    status: mapServerGenerationStatus(artifact),
    reviewStatus: mapServerReviewStatus(artifact.review_state || artifact.status),
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
 * Hydrate local workspace state from the server.
 * Called on workspace switch to ensure local state reflects server truth.
 */
export async function hydrateWorkspaceFromServer(
  workspaceId: string,
  deps: { fetch: (id: string) => Promise<ServerWorkspaceState> }
): Promise<LocalWorkspaceState> {
  const server = await deps.fetch(workspaceId)
  return {
    id: server.id,
    name: server.name ?? "",
    sources: (server.sources ?? []).map(mapServerSourceToLocal),
    artifacts: (server.artifacts ?? []).map(mapServerArtifactToLocal),
    notes: server.notes ?? [],
    version: server.version,
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
    const result = await deps.update(current.id, { ...updates, version: current.version })
    return result
  } catch (err: any) {
    if (err.status === 409 && err.body) {
      return err.body
    }
    throw err
  }
}
