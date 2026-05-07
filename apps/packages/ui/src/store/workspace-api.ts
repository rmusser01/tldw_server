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
  ArtifactReviewStatus,
  ArtifactStatus,
  ArtifactType,
  GeneratedArtifact,
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
  "exported",
  "assigned"
])

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
): GeneratedArtifact => ({
  id: artifact.id,
  type: normalizeArtifactType(artifact.artifact_type),
  title: artifact.title,
  status: mapServerGenerationStatus(artifact),
  reviewStatus: mapServerReviewStatus(artifact.status),
  serverId: artifact.id,
  content: artifact.content || undefined,
  totalTokens: artifact.total_tokens ?? undefined,
  totalCostUsd: artifact.total_cost_usd ?? undefined,
  createdAt: new Date(artifact.created_at),
  completedAt: artifact.completed_at ? new Date(artifact.completed_at) : undefined
})

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
