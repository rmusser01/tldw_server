import type { ChatScope } from "@/types/chat-scope"
import type {
  ArtifactReviewStatus,
  ArtifactStatus,
  ArtifactType,
  WorkspaceNote,
  WorkspaceSource
} from "@/types/workspace"

export const WORKSPACE_SYNC_PAYLOAD_VERSION = 1 as const

export interface WorkspaceSyncSource {
  id: string
  mediaId: number
  title: string
  type: WorkspaceSource["type"]
  url?: string
  addedAt: string
}

export interface WorkspaceSyncArtifact {
  id: string
  type: ArtifactType
  title: string
  status: ArtifactStatus
  reviewStatus?: ArtifactReviewStatus
  content?: string
  totalTokens?: number
  totalCostUsd?: number
  estimatedTokens?: number
  estimatedCostUsd?: number
  createdAt: string
  completedAt?: string
}

export interface WorkspaceSyncPayload {
  version: typeof WORKSPACE_SYNC_PAYLOAD_VERSION
  workspaceId: string
  workspaceTag: string
  updatedAt: string
  snapshot: {
    workspaceName: string
    selectedSourceIds: string[]
    sources: WorkspaceSyncSource[]
    generatedArtifacts: WorkspaceSyncArtifact[]
    currentNote: Pick<WorkspaceNote, "title" | "content" | "keywords" | "isDirty">
  }
}

interface BuildWorkspaceSyncPayloadInput {
  workspaceId: string
  workspaceTag: string
  workspaceName: string
  selectedSourceIds: string[]
  sources: WorkspaceSource[]
  generatedArtifacts: Array<{
    id: string
    type: ArtifactType
    title: string
    status: ArtifactStatus
    reviewStatus?: ArtifactReviewStatus
    content?: string
    totalTokens?: number
    totalCostUsd?: number
    estimatedTokens?: number
    estimatedCostUsd?: number
    createdAt: Date
    completedAt?: Date
  }>
  currentNote: Pick<WorkspaceNote, "title" | "content" | "keywords" | "isDirty">
  updatedAt?: Date
}

const toIsoDate = (date: Date): string => {
  return date.toISOString()
}

export const buildWorkspaceSyncPayload = (
  input: BuildWorkspaceSyncPayloadInput
): WorkspaceSyncPayload => {
  return {
    version: WORKSPACE_SYNC_PAYLOAD_VERSION,
    workspaceId: input.workspaceId,
    workspaceTag: input.workspaceTag,
    updatedAt: toIsoDate(input.updatedAt || new Date()),
    snapshot: {
      workspaceName: input.workspaceName,
      selectedSourceIds: [...input.selectedSourceIds],
      sources: input.sources.map((source) => ({
        id: source.id,
        mediaId: source.mediaId,
        title: source.title,
        type: source.type,
        url: source.url,
        addedAt: toIsoDate(source.addedAt)
      })),
      generatedArtifacts: input.generatedArtifacts.map((artifact) => ({
        id: artifact.id,
        type: artifact.type,
        title: artifact.title,
        status: artifact.status,
        reviewStatus: artifact.reviewStatus,
        content: artifact.content,
        totalTokens: artifact.totalTokens,
        totalCostUsd: artifact.totalCostUsd,
        estimatedTokens: artifact.estimatedTokens,
        estimatedCostUsd: artifact.estimatedCostUsd,
        createdAt: toIsoDate(artifact.createdAt),
        completedAt: artifact.completedAt
          ? toIsoDate(artifact.completedAt)
          : undefined
      })),
      currentNote: {
        title: input.currentNote.title,
        content: input.currentNote.content,
        keywords: [...input.currentNote.keywords],
        isDirty: input.currentNote.isDirty
      }
    }
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value != null

export const isWorkspaceSyncPayload = (
  value: unknown
): value is WorkspaceSyncPayload => {
  if (!isRecord(value)) return false
  if (value.version !== WORKSPACE_SYNC_PAYLOAD_VERSION) return false
  if (typeof value.workspaceId !== "string" || value.workspaceId.length === 0) {
    return false
  }
  if (typeof value.workspaceTag !== "string" || value.workspaceTag.length === 0) {
    return false
  }
  if (typeof value.updatedAt !== "string") return false
  if (!isRecord(value.snapshot)) return false
  if (typeof value.snapshot.workspaceName !== "string") return false
  if (!Array.isArray(value.snapshot.selectedSourceIds)) return false
  if (!Array.isArray(value.snapshot.sources)) return false
  if (!Array.isArray(value.snapshot.generatedArtifacts)) return false
  if (!isRecord(value.snapshot.currentNote)) return false
  if (typeof value.snapshot.currentNote.title !== "string") return false
  if (typeof value.snapshot.currentNote.content !== "string") return false
  if (!Array.isArray(value.snapshot.currentNote.keywords)) return false
  if (typeof value.snapshot.currentNote.isDirty !== "boolean") return false
  return true
}

/**
 * Validate that a cached serverChatId still belongs to the expected scope.
 * Returns the cachedId if scope matches, null otherwise.
 */
export const validateCachedServerChatId = ({
  cachedId,
  serverScope,
  expectedScope,
}: {
  cachedId: string | null
  serverScope: { scope_type: string; workspace_id: string | null } | null
  expectedScope: ChatScope
}): string | null => {
  if (!cachedId || !serverScope) return null
  if (expectedScope.type === "global" && serverScope.scope_type === "global") return cachedId
  if (
    expectedScope.type === "workspace" &&
    serverScope.scope_type === "workspace" &&
    serverScope.workspace_id === expectedScope.workspaceId
  )
    return cachedId
  return null
}
