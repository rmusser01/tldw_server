import type {
  WorkspaceApiResponse,
  WorkspaceSourceApiResponse,
  WorkspaceSourceCreateRequest,
  WorkspaceUpsertRequest
} from "@/services/tldw/domains/workspace-api"
import type { WorkspaceSource } from "@/types/workspace"

export interface ResearchWorkspaceServerClient {
  upsertWorkspace: (
    workspaceId: string,
    data: WorkspaceUpsertRequest
  ) => Promise<WorkspaceApiResponse>
  getWorkspaceSources: (workspaceId: string) => Promise<WorkspaceSourceApiResponse[]>
  addWorkspaceSource: (
    workspaceId: string,
    data: WorkspaceSourceCreateRequest
  ) => Promise<WorkspaceSourceApiResponse>
  updateWorkspaceSourceSelection?: (
    workspaceId: string,
    selectedSourceIds: string[]
  ) => Promise<unknown>
}

export interface ResearchWorkspaceServerReconcileResult {
  workspaceReady: boolean
  sourceRowsChecked: boolean
  addedSourceIds: string[]
  skippedSourceIds: string[]
  errors: string[]
}

export interface ResearchWorkspaceServerReconcileInput {
  client: ResearchWorkspaceServerClient
  workspaceId: string
  workspaceName?: string | null
  sources: WorkspaceSource[]
  selectedSourceIds?: string[]
}

const DEFAULT_RESEARCH_WORKSPACE_NAME = "Research Workspace"
const MAX_RECONCILE_ERROR_MESSAGES = 5
const OMITTED_RECONCILE_ERRORS_MESSAGE = "Additional workspace sync errors omitted."

const describeReconcileError = (error: unknown): string => {
  if (error instanceof Error && error.message.trim()) {
    return error.message.trim()
  }
  if (typeof error === "string" && error.trim()) {
    return error.trim()
  }
  return "Unknown workspace sync error"
}

const isValidMediaId = (mediaId: unknown): mediaId is number =>
  typeof mediaId === "number" && Number.isInteger(mediaId) && mediaId > 0

const normalizeWorkspaceName = (workspaceName?: string | null): string => {
  if (typeof workspaceName === "string" && workspaceName.trim()) {
    return workspaceName.trim()
  }
  return DEFAULT_RESEARCH_WORKSPACE_NAME
}

const appendReconcileError = (
  result: ResearchWorkspaceServerReconcileResult,
  message: string
) => {
  if (result.errors.length < MAX_RECONCILE_ERROR_MESSAGES - 1) {
    result.errors.push(message)
    return
  }
  if (!result.errors.includes(OMITTED_RECONCILE_ERRORS_MESSAGE)) {
    result.errors.push(OMITTED_RECONCILE_ERRORS_MESSAGE)
  }
}

const buildSourceCreateRequest = (
  source: WorkspaceSource,
  position: number,
  selectedSourceIds?: Set<string>
): WorkspaceSourceCreateRequest | null => {
  if (!source.id.trim() || !isValidMediaId(source.mediaId)) {
    return null
  }

  return {
    id: source.id,
    media_id: source.mediaId,
    title: source.title.trim() || "Untitled source",
    source_type: source.type,
    url: source.url || null,
    position,
    selected: selectedSourceIds ? selectedSourceIds.has(source.id) : true
  }
}

export const buildResearchWorkspaceServerSourceSignature = (
  sources: WorkspaceSource[]
): string =>
  sources
    .map((source, index) =>
      [
        index,
        source.id,
        Number.isFinite(source.mediaId) ? source.mediaId : "invalid-media",
        source.title,
        source.type,
        source.url || ""
      ].join(":")
    )
    .join("|")

export const reconcileResearchWorkspaceServerState = async ({
  client,
  workspaceId,
  workspaceName,
  sources,
  selectedSourceIds
}: ResearchWorkspaceServerReconcileInput): Promise<ResearchWorkspaceServerReconcileResult> => {
  const result: ResearchWorkspaceServerReconcileResult = {
    workspaceReady: false,
    sourceRowsChecked: false,
    addedSourceIds: [],
    skippedSourceIds: [],
    errors: []
  }

  try {
    await client.upsertWorkspace(workspaceId, {
      name: normalizeWorkspaceName(workspaceName),
      study_materials_policy: "workspace"
    })
    result.workspaceReady = true
  } catch (error) {
    appendReconcileError(
      result,
      `Failed to sync workspace: ${describeReconcileError(error)}`
    )
    return result
  }

  let existingSources: WorkspaceSourceApiResponse[] = []
  try {
    existingSources = await client.getWorkspaceSources(workspaceId)
    result.sourceRowsChecked = true
  } catch (error) {
    appendReconcileError(
      result,
      `Failed to inspect workspace sources: ${describeReconcileError(error)}`
    )
    return result
  }

  const existingSourceIds = new Set(existingSources.map((source) => source.id))
  const existingMediaIds = new Set(
    existingSources
      .map((source) => source.media_id)
      .filter((mediaId): mediaId is number => isValidMediaId(mediaId))
  )
  const selectedSourceIdSet = Array.isArray(selectedSourceIds)
    ? new Set(selectedSourceIds)
    : undefined

  for (const [position, source] of sources.entries()) {
    const sourceRequest = buildSourceCreateRequest(
      source,
      position,
      selectedSourceIdSet
    )
    if (
      sourceRequest === null ||
      existingSourceIds.has(source.id) ||
      existingMediaIds.has(source.mediaId)
    ) {
      result.skippedSourceIds.push(source.id)
      continue
    }

    try {
      await client.addWorkspaceSource(workspaceId, sourceRequest)
      result.addedSourceIds.push(source.id)
      existingSourceIds.add(source.id)
      existingMediaIds.add(source.mediaId)
    } catch (error) {
      appendReconcileError(
        result,
        `Failed to add source ${source.id}: ${describeReconcileError(error)}`
      )
    }
  }

  if (Array.isArray(selectedSourceIds) && client.updateWorkspaceSourceSelection) {
    try {
      await client.updateWorkspaceSourceSelection(workspaceId, selectedSourceIds)
    } catch (error) {
      appendReconcileError(
        result,
        `Failed to update source selection: ${describeReconcileError(error)}`
      )
    }
  }

  return result
}
