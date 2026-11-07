import { bgRequest, type BgRequestInit } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type WorkspaceHistoryRequestOptions = Pick<
  BgRequestInit, "headers" | "servicePromptConfig" | "abortSignal" | "timeoutMs"
>

const AGENT_ORCHESTRATION_BASE_PATH = "/api/v1/agent-orchestration"
const PROJECTS_PATH = `${AGENT_ORCHESTRATION_BASE_PATH}/projects`

export const listCanonicalWorkspaceProjects = (
  canonicalWorkspaceId: string, options: WorkspaceHistoryRequestOptions
) => {
  const params = new URLSearchParams({
    canonical_workspace_id: canonicalWorkspaceId,
    canonical_workspace_source: "research_workspace"
  })
  return bgRequest<unknown>({
    path: `${PROJECTS_PATH}?${params.toString()}` as AllowedPath,
    method: "GET", ...options
  })
}

export const listWorkspaceProjectTasks = (
  projectId: number, options: WorkspaceHistoryRequestOptions
) => bgRequest<unknown>({
  path: `${PROJECTS_PATH}/${projectId}/tasks` as AllowedPath,
  method: "GET", ...options
})

export const getWorkspaceTaskDetail = <T,>(
  taskId: number, options: WorkspaceHistoryRequestOptions
) => bgRequest<T>({
  path: `${AGENT_ORCHESTRATION_BASE_PATH}/tasks/${taskId}` as AllowedPath,
  method: "GET", ...options
})

export const listCanonicalWorkspaceSessions = (
  canonicalWorkspaceId: string, limit: number, options: WorkspaceHistoryRequestOptions
) => {
  const params = new URLSearchParams({
    workspace_id: canonicalWorkspaceId,
    limit: String(limit)
  })
  return bgRequest<unknown>({
    path: `/api/v1/acp/sessions?${params.toString()}` as AllowedPath,
    method: "GET", ...options
  })
}
