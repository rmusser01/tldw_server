import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "../client-utils"
import { appendPathQuery } from "../path-utils"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import type { OffsetPaginationMeta } from "@/services/response-envelope"
import type { SkillsListResponse } from "@/types/skill"

/**
 * Minimal interface for the TldwApiClient methods referenced via `this`.
 */
export interface TldwApiClientCore {
  ensureConfigForRequest(requireAuth: boolean): Promise<any>
  upload<T>(init: any, requireAuth?: boolean): Promise<T>
  resolveApiPath(key: string, candidates: string[]): Promise<AllowedPath>
  fillPathParams(template: AllowedPath, values: string | string[]): AllowedPath
  buildQuery(params?: Record<string, any>): string
}

type SkillsListPayload = SkillsListResponse & {
  pagination?: OffsetPaginationMeta
}

export type WorkspaceArtifactJsonRecord = Record<string, unknown>

export interface WorkspaceApiResponse {
  id: string
  name: string | null
  archived: boolean
  study_materials_policy: "general" | "workspace"
  deleted: boolean
  banner_title: string | null
  banner_subtitle: string | null
  banner_color: string | null
  audio_provider: string | null
  audio_model: string | null
  audio_voice: string | null
  audio_speed: number | null
  created_at: string
  last_modified: string
  version: number
}

export interface WorkspaceSourceApiResponse {
  id: string
  workspace_id: string
  media_id: number
  title: string
  source_type: string
  url: string | null
  position: number
  selected: boolean
  added_at: string
  version: number
}

export type WorkspaceSourceLifecycleState =
  | "queued"
  | "ingesting"
  | "extracting"
  | "chunking"
  | "indexing"
  | "queryable"
  | "partially_queryable"
  | "failed"
  | "retrying"
  | "missing_media"
  | "blocked_by_permissions"
  | "unknown"

export interface WorkspaceSourceReadiness {
  metadata_ready: boolean
  text_extracted: boolean
  fts_ready: boolean
  vector_ready: boolean
  citation_ready: boolean
  summary_ready: boolean
  tool_accessible: boolean
}

export interface WorkspaceSourceJobStatus {
  id: number | null
  uuid: string | null
  status: string | null
  job_type: string | null
  progress_percent: number | null
  progress_message: string | null
  error_message: string | null
}

export interface WorkspaceSourceStatusApiResponse {
  id: string
  workspace_id: string
  media_id: number | null
  title: string
  source_type: string
  selected: boolean
  state: WorkspaceSourceLifecycleState
  status_reason: string
  readiness: WorkspaceSourceReadiness
  progress_percent: number | null
  progress_message: string | null
  job: WorkspaceSourceJobStatus | null
  updated_at: string | null
}

export interface WorkspaceSourceStatusSummary {
  total: number
  selected: number
  queryable: number
  partially_queryable: number
  processing: number
  failed: number
  missing: number
}

export interface WorkspaceSourceStatusListResponse {
  workspace_id: string
  sources: WorkspaceSourceStatusApiResponse[]
  summary: WorkspaceSourceStatusSummary
}

export type WorkspaceCapabilityServiceState =
  | "available"
  | "private"
  | "not_configured"
  | "unknown"
  | "blocked"
  | "degraded"

export interface WorkspaceCapabilityService {
  state: WorkspaceCapabilityServiceState
  reason_code: string | null
  management_surface: string | null
}

export interface WorkspaceAllowedAction {
  allowed: boolean
  reason_code: string | null
}

export interface WorkspaceCapabilitiesResponse {
  workspace_id: string
  workspace_kind: string
  access_level: string
  source_summary: WorkspaceSourceStatusSummary
  workspace_services: Record<string, WorkspaceCapabilityService>
  allowed_actions: Record<string, WorkspaceAllowedAction>
}

export interface WorkspaceArtifactApiResponse {
  id: string
  workspace_id: string
  artifact_type: string
  title: string
  status: string
  review_state?: string | null
  content_type?: string | null
  content: string | null
  preview_text?: string | null
  summary?: string | null
  total_tokens: number | null
  total_cost_usd: number | null
  owner_scope?: string | null
  owner_id?: string | null
  project_id?: string | null
  task_id?: string | null
  source_collection_id?: string | null
  root_artifact_id?: string | null
  artifact_version_id?: string | null
  previous_version_id?: string | null
  producer_metadata?: WorkspaceArtifactJsonRecord | null
  source_lineage?: WorkspaceArtifactJsonRecord | WorkspaceArtifactJsonRecord[] | null
  review_metadata?: WorkspaceArtifactJsonRecord | null
  version_metadata?: WorkspaceArtifactJsonRecord | null
  export_refs?: WorkspaceArtifactJsonRecord[] | null
  redaction?: WorkspaceArtifactJsonRecord | null
  schema_version?: number | null
  created_at: string
  completed_at: string | null
  version: number
}

export interface WorkspaceNoteApiResponse {
  id: number
  workspace_id: string
  title: string
  content: string
  keywords_json: string
  created_at: string
  last_modified: string
  version: number
}

export interface WorkspaceUpsertRequest {
  name: string
  study_materials_policy?: "general" | "workspace"
}

export interface WorkspaceSourceCreateRequest {
  id: string
  media_id: number
  title: string
  source_type: string
  url?: string | null
  position?: number
  selected?: boolean
}

export interface WorkspaceSourceUpdateRequest {
  title?: string
  source_type?: string
  url?: string | null
  position?: number
  selected?: boolean
  version: number
}

export interface WorkspaceArtifactCreateRequest {
  id: string
  artifact_type: string
  title: string
  status?: string
  content?: string | null
}

export interface WorkspaceArtifactUpdateRequest {
  title?: string
  status?: string
  content?: string | null
  total_tokens?: number | null
  total_cost_usd?: number | null
  completed_at?: string | null
  version: number
}

export interface WorkspaceNoteCreateRequest {
  title?: string
  content?: string
  keywords?: string[]
}

export interface WorkspaceNoteUpdateRequest {
  title?: string
  content?: string
  keywords_json?: string
  version: number
}

export const workspaceApiMethods = {
  // ── Skills API ──

  async listSkills(
    this: TldwApiClientCore,
    params?: {
      limit?: number
      offset?: number
    }
  ): Promise<SkillsListPayload> {
    const query = buildQuery(params)
    const base = await this.resolveApiPath("skills.list", [
      "/api/v1/skills",
      "/api/v1/skills/"
    ])
    return await bgRequest<SkillsListPayload>({
      path: appendPathQuery(base, query),
      method: "GET"
    })
  },

  async getSkill(
    this: TldwApiClientCore,
    name: string
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.get", [
      "/api/v1/skills/{name}",
      "/api/v1/skills/{name}/"
    ])
    const path = this.fillPathParams(base, name)
    return await bgRequest<any>({ path, method: "GET" })
  },

  async createSkill(
    this: TldwApiClientCore,
    payload: {
      name: string
      content: string
      supporting_files?: Record<string, string> | null
    }
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.create", [
      "/api/v1/skills",
      "/api/v1/skills/"
    ])
    return await bgRequest<any>({
      path: base,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async updateSkill(
    this: TldwApiClientCore,
    name: string,
    payload: {
      content?: string
      supporting_files?: Record<string, string | null> | null
    },
    version?: number
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.update", [
      "/api/v1/skills/{name}",
      "/api/v1/skills/{name}/"
    ])
    const path = this.fillPathParams(base, name)
    const headers: Record<string, string> = { "Content-Type": "application/json" }
    if (version != null) {
      headers["If-Match"] = String(version)
    }
    return await bgRequest<any>({ path, method: "PUT", headers, body: payload })
  },

  async deleteSkill(
    this: TldwApiClientCore,
    name: string
  ): Promise<void> {
    const base = await this.resolveApiPath("skills.delete", [
      "/api/v1/skills/{name}",
      "/api/v1/skills/{name}/"
    ])
    const path = this.fillPathParams(base, name)
    await bgRequest<any>({ path, method: "DELETE" })
  },

  async importSkill(
    this: TldwApiClientCore,
    payload: {
      name?: string
      content: string
      supporting_files?: Record<string, string> | null
      overwrite?: boolean
    }
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.import", [
      "/api/v1/skills/import",
      "/api/v1/skills/import/"
    ])
    return await bgRequest<any>({
      path: base,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async importSkillFile(
    this: TldwApiClientCore,
    file: File
  ): Promise<any> {
    const data = await file.arrayBuffer()
    return await this.upload<any>({
      path: "/api/v1/skills/import/file" as AllowedPath,
      method: "POST",
      fileFieldName: "file",
      file: {
        name: file.name || "skill-import",
        type: file.type || "application/octet-stream",
        data
      }
    })
  },

  async seedSkills(
    this: TldwApiClientCore,
    params?: {
      overwrite?: boolean
    }
  ): Promise<any> {
    const query = buildQuery(params)
    const base = await this.resolveApiPath("skills.seed", [
      "/api/v1/skills/seed",
      "/api/v1/skills/seed/"
    ])
    return await bgRequest<any>({
      path: appendPathQuery(base, query),
      method: "POST"
    })
  },

  async exportSkill(
    this: TldwApiClientCore,
    name: string
  ): Promise<Blob> {
    await this.ensureConfigForRequest(true)
    const res = await bgRequest<ArrayBuffer, AllowedPath>({
      path: `/api/v1/skills/${encodeURIComponent(name)}/export` as AllowedPath,
      method: "GET",
      responseType: "arrayBuffer"
    })
    return new Blob([res], { type: "application/zip" })
  },

  async executeSkill(
    this: TldwApiClientCore,
    name: string,
    args?: string
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.execute", [
      "/api/v1/skills/{name}/execute",
      "/api/v1/skills/{name}/execute/"
    ])
    const path = this.fillPathParams(base, name)
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { args: args || "" }
    })
  },

  async getSkillsContext(
    this: TldwApiClientCore
  ): Promise<any> {
    const base = await this.resolveApiPath("skills.context", [
      "/api/v1/skills/context",
      "/api/v1/skills/context/"
    ])
    return await bgRequest<any>({ path: base, method: "GET" })
  },

  // ── Workspace sub-resource methods ──

  async upsertWorkspace(
    workspaceId: string,
    data: WorkspaceUpsertRequest
  ): Promise<WorkspaceApiResponse> {
    return await bgRequest<WorkspaceApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspace(workspaceId: string): Promise<WorkspaceApiResponse> {
    return await bgRequest<WorkspaceApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}`,
      method: "GET"
    })
  },

  async getWorkspaceSources(
    workspaceId: string
  ): Promise<WorkspaceSourceApiResponse[]> {
    return await bgRequest<WorkspaceSourceApiResponse[]>({
      path: `/api/v1/workspaces/${workspaceId}/sources`,
      method: "GET"
    })
  },

  async getWorkspaceSourcesStatus(
    workspaceId: string
  ): Promise<WorkspaceSourceStatusListResponse> {
    return await bgRequest<WorkspaceSourceStatusListResponse>({
      path: `/api/v1/workspaces/${workspaceId}/sources/status`,
      method: "GET"
    })
  },

  async getWorkspaceCapabilities(
    workspaceId: string
  ): Promise<WorkspaceCapabilitiesResponse> {
    return await bgRequest<WorkspaceCapabilitiesResponse>({
      path: `/api/v1/workspaces/${workspaceId}/capabilities`,
      method: "GET"
    })
  },

  async addWorkspaceSource(
    workspaceId: string,
    data: WorkspaceSourceCreateRequest
  ): Promise<WorkspaceSourceApiResponse> {
    return await bgRequest<WorkspaceSourceApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/sources`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async updateWorkspaceSource(
    workspaceId: string,
    sourceId: string,
    data: WorkspaceSourceUpdateRequest
  ): Promise<WorkspaceSourceApiResponse> {
    return await bgRequest<WorkspaceSourceApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/sources/${sourceId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspaceSource(workspaceId: string, sourceId: string): Promise<void> {
    await bgRequest<unknown>({
      path: `/api/v1/workspaces/${workspaceId}/sources/${sourceId}`,
      method: "DELETE"
    })
  },

  async getWorkspaceArtifacts(
    workspaceId: string
  ): Promise<WorkspaceArtifactApiResponse[]> {
    return await bgRequest<WorkspaceArtifactApiResponse[]>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts`,
      method: "GET"
    })
  },

  async addWorkspaceArtifact(
    workspaceId: string,
    data: WorkspaceArtifactCreateRequest
  ): Promise<WorkspaceArtifactApiResponse> {
    return await bgRequest<WorkspaceArtifactApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async updateWorkspaceArtifact(
    workspaceId: string,
    artifactId: string,
    data: WorkspaceArtifactUpdateRequest
  ): Promise<WorkspaceArtifactApiResponse> {
    return await bgRequest<WorkspaceArtifactApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts/${artifactId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspaceArtifact(workspaceId: string, artifactId: string): Promise<void> {
    await bgRequest<unknown>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts/${artifactId}`,
      method: "DELETE"
    })
  },

  async getWorkspaceNotes(
    workspaceId: string
  ): Promise<WorkspaceNoteApiResponse[]> {
    return await bgRequest<WorkspaceNoteApiResponse[]>({
      path: `/api/v1/workspaces/${workspaceId}/notes`,
      method: "GET"
    })
  },

  async addWorkspaceNote(
    workspaceId: string,
    data: WorkspaceNoteCreateRequest
  ): Promise<WorkspaceNoteApiResponse> {
    return await bgRequest<WorkspaceNoteApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/notes`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async updateWorkspaceNote(
    workspaceId: string,
    noteId: number,
    data: WorkspaceNoteUpdateRequest
  ): Promise<WorkspaceNoteApiResponse> {
    return await bgRequest<WorkspaceNoteApiResponse>({
      path: `/api/v1/workspaces/${workspaceId}/notes/${noteId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspaceNote(workspaceId: string, noteId: number): Promise<void> {
    await bgRequest<unknown>({
      path: `/api/v1/workspaces/${workspaceId}/notes/${noteId}`,
      method: "DELETE"
    })
  },

  // ── Watchlists / Monitoring ──

  async listWatchlists(): Promise<any[]> {
    const res = await bgRequest<any>({ path: "/api/v1/monitoring/watchlists", method: "GET" })
    return res?.watchlists ?? (Array.isArray(res) ? res : [])
  },

  async createWatchlist(payload: { name: string; description?: string; scope_type?: string; rules?: any[] }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/monitoring/watchlists",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteWatchlist(id: string): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/monitoring/watchlists/${id}`, method: "DELETE" })
  },

  async listMonitoringAlerts(params?: { rule_severity?: string; source?: string; limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/monitoring/alerts${query}`, method: "GET" })
  },

  async acknowledgeAlert(id: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/monitoring/alerts/${id}/acknowledge`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async dismissAlert(id: number): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/monitoring/alerts/${id}`, method: "DELETE" })
  },

  // ── Runtime Config ──

  async getCleanupSettings(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/cleanup-settings", method: "GET" })
  },

  async updateCleanupSettings(payload: any): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/cleanup-settings",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getRegistrationSettings(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/registration-settings", method: "GET" })
  },

  async updateRegistrationSettings(payload: any): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/registration-settings",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  // ── Rate Limiting / Resource Governor ──

  async getGovernorPolicy(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/resource-governor/policy", method: "GET" })
  },

  async getGovernorCoverage(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/diag/coverage", method: "GET" })
  },

  async listAdminRateLimits(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/rate-limits", method: "GET" })
  },
}

export type WorkspaceApiMethods = typeof workspaceApiMethods
