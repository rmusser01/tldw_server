import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "../client-utils"
import { appendPathQuery } from "../path-utils"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

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

type OffsetPaginationMeta = {
  mode: "offset"
  limit: number
  offset: number
  total?: number | null
  has_more: boolean
  next_offset?: number | null
}

type SkillSummary = {
  name: string
  description?: string | null
  argument_hint?: string | null
  user_invocable: boolean
  disable_model_invocation: boolean
  context: "inline" | "fork"
}

type SkillsListPayload = {
  skills: SkillSummary[]
  count: number
  total: number
  limit: number
  offset: number
  pagination?: OffsetPaginationMeta
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
    data: {
      name: string
      study_materials_policy?: "general" | "workspace"
    }
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspace(workspaceId: string): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/workspaces/${workspaceId}`, method: "GET" })
  },

  async getWorkspaceSources(workspaceId: string): Promise<any[]> {
    return await bgRequest<any>({ path: `/api/v1/workspaces/${workspaceId}/sources`, method: "GET" })
  },

  async addWorkspaceSource(workspaceId: string, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/sources`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async updateWorkspaceSource(workspaceId: string, sourceId: string, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/sources/${sourceId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async deleteWorkspaceSource(workspaceId: string, sourceId: string): Promise<void> {
    await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/sources/${sourceId}`,
      method: "DELETE",
    })
  },

  async getWorkspaceArtifacts(workspaceId: string): Promise<any[]> {
    return await bgRequest<any>({ path: `/api/v1/workspaces/${workspaceId}/artifacts`, method: "GET" })
  },

  async addWorkspaceArtifact(workspaceId: string, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async updateWorkspaceArtifact(workspaceId: string, artifactId: string, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts/${artifactId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async deleteWorkspaceArtifact(workspaceId: string, artifactId: string): Promise<void> {
    await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/artifacts/${artifactId}`,
      method: "DELETE",
    })
  },

  async getWorkspaceNotes(workspaceId: string): Promise<any[]> {
    return await bgRequest<any>({ path: `/api/v1/workspaces/${workspaceId}/notes`, method: "GET" })
  },

  async addWorkspaceNote(workspaceId: string, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/notes`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async updateWorkspaceNote(workspaceId: string, noteId: number, data: Record<string, any>): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/notes/${noteId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data,
    })
  },

  async deleteWorkspaceNote(workspaceId: string, noteId: number): Promise<void> {
    await bgRequest<any>({
      path: `/api/v1/workspaces/${workspaceId}/notes/${noteId}`,
      method: "DELETE",
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
