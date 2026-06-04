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

export type WorkspaceProfile = "research" | "project"
export type WorkspaceAttentionState =
  | "ready"
  | "setup_pending"
  | "working"
  | "needs_attention"
  | "blocked"
  | "archived"

export type WorkspaceOperationStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "conflicted"
  | "expired"

export type WorkspaceProjectRootBackend = "host_local" | "sandbox_volume"
export type WorkspaceProjectRootState =
  | "not_configured"
  | "provisioning"
  | "attached"
  | "unavailable"
  | "missing"
  | "detached"
  | "failed"
  | "cleanup_pending"
  | "archived"

export type WorkspaceResolutionStatus = "complete" | "partial" | "failed"

export interface WorkspaceApiResponse {
  id: string
  name: string | null
  archived: boolean
  study_materials_policy: "general" | "workspace"
  workspace_profile: WorkspaceProfile
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

export interface WorkspaceListApiResponse {
  items: WorkspaceApiResponse[]
  total: number
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
  url: string | null
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
  | "needs_approval"
  | "unknown"
  | "blocked"
  | "degraded"
  | "external_provider_warning"

export interface WorkspaceCapabilityService {
  state: WorkspaceCapabilityServiceState
  reason_code: string | null
  management_surface: string | null
}

export interface WorkspaceAllowedAction {
  allowed: boolean
  reason_code: string | null
}

export interface WorkspaceContextPartialError {
  scope: string
  code: string
  message: string
}

export interface WorkspaceResolution {
  status: WorkspaceResolutionStatus
  partial_errors: WorkspaceContextPartialError[]
}

export interface WorkspaceFileInventory {
  state: string | null
  indexed_file_count: number | null
  total_file_count: number | null
  updated_at: string | null
  available: boolean
}

export interface WorkspaceProjectRoot {
  state: WorkspaceProjectRootState
  root_id: string | null
  backend: WorkspaceProjectRootBackend | null
  display_name: string | null
  path_hint: string | null
  git_state: string | null
  file_inventory_state: string | null
  file_inventory: WorkspaceFileInventory
  indexing_state: string | null
  sandbox_mount_state: string | null
  mcp_trust_state: string | null
}

export interface WorkspaceRootResponse extends WorkspaceProjectRoot {
  workspace_id: string | null
  is_primary: boolean
  version: number | null
  updated_at: string | null
}

export interface WorkspaceRootsResponse {
  workspace_id: string
  workspace_profile: WorkspaceProfile
  primary_root: WorkspaceRootResponse | null
  roots: WorkspaceRootResponse[]
}

export interface WorkspaceCapabilitiesResponse {
  workspace_id: string
  workspace_profile: WorkspaceProfile
  workspace_kind: string
  access_level: string
  resolution?: WorkspaceResolution
  project_root?: WorkspaceProjectRoot
  source_summary: WorkspaceSourceStatusSummary
  workspace_services: Record<string, WorkspaceCapabilityService>
  allowed_actions: Record<string, WorkspaceAllowedAction>
}

export interface WorkspaceSourcePreviewSummary {
  available: boolean
  detail_href: string | null
  snippet_count: number | null
  total_chars: number | null
  unavailable_reason: string | null
}

export interface WorkspaceContextSource extends WorkspaceSourceApiResponse {
  state: WorkspaceSourceLifecycleState
  status_reason: string
  readiness: WorkspaceSourceReadiness
  progress_percent: number | null
  progress_message: string | null
  job: WorkspaceSourceJobStatus | null
  updated_at: string | null
  preview: WorkspaceSourcePreviewSummary
}

export interface WorkspaceContextResponse {
  workspace_id: string
  workspace_profile: WorkspaceProfile
  workspace_kind: string
  schema_version: number
  generated_at: string
  workspace: WorkspaceApiResponse
  attention_state: WorkspaceAttentionState
  resolution: WorkspaceResolution
  project_root: WorkspaceProjectRoot
  sources: {
    items: WorkspaceContextSource[]
    summary: WorkspaceSourceStatusSummary
  }
  capabilities: WorkspaceCapabilitiesResponse
  services: Record<string, WorkspaceCapabilityService>
  allowed_actions: Record<string, WorkspaceAllowedAction>
  active_jobs: WorkspaceSourceJobStatus[]
  active_operations: WorkspaceOperationResponse[]
  partial_errors: WorkspaceContextPartialError[]
}

export type WorkspaceSourcePreviewMode =
  | "available"
  | "pending"
  | "failed"
  | "missing_media"
  | "empty"

export interface WorkspaceSourcePreviewSnippet {
  id: string
  source_id: string
  media_id: number | null
  kind: "content_excerpt" | "chunk"
  text: string
  start_char: number | null
  end_char: number | null
  chunk_index: number | null
  chunk_uuid: string | null
  chunk_type: string | null
}

export interface WorkspaceSourcePreviewResponse {
  workspace_id: string
  source_id: string
  media_id: number | null
  title: string
  source_type: string
  url: string | null
  state: WorkspaceSourceLifecycleState
  status_reason: string
  readiness: WorkspaceSourceReadiness
  content_available: boolean
  preview_mode: WorkspaceSourcePreviewMode
  unavailable_reason: string | null
  text_preview: string | null
  text_total_chars: number | null
  text_truncated: boolean
  snippets: WorkspaceSourcePreviewSnippet[]
  generated_at: string
}

export type WorkspaceMigrationStatus = "created" | "finalized" | "failed"

export interface WorkspaceMigrationChunkDeclaration {
  id: string
  sha256: string
  byte_count: number
  chunk_kind?: string
}

export interface WorkspaceMigrationCreateRequest {
  id: string
  idempotency_key: string
  target_workspace_id: string
  target_workspace_name: string
  source_product?: string
  manifest_hash: string
  declared_chunks?: WorkspaceMigrationChunkDeclaration[]
  manifest?: Record<string, unknown>
  diagnostics?: Record<string, unknown>
}

export interface WorkspaceMigrationChunkUploadRequest {
  sha256: string
  byte_count: number
  chunk_kind?: string
  metadata?: Record<string, unknown>
}

export interface WorkspaceMigrationFinalizeRequest {
  manifest_hash: string
}

export interface WorkspaceMigrationClientDeleteAckRequest {
  acknowledged_manifest_hash: string
}

export interface WorkspaceMigrationChunkReceiptResponse {
  id: string
  migration_id: string
  sha256: string
  byte_count: number
  chunk_kind: string
  metadata: Record<string, unknown>
  status: "accepted"
  accepted_at: string
}

export interface WorkspaceMigrationResponse {
  id: string
  idempotency_key: string
  target_workspace_id: string
  target_workspace_name: string
  source_product: string
  manifest_hash: string
  status: WorkspaceMigrationStatus
  declared_chunk_count: number
  accepted_chunk_count: number
  missing_chunk_ids: string[]
  client_delete_eligible: boolean
  created_at: string
  updated_at: string
  finalized_at: string | null
  recovery_manifest: Record<string, unknown>
  chunks: WorkspaceMigrationChunkReceiptResponse[]
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
  workspace_profile?: WorkspaceProfile
}

export interface WorkspacePatchRequest {
  name?: string | null
  archived?: boolean | null
  study_materials_policy?: "general" | "workspace" | null
  workspace_profile?: WorkspaceProfile | null
  banner_title?: string | null
  banner_subtitle?: string | null
  banner_color?: string | null
  audio_provider?: string | null
  audio_model?: string | null
  audio_voice?: string | null
  audio_speed?: number | null
  version: number
}

export interface WorkspacePrimaryRootAttachRequest {
  backend: WorkspaceProjectRootBackend
  root_id?: string | null
  absolute_root?: string | null
  sandbox_volume_id?: string | null
  display_name?: string | null
  replace_existing?: boolean
  expected_workspace_version?: number | null
  strict_sandbox_validation?: boolean
}

export interface WorkspaceSandboxRootProvisionRequest {
  display_name?: string | null
  requested_runtime?: string | null
  root_id?: string | null
  replace_existing?: boolean
  expected_workspace_version?: number | null
}

export interface WorkspaceSandboxRootProvisionResponse {
  workspace_id: string
  workspace_profile: WorkspaceProfile
  operation: WorkspaceOperationResponse
  primary_root: WorkspaceRootResponse | null
}

export type WorkspaceFileInventoryState =
  | "not_started"
  | "queued"
  | "scanning"
  | "current"
  | "partial"
  | "stale"
  | "failed"
  | "disabled"

export type WorkspaceFileInventoryEntryKind =
  | "file"
  | "directory"
  | "symlink"
  | "other"

export interface WorkspaceFileInventoryScanRequest {
  force?: boolean
  expected_root_version?: number | null
}

export interface WorkspaceFileInventoryJobStatus {
  id: number | null
  uuid: string | null
  status: string | null
  job_type: string | null
  progress_percent: number | null
  progress_message: string | null
  error_message: string | null
}

export interface WorkspaceFileInventoryCounts {
  files: number
  directories: number
  symlinks: number
  ignored: number
  indexing_candidates: number
  diagnostics: number
  total_entries: number
}

export interface WorkspaceFileInventoryDiagnostic {
  code: string
  message: string
  path_hint: string | null
}

export interface WorkspaceFileInventoryStatusResponse {
  workspace_id: string
  root_id: string | null
  state: WorkspaceFileInventoryState
  durable_state: string | null
  stale: boolean
  last_scan_id: string | null
  last_scan_started_at: string | null
  last_scan_completed_at: string | null
  root_version: number | null
  scan_root_version: number | null
  ignore_policy_fingerprint: string | null
  root_snapshot_token: string | null
  counts: WorkspaceFileInventoryCounts
  diagnostics: WorkspaceFileInventoryDiagnostic[]
  job: WorkspaceFileInventoryJobStatus | null
  updated_at: string | null
}

export interface WorkspaceFileInventoryItemResponse {
  relative_path: string
  entry_kind: WorkspaceFileInventoryEntryKind
  size_bytes: number | null
  mtime_ns: number | null
  mode_bits: number | null
  extension: string | null
  mime_hint: string | null
  language_hint: string | null
  ignored: boolean
  ignore_reason: string | null
  indexing_candidate: boolean
}

export interface WorkspaceFileInventoryItemsResponse {
  workspace_id: string
  root_id: string | null
  items: WorkspaceFileInventoryItemResponse[]
  next_cursor: string | null
  limit: number
}

export interface WorkspaceFileInventoryItemsRequest {
  prefix?: string | null
  limit?: number | null
  cursor?: string | null
  include_ignored?: boolean | null
  entry_kind?: WorkspaceFileInventoryEntryKind | null
}

export interface WorkspaceOperationResponse {
  operation_id: string
  workspace_id: string
  command: string
  status: WorkspaceOperationStatus
  started_at: string
  updated_at: string
  retryable: boolean
  diagnostics: Record<string, unknown>
  poll_href: string
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

export type WorkspaceArtifactReviewState =
  | "draft"
  | "reviewing"
  | "accepted"
  | "needs_revision"
  | "rejected"
  | "exported"
  | "assigned"
  | "archived"

export interface WorkspaceArtifactRedaction {
  support_safe?: boolean
  redacted?: boolean
  retention_class?: string | null
  redacted_fields?: string[]
  [key: string]: unknown
}

export interface WorkspaceArtifactCreateRequest {
  id: string
  artifact_type: string
  title: string
  status?: string
  content?: string | null
  content_type?: string
  preview_text?: string | null
  summary?: string | null
  review_state?: WorkspaceArtifactReviewState
  owner_scope?: string
  owner_id?: string | null
  project_id?: string | null
  task_id?: string | null
  source_collection_id?: string | null
  producer_metadata?: WorkspaceArtifactJsonRecord
  source_lineage?: WorkspaceArtifactJsonRecord
  review_metadata?: WorkspaceArtifactJsonRecord
  version_metadata?: WorkspaceArtifactJsonRecord
  export_refs?: WorkspaceArtifactJsonRecord[]
  redaction?: WorkspaceArtifactRedaction
  schema_version?: number
}

export interface WorkspaceArtifactUpdateRequest {
  title?: string
  status?: string
  content?: string | null
  content_type?: string | null
  preview_text?: string | null
  summary?: string | null
  review_state?: WorkspaceArtifactReviewState | null
  owner_scope?: string | null
  owner_id?: string | null
  project_id?: string | null
  task_id?: string | null
  source_collection_id?: string | null
  producer_metadata?: WorkspaceArtifactJsonRecord | null
  source_lineage?: WorkspaceArtifactJsonRecord | null
  review_metadata?: WorkspaceArtifactJsonRecord | null
  version_metadata?: WorkspaceArtifactJsonRecord | null
  export_refs?: WorkspaceArtifactJsonRecord[] | null
  redaction?: WorkspaceArtifactRedaction | null
  schema_version?: number | null
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

const encodeWorkspacePathSegment = (value: string, label: string): string => {
  if (value.includes("/")) {
    throw new Error(`${label} cannot contain "/" in Workspace API path segments`)
  }
  return encodeURIComponent(value)
}

const workspacePath = (workspaceId: string, suffix = ""): AllowedPath =>
  `/api/v1/workspaces/${encodeWorkspacePathSegment(
    workspaceId,
    "workspaceId"
  )}${suffix}` as AllowedPath

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

  async listWorkspaces(): Promise<WorkspaceListApiResponse> {
    return await bgRequest<WorkspaceListApiResponse>({
      path: "/api/v1/workspaces/",
      method: "GET"
    })
  },

  async upsertWorkspace(
    workspaceId: string,
    data: WorkspaceUpsertRequest
  ): Promise<WorkspaceApiResponse> {
    return await bgRequest<WorkspaceApiResponse>({
      path: workspacePath(workspaceId),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspace(workspaceId: string): Promise<WorkspaceApiResponse> {
    return await bgRequest<WorkspaceApiResponse>({
      path: workspacePath(workspaceId),
      method: "GET"
    })
  },

  async patchWorkspace(
    workspaceId: string,
    data: WorkspacePatchRequest
  ): Promise<WorkspaceApiResponse> {
    return await bgRequest<WorkspaceApiResponse>({
      path: workspacePath(workspaceId),
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspace(workspaceId: string): Promise<void> {
    await bgRequest<unknown>({
      path: workspacePath(workspaceId),
      method: "DELETE"
    })
  },

  async getWorkspaceRoots(workspaceId: string): Promise<WorkspaceRootsResponse> {
    return await bgRequest<WorkspaceRootsResponse>({
      path: workspacePath(workspaceId, "/roots"),
      method: "GET"
    })
  },

  async attachWorkspacePrimaryRoot(
    workspaceId: string,
    data: WorkspacePrimaryRootAttachRequest
  ): Promise<WorkspaceRootsResponse> {
    return await bgRequest<WorkspaceRootsResponse>({
      path: workspacePath(workspaceId, "/roots/primary"),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async provisionWorkspaceSandboxRoot(
    workspaceId: string,
    data: WorkspaceSandboxRootProvisionRequest,
    idempotencyKey: string
  ): Promise<WorkspaceSandboxRootProvisionResponse> {
    const key = idempotencyKey.trim()
    if (!key) {
      throw new Error("idempotencyKey is required for sandbox root provisioning")
    }
    return await bgRequest<WorkspaceSandboxRootProvisionResponse>({
      path: workspacePath(workspaceId, "/roots/primary/sandbox-volume"),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": key
      },
      body: data
    })
  },

  async getWorkspaceOperation(
    workspaceId: string,
    operationId: string
  ): Promise<WorkspaceOperationResponse> {
    const encodedOperationId = encodeWorkspacePathSegment(
      operationId,
      "operationId"
    )
    return await bgRequest<WorkspaceOperationResponse>({
      path: workspacePath(workspaceId, `/operations/${encodedOperationId}`),
      method: "GET"
    })
  },

  async queueWorkspaceFileInventoryScan(
    workspaceId: string,
    data: WorkspaceFileInventoryScanRequest = {}
  ): Promise<WorkspaceFileInventoryStatusResponse> {
    return await bgRequest<WorkspaceFileInventoryStatusResponse>({
      path: workspacePath(workspaceId, "/file-inventory/scan"),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspaceFileInventoryStatus(
    workspaceId: string
  ): Promise<WorkspaceFileInventoryStatusResponse> {
    return await bgRequest<WorkspaceFileInventoryStatusResponse>({
      path: workspacePath(workspaceId, "/file-inventory/status"),
      method: "GET"
    })
  },

  async getWorkspaceFileInventoryItems(
    workspaceId: string,
    params?: WorkspaceFileInventoryItemsRequest
  ): Promise<WorkspaceFileInventoryItemsResponse> {
    const query = buildQuery(params as Record<string, any> | undefined)
    return await bgRequest<WorkspaceFileInventoryItemsResponse>({
      path: appendPathQuery(
        workspacePath(workspaceId, "/file-inventory/items"),
        query
      ),
      method: "GET"
    })
  },

  async createWorkspaceMigration(
    data: WorkspaceMigrationCreateRequest
  ): Promise<WorkspaceMigrationResponse> {
    return await bgRequest<WorkspaceMigrationResponse>({
      path: "/api/v1/workspaces/migrations",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspaceMigration(
    migrationId: string
  ): Promise<WorkspaceMigrationResponse> {
    const encodedMigrationId = encodeWorkspacePathSegment(
      migrationId,
      "migrationId"
    )
    return await bgRequest<WorkspaceMigrationResponse>({
      path: `/api/v1/workspaces/migrations/${encodedMigrationId}`,
      method: "GET"
    })
  },

  async putWorkspaceMigrationChunk(
    migrationId: string,
    chunkId: string,
    data: WorkspaceMigrationChunkUploadRequest
  ): Promise<WorkspaceMigrationChunkReceiptResponse> {
    const encodedMigrationId = encodeWorkspacePathSegment(
      migrationId,
      "migrationId"
    )
    const encodedChunkId = encodeWorkspacePathSegment(chunkId, "chunkId")
    return await bgRequest<WorkspaceMigrationChunkReceiptResponse>({
      path: `/api/v1/workspaces/migrations/${encodedMigrationId}/chunks/${encodedChunkId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async finalizeWorkspaceMigration(
    migrationId: string,
    data: WorkspaceMigrationFinalizeRequest
  ): Promise<WorkspaceMigrationResponse> {
    const encodedMigrationId = encodeWorkspacePathSegment(
      migrationId,
      "migrationId"
    )
    return await bgRequest<WorkspaceMigrationResponse>({
      path: `/api/v1/workspaces/migrations/${encodedMigrationId}/finalize`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async ackWorkspaceMigrationClientDelete(
    migrationId: string,
    data: WorkspaceMigrationClientDeleteAckRequest
  ): Promise<{ ok: boolean }> {
    const encodedMigrationId = encodeWorkspacePathSegment(
      migrationId,
      "migrationId"
    )
    return await bgRequest<{ ok: boolean }>({
      path: `/api/v1/workspaces/migrations/${encodedMigrationId}/client-delete-ack`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async getWorkspaceSources(
    workspaceId: string
  ): Promise<WorkspaceSourceApiResponse[]> {
    return await bgRequest<WorkspaceSourceApiResponse[]>({
      path: workspacePath(workspaceId, "/sources"),
      method: "GET"
    })
  },

  async getWorkspaceSourcesStatus(
    workspaceId: string
  ): Promise<WorkspaceSourceStatusListResponse> {
    return await bgRequest<WorkspaceSourceStatusListResponse>({
      path: workspacePath(workspaceId, "/sources/status"),
      method: "GET"
    })
  },

  async getWorkspaceCapabilities(
    workspaceId: string
  ): Promise<WorkspaceCapabilitiesResponse> {
    return await bgRequest<WorkspaceCapabilitiesResponse>({
      path: workspacePath(workspaceId, "/capabilities"),
      method: "GET"
    })
  },

  async getWorkspaceContext(
    workspaceId: string
  ): Promise<WorkspaceContextResponse> {
    return await bgRequest<WorkspaceContextResponse>({
      path: workspacePath(workspaceId, "/context"),
      method: "GET"
    })
  },

  async getWorkspaceSourcePreview(
    workspaceId: string,
    sourceId: string,
    params?: {
      max_chars?: number
      chunk_limit?: number
    }
  ): Promise<WorkspaceSourcePreviewResponse> {
    const query = buildQuery(params)
    const encodedSourceId = encodeWorkspacePathSegment(sourceId, "sourceId")
    return await bgRequest<WorkspaceSourcePreviewResponse>({
      path: appendPathQuery(
        workspacePath(workspaceId, `/sources/${encodedSourceId}/preview`),
        query
      ),
      method: "GET"
    })
  },

  async addWorkspaceSource(
    workspaceId: string,
    data: WorkspaceSourceCreateRequest
  ): Promise<WorkspaceSourceApiResponse> {
    return await bgRequest<WorkspaceSourceApiResponse>({
      path: workspacePath(workspaceId, "/sources"),
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
    const encodedSourceId = encodeWorkspacePathSegment(sourceId, "sourceId")
    return await bgRequest<WorkspaceSourceApiResponse>({
      path: workspacePath(workspaceId, `/sources/${encodedSourceId}`),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async updateWorkspaceSourceSelection(
    workspaceId: string,
    selectedSourceIds: string[]
  ): Promise<void> {
    await bgRequest<unknown>({
      path: workspacePath(workspaceId, "/sources/selection"),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { selected_ids: selectedSourceIds }
    })
  },

  async deleteWorkspaceSource(workspaceId: string, sourceId: string): Promise<void> {
    const encodedSourceId = encodeWorkspacePathSegment(sourceId, "sourceId")
    await bgRequest<unknown>({
      path: workspacePath(workspaceId, `/sources/${encodedSourceId}`),
      method: "DELETE"
    })
  },

  async getWorkspaceArtifacts(
    workspaceId: string
  ): Promise<WorkspaceArtifactApiResponse[]> {
    return await bgRequest<WorkspaceArtifactApiResponse[]>({
      path: workspacePath(workspaceId, "/artifacts"),
      method: "GET"
    })
  },

  async addWorkspaceArtifact(
    workspaceId: string,
    data: WorkspaceArtifactCreateRequest
  ): Promise<WorkspaceArtifactApiResponse> {
    return await bgRequest<WorkspaceArtifactApiResponse>({
      path: workspacePath(workspaceId, "/artifacts"),
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
    const encodedArtifactId = encodeWorkspacePathSegment(
      artifactId,
      "artifactId"
    )
    return await bgRequest<WorkspaceArtifactApiResponse>({
      path: workspacePath(workspaceId, `/artifacts/${encodedArtifactId}`),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspaceArtifact(workspaceId: string, artifactId: string): Promise<void> {
    const encodedArtifactId = encodeWorkspacePathSegment(
      artifactId,
      "artifactId"
    )
    await bgRequest<unknown>({
      path: workspacePath(workspaceId, `/artifacts/${encodedArtifactId}`),
      method: "DELETE"
    })
  },

  async getWorkspaceNotes(
    workspaceId: string
  ): Promise<WorkspaceNoteApiResponse[]> {
    return await bgRequest<WorkspaceNoteApiResponse[]>({
      path: workspacePath(workspaceId, "/notes"),
      method: "GET"
    })
  },

  async addWorkspaceNote(
    workspaceId: string,
    data: WorkspaceNoteCreateRequest
  ): Promise<WorkspaceNoteApiResponse> {
    return await bgRequest<WorkspaceNoteApiResponse>({
      path: workspacePath(workspaceId, "/notes"),
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
      path: workspacePath(workspaceId, `/notes/${noteId}`),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async deleteWorkspaceNote(workspaceId: string, noteId: number): Promise<void> {
    await bgRequest<unknown>({
      path: workspacePath(workspaceId, `/notes/${noteId}`),
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
