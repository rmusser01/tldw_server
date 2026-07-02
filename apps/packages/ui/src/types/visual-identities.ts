export type VisualIdentityActorKind = "character" | "persona"
export type VisualIdentityPackStatus = "active" | "archived" | "deleted"
export type VisualIdentityDraftStatus =
  | "importing"
  | "ready_for_review"
  | "failed"
  | "abandoned"
  | "activated"
export type VisualIdentityBindingStatus = "active" | "deleted"

export interface VisualIdentityCapabilitiesResponse {
  upload_max_bytes: number
  archive_max_bytes: number
  max_dimension: number
  max_frame_count: number
  supported_mime_types: string[]
  avif_enabled: boolean
}

export interface VisualIdentityExpressionSlotResponse {
  key: string
  label: string
  canonical: boolean
  aliases: string[]
}

export interface VisualIdentityPackCreate {
  title: string
  description?: string
  default_expression_key?: string
  source_kind?: string
  source_context?: Record<string, unknown>
}

export interface VisualIdentityPackUpdate {
  title?: string
  description?: string | null
  status?: Extract<VisualIdentityPackStatus, "active" | "archived">
  default_expression_key?: string | null
  source_kind?: string | null
  source_context?: Record<string, unknown> | null
}

export interface VisualIdentityPackResponse {
  id: number
  owner_user_id: number
  title: string
  description: string
  status: VisualIdentityPackStatus | string
  active_version_id: number | null
  default_expression_key: string
  source_kind: string
  source_context: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
  version: number
}

export interface VisualIdentityAssetResponse {
  id: number
  owner_user_id: number
  pack_id: number | null
  draft_id: number | null
  pack_version_id: number | null
  expression_key: string
  original_expression_key: string
  display_label: string
  source_filename: string
  content_type: string
  bytes: number
  sha256: string
  width: number
  height: number
  is_animated: boolean
  frame_count: number | null
  duration_ms: number | null
  preview_relpath: string | null
  created_at: string | null
  updated_at: string | null
}

export interface VisualIdentityDraftResponse {
  id: number
  owner_user_id: number
  pack_id: number | null
  title: string
  status: VisualIdentityDraftStatus | string
  source_kind: string
  source_filename: string
  import_job_id: string | null
  validation_summary: Record<string, unknown>
  slot_map: Record<string, unknown>
  default_expression_key: string
  error: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
  version: number
  assets: VisualIdentityAssetResponse[]
  pack_version_id: number | null
  asset_ids: number[]
  binding_id: number | null
}

export interface VisualIdentityDraftSlotUpdate {
  asset_id?: number | null
  expression_key?: string | null
  display_label?: string | null
  metadata?: Record<string, unknown>
}

export interface VisualIdentityDraftActivateRequest {
  actor_kind?: VisualIdentityActorKind | null
  actor_id?: number | string | null
}

export interface VisualIdentityBindingRequest {
  actor_kind: VisualIdentityActorKind
  actor_id: number | string
  pack_id: number
  active_version_id?: number | null
}

export interface VisualIdentityBindingResponse {
  id: number
  owner_user_id: number
  actor_kind: VisualIdentityActorKind
  actor_id: string
  pack_id: number
  active_version_id: number
  status: VisualIdentityBindingStatus | string
  created_at: string | null
  updated_at: string | null
  version: number
}

export interface VisualIdentityResolveRequest {
  actor_kind: VisualIdentityActorKind
  actor_id: number | string
  expression_key?: string
  manual_override_expression_key?: string | null
  mood_expression_key?: string | null
}

export interface VisualIdentityResolveResponse {
  actor_kind: VisualIdentityActorKind
  actor_id: number | string
  pack_id: number | null
  pack_version_id: number | null
  expression_key: string | null
  requested_expression_key: string | null
  asset_id: number | null
  storage_relpath: string | null
  fallback_reason: string | null
  is_animated: boolean
  content_type: string | null
  asset_url: string | null
}

export interface VisualIdentityImportZipStartResponse {
  draft_id: number
  job_id: string | number | null
  status: string
  source_filename: string
  import_job_id: string | null
}

export interface VisualIdentityGeneratedFileAssetRequest {
  generated_file_id: number
  expression_key: string
  draft_id?: number | null
  source_feature?: string
  idempotency_key: string
}

export interface VisualIdentityUploadFile {
  name?: string
  type?: string
  data: ArrayBuffer | Uint8Array | number[]
}

export interface VisualIdentityAssetUploadRequest {
  expression_key: string
  draft_id?: number | null
  file: VisualIdentityUploadFile
  timeoutMs?: number
}

export interface VisualIdentityZipImportRequest {
  archive: VisualIdentityUploadFile
  title?: string
  pack_id?: number | null
  idempotency_key: string
  timeoutMs?: number
}
