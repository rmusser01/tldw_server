export type PersonaVisualBuiltinStateId =
  | "idle"
  | "wake_armed"
  | "listening"
  | "thinking"
  | "speaking"
  | "tool_running"
  | "approval_needed"
  | "error"
  | "offline"

export type PersonaVisualCustomStateId = string & {}
export type PersonaVisualStateId =
  | PersonaVisualBuiltinStateId
  | PersonaVisualCustomStateId

export type PersonaVisualRendererType =
  | "sprite_frames"
  | "sprite_sheet"
  | "static_image"
  | "live2d"

export type PersonaVisualRendererSetupStatus =
  | "supported"
  | "unsupported_renderer"
  | "feature_gated"
  | "dependency_missing"
  | "license_review_required"

export interface PersonaVisualRendererCapability {
  renderer_type: string
  display_name: string
  manifest_versions: number[]
  can_validate: boolean
  can_activate: boolean
  buddy_runtime_supported: boolean
  import_supported: boolean
  export_supported: boolean
  disabled_reason?: string | null
  renderer_contract_versions?: number[]
  supported_asset_roles?: string[]
  required_role_categories?: string[]
  role_category_map?: Record<string, string[]>
  allowed_mime_types?: string[]
  allowed_extensions?: string[]
  max_file_count?: number | null
  max_total_bytes?: number | null
  max_texture_width?: number | null
  max_texture_height?: number | null
  feature_flag?: string | null
  setup_status?: PersonaVisualRendererSetupStatus
  setup_blockers?: string[]
  requires_static_fallback?: boolean
  requires_license_ack?: boolean
}

export interface PersonaVisualRendererCapabilitiesResponse {
  renderers: PersonaVisualRendererCapability[]
}

export type PersonaVisualPackStatus =
  | "draft"
  | "review"
  | "active"
  | "archived"
  | "failed"

export type PersonaVisualAssetRole =
  | "frame"
  | "still_pose"
  | "sprite_sheet"
  | "preview"
  | "generated_candidate"

export type PersonaVisualCandidateStatus =
  | "review"
  | "accepted"
  | "rejected"
  | "failed"

export type PersonaVisualPortabilityOperation =
  | "export"
  | "import_preview"
  | "import_commit"

export interface PersonaVisualRegion {
  x: number
  y: number
  width: number
  height: number
}

export interface PersonaVisualFrame {
  asset_id: string
  region?: PersonaVisualRegion | null
  duration_ms?: number
}

export interface PersonaVisualAnimation {
  frames?: PersonaVisualFrame[]
  asset_ids?: string[]
  frame_rate?: number
  loop?: boolean
  alignment?: { x: number; y: number }
  preview_frame?: number
  preview_asset_id?: string
}

export type PersonaVisualStateCatalogKind =
  | "tool_variant"
  | "reaction"
  | "live_variant"
  | "mcp_runtime"
  | "mood"
  | "pack_private"

export interface PersonaVisualStateCatalogEntry {
  label: string
  kind: PersonaVisualStateCatalogKind
  description?: string | null
  tags?: string[]
}

export type PersonaVisualAuthoredTriggerSource =
  | "live_state"
  | "tool_category"
  | "mcp_runtime"
  | "tool_name"

export interface PersonaVisualAuthoredTrigger {
  id: string
  source: PersonaVisualAuthoredTriggerSource
  match: string
  state: PersonaVisualStateId
  duration_ms: number
  priority: number
}

export interface PersonaVisualManifest {
  manifest_version: 1
  renderer_type: PersonaVisualRendererType
  states: Partial<Record<PersonaVisualStateId, { animation_id: string }>>
  animations: Record<string, PersonaVisualAnimation>
  fallbacks?: Partial<Record<PersonaVisualStateId, PersonaVisualStateId[]>>
  state_catalog?: Record<PersonaVisualCustomStateId, PersonaVisualStateCatalogEntry>
  authored_triggers?: PersonaVisualAuthoredTrigger[]
}

export interface PersonaVisualAsset {
  id: string
  pack_id?: string
  persona_id?: string
  asset_role: PersonaVisualAssetRole | string
  storage_key?: string
  url: string
  original_filename?: string | null
  mime_type: string
  byte_size?: number
  checksum_sha256?: string
  width?: number | null
  height?: number | null
  duration_ms?: number | null
  provenance?: string
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualPack {
  id: string
  persona_id: string
  user_id?: string
  title: string
  renderer_type: PersonaVisualRendererType
  status: PersonaVisualPackStatus
  manifest_version?: number
  manifest: PersonaVisualManifest
  parent_pack_id?: string | null
  revision_number?: number
  provenance?: string
  active_at?: string | null
  assets?: PersonaVisualAsset[]
  assets_by_id?: Record<string, PersonaVisualAsset>
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualPackCreate {
  title: string
  manifest?: Partial<PersonaVisualManifest> | Record<string, unknown>
}

export interface PersonaVisualPackDuplicateRequest {
  target_persona_id: string
  title?: string | null
}

export interface PersonaVisualDuplicateTarget {
  id: string
  name?: string | null
}

export interface PersonaVisualLibraryItem {
  id: string
  user_id?: string
  source_persona_id?: string | null
  source_pack_id?: string | null
  title: string
  notes?: string | null
  tags: string[]
  source_persona_name?: string | null
  source_pack_title?: string | null
  source_pack_version?: number | null
  source_current_version?: number | null
  source_available: boolean
  source_changed: boolean
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualLibrarySaveRequest {
  title?: string | null
  notes?: string | null
  tags?: string[]
}

export interface PersonaVisualLibraryUpdateRequest {
  title?: string | null
  notes?: string | null
  tags?: string[] | null
  expected_version?: number | null
}

export interface PersonaVisualLibraryUseRequest {
  target_persona_id: string
  title?: string | null
}

export interface PersonaVisualLibraryListResponse {
  items: PersonaVisualLibraryItem[]
}

export interface PersonaVisualLibraryDeleteResponse {
  status: "deleted"
  item_id: string
}

export interface PersonaVisualManifestUpdate {
  manifest: PersonaVisualManifest | Record<string, unknown>
  expected_version?: number | null
}

export interface PersonaVisualPackListResponse {
  packs: PersonaVisualPack[]
  active_pack?: PersonaVisualPack | null
}

export interface PersonaVisualCandidate {
  id: string
  pack_id: string
  persona_id: string
  user_id?: string
  job_id?: string | null
  status: PersonaVisualCandidateStatus
  proposed_manifest_patch?: Record<string, unknown>
  generated_asset_ids?: string[]
  generated_assets?: PersonaVisualAsset[]
  prompt?: string | null
  failure_reason?: string | null
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualCandidateListResponse {
  candidates: PersonaVisualCandidate[]
}

export interface PersonaVisualGenerationRequest {
  prompt: string
  target_state?: PersonaVisualStateId | string | null
  backend?: string | null
}

export interface PersonaVisualGenerationJobResponse {
  job_id: string
  status?: string | null
}

export interface PersonaVisualGenerationReadinessResponse {
  available: boolean
  worker_enabled: boolean
  queue: string
  image_backend_available: boolean
  default_backend?: string | null
  requested_backend?: string | null
  requested_backend_available?: boolean | null
  enabled_backends: string[]
  reasons: string[]
}

export interface PersonaVisualPackExportRequest {
  request_id?: string | null
  strict?: boolean
  include_full_provenance?: boolean
  warn_for_sharing?: boolean
}

export interface PersonaVisualPortabilityJobResponse {
  job_id: string
  portability_job_id: string
  operation: PersonaVisualPortabilityOperation
  persona_id?: string | null
  pack_id?: string | null
  status: string
  visual_status: string
  stage: string
  progress: Record<string, unknown>
  warnings: unknown[]
  archive_sha256?: string | null
  canonical_payload_fingerprint?: string | null
  download_url?: string | null
  error_code?: string | null
  error_message?: string | null
  expires_at?: string | null
}

export interface PersonaVisualPackExportResponse {
  job_id: string
  portability_job_id: string
  operation: "export"
  persona_id: string
  pack_id: string
  status: string
  stage: string
  download_url?: string | null
}

export interface PersonaVisualImportPreviewStartResponse {
  preview_id: string
  job_id: string
  portability_job_id: string
  operation: "import_preview"
  target_persona_id?: string | null
  status: string
  stage: string
}

export type PersonaVisualImportTargetMode = "create_new" | "replace_draft"

export interface PersonaVisualImportConflict {
  conflict_id?: string
  type?: string
  severity?: string
  message?: string
  pack_id?: string
  pack_title?: string
  pack_status?: string
  allowed_choices?: PersonaVisualImportTargetMode[]
}

export interface PersonaVisualImportRequiredChoice {
  choice_id: string
  reason?: string
  default_target_mode?: PersonaVisualImportTargetMode
  allowed_target_modes?: PersonaVisualImportTargetMode[]
  replaceable_pack_ids?: string[]
}

export interface PersonaVisualRendererImportPreview {
  status?: string | null
  renderer_type?: string | null
  manifest_version?: number | null
  renderer_contract_version?: number | null
  can_commit?: boolean | null
  activation_eligible?: boolean | null
  blockers?: string[]
  warnings?: string[]
  normalized_role_categories?: Record<string, string[]>
  setup_status?: string | null
  setup_blockers?: string[]
  disabled_reason?: string | null
}

export interface PersonaVisualImportProposedPlan extends Record<string, unknown> {
  target_mode?: PersonaVisualImportTargetMode | string
  target_modes?: Array<PersonaVisualImportTargetMode | string>
  default_target_mode?: PersonaVisualImportTargetMode | string
  commit_eligible?: boolean
  activation_eligible?: boolean
  commit_blockers?: string[]
  renderer_import_preview?: PersonaVisualRendererImportPreview
}

export interface PersonaVisualImportPreviewResponse {
  preview_id: string
  job_id: string
  portability_job_id: string
  operation: "import_preview"
  target_persona_id?: string | null
  status: string
  visual_status: string
  stage: string
  archive_sha256?: string | null
  canonical_payload_fingerprint?: string | null
  schema_version?: string | null
  bundle_summary: Record<string, unknown>
  validation_warnings: unknown[]
  conflicts: PersonaVisualImportConflict[]
  proposed_plan: PersonaVisualImportProposedPlan
  quota_estimate: Record<string, unknown>
  required_choices: PersonaVisualImportRequiredChoice[]
  target_warnings: unknown[]
  error_code?: string | null
  error_message?: string | null
  expires_at?: string | null
}

export interface PersonaVisualImportCommitRequest {
  request_id?: string | null
  trust_mode?: "trusted_restore" | "untrusted_import"
  target_mode?: PersonaVisualImportTargetMode
  target_pack_id?: string | null
  title?: string | null
}

export interface PersonaVisualImportCommitStartResponse {
  job_id: string
  portability_job_id: string
  operation: "import_commit"
  preview_id: string
  target_persona_id: string
  status: string
  stage: string
}

export interface PersonaVisualCandidateReviewRequest {
  status: "accepted" | "rejected" | "failed"
  failure_reason?: string | null
}

export interface PersonaVisualDeactivateResponse {
  status: "deactivated"
  persona_id: string
}
