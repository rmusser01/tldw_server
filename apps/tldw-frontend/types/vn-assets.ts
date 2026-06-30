export type VNAssetReviewStatus = 'draft' | 'approved' | 'rejected' | 'hidden';
export type VNAssetSlotStatus =
  | 'planned'
  | 'queued'
  | 'generating'
  | 'reviewing'
  | 'approved'
  | 'failed'
  | 'skipped'
  | 'cancelled';

export interface VNAssetPack {
  id: number;
  owner_user_id?: number;
  title: string;
  primary_character_id: number;
  description?: string | null;
  status?: string;
  content_rating?: string;
  source_world_book_ids?: number[];
  scenario_notes?: string | null;
  style_prompt?: string | null;
  negative_prompt?: string | null;
  default_backend?: string | null;
  default_model?: string | null;
  default_dimensions?: Record<string, unknown> | null;
  style_lock?: Record<string, unknown> | null;
  generation_budget?: Record<string, unknown> | null;
  planned_output_count?: number;
  created_at?: string | null;
  updated_at?: string | null;
  version?: number;
  deleted?: boolean;
}

export interface VNAssetPackCreate {
  title: string;
  primary_character_id: number;
  description?: string | null;
  content_rating?: string;
  source_world_book_ids?: number[];
  scenario_notes?: string | null;
  style_prompt?: string | null;
  negative_prompt?: string | null;
  default_backend?: string | null;
  default_model?: string | null;
  default_dimensions?: Record<string, unknown> | null;
  style_lock?: Record<string, unknown> | null;
  generation_budget?: Record<string, unknown> | null;
  apply_starter_matrix?: boolean;
  starter_matrix_variant_count?: number;
}

export type VNAssetPackUpdate = Partial<Omit<VNAssetPackCreate, 'primary_character_id' | 'apply_starter_matrix' | 'starter_matrix_variant_count'>>;

export interface VNAssetStarterMatrix {
  key: string;
  title: string;
  slot_count: number;
  planned_output_count: number;
  asset_types: string[];
}

export interface VNAssetStarterMatricesResponse {
  matrices: VNAssetStarterMatrix[];
}

export interface VNAssetSlot {
  id: number;
  pack_id: number;
  asset_type: string;
  slot_key: string;
  labels?: Record<string, unknown>;
  prompt_template?: string | null;
  negative_prompt_template?: string | null;
  variant_count: number;
  width?: number | null;
  height?: number | null;
  backend_override?: string | null;
  model_override?: string | null;
  seed_policy?: Record<string, unknown> | null;
  requires_review?: boolean;
  required_for_runtime?: boolean;
  depends_on_slot_id?: number | null;
  status: VNAssetSlotStatus | string;
  last_error?: string | null;
}

export type VNAssetSlotUpdate = Partial<Omit<VNAssetSlot, 'id' | 'pack_id'>>;

export interface VNAssetItem {
  id: number;
  pack_id: number;
  slot_id: number;
  variant_index: number;
  file_artifact_id?: string | null;
  generated_file_id?: number | null;
  storage_ref?: string | null;
  mime_type?: string | null;
  width?: number | null;
  height?: number | null;
  bytes?: number | null;
  review_status: VNAssetReviewStatus | string;
  preferred: boolean;
  source: string;
  generation_job_id?: string | null;
  depth_kind?: string | null;
  parent_item_id?: number | null;
}

export interface VNAssetGenerationRequest {
  slot_ids?: number[];
  variant_count?: number | null;
  options?: Record<string, unknown>;
}

export interface VNAssetGenerationStatus {
  batch_id?: number | null;
  job_batch_id?: string | null;
  status: string;
  total_slots?: number;
  total_variants?: number;
  planned_count?: number;
  enqueued_count?: number;
  completed_count?: number;
  failed_count?: number;
  cancelled_count?: number;
  enqueue_error?: string | null;
}

export interface VNAssetReviewRequest {
  review_status: VNAssetReviewStatus;
  preferred?: boolean | null;
}

export interface VNAssetBulkReviewRequest {
  item_ids: number[];
  review_status: VNAssetReviewStatus;
}

export interface VNAssetReadiness {
  ready: boolean;
  status: string;
  warnings: string[];
  errors: string[];
}

export interface VNAssetManifest {
  schema_version: string;
  pack_id: number;
  title: string;
  primary_character_id: number;
  content_rating: string;
  assets: Record<string, Array<Record<string, unknown>>>;
}

export interface VNAssetPromptPreviewRequest {
  slot_id: number;
  variant_index?: number;
  budgets?: Record<string, number> | null;
}

export interface VNAssetPromptPreview {
  prompt: string;
  negative_prompt: string;
  omitted_source_counts: Record<string, number>;
  token_estimates: Record<string, number>;
  warnings: string[];
}

export interface VNPackExportRequest {
  include_character_payload?: boolean;
  include_world_book_payloads?: boolean;
  include_full_provenance?: boolean;
  strict?: boolean;
  warn_for_sharing?: boolean;
  idempotency_key: string;
  request_id?: string | null;
}

export interface VNPackExportResponse {
  job_id: string;
  portability_job_id: number;
  operation: string;
  pack_id?: number | null;
  status: string;
  stage: string;
  download_url?: string | null;
}

export interface VNPackPortabilityJob {
  job_id: string;
  portability_job_id: number;
  operation: string;
  pack_id?: number | null;
  status: string;
  vn_status: string;
  stage: string;
  progress?: Record<string, unknown>;
  warnings?: unknown[];
  archive_sha256?: string | null;
  canonical_payload_fingerprint?: string | null;
  download_url?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  expires_at?: string | null;
}

export interface VNPackImportPreviewStartResponse {
  job_id: string;
  portability_job_id: number;
  operation: string;
  preview_id: number;
  status: string;
  stage: string;
}

export interface VNPackImportPreview {
  preview_id: number;
  job_id: string;
  portability_job_id: number;
  operation: string;
  status: string;
  vn_status: string;
  stage: string;
  archive_sha256?: string | null;
  canonical_payload_fingerprint?: string | null;
  schema_version?: string | null;
  bundle_summary?: Record<string, unknown>;
  validation_warnings?: unknown[];
  conflicts?: unknown[];
  proposed_plan?: Record<string, unknown>;
  quota_estimate?: Record<string, unknown>;
  required_choices?: unknown[];
  error_code?: string | null;
  error_message?: string | null;
  expires_at?: string | null;
}

export interface VNPackImportCommitRequest {
  preview_id: number;
  trust_mode: 'trusted_restore' | 'untrusted_import';
  target_mode?: 'create_new' | 'update_existing';
  character_action:
    | 'import_included_character'
    | 'link_existing_character'
    | 'create_placeholder_character'
    | 'fail_import';
  target_character_id?: number | null;
  target_pack_id?: number | null;
  conflict_decisions?: Record<string, unknown>;
  idempotency_key: string;
  request_id?: string | null;
}

export interface VNPackImportCommitStartResponse {
  job_id: string;
  portability_job_id: number;
  operation: string;
  preview_id: number;
  import_id: number;
  status: string;
  stage: string;
}

export interface VNPackImportJob {
  job_id: string;
  portability_job_id: number;
  operation: string;
  preview_id: number;
  import_id: number;
  status: string;
  vn_status: string;
  stage: string;
  pack_id?: number | null;
  id_maps?: Record<string, unknown>;
  created_records?: Record<string, unknown>;
  cleanup_status?: Record<string, unknown>;
  warnings?: unknown[];
  archive_sha256?: string | null;
  canonical_payload_fingerprint?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  completed_at?: string | null;
}
