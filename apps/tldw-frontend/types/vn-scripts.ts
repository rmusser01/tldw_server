export type VNScriptStatus = 'draft' | 'ready' | 'archived';
export type VNScriptContentRating = 'general' | 'teen' | 'suggestive' | 'mature';

export interface VNScriptCreate {
  title: string;
  description?: string | null;
  primary_asset_pack_id: number;
  policy_profile_id?: string;
  generation_profile_id?: string;
  generation_profiles?: Record<string, string>;
  generation_profile_ids?: Record<string, string> | null;
  content_rating?: VNScriptContentRating;
}

export interface VNScriptPatch {
  title?: string | null;
  description?: string | null;
  status?: VNScriptStatus | null;
  primary_asset_pack_id?: number | null;
  policy_profile_id?: string | null;
  generation_profile_id?: string | null;
  generation_profiles?: Record<string, string> | null;
  generation_profile_ids?: Record<string, string> | null;
  content_rating?: VNScriptContentRating | null;
}

export interface VNScriptResponse {
  id: number;
  title: string;
  description?: string | null;
  status: string;
  primary_asset_pack_id: number;
  policy_profile_id: string;
  generation_profile_id: string;
  generation_profiles: Record<string, string>;
  content_rating: string;
}

export interface VNScriptOffsetPagination {
  limit: number;
  offset: number;
  total?: number | null;
  has_more: boolean;
  next_offset?: number | null;
}

export interface VNScriptListResponse {
  items: VNScriptResponse[];
  limit: number;
  offset: number;
  total: number;
  has_more: boolean;
  next_offset?: number | null;
  pagination: VNScriptOffsetPagination;
}

export interface VNScriptTemplateSummary {
  id: string;
  label: string;
  description: string;
  category: string;
  recommended_content_rating: VNScriptContentRating;
  required_capabilities: string[];
  preview: Record<string, unknown>;
  default_title: string;
  default_description?: string | null;
}

export interface VNScriptTemplateListResponse {
  items: VNScriptTemplateSummary[];
}

export type VNScriptCreateFromTemplateRequest = VNScriptCreate;

export interface VNScriptCreateFromTemplateResponse {
  script: VNScriptResponse;
  draft: VNScriptDraftResponse;
}

export interface VNScriptDraftResponse {
  script_id: number;
  revision: number;
  draft: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
}

export interface VNScriptDraftPutRequest {
  if_revision: number;
  draft: Record<string, unknown>;
}

export interface VNScriptValidateRequest {
  draft?: Record<string, unknown> | null;
}

export interface VNScriptValidationResponse {
  valid: boolean;
  errors: Array<Record<string, unknown>>;
  warnings: Array<Record<string, unknown>>;
}

export interface VNScriptDiagnosticsResponse {
  script_id: number;
  revision: number;
  diagnostics: Record<string, unknown>;
}

export interface VNScriptPublishRequest {
  draft_revision: number;
  label?: string | null;
  idempotency_key: string;
  acknowledgements?: string[];
}

export interface VNScriptPublishResponse {
  script_id: number;
  version_id: number;
  version_number: number;
  status: string;
  asset_pack_id: number;
  manifest_snapshot_id: number;
  policy_snapshot_id: number;
  generation_profile_snapshot_id: number;
  generation_profile_snapshots: Record<string, number>;
  validation: Record<string, unknown>;
  created_at: string;
}

export interface VNScriptVersionResponse {
  id: number;
  script_id: number;
  version_number: number;
  label?: string | null;
  draft_revision: number;
  program: Record<string, unknown>;
  asset_pack_id: number;
  manifest_snapshot_id: number;
  policy_snapshot_id: number;
  generation_profile_snapshot_id: number;
  generation_profile_snapshots: Record<string, number>;
  script_defaults: Record<string, unknown>;
  validation: Record<string, unknown>;
  created_at: string;
}

export interface VNScriptVersionListResponse {
  items: VNScriptVersionResponse[];
  limit: number;
  offset: number;
  total: number;
  has_more: boolean;
  next_offset?: number | null;
  pagination: VNScriptOffsetPagination;
}

export interface VNScriptManifestSnapshotResponse {
  id: number;
  script_id: number;
  version_id?: number | null;
  asset_pack_id: number;
  manifest: Record<string, unknown>;
  manifest_hash: string;
  created_at: string;
}

export interface VNScriptVersionPolicyEvaluateRequest {
  context?: Record<string, unknown>;
}

export interface VNScriptVersionPolicyEvaluateResponse {
  decision: string;
  profile_id: string;
  reasons: Array<Record<string, unknown>>;
  blocked: boolean;
  requires_acknowledgement: boolean;
  remediation: string[];
}

export interface VNScriptListQuery {
  limit?: number;
  offset?: number;
}

export interface VNScriptVersionListQuery {
  limit?: number;
  offset?: number;
}
