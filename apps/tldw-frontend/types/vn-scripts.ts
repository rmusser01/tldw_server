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

export interface VNScriptAuthoringOperation {
  op: string;
  label: string;
  category: string;
  description?: string | null;
  fields: Array<Record<string, unknown>>;
  capability_tokens: string[];
  forbidden_fields: string[];
  supports_condition: boolean;
  preview?: Record<string, unknown> | null;
  output_compatibility: Record<string, unknown>;
  notes: string[];
}

export interface VNScriptAuthoringSnippet {
  id: string;
  schema_version: 'vn_script_program.v1';
  label: string;
  operation_sequence: string[];
  required_capability_tokens: string[];
  parameters_schema: Record<string, unknown>;
  default_parameters: Record<string, unknown>;
  preview: Array<Record<string, unknown>>;
}

export interface VNScriptAuthoringCatalogResponse {
  schema_version: 'vn_script_authoring_catalog.v1';
  program_schema_version: 'vn_script_program.v1';
  capability_tokens: string[];
  generation_output_schemas: string[];
  operation_categories: Record<string, string[]>;
  operations: VNScriptAuthoringOperation[];
  snippets: VNScriptAuthoringSnippet[];
  limits: Record<string, number>;
}

export type VNScriptSnippetAnchor =
  | { label: string; mode?: 'append'; op_index?: number | null }
  | { label: string; mode: 'before' | 'after'; op_index: number };

export interface VNScriptSnippetPreviewRequest {
  snippet_id: string;
  anchor: VNScriptSnippetAnchor;
  parameters?: Record<string, unknown>;
  draft?: Record<string, unknown> | null;
  draft_revision?: number | null;
}

export interface VNScriptSnippetApplyRequest {
  if_revision: number;
  snippet_id: string;
  anchor: VNScriptSnippetAnchor;
  parameters?: Record<string, unknown>;
}

export interface VNScriptSnippetPatchSummary {
  inserted_ops: number;
  created_labels: string[];
  changed_paths: string[];
}

export interface VNScriptSnippetPreviewResponse {
  script_id: number;
  base_revision: number;
  snippet_id: string;
  draft: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  patch_summary: VNScriptSnippetPatchSummary;
  warnings: Array<Record<string, unknown>>;
}

export interface VNScriptSnippetApplyResponse {
  script_id: number;
  revision: number;
  snippet_id: string;
  draft: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  patch_summary: VNScriptSnippetPatchSummary;
}

export type VNScriptCreateFromTemplateRequest = Omit<VNScriptCreate, 'title'> & {
  title?: string | null;
};

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
