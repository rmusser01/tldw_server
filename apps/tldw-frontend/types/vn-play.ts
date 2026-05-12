export type VNPlayMode = 'freeform' | 'story';
export type VNPlaySessionStatus = 'active' | 'paused' | 'completed' | 'archived' | 'failed';
export type VNPlayTrustLevel = 'local' | 'trusted_restore' | 'untrusted_import' | 'mixed';
export type VNPlayLinkedChatMode = 'read_only_context';
export type VNPlaySetupTrustLevel = 'local' | 'trusted_restore' | 'untrusted_import' | 'unknown';
export type VNPlaySetupTrustSource = 'local_pack' | 'latest_import_journal' | 'unknown';
export type VNPlaySetupWarningSeverity = 'info' | 'warning' | 'high_risk';
export type VNPlaySetupCompatibilityStatus = 'compatible' | 'different_character' | 'unknown';
export type VNPlaySetupEmptyStateScope = 'global' | 'filter' | 'page';
export type VNPlayTurnStatus =
  | 'pending'
  | 'model_calling'
  | 'model_failed'
  | 'parse_failed'
  | 'completed'
  | 'abandoned'
  | 'cancelled';
export type VNPlayGenerationRawDebugState = 'absent' | 'available' | 'redacted' | 'revealed';

export interface VNPlayChoice {
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
  source?: string;
  generation_id?: number;
  revision_id?: number;
}

export interface VNPlaySceneAsset {
  item_id?: number;
  content_url?: string;
  url?: string;
  src?: string;
  labels?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface VNPlaySceneState {
  session_id?: number;
  owner_user_id?: number;
  last_event_id?: number | null;
  background?: VNPlaySceneAsset | null;
  depth?: VNPlaySceneAsset | null;
  active_sprites?: VNPlaySceneAsset[];
  current_background_item_id?: number | null;
  current_depth_item_id?: number | null;
  active_sprite_items?: Array<Record<string, unknown>>;
  location_key?: string | null;
  mood?: string | null;
  time_of_day?: string | null;
  weather?: string | null;
  active_branch_node_id?: number | null;
  visible_choices?: VNPlayChoice[] | Array<Record<string, unknown>>;
  waiting_generation_request_id?: number | null;
  waiting_generation_confirmation?: Record<string, unknown> | null;
  waiting_reason?: string | null;
  transcript_cursor?: number | null;
  scene_version: number;
  warnings?: unknown[];
  updated_at?: string | null;
}

export interface VNPlaySession {
  id: number;
  owner_user_id?: number;
  mode: VNPlayMode;
  title: string;
  status?: VNPlaySessionStatus;
  primary_character_id: number;
  additional_character_ids?: number[];
  linked_chat_id?: string | null;
  vn_asset_pack_id: number;
  asset_manifest_version?: string | null;
  source_world_book_ids?: number[];
  content_rating?: string;
  trust_level?: VNPlayTrustLevel;
  linked_chat_mode?: VNPlayLinkedChatMode;
  seed?: string | null;
  settings?: Record<string, unknown>;
  scene_version: number;
  active_turn_request_id?: number | null;
  current_scene?: VNPlaySceneState | null;
  scene_state?: VNPlaySceneState | null;
  created_at?: string | null;
  updated_at?: string | null;
  deleted?: boolean;
}

export interface VNPlaySessionCreate {
  mode: VNPlayMode;
  title: string;
  primary_character_id: number;
  vn_asset_pack_id: number;
  additional_character_ids?: number[];
  linked_chat_id?: string | null;
  asset_manifest_version?: string | null;
  source_world_book_ids?: number[];
  content_rating?: string;
  trust_level?: VNPlayTrustLevel;
  linked_chat_mode?: VNPlayLinkedChatMode;
  seed?: string | null;
  settings?: Record<string, unknown>;
}

export interface VNPlaySetupCharacterOption {
  id: number;
  name: string;
  description_preview?: string | null;
  tags: string[];
  favorite: boolean;
  deleted: boolean;
  has_image: boolean;
}

export interface VNPlaySetupCompatibility {
  status: VNPlaySetupCompatibilityStatus;
  reason_codes: string[];
}

export interface VNPlaySetupWarning {
  code: string;
  severity: VNPlaySetupWarningSeverity;
  message: string;
  requires_acknowledgement: boolean;
}

export interface VNPlaySetupWarningSummary {
  highest_severity: VNPlaySetupWarningSeverity;
  requires_acknowledgement: boolean;
  warnings: VNPlaySetupWarning[];
}

export interface VNPlaySetupAssetPackOption {
  id: number;
  title: string;
  primary_character_id: number;
  content_rating: string;
  status: string;
  trust_level: VNPlaySetupTrustLevel;
  trust_source: VNPlaySetupTrustSource;
  ready: boolean;
  readiness_status: string;
  readiness_warnings: string[];
  readiness_errors: string[];
  compatibility: VNPlaySetupCompatibility;
  warning_summary: VNPlaySetupWarningSummary;
  recommended: boolean;
}

export interface VNPlaySetupDefaults {
  mode?: VNPlayMode | null;
  character_id?: number | null;
  asset_pack_id?: number | null;
  content_rating?: string | null;
}

export interface VNPlaySetupPagination {
  limit: number;
  offset: number;
  has_more: boolean;
  total?: number | null;
}

export interface VNPlaySetupEmptyState {
  code: string;
  scope: VNPlaySetupEmptyStateScope;
  message: string;
}

export interface VNPlaySetupOptionsResponse {
  characters: VNPlaySetupCharacterOption[];
  selected_character?: VNPlaySetupCharacterOption | null;
  asset_packs: VNPlaySetupAssetPackOption[];
  defaults: VNPlaySetupDefaults;
  pagination: {
    characters: VNPlaySetupPagination;
    asset_packs: VNPlaySetupPagination;
  };
  empty_states: VNPlaySetupEmptyState[];
  generated_at: string;
}

export interface VNPlaySetupOptionsQuery {
  mode?: VNPlayMode;
  character_query?: string;
  pack_query?: string;
  character_limit?: number;
  character_offset?: number;
  pack_limit?: number;
  pack_offset?: number;
  selected_character_id?: number;
  content_rating?: string;
}

export type VNPlaySessionUpdate = Partial<
  Pick<VNPlaySession, 'title' | 'status' | 'linked_chat_id' | 'linked_chat_mode' | 'settings' | 'deleted'>
>;

export interface VNPlayEvent {
  id: number;
  session_id: number;
  owner_user_id: number;
  sequence_number: number;
  event_type: string;
  event_payload: Record<string, unknown>;
  source: 'user' | 'model' | 'runtime' | 'system' | string;
  model_provider?: string | null;
  model_name?: string | null;
  branch_node_id?: number | null;
  created_at?: string | null;
}

export interface VNPlayTurnRequest {
  input_text?: string | null;
  choice_id?: string | null;
  custom_action?: Record<string, unknown> | null;
  client_scene_version: number;
  idempotency_key: string;
  provider?: string | null;
  model?: string | null;
  options?: Record<string, unknown>;
}

export interface VNPlayRetryTurnRequest {
  client_scene_version: number;
  idempotency_key: string;
}

export interface VNPlayTurnResponse {
  turn_request_id: number;
  status: VNPlayTurnStatus;
  scene_version: number;
  replayed?: boolean;
  session?: VNPlaySession | null;
  current_scene?: VNPlaySceneState | null;
  scene_state?: VNPlaySceneState | null;
  events?: VNPlayEvent[];
  warnings?: unknown[];
  error_code?: string | null;
  error_message?: string | null;
}

export interface VNPlayCheckpointCreate {
  label: string;
  event_id?: number | null;
  scene_version?: number | null;
}

export interface VNPlayCheckpoint {
  id: number;
  session_id: number;
  owner_user_id: number;
  label: string;
  event_id?: number | null;
  scene_version: number;
  scene_state_snapshot?: Record<string, unknown>;
  created_at?: string | null;
}

export interface VNPlayRestoreRequest {
  checkpoint_id: number;
  client_scene_version: number;
  idempotency_key: string;
}

export interface VNPlayGenerationActionRequest {
  client_scene_version: number;
  idempotency_key: string;
}

export interface VNPlayGenerationProfileSummary {
  profile_key: string;
  snapshot_id: number;
  provider_class?: string | null;
  moderation_required?: boolean | null;
  estimated_cost_class?: string | null;
}

export interface VNPlayGenerationHistoryItem {
  id: number;
  generation_id: number;
  generation_point_key: string;
  revision_number: number;
  status: string;
  active: boolean;
  output_schema: string;
  public_output: Record<string, unknown>;
  applied_visuals?: Array<Record<string, unknown>>;
  rejected_visuals?: Array<Record<string, unknown>>;
  public_error_code?: string | null;
  source?: string;
  profile: VNPlayGenerationProfileSummary;
  created_at?: string | null;
}

export interface VNPlayOffsetPagination {
  mode: 'offset';
  total?: number | null;
  limit: number;
  offset: number;
  has_more: boolean;
  next_offset?: number | null;
}

export interface VNPlayGenerationHistoryResponse {
  items: VNPlayGenerationHistoryItem[];
  pagination: VNPlayOffsetPagination;
  total?: number | null;
  limit?: number | null;
  offset?: number | null;
  has_more?: boolean | null;
  next_offset?: number | null;
}

export type VNPlayGenerationRevisionListResponse = VNPlayGenerationHistoryResponse;

export interface VNPlayGenerationListQuery {
  generation_id?: number;
  generation_point_key?: string;
  status?: string;
  active?: boolean;
  source?: string;
  created_after?: string;
  created_before?: string;
  limit?: number;
  offset?: number;
}

export interface VNPlayGenerationDebugQuery {
  include_blocked_raw?: boolean;
  confirm?: string;
}

export interface VNPlayGenerationRevisionDebugResponse {
  id: number;
  generation_id: number;
  generation_request_id: number;
  generation_point_key: string;
  revision_number: number;
  status: string;
  output_schema: string;
  public_output: Record<string, unknown>;
  raw_output_debug_state: VNPlayGenerationRawDebugState;
  raw_output_debug?: Record<string, unknown> | null;
  parser_diagnostics?: Record<string, unknown>;
  moderation_diagnostics?: Record<string, unknown>;
  model_metadata?: Record<string, unknown>;
  usage_metadata?: Record<string, unknown>;
  request?: Record<string, unknown>;
  profile: VNPlayGenerationProfileSummary;
  created_at?: string | null;
}

export interface VNPlayBranch {
  id: number;
  session_id: number;
  owner_user_id: number;
  parent_event_id?: number | null;
  branch_label?: string | null;
  branch_path?: unknown[];
  status?: string;
  created_at?: string | null;
  updated_at?: string | null;
}
