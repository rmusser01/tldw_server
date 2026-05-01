export type VNPlayMode = 'freeform' | 'story';
export type VNPlaySessionStatus = 'active' | 'paused' | 'completed' | 'archived' | 'failed';
export type VNPlayTrustLevel = 'local' | 'trusted_restore' | 'untrusted_import' | 'mixed';
export type VNPlayLinkedChatMode = 'read_only_context';
export type VNPlayTurnStatus =
  | 'pending'
  | 'model_calling'
  | 'model_failed'
  | 'parse_failed'
  | 'completed'
  | 'abandoned'
  | 'cancelled';

export interface VNPlayChoice {
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
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
  idempotency_key: string;
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
