export interface LlamacppSavedConfig {
  enabled: boolean
  executable_path?: string | null
  models_dir?: string | null
  default_host?: string | null
  default_port?: number | null
  default_threads?: number | null
  default_n_gpu_layers?: number | null
  default_ctx_size?: number | null
  allow_unvalidated_args?: boolean | null
  allow_cli_secrets?: boolean | null
  port_autoselect?: boolean | null
  port_probe_max?: number | null
  allowed_paths: string[]
  registered_model_paths: string[]
  imported_asset_folders: string[]
  log_output_file?: string | null
}

export interface LlamacppActiveConfig {
  handler_configured: boolean
  enabled?: boolean | null
  executable_path?: string | null
  models_dir?: string | null
  default_host?: string | null
  default_port?: number | null
  active_model?: string | null
  active_host?: string | null
  active_port?: number | null
  active_pid?: number | null
}

export interface LlamacppConfigResponse {
  saved_config: LlamacppSavedConfig
  active_config: LlamacppActiveConfig
  restart_required: boolean
  restart_reasons: string[]
  env_overrides: Record<string, boolean>
  warnings: string[]
}

export interface LlamacppConfigUpdateRequest {
  enabled?: boolean
  executable_path?: string | null
  models_dir?: string | null
  default_host?: string | null
  default_port?: number | null
  default_threads?: number | null
  default_n_gpu_layers?: number | null
  default_ctx_size?: number | null
  allow_unvalidated_args?: boolean
  allow_cli_secrets?: boolean
  port_autoselect?: boolean
  port_probe_max?: number | null
  allowed_paths?: string[] | null
  log_output_file?: string | null
}

export interface LlamacppValidationRequest {
  binary_path: string
  timeout_seconds?: number
  run_probe?: boolean
}

export interface LlamacppValidationResponse {
  valid: boolean
  exists: boolean
  executable: boolean
  resolved_path?: string | null
  version_output?: string | null
  help_output?: string | null
  warnings: string[]
}

export interface LlamacppModelMetadata {
  quantization?: string | null
  parameter_hint?: string | null
  context_hint?: number | null
}

export interface LlamacppInventoryItem {
  model_id: string
  display_name: string
  basename: string
  source: string
  path: string
  size_bytes?: number | null
  modified_at?: string | null
  metadata: LlamacppModelMetadata
  warnings: string[]
}

export interface LlamacppInventoryResponse {
  models: LlamacppInventoryItem[]
  warnings: string[]
  scan_limited: boolean
}

export type LlamacppAssetKind = "gguf" | "mmproj" | "folder" | "unknown"
export type LlamacppAssetSource =
  | "models_dir"
  | "registered_path"
  | "imported_folder"

export interface LlamacppAssetMetadata {
  quantization?: string | null
  parameter_hint?: string | null
  context_hint?: number | null
  family_hint?: string | null
}

export interface LlamacppAsset {
  asset_id: string
  kind: LlamacppAssetKind
  identity_basis: "resolved_path" | "manual"
  path: string
  resolved_path?: string | null
  display_name: string
  source: LlamacppAssetSource
  size_bytes?: number | null
  modified_at?: string | null
  metadata: LlamacppAssetMetadata
  capabilities: string[]
  mmproj_asset_ids: string[]
  base_model_asset_ids: string[]
  warnings: string[]
}

export interface LlamacppAssetsResponse {
  assets: LlamacppAsset[]
  warnings: string[]
  scan_limited: boolean
}

export interface LlamacppUseInChatResponse {
  provider: string
  endpoint: string
  updated: boolean
  effective: boolean
  warnings: string[]
}

export interface LlamacppLogTailResponse {
  lines: string[]
  truncated: boolean
  warnings: string[]
}

export type LlamacppProfileMode =
  | "chat"
  | "vision"
  | "embedding"
  | "rerank"
  | "server_generic"

export type LlamacppCapabilityKey = "chat" | "vision" | "embeddings" | "rerank"
export type LlamacppModalityDirection = "input" | "output"
export type LlamacppCapabilityMap = Partial<Record<LlamacppCapabilityKey, boolean>>
export type LlamacppModalities = Partial<Record<LlamacppModalityDirection, string[]>>

export type LlamacppPortPolicy = "explicit" | "autoselect"

export type LlamacppRuntimeState =
  | "defined"
  | "starting"
  | "running"
  | "stopped"
  | "failed"
  | "paused"

export interface LlamacppProfile {
  profile_id: string
  name: string
  enabled: boolean
  mode: LlamacppProfileMode
  model_id?: string | null
  model_path?: string | null
  mmproj_model_id?: string | null
  mmproj_path?: string | null
  mmproj_display_name?: string | null
  capabilities?: LlamacppCapabilityMap
  modalities?: LlamacppModalities
  capability_warnings?: string[]
  host: string
  port: number
  port_policy: LlamacppPortPolicy
  server_args: Record<string, unknown>
  autostart: boolean
  restart_policy: Record<string, unknown>
  provider_alias?: string | null
  tags: string[]
}

export type LlamacppProfileCreateRequest = Partial<
  Omit<
    LlamacppProfile,
    | "profile_id"
    | "server_args"
    | "restart_policy"
    | "tags"
    | "mmproj_path"
    | "mmproj_display_name"
    | "capabilities"
    | "modalities"
    | "capability_warnings"
  >
> & {
  profile_id?: string | null
  name: string
  server_args?: Record<string, unknown>
  restart_policy?: Record<string, unknown>
  tags?: string[]
}

export type LlamacppProfileUpdateRequest = Partial<
  Omit<
    LlamacppProfile,
    | "profile_id"
    | "mmproj_path"
    | "mmproj_display_name"
    | "capabilities"
    | "modalities"
    | "capability_warnings"
  >
>

export interface LlamacppProfileListResponse {
  profiles: LlamacppProfile[]
}

export interface LlamacppRuntime {
  profile_id: string
  state: LlamacppRuntimeState
  pid?: number | null
  host?: string | null
  port?: number | null
  endpoint?: string | null
  model_id?: string | null
  model_path?: string | null
  mmproj_model_id?: string | null
  mmproj_path?: string | null
  mmproj_display_name?: string | null
  capabilities?: LlamacppCapabilityMap
  modalities?: LlamacppModalities
  capability_warnings?: string[]
  resolved_args: string[]
  started_at?: string | null
  stopped_at?: string | null
  last_health_at?: string | null
  restart_count: number
  next_restart_at?: string | null
  exit_code?: number | null
  last_error?: string | null
  log_tail_available: boolean
  warnings: string[]
  health: Record<string, unknown>
  message?: string | null
}

export interface LlamacppRuntimeListResponse {
  runtimes: LlamacppRuntime[]
}

export interface LlamacppLifecycleActionResponse {
  profile_id: string
  action: string
  state: LlamacppRuntimeState
  accepted: boolean
  message?: string | null
}

export interface LlamacppGpuSnapshot {
  index: number
  name?: string | null
  memory_total_bytes?: number | null
  memory_free_bytes?: number | null
  memory_used_bytes?: number | null
}

export interface LlamacppHardwareSnapshotResponse {
  ram_total_bytes?: number | null
  ram_available_bytes?: number | null
  cpu_count?: number | null
  gpus: LlamacppGpuSnapshot[]
  warnings: string[]
}
