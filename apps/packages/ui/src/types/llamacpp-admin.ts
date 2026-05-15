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
