export type FirstRunStatus =
  | "not_started"
  | "in_progress"
  | "blocked"
  | "skipped"
  | "first_chat_complete"
  | "completed"

export type FirstRunChatResult = {
  completed: boolean
  provider?: string | null
  model?: string | null
  response_id?: string | null
  completed_at?: string | null
}

export type FirstRunState = {
  status: FirstRunStatus
  current_step?: string | null
  completed_steps: string[]
  skipped_steps: string[]
  step_data: Record<string, Record<string, unknown>>
  first_chat: FirstRunChatResult
  acknowledged_steps: string[]
  skip_reason?: string | null
  created_at?: string
  updated_at?: string
  completed_at?: string | null
}

export type SetupProviderType = "hosted_api_key" | "local_endpoint"

export type SetupProviderCatalogEntry = {
  provider_key: string
  label: string
  provider_type: SetupProviderType
  config_section?: string
  api_key_field?: string | null
  base_url_field?: string | null
  model_field?: string | null
  default_base_url?: string | null
  supports_preflight: boolean
  recommended_for_first_chat: boolean
}

export type SetupProviderCatalogResponse = {
  providers: SetupProviderCatalogEntry[]
}

export type SetupProviderSaveRequest = {
  provider_key: string
  api_key?: string | null
  base_url?: string | null
  model?: string | null
  make_default?: boolean
}

export type SetupProviderSaveResponse = {
  provider_key: string
  status: "saved" | "failed"
  masked_api_key?: string | null
  credential_configured?: boolean
  base_url?: string | null
  model?: string | null
  make_default?: boolean
  requires_restart?: boolean
  failure_category?: string | null
  message?: string | null
}

export type SetupProviderValidationResponse = {
  provider_key: string
  status: string
  failure_category?: string | null
  message?: string | null
  models: string[]
  validation_level?: string | null
  can_gate_first_chat?: boolean
}

export type FirstRunMetadata = {
  auth_mode: string
  bundled_single_user_auth_available: boolean
  manual_auth_required: boolean
  setup_required: boolean
  setup_completed: boolean
  remote_setup_enabled: boolean
  connection: {
    frontend_origin?: string | null
    api_origin?: string | null
    browser_access?: "local" | "lan" | "remote" | "unknown" | string | null
  }
  setup_paths: Array<{
    key: string
    label: string
    recommended: boolean
    guide_path?: string | null
  }>
  multi_user_exit: {
    guide_path: string
    checklist_path?: string | null
  }
}

export type FirstRunStepUpdateRequest = {
  step: string
  data?: Record<string, unknown>
}

export type FirstRunSkipRequest = {
  reason?: string | null
}

export type IngestDefaultsRequest = {
  allow_local_file_ingest?: boolean
  chunking_profile?: string
  metadata_mode?: string
  allowed_local_roots?: string[]
}

export type AudioDefaultsRequest = {
  mode?: "defaults" | "configure" | "skip"
  stt_provider?: string | null
  tts_provider?: string | null
  tts_voice?: string | null
}

export type AudioRecommendationsResponse = {
  machine_profile: Record<string, unknown>
  catalog: Array<Record<string, unknown>>
  recommendations: Array<Record<string, unknown>>
  excluded: Array<Record<string, unknown>>
}

export type OptionalAdvancedRequest = {
  rag?: "configure" | "skip" | "defer"
  storage_paths?: "configure" | "skip" | "defer"
  values?: Record<string, unknown>
}

export type FirstRunStepSaveResponse = {
  status: string
  step: string
  requires_restart: boolean
}

export type FirstChatVerifyRequest = {
  provider: string
  model: string
  prompt?: string
}

export type FirstChatVerifyResponse = {
  status: string
  provider: string
  model: string
  response_id?: string | null
  response_text?: string | null
  failure_category?: string | null
  message?: string | null
}

export type FirstRunCompleteRequest = {
  acknowledged_steps?: string[]
}

export type SetupCompleteResponse = {
  success: boolean
  message: string
  requires_restart: boolean
  install_plan_submitted: boolean
}
