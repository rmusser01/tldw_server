export type WebhookRegistration = {
  id: number;
  description: string;
  target_display: string;
  target_hostname: string;
  event_types: string[];
  active: boolean;
  timeout_seconds: number;
  revision: number;
  delivery_config_version: number;
  secret_version: number;
  secret_rotation_required: boolean;
  created_by: number | null;
  updated_by: number | null;
  created_at: string;
  updated_at: string;
};

export type WebhookCreateRequest = {
  url: string;
  event_types: string[];
  description?: string;
  timeout_seconds?: number;
};

export type WebhookPatchRequest = {
  description?: string;
  url?: string;
  event_types?: string[];
  active?: boolean;
  timeout_seconds?: number;
};

export type WebhookSecretResponse = {
  registration: WebhookRegistration;
  signing_secret: string;
  replayed: boolean;
};

export type WebhookDeleteResponse = {
  deleted: true;
  id: number;
};

export type WebhookListResponse = {
  items: WebhookRegistration[];
  total: number;
  limit: number;
  offset: number;
};

export type WebhookCatalogItem = {
  event_type: string;
  description: string;
};

export type WebhookCatalog = {
  api_version: string;
  events: WebhookCatalogItem[];
  registration_limit: number;
  active_limit: number;
};

export type WebhookLimits = {
  registrations: number;
  active_registrations: number;
  current_registrations: number;
  current_active_registrations: number;
  registrations_over_limit: boolean;
  active_registrations_over_limit: boolean;
};

export type WebhookMigrationStatus = {
  phase: string;
  imported_count: number;
  unresolved_count: number;
  rejected_count: number;
  secret_rotation_required_count: number;
  legacy_file_restore_permitted: boolean;
  rollback_window_expires_at: string | null;
};

export type WebhookDeliveryRuntimeReason =
  | 'mode_off'
  | 'mode_migrate'
  | 'schema_unready'
  | 'migration_pending'
  | 'key_unavailable'
  | 'key_configuration_mismatch'
  | 'jobs_unavailable'
  | 'database_unavailable'
  | 'worker_unavailable'
  | 'reconciler_unavailable'
  | 'retention_unavailable'
  | 'heartbeat_stale';

export type WebhookDeliveryComponent = {
  component: 'worker' | 'reconciler' | 'retention';
  ready: boolean;
  reason_code: WebhookDeliveryRuntimeReason | null;
  heartbeat_age_seconds: number | null;
};

export type WebhookDeliveryBacklog = {
  pending: number;
  enqueue_claimed: number;
  queued: number;
  processing: number;
  retry_wait: number;
};

export type WebhookDeliveryCapability = {
  canonical_schema_version: number;
  schema_ready: boolean;
  delivery_schema_ready: boolean;
  migration_complete: boolean;
  key_ready: boolean;
  key_primary_match: boolean;
  jobs_database_ready: boolean;
  queue_ready: boolean;
  job_type_ready: boolean;
  jobs_backend: 'sqlite' | 'postgres' | 'unavailable';
  worker: WebhookDeliveryComponent;
  reconciler: WebhookDeliveryComponent;
  retention: WebhookDeliveryComponent;
  backlog: WebhookDeliveryBacklog;
  oldest_nonterminal_age_seconds: number | null;
  acquisition_ready: boolean;
  acquisition_reason_code: WebhookDeliveryRuntimeReason | null;
  delivery_capability_ready: boolean;
};

export type WebhookStatus = {
  mode: 'off' | 'migrate' | 'on';
  route_selection: 'canonical';
  schema_ready: boolean;
  key_state: string;
  delivery_capability_ready: boolean;
  delivery: WebhookDeliveryCapability;
  limits: WebhookLimits;
  migration: WebhookMigrationStatus;
};

export type WebhookDeliveryKind = 'automatic' | 'manual' | 'test';

export type WebhookDeliveryState =
  | 'pending'
  | 'enqueue_claimed'
  | 'queued'
  | 'processing'
  | 'retry_wait'
  | 'succeeded'
  | 'dead'
  | 'canceled'
  | 'superseded';

export type WebhookDeliveryAttemptState =
  | 'processing'
  | 'succeeded'
  | 'retryable'
  | 'failed'
  | 'canceled'
  | 'superseded'
  | 'outcome_unknown';

export type WebhookDeliveryReason =
  | 'attempt_budget_exhausted'
  | 'canceled_deleted'
  | 'canceled_disabled'
  | 'canceled_secret_rotation'
  | 'delivery_expired'
  | 'jobs_identity_conflict'
  | 'outcome_unknown'
  | 'superseded_config'
  | 'test_attempt_interrupted'
  | 'target_invalid'
  | 'target_rejected'
  | 'policy_error'
  | 'clock_error'
  | 'transport_error'
  | 'http_redirect'
  | 'http_client_error'
  | 'http_request_timeout'
  | 'http_rate_limited'
  | 'http_server_error'
  | 'http_status_invalid'
  | 'http_hop_invalid_request'
  | 'http_hop_dns_resolution_failed'
  | 'http_hop_dns_timeout'
  | 'http_hop_dns_address_denied'
  | 'http_hop_connect_timeout'
  | 'http_hop_read_timeout'
  | 'http_hop_write_timeout'
  | 'http_hop_total_timeout'
  | 'http_hop_peer_verification_failed'
  | 'http_hop_tls_error'
  | 'http_hop_protocol_error'
  | 'http_hop_response_headers_too_large'
  | 'http_hop_response_too_large'
  | 'http_hop_decompressed_response_too_large'
  | 'http_hop_parser_input_too_large'
  | 'http_hop_unsupported_content_encoding'
  | 'http_hop_invalid_content_encoding'
  | 'http_hop_transport_error';

export type WebhookDeliveryAttempt = {
  id: string;
  sequence: number;
  state: WebhookDeliveryAttemptState;
  request_timeout_seconds: number | null;
  status_code: number | null;
  latency_ms: number | null;
  reason_code: WebhookDeliveryReason | null;
  requested_retry_delay_seconds: number | null;
  started_at: string;
  finished_at: string | null;
};

export type WebhookDelivery = {
  id: string;
  event_id: string;
  event_type: string;
  webhook_id: number;
  kind: WebhookDeliveryKind;
  state: WebhookDeliveryState;
  delivery_config_version: number;
  secret_version: number;
  attempt_count: number;
  status_code: number | null;
  latency_ms: number | null;
  reason_code: WebhookDeliveryReason | null;
  expires_at: string;
  created_at: string;
  updated_at: string;
  terminal_at: string | null;
  redelivery_of_id: string | null;
  completed_after_config_change: boolean;
};

export type WebhookDeliveryHistoryItem = {
  delivery: WebhookDelivery;
  attempts: WebhookDeliveryAttempt[];
};

export type WebhookDeliveryListResponse = {
  items: WebhookDeliveryHistoryItem[];
  total: number;
  limit: number;
  offset: number;
};

export type WebhookTestRequest = {
  delivery_config_version: number;
};

export type WebhookTestResponse = {
  delivery: WebhookDelivery;
  attempt: WebhookDeliveryAttempt;
  idempotent_replay: boolean;
  in_progress: boolean;
};

export type WebhookRedeliveryRequest = {
  delivery_config_version: number;
  confirm_changed_configuration: boolean;
};

export type WebhookRedeliveryResponse = {
  delivery: WebhookDelivery;
  idempotent_replay: boolean;
};

export type WebhookErrorResponse = {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
};
