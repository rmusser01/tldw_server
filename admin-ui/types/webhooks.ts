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

export type WebhookStatus = {
  mode: 'off' | 'migrate' | 'on';
  route_selection: 'canonical' | 'legacy';
  schema_ready: boolean;
  key_state: string;
  delivery_capability_ready: boolean;
  limits: WebhookLimits;
  migration: WebhookMigrationStatus;
};

export type WebhookErrorResponse = {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
};

/** @deprecated Removed with the Task 10 canonical page rewrite. */
export type AdminWebhook = {
  id: number;
  url: string;
  event_types: string[];
  description: string;
  active: boolean;
  retry_count: number;
  timeout_seconds: number;
  created_by: number | null;
  created_at: string | null;
  updated_at: string | null;
};

/** @deprecated Removed with the Task 10 canonical page rewrite. */
export type AdminWebhooksResponse = {
  items: AdminWebhook[];
  total: number;
};

/** @deprecated Removed with the Task 10 canonical page rewrite. */
export type AdminWebhookDeliveryLogEntry = {
  id: number;
  webhook_id: number;
  event_type: string;
  status_code: number | null;
  latency_ms: number | null;
  retry_attempt: number;
  error_message: string | null;
  delivered_at: string | null;
  created_at: string | null;
};

/** @deprecated Removed with the Task 10 canonical page rewrite. */
export type AdminWebhookDeliveryLogResponse = {
  items: AdminWebhookDeliveryLogEntry[];
  total: number;
};

/** @deprecated Removed with the Task 10 canonical page rewrite. */
export type AdminWebhookTestResult = {
  success: boolean;
  status_code: number | null;
  latency_ms: number | null;
  error: string | null;
};
