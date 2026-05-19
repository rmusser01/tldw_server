// tldw_server Admin Types

export interface User {
  id: number;
  uuid: string;
  username: string;
  email: string;
  role: string;
  roles?: string[];
  is_active: boolean;
  is_verified: boolean;
  mfa_enabled: boolean;
  storage_quota_mb: number;
  storage_used_mb: number;
  created_at: string;
  updated_at: string;
  last_login?: string;
}

export interface UserWithKeyCount extends User {
  api_key_count?: number;
}

export interface Organization {
  id: number;
  name: string;
  slug: string;
  description?: string;
  owner_user_id: number;
  created_at: string;
  updated_at: string;
}

export interface Team {
  id: number;
  org_id: number;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface RegistrationCode {
  id: number;
  code: string;
  max_uses: number;
  times_used: number;
  expires_at: string;
  created_at: string;
  role_to_grant: string;
  is_valid?: boolean;
}

export interface RegistrationSettings {
  enable_registration: boolean;
  require_registration_code: boolean;
  auth_mode?: string;
  profile?: string;
  self_registration_allowed?: boolean;
}

export interface WatchlistSettings {
  watchlists_enabled?: boolean;
  default_threshold?: number;
  notification_email?: string;
  alert_on_breach?: boolean;
}

export interface OrgMember {
  org_id: number;
  user_id: number;
  role: string;
  joined_at: string;
  user?: User;
}

export interface TeamMember {
  team_id: number;
  user_id: number;
  role: string;
  joined_at: string;
  user?: User;
}

export interface OrgMembership {
  org_id: number;
  role: string;
  org_name?: string;
}

export interface TeamMembership {
  team_id: number;
  org_id: number;
  role: string;
  team_name?: string;
  org_name?: string;
}

export interface ApiKey {
  id: string;
  user_id: number;
  name?: string;
  key_prefix: string;
  scope: string;
  created_at: string;
  expires_at?: string;
  revoked_at?: string;
  last_used_at?: string;
}

export interface ApiKeyMutationResponse {
  key?: string;
  id?: string;
  key_prefix?: string;
}

export interface ApiKeyDailySnapshot {
  date: string;
  requests: number;
  tokens: number;
  cost_usd: number;
}

export interface ApiKeyUsageSummary {
  key_id: string;
  request_count: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  estimated_cost_usd: number;
  last_used_at: string | null;
  daily_snapshots: ApiKeyDailySnapshot[];
}

export interface ApiKeyUsageTopItem {
  key_id: string;
  request_count: number;
  total_tokens: number;
  estimated_cost_usd: number;
  last_used_at: string | null;
}

export interface ApiKeyUsageTopResponse {
  items: ApiKeyUsageTopItem[];
}

export interface Role {
  id: number;
  name: string;
  description?: string;
  is_system: boolean;
}

export interface Permission {
  id: number;
  name: string;
  description?: string;
}

export interface AuditLog {
  id: string;
  timestamp: string;
  user_id: number;
  action: string;
  resource: string;
  details?: Record<string, unknown>;
  ip_address?: string;
  username?: string;
  request_id?: string;
  raw?: Record<string, unknown>;
}

export interface BackupItem {
  id: string;
  dataset: string;
  user_id?: number | null;
  status: string;
  size_bytes: number;
  created_at: string;
}

export interface BackupsResponse {
  items: BackupItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface BackupScheduleItem {
  id: string;
  dataset: string;
  target_user_id?: number | null;
  frequency: 'daily' | 'weekly' | 'monthly';
  time_of_day: string;
  timezone: string;
  anchor_day_of_week?: number | null;
  anchor_day_of_month?: number | null;
  retention_count: number;
  is_paused: boolean;
  schedule_description: string;
  next_run_at?: string | null;
  last_run_at?: string | null;
  last_status?: string | null;
  last_job_id?: string | null;
  last_error?: string | null;
  created_at: string;
  updated_at: string;
  deleted_at?: string | null;
}

export interface BackupScheduleListResponse {
  items: BackupScheduleItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface BackupScheduleMutationResponse {
  status: string;
  item: BackupScheduleItem;
}

export interface MaintenanceRotationRunItem {
  id: string;
  mode: 'dry_run' | 'execute';
  status: 'queued' | 'running' | 'complete' | 'failed';
  domain?: string | null;
  queue?: string | null;
  job_type?: string | null;
  fields_json: string;
  limit?: number | null;
  affected_count?: number | null;
  requested_by_user_id?: number | null;
  requested_by_label?: string | null;
  confirmation_recorded: boolean;
  job_id?: string | null;
  scope_summary: string;
  key_source: string;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface MaintenanceRotationRunListResponse {
  items: MaintenanceRotationRunItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface MaintenanceRotationRunCreateRequest {
  mode: 'dry_run' | 'execute';
  domain?: string;
  queue?: string;
  job_type?: string;
  fields: string[];
  limit: number;
  confirmed: boolean;
}

export interface MaintenanceRotationRunCreateResponse {
  item: MaintenanceRotationRunItem;
}

export interface ByokValidationRunItem {
  id: string;
  status: 'queued' | 'running' | 'complete' | 'failed';
  org_id?: number | null;
  provider?: string | null;
  keys_checked?: number | null;
  valid_count?: number | null;
  invalid_count?: number | null;
  error_count?: number | null;
  requested_by_user_id?: number | null;
  requested_by_label?: string | null;
  job_id?: string | null;
  scope_summary: string;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ByokValidationRunListResponse {
  items: ByokValidationRunItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ByokValidationRunCreateRequest {
  org_id?: number;
  provider?: string;
}

export interface RetentionPolicy {
  key: string;
  days?: number | null;
  description?: string | null;
}

export interface RetentionPoliciesResponse {
  policies: RetentionPolicy[];
}

export interface RetentionPolicyPreviewCounts {
  audit_log_entries: number;
  job_records: number;
  backup_files: number;
}

export interface RetentionPolicyPreviewResponse {
  key: string;
  current_days: number;
  new_days: number;
  counts: RetentionPolicyPreviewCounts;
  preview_signature: string;
  notes: string[];
}

export interface ProviderSecret {
  provider: string;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderOverride {
  provider: string;
  is_enabled?: boolean;
  allowed_models?: string[];
  config?: Record<string, unknown>;
  credential_fields?: Record<string, unknown>;
  has_api_key?: boolean;
  api_key_hint?: string;
  created_at?: string;
  updated_at?: string;
}

export interface LLMProvider {
  name: string;
  enabled: boolean;
  models?: string[];
  default_model?: string;
  override?: LLMProviderOverride;
}

export interface DashboardStats {
  users: number;
  organizations: number;
  teams: number;
  apiKeys: number;
  providers: number;
  storageUsedMb: number;
}

export type SecurityHealthData = {
  risk_score?: number;
  recent_security_events?: number;
  failed_logins_24h?: number;
  suspicious_activity?: number;
  mfa_adoption_rate?: number;
  active_sessions?: number;
  api_keys_active?: number;
  last_security_scan?: string;
};

export interface SecurityAlertStatus {
  total_alerts?: number;
  critical_alerts?: number;
  warning_alerts?: number;
  unacknowledged?: number;
  recent_alerts?: {
    id: string;
    severity: string;
    message: string;
    timestamp: string;
    source?: string;
  }[];
}

export interface CompliancePosture {
  overall_score: number;
  mfa_adoption_pct: number;
  mfa_enabled_count: number;
  total_users: number;
  key_rotation_compliance_pct: number;
  keys_needing_rotation: number;
  keys_total: number;
  rotation_threshold_days: number;
  audit_logging_enabled: boolean;
}

export type SystemDependencyStatus = 'healthy' | 'degraded' | 'down' | 'unknown';

export interface SystemDependencyItem {
  name: string;
  status: SystemDependencyStatus;
  latency_ms: number;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface SystemDependenciesResponse {
  items: SystemDependencyItem[];
  checked_at: string;
}

export interface DependencyUptimeStats {
  dependency_name: string;
  days: number;
  total_checks: number;
  healthy_checks: number;
  uptime_pct: number;
  avg_latency_ms: number;
  downtime_minutes: number;
  sparkline: number[];
}

export interface EffectivePermissionsResponse {
  user_id: number;
  permissions: string[];
}

export interface AuthContextType {
  user: User | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  isAuthenticated: boolean;
}

// Voice Commands
export type VoiceActionType = 'mcp_tool' | 'workflow' | 'custom' | 'llm_chat';

export interface VoiceCommand {
  id: string;
  user_id: number;
  name: string;
  phrases: string[];
  action_type: VoiceActionType;
  action_config: Record<string, unknown>;
  priority: number;
  enabled: boolean;
  requires_confirmation: boolean;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

export interface VoiceCommandValidationStep {
  name: string;
  passed: boolean;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface VoiceCommandValidationResponse {
  command_id: string;
  command_name: string;
  action_type: VoiceActionType;
  valid: boolean;
  steps: VoiceCommandValidationStep[];
}

export interface VoiceSession {
  session_id: string;
  user_id: number;
  state: string;
  created_at: string;
  last_activity: string;
  turn_count: number;
}

export interface VoiceCommandListResponse {
  commands?: VoiceCommand[];
  items?: VoiceCommand[];
  total?: number;
}

export interface VoiceSessionListResponse {
  sessions?: VoiceSession[];
  items?: VoiceSession[];
  total?: number;
}

export interface VoiceCommandUsage {
  command_id: string;
  command_name: string;
  total_invocations: number;
  success_count: number;
  error_count: number;
  avg_response_time_ms: number;
  last_used?: string;
}

export interface VoiceAnalytics {
  date: string;
  total_commands: number;
  unique_users: number;
  success_rate: number;
  avg_response_time_ms: number;
  top_commands: Array<{
    command_id: string;
    command_name: string;
    count: number;
  }>;
}

export interface VoiceAnalyticsSummary {
  total_commands_processed: number;
  active_sessions: number;
  total_voice_commands: number;
  enabled_commands: number;
  success_rate: number;
  avg_response_time_ms: number;
  top_commands: VoiceCommandUsage[];
  usage_by_day: VoiceAnalytics[];
}

export type { IncidentEvent, IncidentItem, IncidentNotifyResponse, IncidentsResponse } from './incidents';
export type { WebhookItem, WebhookCreateResponse, WebhookListResponse, WebhookDeliveryItem, WebhookDeliveryListResponse } from './webhooks';
export type { EmailDeliveryStatus, EmailDeliveryItem, EmailDeliveryListResponse } from './email-deliveries';

// ============================================
// Billing & Subscription Types
// ============================================

export type PlanTier = 'free' | 'pro' | 'enterprise';
export type SubscriptionStatus = 'active' | 'past_due' | 'canceled' | 'trialing' | 'incomplete';

export interface Plan {
  id: string;
  name: string;
  tier: PlanTier;
  stripe_product_id: string;
  stripe_price_id: string;
  monthly_price_cents: number;
  included_token_credits: number;
  overage_rate_per_1k_tokens_cents: number;
  features: string[];
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface Subscription {
  id: string;
  org_id: number;
  org_name?: string;
  plan_id: string;
  plan?: Plan;
  stripe_subscription_id: string;
  status: SubscriptionStatus;
  current_period_start: string;
  current_period_end: string;
  trial_end?: string;
  cancel_at?: string;
  cancel_at_period_end?: boolean;
  created_at: string;
  updated_at: string;
  // Computed lifecycle fields
  days_since_created?: number | null;
  days_past_due?: number;
  days_until_period_end?: number | null;
  usage_pct?: number | null;
  at_risk?: boolean;
  at_risk_reasons?: string[];
  billing_cycle?: string;
}

export interface OrgUsageSummary {
  org_id: number;
  period_start: string;
  period_end: string;
  tokens_used: number;
  tokens_included: number;
  tokens_overage: number;
  overage_cost_cents: number;
  breakdown_by_provider: Record<string, number>;
}

export interface Invoice {
  id: string;
  stripe_invoice_id: string;
  amount_cents: number;
  currency: string;
  status: 'paid' | 'open' | 'void' | 'draft' | 'uncollectible';
  invoice_pdf?: string;
  period_start: string;
  period_end: string;
  created_at: string;
}

export interface PlanDistributionEntry {
  plan_name: string;
  count: number;
}

export interface BillingAnalytics {
  mrr_cents: number;
  subscriber_count: number;
  active_count: number;
  trialing_count: number;
  past_due_count: number;
  canceled_count: number;
  plan_distribution: PlanDistributionEntry[];
  trial_conversion_rate_pct: number;
}

export interface FeatureRegistryEntry {
  feature_key: string;
  display_name: string;
  description: string;
  plans: string[];
  category: string;
  availability_type?: 'core' | 'plan_gated' | 'add_on';
}

// Admin Webhooks
export interface AdminWebhook {
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
}

export interface AdminWebhooksResponse {
  items: AdminWebhook[];
  total: number;
}

export interface AdminWebhookDeliveryLogEntry {
  id: number;
  webhook_id: number;
  event_type: string;
  status_code: number | null;
  latency_ms: number | null;
  retry_attempt: number;
  error_message: string | null;
  delivered_at: string | null;
  created_at: string | null;
}

export interface AdminWebhookDeliveryLogResponse {
  items: AdminWebhookDeliveryLogEntry[];
  total: number;
}

export interface AdminWebhookTestResult {
  success: boolean;
  status_code: number | null;
  latency_ms: number | null;
  error: string | null;
}

export interface ComplianceReportSchedule {
  id: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  recipients: string[];
  format: 'html' | 'json';
  enabled: boolean;
  created_at: string | null;
  last_sent_at: string | null;
}

export interface DigestPreference {
  id?: string;
  user_id: string;
  email: string;
  frequency: 'daily' | 'weekly' | 'off';
  enabled: boolean;
  created_at?: string;
}
