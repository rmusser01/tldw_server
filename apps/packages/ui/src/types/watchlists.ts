/**
 * Watchlists module types
 * Corresponds to tldw_server2 /api/v1/watchlists endpoints
 */

// ─────────────────────────────────────────────────────────────────────────────
// Source Types
// ─────────────────────────────────────────────────────────────────────────────

export type SourceType = "rss" | "site" | "forum"
export type WatchlistDomain = "cti_osint" | "news" | "general"
export type WatchlistStatus = "active" | "paused" | "archived"
export type WatchlistPriority = "low" | "medium" | "high" | "critical"

export interface WatchlistContainer {
  id: number
  name: string
  description?: string | null
  objective?: string | null
  domain: WatchlistDomain
  status: WatchlistStatus
  priority: WatchlistPriority
  tags: string[]
  archived_at?: string | null
  deleted_at?: string | null
  restore_expires_at?: string | null
  created_at: string
  updated_at?: string | null
}

export interface WatchlistCreate {
  name: string
  description?: string | null
  objective?: string | null
  domain?: WatchlistDomain
  status?: WatchlistStatus
  priority?: WatchlistPriority
  tags?: string[]
}

export interface WatchlistUpdate {
  name?: string
  description?: string | null
  objective?: string | null
  domain?: WatchlistDomain
  status?: WatchlistStatus
  priority?: WatchlistPriority
  tags?: string[]
}

export interface WatchlistSource {
  id: number
  name: string
  url: string
  source_type: SourceType
  active: boolean
  tags: string[]
  group_ids?: number[]
  watchlist_ids?: number[]
  settings?: Record<string, unknown> | null
  last_scraped_at?: string | null
  status?: string | null
  created_at: string
  updated_at?: string | null
}

export interface WatchlistSourceCreate {
  name: string
  url: string
  source_type: SourceType
  active?: boolean
  tags?: string[]
  settings?: Record<string, unknown> | null
  group_ids?: number[]
  watchlist_id?: number
}

export interface WatchlistSourceUpdate {
  name?: string
  url?: string
  source_type?: SourceType
  active?: boolean
  tags?: string[]
  settings?: Record<string, unknown> | null
  group_ids?: number[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Group Types
// ─────────────────────────────────────────────────────────────────────────────

export interface WatchlistGroup {
  id: number
  name: string
  description?: string | null
  parent_group_id?: number | null
}

export interface WatchlistGroupCreate {
  name: string
  description?: string
  parent_group_id?: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag Types
// ─────────────────────────────────────────────────────────────────────────────

export interface WatchlistTag {
  id: number
  name: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter Types
// ─────────────────────────────────────────────────────────────────────────────

export type FilterType = "keyword" | "author" | "date_range" | "regex" | "all"
export type FilterAction = "include" | "exclude" | "flag"

export interface WatchlistFilter {
  type: FilterType
  action: FilterAction
  value: Record<string, unknown>
  priority?: number
  is_active?: boolean
}

export interface WatchlistFiltersPayload {
  filters: WatchlistFilter[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Job Types
// ─────────────────────────────────────────────────────────────────────────────

export interface JobScope {
  sources?: number[]
  groups?: number[]
  tags?: string[]
}

export interface JobOutputPrefs {
  auto_output?: {
    enabled?: boolean
    type?: string
    format?: "md" | "html"
    template_name?: string
    template_version?: number
  }
  generate_audio?: boolean
  target_audio_minutes?: number
  audio_model?: string
  audio_voice?: string
  audio_speed?: number
  background_audio_uri?: string
  background_volume?: number
  background_delay_ms?: number
  background_fade_seconds?: number
  audio_language?: string
  llm_provider?: string
  llm_model?: string
  persona_summarize?: boolean
  persona_id?: string
  persona_provider?: string
  persona_model?: string
  voice_map?: Record<string, string>
  audio_cast?: WatchlistAudioCast
  retention?: {
    default_seconds?: number
    temporary_seconds?: number
  }
  template?: {
    default_name?: string
    default_format?: "md" | "html"
    default_version?: number
  }
  deliveries?: {
    email?: {
      enabled?: boolean
      recipients?: string[]
      body_format?: "auto" | "text" | "html"
      attach_file?: boolean
      subject?: string
      sender?: string
      reply_to?: string
    }
    chatbook?: {
      enabled?: boolean
      title?: string
      description?: string
      conversation_id?: number
      provider?: string
      model?: string
      metadata?: Record<string, unknown>
    }
  }
  ingest?: {
    persist_to_media_db?: boolean
  }
  retention_days?: number
  template_name?: string
  delivery_config?: {
    email_recipients?: string[]
    email_format?: "auto" | "text" | "html"
    create_chatbook?: boolean
  }
}

export interface WatchlistJob {
  id: number
  name: string
  description?: string | null
  watchlist_id?: number | null
  scope: JobScope
  schedule_expr?: string | null
  timezone?: string | null
  active: boolean
  max_concurrency?: number | null
  per_host_delay_ms?: number | null
  retry_policy?: Record<string, unknown> | null
  output_prefs?: JobOutputPrefs | null
  job_filters?: WatchlistFiltersPayload | null
  created_at: string
  updated_at?: string | null
  last_run_at?: string | null
  next_run_at?: string | null
  wf_schedule_id?: string | null
}

export interface WatchlistJobCreate {
  name: string
  description?: string
  scope: JobScope
  schedule_expr?: string
  timezone?: string
  active?: boolean
  max_concurrency?: number
  per_host_delay_ms?: number
  retry_policy?: Record<string, unknown>
  output_prefs?: JobOutputPrefs
  job_filters?: WatchlistFiltersPayload
  watchlist_id?: number
}

export interface WatchlistJobUpdate {
  name?: string
  description?: string
  scope?: JobScope
  schedule_expr?: string | null
  timezone?: string | null
  active?: boolean
  max_concurrency?: number | null
  per_host_delay_ms?: number | null
  retry_policy?: Record<string, unknown> | null
  output_prefs?: JobOutputPrefs | null
  job_filters?: WatchlistFiltersPayload | null
  watchlist_id?: number | null
}

// ─────────────────────────────────────────────────────────────────────────────
// Run Types
// ─────────────────────────────────────────────────────────────────────────────

export type RunStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"

export type RunStatsValue =
  | number
  | string
  | boolean
  | null
  | Record<string, unknown>
  | unknown[]

export interface RunStats {
  items_found?: number
  items_ingested?: number
  items_filtered?: number
  items_errored?: number
  [key: string]: RunStatsValue | undefined
}

export interface WatchlistRun {
  id: number
  job_id: number
  status: RunStatus
  started_at?: string | null
  finished_at?: string | null
  stats?: RunStats | null
  error_msg?: string | null
  log_path?: string | null
}

export interface RunDetailResponse {
  id: number
  job_id: number
  status: string
  started_at?: string | null
  finished_at?: string | null
  stats: Record<string, RunStatsValue>
  filter_tallies?: Record<string, number> | null
  error_msg?: string | null
  log_text?: string | null
  log_path?: string | null
  truncated?: boolean
  filtered_sample?: Array<Record<string, unknown>> | null
}

export interface WatchlistAudioCastSpeaker {
  id: string
  label: string
  role?: string
  voice: string
  persona?: string
}

export interface WatchlistAudioCast {
  speaker_count: 1 | 2 | 3 | 4
  speakers: WatchlistAudioCastSpeaker[]
}

export interface WatchlistRunAudioStatus {
  run_id: number
  task_id?: string | null
  queue_name?: string | null
  status: "pending" | "running" | "completed" | "failed" | "unknown" | string
  audio_uri?: string | null
  download_url?: string | null
  artifact_id?: string | number | null
  size_bytes?: number | null
  mime_type?: string | null
  script_artifact?: Record<string, unknown> | null
  speaker_artifacts?: Array<Record<string, unknown>>
  final_artifact?: Record<string, unknown> | null
  fallback_reason?: string | null
  error?: string | null
}

export interface WatchlistRunStageRetryResponse {
  run_id: number
  stage: "audio" | "delivery" | string
  retried: boolean
  task_id?: string | null
  output_id?: number | null
  delivery_results?: Array<Record<string, unknown>>
  message?: string | null
}

export interface WatchlistRunDiagnostics {
  run_id: number
  generated_at: string
  run: Record<string, unknown>
  job?: Record<string, unknown> | null
  outputs?: Array<Record<string, unknown>>
  audio?: Record<string, unknown> | null
  recovery?: Record<string, unknown> | null
}

// ─────────────────────────────────────────────────────────────────────────────
// Scraped Item Types
// ─────────────────────────────────────────────────────────────────────────────

export type ItemStatus = "ingested" | "filtered"
export type ItemMutableStatus = ItemStatus | "ignored" | "reviewed"
export type ScrapedItemSortMode =
  | "created_desc"
  | "created_asc"
  | "published_desc"
  | "published_asc"
  | "unread_first"
  | "source_asc"
  | "alert_severity_desc"

export interface ScrapedItemAlertSummary {
  total: number
  unread: number
  read: number
  dismissed: number
  highest_severity?: WatchlistContentAlertSeverity | null
  latest_alert_id?: number | null
  latest_alert_status?: WatchlistContentAlertStatus | null
  latest_alert_created_at?: string | null
  latest_matched_text?: string | null
  rule_ids: number[]
  severities: WatchlistContentAlertSeverity[]
}

export interface ScrapedItem {
  id: number
  run_id: number
  job_id: number
  source_id: number
  media_id?: number | null
  media_uuid?: string | null
  url?: string | null
  title?: string | null
  summary?: string | null
  content?: string | null
  published_at?: string | null
  tags: string[]
  status: ItemStatus
  reviewed: boolean
  queued_for_briefing?: boolean
  created_at: string
  alert_summary?: ScrapedItemAlertSummary | null
}

export interface ScrapedItemSmartCounts {
  all: number
  today: number
  today_unread: number
  unread: number
  reviewed: number
  queued: number
}

export interface ScrapedItemUpdate {
  reviewed?: boolean
  status?: ItemMutableStatus
  queued_for_briefing?: boolean
}

export interface ScrapedItemAlertFilterParams {
  has_alert?: boolean
  alert_status?: WatchlistContentAlertStatus
  alert_severity?: WatchlistContentAlertSeverity
  alert_rule_id?: number
}

export interface ScrapedItemBatchScope extends ScrapedItemAlertFilterParams {
  run_id?: number
  job_id?: number
  source_id?: number
  status?: string
  reviewed?: boolean
  queued_for_briefing?: boolean
  q?: string
  search?: string
  since?: string
  until?: string
}

export interface ScrapedItemBatchUpdateRequest {
  watchlist_id: number
  item_ids?: number[]
  scope?: ScrapedItemBatchScope
  reviewed?: boolean
  status?: ItemMutableStatus
  queued_for_briefing?: boolean
  limit?: number
}

export interface ScrapedItemBatchUpdateResponse {
  matched: number
  changed: number
  unchanged: number
  failed: number
  matched_ids: number[]
  changed_ids: number[]
  unchanged_ids: number[]
  failed_ids: number[]
  capped: boolean
  exhausted: boolean
  limit: number
}

export type WatchlistItemSavedViewSmartFilter =
  | "all"
  | "today"
  | "today_unread"
  | "todayUnread"
  | "unread"
  | "reviewed"
  | "queued"

export interface WatchlistItemSavedViewFilters extends ScrapedItemAlertFilterParams {
  run_id?: number
  job_id?: number
  source_id?: number
  status?: string
  reviewed?: boolean
  queued_for_briefing?: boolean
  q?: string
  search?: string
  since?: string
  until?: string
  smart_filter?: WatchlistItemSavedViewSmartFilter
}

export interface WatchlistItemSavedViewCreate {
  name: string
  filters: WatchlistItemSavedViewFilters
  sort: ScrapedItemSortMode
  is_default?: boolean
}

export interface WatchlistItemSavedViewUpdate {
  name?: string
  filters?: WatchlistItemSavedViewFilters
  sort?: ScrapedItemSortMode
  is_default?: boolean
}

export interface WatchlistItemSavedView {
  id: number
  watchlist_id: number
  name: string
  filters: WatchlistItemSavedViewFilters
  sort: ScrapedItemSortMode
  is_default: boolean
  created_at: string
  updated_at: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Alert Types
// ─────────────────────────────────────────────────────────────────────────────

export type WatchlistContentAlertRuleKind =
  | "keyword"
  | "regex"
  | "descriptor"
  | "classification"
  | "entity"
  | "ioc"
  | "cve"

export type WatchlistContentAlertMatchMode = "contains" | "exact" | "regex"
export type WatchlistContentAlertSeverity = "info" | "low" | "medium" | "high" | "critical"
export type WatchlistContentAlertStatus = "unread" | "read" | "dismissed"

export interface WatchlistContentAlertSourceConstraints {
  source_ids?: number[]
  source_types?: SourceType[]
  source_tags?: string[]
  url_contains?: string[]
  [key: string]: unknown
}

export interface WatchlistContentAlertRuleCreate {
  name: string
  rule_kind: WatchlistContentAlertRuleKind
  match_mode?: WatchlistContentAlertMatchMode
  pattern: string
  severity?: WatchlistContentAlertSeverity
  enabled?: boolean
  classification?: string | null
  descriptor?: string | null
  entity_type?: string | null
  source_constraints?: WatchlistContentAlertSourceConstraints | null
  metadata?: Record<string, unknown> | null
}

export interface WatchlistContentAlertRuleUpdate {
  name?: string
  enabled?: boolean
  rule_kind?: WatchlistContentAlertRuleKind
  match_mode?: WatchlistContentAlertMatchMode
  pattern?: string
  severity?: WatchlistContentAlertSeverity
  classification?: string | null
  descriptor?: string | null
  entity_type?: string | null
  source_constraints?: WatchlistContentAlertSourceConstraints | null
  metadata?: Record<string, unknown> | null
}

export interface WatchlistContentAlertRule {
  id: number
  watchlist_id: number
  name: string
  enabled: boolean
  rule_kind: WatchlistContentAlertRuleKind
  match_mode: WatchlistContentAlertMatchMode
  pattern: string
  severity: WatchlistContentAlertSeverity
  classification?: string | null
  descriptor?: string | null
  entity_type?: string | null
  source_constraints?: WatchlistContentAlertSourceConstraints | null
  metadata?: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface WatchlistContentAlertEvidence {
  url?: string | null
  title?: string | null
  summary?: string | null
  published_at?: string | null
  source_id?: number
  source_name?: string | null
  source_url?: string | null
  source_type?: SourceType | string | null
  source_tags?: string[]
  rule_kind?: WatchlistContentAlertRuleKind | string
  match_mode?: WatchlistContentAlertMatchMode | string
  pattern?: string | null
  matched_text?: string | null
  [key: string]: unknown
}

export interface WatchlistContentAlert {
  id: number
  watchlist_id: number
  rule_id: number
  item_id: number
  run_id: number
  job_id: number
  source_id: number
  severity: WatchlistContentAlertSeverity
  status: WatchlistContentAlertStatus
  title?: string | null
  snippet?: string | null
  matched_text?: string | null
  evidence: WatchlistContentAlertEvidence
  dedupe_key: string
  created_at: string
  read_at?: string | null
  dismissed_at?: string | null
}

export interface WatchlistContentAlertUpdate {
  status: WatchlistContentAlertStatus
}

// ─────────────────────────────────────────────────────────────────────────────
// Output Types
// ─────────────────────────────────────────────────────────────────────────────

export type OutputFormat = "md" | "html" | "mp3" | "wav" | "ogg" | "m4a" | "aac" | "flac" | string

export type WatchlistReportPreset = "auto" | "cti_osint" | "news_briefing" | "general_research"
export type WatchlistReportReadinessState = "ready" | "warning" | "blocked" | "legacy_live_only"
export type WatchlistReportReadinessWarningSeverity = "info" | "warning" | "blocking"

export interface WatchlistReportReadinessWarning {
  code: string
  severity: WatchlistReportReadinessWarningSeverity
  message: string
  affected_item_ids: number[]
}

export interface WatchlistReportReadiness {
  state: WatchlistReportReadinessState
  score: number
  warnings: WatchlistReportReadinessWarning[]
}

export interface WatchlistReportEvidenceAlert {
  id: number
  rule_id: number
  rule_name?: string | null
  severity: string
  status: string
  title?: string | null
  snippet?: string | null
  matched_text?: string | null
  evidence: Record<string, unknown>
  created_at?: string | null
}

export interface WatchlistReportEvidenceItem {
  id: number
  title?: string | null
  url?: string | null
  source_id?: number | null
  source_name?: string | null
  published_at?: string | null
  summary?: string | null
  tags: string[]
  reviewed: boolean
  queued_for_briefing: boolean
  alerts: WatchlistReportEvidenceAlert[]
}

export interface WatchlistReportExcludedItem {
  id: number
  title?: string | null
  url?: string | null
  reason: string
}

export interface WatchlistReportEvidenceSnapshot {
  schema_version: number
  snapshot_id: string
  generated_at: string
  preset: WatchlistReportPreset
  watchlist_id?: number | null
  job_id: number
  run_id: number
  output_id?: number | null
  included_items: WatchlistReportEvidenceItem[]
  excluded_items: WatchlistReportExcludedItem[]
  source_summary: Record<string, unknown>
  included_count: number
  excluded_count: number
  excluded_total_count?: number | null
  excluded_items_truncated: boolean
  alert_count: number
  critical_alert_count: number
  readiness: WatchlistReportReadiness
}

export interface WatchlistOutputEvidenceResponse {
  output_id: number
  immutable_snapshot: boolean
  snapshot?: WatchlistReportEvidenceSnapshot | null
  readiness: WatchlistReportReadiness
}

export interface WatchlistOutput {
  id: number
  run_id: number
  job_id: number
  type: string
  format: OutputFormat
  title?: string | null
  content?: string | null
  storage_path?: string | null
  metadata?: Record<string, unknown> | null
  media_item_id?: number | null
  chatbook_path?: string | null
  version: number
  expires_at?: string | null
  expired: boolean
  created_at: string
}

export interface WatchlistOutputCreate {
  run_id: number
  item_ids?: number[]
  title?: string
  type?: string
  format?: OutputFormat
  metadata?: Record<string, unknown>
  report_preset?: WatchlistReportPreset
  include_evidence_table?: boolean
  include_excluded_items?: boolean
  require_reviewed_items?: boolean
  allow_weak_evidence?: boolean
  template_name?: string
  template_version?: number
  generate_audio?: boolean
  target_audio_minutes?: number
  audio_model?: string
  audio_voice?: string
  audio_speed?: number
  background_audio_uri?: string
  background_volume?: number
  background_delay_ms?: number
  background_fade_seconds?: number
  audio_language?: string
  llm_provider?: string
  llm_model?: string
  persona_summarize?: boolean
  persona_id?: string
  persona_provider?: string
  persona_model?: string
  voice_map?: Record<string, string>
  audio_cast?: WatchlistAudioCast
  retention_seconds?: number
  temporary?: boolean
  deliveries?: {
    email?: {
      recipients?: string[]
      format?: "auto" | "text" | "html"
      subject?: string
      sender?: string
      reply_to?: string
    }
    chatbook?: {
      enabled?: boolean
      title?: string
      description?: string
      conversation_id?: number
      provider?: string
      model?: string
      metadata?: Record<string, unknown>
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Template Types
// ─────────────────────────────────────────────────────────────────────────────

export interface WatchlistTemplate {
  name: string
  description?: string | null
  content?: string
  format: "md" | "html"
  updated_at?: string | null
  version?: number
  history_count?: number
  available_versions?: number[]
  composer_ast?: Record<string, unknown> | null
  composer_schema_version?: string | null
  composer_sync_hash?: string | null
  composer_sync_status?: "in_sync" | "needs_repair" | "recovered_from_code" | null
}

export interface WatchlistTemplateCreate {
  name: string
  description?: string | null
  content: string
  format?: "md" | "html"
  overwrite?: boolean
  composer_ast?: Record<string, unknown> | null
  composer_schema_version?: string | null
  composer_sync_hash?: string | null
  composer_sync_status?: "in_sync" | "needs_repair" | "recovered_from_code" | null
}

export interface WatchlistTemplateVersionSummary {
  version: number
  format: "md" | "html"
  description?: string | null
  updated_at: string
  is_current: boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings Types
// ─────────────────────────────────────────────────────────────────────────────

export interface WatchlistSettingsStats {
  sources_count?: number
  jobs_count?: number
  runs_count?: number
  items_count?: number
}

export interface WatchlistSettings {
  default_output_ttl_seconds?: number
  temporary_output_ttl_seconds?: number
  forums_enabled?: boolean
  forum_default_top_n?: number
  sharing_mode?: string
  watchlists_backend?: "sqlite" | "postgres" | string
}

export interface WatchlistsOnboardingTelemetryPayload {
  session_id: string
  event_type: string
  event_at?: string | null
  details?: Record<string, string | number | boolean | null> | null
}

export interface WatchlistsOnboardingTelemetryResponse {
  accepted: boolean
  code?: string | null
}

export interface WatchlistsOnboardingTelemetrySummaryResponse {
  counters: Record<string, number>
  rates: Record<string, number>
  timings: Record<string, number>
  since?: string | null
  until?: string | null
}

export type WatchlistsIaExperimentVariant = "baseline" | "experimental"

export interface WatchlistsIaExperimentTelemetryPayload {
  variant: WatchlistsIaExperimentVariant
  session_id: string
  previous_tab?: string | null
  current_tab: string
  transitions: number
  visited_tabs: string[]
  first_seen_at?: string | null
  last_seen_at?: string | null
}

export interface WatchlistsIaExperimentTelemetryResponse {
  accepted: boolean
}

export interface WatchlistsIaExperimentVariantSummary {
  variant: WatchlistsIaExperimentVariant
  events: number
  sessions: number
  reached_target_sessions: number
  avg_transitions: number
  avg_visited_tabs: number
  avg_session_seconds: number
}

export interface WatchlistsIaExperimentTelemetrySummaryResponse {
  items: WatchlistsIaExperimentVariantSummary[]
  since?: string | null
  until?: string | null
}

export interface WatchlistsRcTelemetryThresholdSummary {
  id: string
  label: string
  status: "ok" | "potential_breach"
  reporting_only: boolean
  metric_value?: number | null
  baseline_value?: number | null
  delta?: number | null
  notes?: string | null
}

export interface WatchlistsRcTelemetrySummaryResponse {
  onboarding: WatchlistsOnboardingTelemetrySummaryResponse
  uc2_backend: Record<string, number>
  ia_experiment: Record<string, unknown>
  baseline: Record<string, number>
  thresholds: WatchlistsRcTelemetryThresholdSummary[]
  since?: string | null
  until?: string | null
}

export type WatchlistsIaExperimentTelemetryIngestRequest =
  WatchlistsIaExperimentTelemetryPayload
export type WatchlistsIaExperimentTelemetryIngestResponse =
  WatchlistsIaExperimentTelemetryResponse

// ─────────────────────────────────────────────────────────────────────────────
// Claim Cluster Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ClaimCluster {
  id: number
  canonical_claim_text?: string | null
  summary?: string | null
  member_count?: number
  updated_at?: string | null
  watchlist_count?: number
}

export interface WatchlistClusterSubscription {
  cluster_id: number
  created_at?: string | null
}

// ─────────────────────────────────────────────────────────────────────────────
// API Response Types
// ─────────────────────────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page?: number
  size?: number
  has_more?: boolean
}

export interface SourceCheckNowItem {
  source_id: number
  status: "ok" | "error" | "not_found" | "inactive"
  detail?: string | null
  last_scraped_at?: string | null
  run_id?: number | null
}

export interface SourcesCheckNowResponse {
  items: SourceCheckNowItem[]
  total: number
  success: number
  failed: number
}

export interface SourcesBulkCreateItem {
  name?: string | null
  url: string
  id?: number | null
  status: "created" | "error"
  error?: string | null
}

export interface SourcesBulkCreateResponse {
  items: SourcesBulkCreateItem[]
  total: number
  created: number
  errors: number
}

export interface SourcesImportItem {
  url: string
  name?: string | null
  id?: number | null
  status: "created" | "skipped" | "error"
  error?: string | null
}

export interface SourcesImportResponse {
  items: SourcesImportItem[]
  total: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Preview Types
// ─────────────────────────────────────────────────────────────────────────────

export interface PreviewItem {
  source_id: number
  source_type: SourceType
  url?: string | null
  title?: string | null
  summary?: string | null
  published_at?: string | null
  decision: "ingest" | "filtered"
  matched_action?: "include" | "exclude" | "flag" | null
  matched_filter_key?: string | null
  matched_filter_id?: number | null
  matched_filter_type?: FilterType | null
  flagged?: boolean
}

export interface JobPreviewResult {
  items: PreviewItem[]
  total: number
  ingestable: number
  filtered: number
  diagnostics?: SourcePreviewDiagnostics | null
}

export interface SourcePreviewDiagnostics {
  fetch_mode?: string | null
  fetch_status?: number | null
  fetch_error?: string | null
  selector_errors?: string[]
  selector_warnings?: string[]
  no_match_warnings?: string[]
  non_unique_warnings?: string[]
  fragile_selector_warnings?: string[]
  dedupe_preview_key?: string | null
}

// ─────────────────────────────────────────────────────────────────────────────
// Dedup / Seen Types
// ─────────────────────────────────────────────────────────────────────────────

export interface SourceSeenStats {
  source_id: number
  user_id: number
  seen_count: number
  latest_seen_at: string | null
  defer_until: string | null
  consec_not_modified: number | null
  recent_keys: string[]
}

export interface SourceSeenResetResponse {
  source_id: number
  user_id: number
  cleared: number
  cleared_backoff: boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// UI State Types
// ─────────────────────────────────────────────────────────────────────────────

export type WatchlistTab =
  | "overview"
  | "sources"
  | "jobs"
  | "runs"
  | "items"
  | "alerts"
  | "outputs"
  | "templates"
  | "settings"
