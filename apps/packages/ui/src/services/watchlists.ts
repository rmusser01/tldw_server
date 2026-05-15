/**
 * Watchlists API service layer
 * Provides typed access to tldw_server2 /api/v1/watchlists endpoints
 */

import { bgRequest, bgUpload } from "@/services/background-proxy"
import { getTldwTTSModel, getTldwTTSVoice } from "@/services/tts"
import type {
  ClaimCluster,
  JobPreviewResult,
  PaginatedResponse,
  RunDetailResponse,
  ScrapedItem,
  ScrapedItemAlertFilterParams,
  ScrapedItemBatchUpdateRequest,
  ScrapedItemBatchUpdateResponse,
  ScrapedItemSortMode,
  ScrapedItemSmartCounts,
  ScrapedItemUpdate,
  SourceSeenResetResponse,
  SourceSeenStats,
  SourcesCheckNowResponse,
  SourcesBulkCreateResponse,
  SourcesImportResponse,
  WatchlistClusterSubscription,
  WatchlistFilter,
  WatchlistGroup,
  WatchlistGroupCreate,
  WatchlistsOnboardingTelemetryPayload,
  WatchlistsOnboardingTelemetryResponse,
  WatchlistsOnboardingTelemetrySummaryResponse,
  WatchlistsIaExperimentTelemetryPayload,
  WatchlistsIaExperimentTelemetryResponse,
  WatchlistsIaExperimentTelemetrySummaryResponse,
  WatchlistsRcTelemetrySummaryResponse,
  WatchlistContainer,
  WatchlistContentAlert,
  WatchlistContentAlertRule,
  WatchlistContentAlertRuleCreate,
  WatchlistContentAlertRuleUpdate,
  WatchlistContentAlertSeverity,
  WatchlistContentAlertStatus,
  WatchlistContentAlertUpdate,
  WatchlistCreate,
  WatchlistItemSavedView,
  WatchlistItemSavedViewCreate,
  WatchlistItemSavedViewUpdate,
  WatchlistJob,
  WatchlistJobCreate,
  WatchlistJobUpdate,
  WatchlistOutput,
  WatchlistOutputCreate,
  WatchlistRun,
  WatchlistSettings,
  WatchlistSource,
  WatchlistSourceCreate,
  WatchlistSourceUpdate,
  WatchlistTag,
  WatchlistTemplate,
  WatchlistTemplateCreate,
  WatchlistTemplateVersionSummary,
  WatchlistUpdate
} from "@/types/watchlists"

// Helper to build query string (supports array params)
const buildQuery = (params?: Record<string, unknown> | object | null): string => {
  if (!params) return ""
  const query = new URLSearchParams()
  Object.entries(params as Record<string, unknown>).forEach(([key, value]) => {
    if (value == null) return
    if (Array.isArray(value)) {
      value.forEach((entry) => {
        if (entry != null) query.append(key, String(entry))
      })
      return
    }
    query.set(key, String(value))
  })
  const str = query.toString()
  return str ? `?${str}` : ""
}

// ─────────────────────────────────────────────────────────────────────────────
// Watchlists API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchWatchlistsParams {
  status?: "active" | "paused" | "archived"
  include_deleted?: boolean
  page?: number
  size?: number
}

export const fetchWatchlists = async (
  params?: FetchWatchlistsParams
): Promise<PaginatedResponse<WatchlistContainer>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistContainer>>({
    path: `/api/v1/watchlists${qs}` as any,
    method: "GET"
  })
}

export const getWatchlist = async (watchlistId: number): Promise<WatchlistContainer> => {
  return bgRequest<WatchlistContainer>({
    path: `/api/v1/watchlists/${watchlistId}` as any,
    method: "GET"
  })
}

export const createWatchlist = async (
  watchlist: WatchlistCreate
): Promise<WatchlistContainer> => {
  return bgRequest<WatchlistContainer>({
    path: "/api/v1/watchlists",
    method: "POST",
    body: watchlist
  })
}

export const updateWatchlist = async (
  watchlistId: number,
  updates: WatchlistUpdate
): Promise<WatchlistContainer> => {
  return bgRequest<WatchlistContainer>({
    path: `/api/v1/watchlists/${watchlistId}` as any,
    method: "PATCH",
    body: updates
  })
}

export interface FetchSourcesParams {
  q?: string
  tags?: string[]
  watchlist_id?: number
  page?: number
  size?: number
}

export interface ReversibleDeleteResponse {
  success: boolean
  restore_window_seconds: number
  restore_expires_at: string
}

export interface SourceDeleteResponse extends ReversibleDeleteResponse {
  source_id: number
}

export interface WatchlistDeleteResponse extends ReversibleDeleteResponse {
  watchlist_id: number
}

export const deleteWatchlist = async (
  watchlistId: number
): Promise<WatchlistDeleteResponse> => {
  return bgRequest<WatchlistDeleteResponse>({
    path: `/api/v1/watchlists/${watchlistId}` as any,
    method: "DELETE"
  })
}

export const restoreWatchlist = async (
  watchlistId: number
): Promise<WatchlistContainer> => {
  return bgRequest<WatchlistContainer>({
    path: `/api/v1/watchlists/${watchlistId}/restore` as any,
    method: "POST"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Alerts API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchContentAlertRulesParams {
  enabled?: boolean
  page?: number
  size?: number
}

export interface FetchContentAlertsParams {
  status?: WatchlistContentAlertStatus
  severity?: WatchlistContentAlertSeverity
  rule_id?: number
  source_id?: number
  page?: number
  size?: number
}

export interface ContentAlertRuleDeleteResponse {
  deleted: boolean
}

export const fetchWatchlistContentAlertRules = async (
  watchlistId: number,
  params?: FetchContentAlertRulesParams
): Promise<PaginatedResponse<WatchlistContentAlertRule>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistContentAlertRule>>({
    path: `/api/v1/watchlists/${watchlistId}/content-alert-rules${qs}` as any,
    method: "GET"
  })
}

export const createWatchlistContentAlertRule = async (
  watchlistId: number,
  rule: WatchlistContentAlertRuleCreate
): Promise<WatchlistContentAlertRule> => {
  return bgRequest<WatchlistContentAlertRule>({
    path: `/api/v1/watchlists/${watchlistId}/content-alert-rules` as any,
    method: "POST",
    body: rule
  })
}

export const updateWatchlistContentAlertRule = async (
  watchlistId: number,
  ruleId: number,
  updates: WatchlistContentAlertRuleUpdate
): Promise<WatchlistContentAlertRule> => {
  return bgRequest<WatchlistContentAlertRule>({
    path: `/api/v1/watchlists/${watchlistId}/content-alert-rules/${ruleId}` as any,
    method: "PATCH",
    body: updates
  })
}

export const deleteWatchlistContentAlertRule = async (
  watchlistId: number,
  ruleId: number
): Promise<ContentAlertRuleDeleteResponse> => {
  return bgRequest<ContentAlertRuleDeleteResponse>({
    path: `/api/v1/watchlists/${watchlistId}/content-alert-rules/${ruleId}` as any,
    method: "DELETE"
  })
}

export const fetchWatchlistContentAlerts = async (
  watchlistId: number,
  params?: FetchContentAlertsParams
): Promise<PaginatedResponse<WatchlistContentAlert>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistContentAlert>>({
    path: `/api/v1/watchlists/${watchlistId}/alerts${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistContentAlert = async (
  watchlistId: number,
  alertId: number
): Promise<WatchlistContentAlert> => {
  return bgRequest<WatchlistContentAlert>({
    path: `/api/v1/watchlists/${watchlistId}/alerts/${alertId}` as any,
    method: "GET"
  })
}

export const updateWatchlistContentAlert = async (
  watchlistId: number,
  alertId: number,
  updates: WatchlistContentAlertUpdate
): Promise<WatchlistContentAlert> => {
  return bgRequest<WatchlistContentAlert>({
    path: `/api/v1/watchlists/${watchlistId}/alerts/${alertId}` as any,
    method: "PATCH",
    body: updates
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Sources API
// ─────────────────────────────────────────────────────────────────────────────

export const fetchWatchlistSources = async (
  params?: FetchSourcesParams
): Promise<PaginatedResponse<WatchlistSource>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistSource>>({
    path: `/api/v1/watchlists/sources${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistSource = async (sourceId: number): Promise<WatchlistSource> => {
  return bgRequest<WatchlistSource>({
    path: `/api/v1/watchlists/sources/${sourceId}` as any,
    method: "GET"
  })
}

export const createWatchlistSource = async (
  source: WatchlistSourceCreate
): Promise<WatchlistSource> => {
  return bgRequest<WatchlistSource>({
    path: "/api/v1/watchlists/sources",
    method: "POST",
    body: source
  })
}

export const updateWatchlistSource = async (
  sourceId: number,
  updates: WatchlistSourceUpdate
): Promise<WatchlistSource> => {
  return bgRequest<WatchlistSource>({
    path: `/api/v1/watchlists/sources/${sourceId}` as any,
    method: "PATCH",
    body: updates
  })
}

export const deleteWatchlistSource = async (
  sourceId: number
): Promise<SourceDeleteResponse> => {
  return bgRequest<SourceDeleteResponse>({
    path: `/api/v1/watchlists/sources/${sourceId}` as any,
    method: "DELETE"
  })
}

export const restoreWatchlistSource = async (
  sourceId: number
): Promise<WatchlistSource> => {
  return bgRequest<WatchlistSource>({
    path: `/api/v1/watchlists/sources/${sourceId}/restore` as any,
    method: "POST"
  })
}

export const bulkCreateSources = async (
  sources: WatchlistSourceCreate[]
): Promise<SourcesBulkCreateResponse> => {
  return bgRequest<SourcesBulkCreateResponse>({
    path: "/api/v1/watchlists/sources/bulk",
    method: "POST",
    body: { sources }
  })
}

export const importOpml = async (
  file: File,
  options?: { active?: boolean; tags?: string[]; group_id?: number; watchlist_id?: number }
): Promise<SourcesImportResponse> => {
  const data = await file.arrayBuffer()
  const fields: Record<string, unknown> = {}
  if (options?.active != null) fields.active = String(options.active)
  if (options?.group_id != null) fields.group_id = String(options.group_id)
  if (options?.watchlist_id != null) fields.watchlist_id = String(options.watchlist_id)
  if (options?.tags?.length) fields.tags = options.tags
  return bgUpload<SourcesImportResponse>({
    path: "/api/v1/watchlists/sources/import",
    method: "POST",
    file: { name: file.name, type: file.type || "text/xml", data },
    fields: Object.keys(fields).length ? fields : undefined
  })
}

export const exportOpml = async (params?: {
  tag?: string[]
  group?: number[]
  type?: string
}): Promise<string> => {
  const qs = buildQuery(params || {})
  return bgRequest<string>({
    path: `/api/v1/watchlists/sources/export${qs}` as any,
    method: "GET"
  })
}

export const getSourceSeenStats = async (
  sourceId: number,
  params?: { target_user_id?: number; keys_limit?: number }
): Promise<SourceSeenStats> => {
  const qs = buildQuery(params || {})
  return bgRequest<SourceSeenStats>({
    path: `/api/v1/watchlists/sources/${sourceId}/seen${qs}` as any,
    method: "GET"
  })
}

export const clearSourceSeen = async (
  sourceId: number,
  params?: { target_user_id?: number; clear_backoff?: boolean }
): Promise<SourceSeenResetResponse> => {
  const qs = buildQuery(params || {})
  return bgRequest<SourceSeenResetResponse>({
    path: `/api/v1/watchlists/sources/${sourceId}/seen${qs}` as any,
    method: "DELETE"
  })
}

export const checkWatchlistSourcesNow = async (
  sourceIds: number[]
): Promise<SourcesCheckNowResponse> => {
  return bgRequest<SourcesCheckNowResponse>({
    path: "/api/v1/watchlists/sources/check-now" as any,
    method: "POST",
    body: { source_ids: sourceIds }
  })
}

export const testWatchlistSource = async (
  sourceId: number,
  params?: { limit?: number }
): Promise<JobPreviewResult> => {
  const qs = buildQuery(params || {})
  return bgRequest<JobPreviewResult>({
    path: `/api/v1/watchlists/sources/${sourceId}/test${qs}` as any,
    method: "POST"
  })
}

export interface WatchlistSourceDraftTestRequest {
  url: string
  source_type: "rss" | "site" | "forum"
  settings?: Record<string, unknown> | null
}

export const testWatchlistSourceDraft = async (
  payload: WatchlistSourceDraftTestRequest,
  params?: { limit?: number }
): Promise<JobPreviewResult> => {
  const qs = buildQuery(params || {})
  return bgRequest<JobPreviewResult>({
    path: `/api/v1/watchlists/sources/test${qs}` as any,
    method: "POST",
    body: payload
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Groups API
// ─────────────────────────────────────────────────────────────────────────────

export const fetchWatchlistGroups = async (
  params?: { q?: string; page?: number; size?: number }
): Promise<PaginatedResponse<WatchlistGroup>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistGroup>>({
    path: `/api/v1/watchlists/groups${qs}` as any,
    method: "GET"
  })
}

export const createWatchlistGroup = async (
  group: WatchlistGroupCreate
): Promise<WatchlistGroup> => {
  return bgRequest<WatchlistGroup>({
    path: "/api/v1/watchlists/groups",
    method: "POST",
    body: group
  })
}

export const updateWatchlistGroup = async (
  groupId: number,
  updates: Partial<WatchlistGroupCreate>
): Promise<WatchlistGroup> => {
  return bgRequest<WatchlistGroup>({
    path: `/api/v1/watchlists/groups/${groupId}` as any,
    method: "PATCH",
    body: updates
  })
}

export const deleteWatchlistGroup = async (groupId: number): Promise<void> => {
  return bgRequest<void>({
    path: `/api/v1/watchlists/groups/${groupId}` as any,
    method: "DELETE"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tags API
// ─────────────────────────────────────────────────────────────────────────────

export const fetchWatchlistTags = async (
  params?: { q?: string; page?: number; size?: number }
): Promise<PaginatedResponse<WatchlistTag>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistTag>>({
    path: `/api/v1/watchlists/tags${qs}` as any,
    method: "GET"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Jobs API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchJobsParams {
  q?: string
  watchlist_id?: number
  page?: number
  size?: number
}

export interface JobDeleteResponse extends ReversibleDeleteResponse {
  job_id: number
}

export const fetchWatchlistJobs = async (
  params?: FetchJobsParams
): Promise<PaginatedResponse<WatchlistJob>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistJob>>({
    path: `/api/v1/watchlists/jobs${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistJob = async (jobId: number): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: `/api/v1/watchlists/jobs/${jobId}` as any,
    method: "GET"
  })
}

export const createWatchlistJob = async (job: WatchlistJobCreate): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: "/api/v1/watchlists/jobs",
    method: "POST",
    body: job
  })
}

export const updateWatchlistJob = async (
  jobId: number,
  updates: WatchlistJobUpdate
): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: `/api/v1/watchlists/jobs/${jobId}` as any,
    method: "PATCH",
    body: updates
  })
}

export const deleteWatchlistJob = async (jobId: number): Promise<JobDeleteResponse> => {
  return bgRequest<JobDeleteResponse>({
    path: `/api/v1/watchlists/jobs/${jobId}` as any,
    method: "DELETE"
  })
}

export const restoreWatchlistJob = async (jobId: number): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: `/api/v1/watchlists/jobs/${jobId}/restore` as any,
    method: "POST"
  })
}

export const previewWatchlistJob = async (
  jobId: number,
  params?: { limit?: number; per_source?: number }
): Promise<JobPreviewResult> => {
  const qs = buildQuery(params || {})
  return bgRequest<JobPreviewResult>({
    path: `/api/v1/watchlists/jobs/${jobId}/preview${qs}` as any,
    method: "POST"
  })
}

export interface WatchlistAudioSettingsTestRequest {
  text: string
  voice: string
  speed: number
  model?: string
  response_format?: "mp3" | "wav" | "ogg" | "opus" | "flac" | "aac" | "webm"
}

export const testWatchlistAudioSettings = async (
  payload: WatchlistAudioSettingsTestRequest
): Promise<ArrayBuffer> => {
  const model = payload.model || (await getTldwTTSModel())
  const voice = payload.voice || (await getTldwTTSVoice())

  return bgRequest<ArrayBuffer>({
    path: "/api/v1/audio/speech",
    method: "POST",
    headers: {
      Accept: "audio/mpeg"
    },
    body: {
      input: payload.text,
      text: payload.text,
      model,
      voice,
      speed: payload.speed,
      ...(payload.response_format ? { response_format: payload.response_format } : {})
    },
    responseType: "arrayBuffer"
  })
}

export const updateJobFilters = async (
  jobId: number,
  filters: WatchlistFilter[]
): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: `/api/v1/watchlists/jobs/${jobId}/filters` as any,
    method: "PATCH",
    body: { filters }
  })
}

export const addJobFilters = async (
  jobId: number,
  filters: WatchlistFilter[]
): Promise<WatchlistJob> => {
  return bgRequest<WatchlistJob>({
    path: `/api/v1/watchlists/jobs/${jobId}/filters:add` as any,
    method: "POST",
    body: { filters }
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Runs API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchRunsParams {
  q?: string
  watchlist_id?: number
  page?: number
  size?: number
}

export interface ExportRunsCsvParams extends FetchRunsParams {
  scope?: "global" | "job"
  job_id?: number
  include_tallies?: boolean
  tallies_mode?: "per_run" | "aggregate"
}

export const fetchWatchlistRuns = async (
  params?: FetchRunsParams
): Promise<PaginatedResponse<WatchlistRun>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistRun>>({
    path: `/api/v1/watchlists/runs${qs}` as any,
    method: "GET"
  })
}

export const fetchJobRuns = async (
  jobId: number,
  params?: { page?: number; size?: number }
): Promise<PaginatedResponse<WatchlistRun>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistRun>>({
    path: `/api/v1/watchlists/jobs/${jobId}/runs${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistRun = async (runId: number): Promise<WatchlistRun> => {
  return bgRequest<WatchlistRun>({
    path: `/api/v1/watchlists/runs/${runId}` as any,
    method: "GET"
  })
}

export const getRunDetails = async (runId: number): Promise<RunDetailResponse> => {
  return bgRequest<RunDetailResponse>({
    path: `/api/v1/watchlists/runs/${runId}/details?include_tallies=true` as any,
    method: "GET"
  })
}

export const triggerWatchlistRun = async (jobId: number): Promise<WatchlistRun> => {
  return bgRequest<WatchlistRun>({
    path: `/api/v1/watchlists/jobs/${jobId}/run` as any,
    method: "POST"
  })
}

export interface CancelWatchlistRunResponse {
  run_id: number
  status: string
  cancelled: boolean
  message?: string | null
}

export const cancelWatchlistRun = async (
  runId: number
): Promise<CancelWatchlistRunResponse> => {
  return bgRequest<CancelWatchlistRunResponse>({
    path: `/api/v1/watchlists/runs/${runId}/cancel` as any,
    method: "POST"
  })
}

export const exportRunsCsv = async (params?: ExportRunsCsvParams): Promise<string> => {
  const qs = buildQuery(params || {})
  return bgRequest<string>({
    path: `/api/v1/watchlists/runs/export.csv${qs}` as any,
    method: "GET"
  })
}

export const exportRunTalliesCsv = async (runId: number): Promise<string> => {
  return bgRequest<string>({
    path: `/api/v1/watchlists/runs/${runId}/tallies.csv` as any,
    method: "GET"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Scraped Items API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchItemsParams extends ScrapedItemAlertFilterParams {
  run_id?: number
  job_id?: number
  source_id?: number
  watchlist_id?: number
  status?: string
  reviewed?: boolean
  queued_for_briefing?: boolean
  sort?: ScrapedItemSortMode
  include_alert_summary?: boolean
  q?: string
  since?: string
  until?: string
  page?: number
  size?: number
}

export const fetchScrapedItems = async (
  params?: FetchItemsParams
): Promise<PaginatedResponse<ScrapedItem>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<ScrapedItem>>({
    path: `/api/v1/watchlists/items${qs}` as any,
    method: "GET"
  })
}

export interface FetchItemSmartCountsParams extends ScrapedItemAlertFilterParams {
  run_id?: number
  job_id?: number
  source_id?: number
  watchlist_id?: number
  status?: string
  q?: string
  since?: string
  until?: string
  queue_run_id?: number
}

export const fetchScrapedItemSmartCounts = async (
  params?: FetchItemSmartCountsParams
): Promise<ScrapedItemSmartCounts> => {
  const qs = buildQuery(params || {})
  return bgRequest<ScrapedItemSmartCounts>({
    path: `/api/v1/watchlists/items/smart-counts${qs}` as any,
    method: "GET"
  })
}

export const getScrapedItem = async (itemId: number): Promise<ScrapedItem> => {
  return bgRequest<ScrapedItem>({
    path: `/api/v1/watchlists/items/${itemId}` as any,
    method: "GET"
  })
}

export const updateScrapedItem = async (
  itemId: number,
  updates: ScrapedItemUpdate
): Promise<ScrapedItem> => {
  return bgRequest<ScrapedItem>({
    path: `/api/v1/watchlists/items/${itemId}` as any,
    method: "PATCH",
    body: updates
  })
}

export const batchUpdateScrapedItems = async (
  payload: ScrapedItemBatchUpdateRequest
): Promise<ScrapedItemBatchUpdateResponse> => {
  return bgRequest<ScrapedItemBatchUpdateResponse>({
    path: "/api/v1/watchlists/items/batch-update" as any,
    method: "POST",
    body: payload
  })
}

export const fetchWatchlistItemViews = async (
  watchlistId: number
): Promise<{ items: WatchlistItemSavedView[] }> => {
  return bgRequest<{ items: WatchlistItemSavedView[] }>({
    path: `/api/v1/watchlists/${watchlistId}/item-views` as any,
    method: "GET"
  })
}

export const createWatchlistItemView = async (
  watchlistId: number,
  payload: WatchlistItemSavedViewCreate
): Promise<WatchlistItemSavedView> => {
  return bgRequest<WatchlistItemSavedView>({
    path: `/api/v1/watchlists/${watchlistId}/item-views` as any,
    method: "POST",
    body: payload
  })
}

export const updateWatchlistItemView = async (
  watchlistId: number,
  viewId: number,
  payload: WatchlistItemSavedViewUpdate
): Promise<WatchlistItemSavedView> => {
  return bgRequest<WatchlistItemSavedView>({
    path: `/api/v1/watchlists/${watchlistId}/item-views/${viewId}` as any,
    method: "PATCH",
    body: payload
  })
}

export const deleteWatchlistItemView = async (
  watchlistId: number,
  viewId: number
): Promise<void> => {
  return bgRequest<void>({
    path: `/api/v1/watchlists/${watchlistId}/item-views/${viewId}` as any,
    method: "DELETE"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Outputs API
// ─────────────────────────────────────────────────────────────────────────────

export interface FetchOutputsParams {
  job_id?: number
  run_id?: number
  watchlist_id?: number
  page?: number
  size?: number
}

export const fetchWatchlistOutputs = async (
  params?: FetchOutputsParams
): Promise<PaginatedResponse<WatchlistOutput>> => {
  const qs = buildQuery(params || {})
  return bgRequest<PaginatedResponse<WatchlistOutput>>({
    path: `/api/v1/watchlists/outputs${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistOutput = async (outputId: number): Promise<WatchlistOutput> => {
  return bgRequest<WatchlistOutput>({
    path: `/api/v1/watchlists/outputs/${outputId}` as any,
    method: "GET"
  })
}

export const createWatchlistOutput = async (
  output: WatchlistOutputCreate
): Promise<WatchlistOutput> => {
  return bgRequest<WatchlistOutput>({
    path: "/api/v1/watchlists/outputs",
    method: "POST",
    body: output
  })
}

export const downloadWatchlistOutput = async (outputId: number): Promise<string> => {
  return bgRequest<string>({
    path: `/api/v1/watchlists/outputs/${outputId}/download` as any,
    method: "GET"
  })
}

export const downloadWatchlistOutputBinary = async (outputId: number): Promise<ArrayBuffer> => {
  return bgRequest<ArrayBuffer>({
    path: `/api/v1/watchlists/outputs/${outputId}/download` as any,
    method: "GET",
    responseType: "arrayBuffer"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Templates API
// ─────────────────────────────────────────────────────────────────────────────

export interface JobOutputTemplateSummary {
  id: string
  name: string
  format: "md" | "html" | "mp3" | string
  updated_at?: string | null
}

export const fetchJobOutputTemplates = async (
  params?: { q?: string; limit?: number; offset?: number }
): Promise<{ items: JobOutputTemplateSummary[]; total: number }> => {
  const qs = buildQuery(params || {})
  const data = await bgRequest<{ items?: Array<Record<string, unknown>>; total?: number }>({
    path: `/api/v1/outputs/templates${qs}` as any,
    method: "GET"
  })
  const items = Array.isArray(data?.items)
    ? data.items
      .filter((item) => typeof item?.name === "string" && item.name.trim().length > 0)
      .map((item) => ({
        id: String(item.id),
        name: String(item.name).trim(),
        format: typeof item.format === "string" ? item.format : "md",
        updated_at: typeof item.updated_at === "string" ? item.updated_at : null
      }))
    : []
  return {
    items,
    total: Number.isFinite(data?.total) ? Number(data.total) : items.length
  }
}

export const fetchWatchlistTemplates = async (): Promise<{ items: WatchlistTemplate[] }> => {
  return bgRequest<{ items: WatchlistTemplate[] }>({
    path: "/api/v1/watchlists/templates",
    method: "GET"
  })
}

export const getWatchlistTemplate = async (
  templateName: string,
  options?: { version?: number }
): Promise<WatchlistTemplate> => {
  const qs = buildQuery(options || {})
  return bgRequest<WatchlistTemplate>({
    path: `/api/v1/watchlists/templates/${encodeURIComponent(templateName)}${qs}` as any,
    method: "GET"
  })
}

export const getWatchlistTemplateVersions = async (
  templateName: string
): Promise<{ items: WatchlistTemplateVersionSummary[] }> => {
  return bgRequest<{ items: WatchlistTemplateVersionSummary[] }>({
    path: `/api/v1/watchlists/templates/${encodeURIComponent(templateName)}/versions` as any,
    method: "GET"
  })
}

export const createWatchlistTemplate = async (
  template: WatchlistTemplateCreate
): Promise<WatchlistTemplate> => {
  return bgRequest<WatchlistTemplate>({
    path: "/api/v1/watchlists/templates",
    method: "POST",
    body: template
  })
}

export const deleteWatchlistTemplate = async (templateName: string): Promise<void> => {
  return bgRequest<void>({
    path: `/api/v1/watchlists/templates/${encodeURIComponent(templateName)}` as any,
    method: "DELETE"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings API
// ─────────────────────────────────────────────────────────────────────────────

export const getWatchlistSettings = async (): Promise<WatchlistSettings> => {
  return bgRequest<WatchlistSettings>({
    path: "/api/v1/watchlists/settings",
    method: "GET"
  })
}

export const recordWatchlistsOnboardingTelemetry = async (
  payload: WatchlistsOnboardingTelemetryPayload
): Promise<WatchlistsOnboardingTelemetryResponse> => {
  return bgRequest<WatchlistsOnboardingTelemetryResponse>({
    path: "/api/v1/watchlists/telemetry/onboarding" as any,
    method: "POST",
    body: payload
  })
}

export const fetchWatchlistsOnboardingTelemetrySummary = async (params?: {
  since?: string
  until?: string
}): Promise<WatchlistsOnboardingTelemetrySummaryResponse> => {
  const qs = buildQuery(params || {})
  return bgRequest<WatchlistsOnboardingTelemetrySummaryResponse>({
    path: `/api/v1/watchlists/telemetry/onboarding/summary${qs}` as any,
    method: "GET"
  })
}

export const fetchWatchlistsRcTelemetrySummary = async (params?: {
  since?: string
  until?: string
}): Promise<WatchlistsRcTelemetrySummaryResponse> => {
  const qs = buildQuery(params || {})
  return bgRequest<WatchlistsRcTelemetrySummaryResponse>({
    path: `/api/v1/watchlists/telemetry/rc-summary${qs}` as any,
    method: "GET"
  })
}

export const recordWatchlistsIaExperimentTelemetry = async (
  payload: WatchlistsIaExperimentTelemetryPayload
): Promise<WatchlistsIaExperimentTelemetryResponse> => {
  return bgRequest<WatchlistsIaExperimentTelemetryResponse>({
    path: "/api/v1/watchlists/telemetry/ia-experiment" as any,
    method: "POST",
    body: payload
  })
}

export const fetchWatchlistsIaExperimentTelemetrySummary = async (params?: {
  since?: string
  until?: string
}): Promise<WatchlistsIaExperimentTelemetrySummaryResponse> => {
  const qs = buildQuery(params || {})
  return bgRequest<WatchlistsIaExperimentTelemetrySummaryResponse>({
    path: `/api/v1/watchlists/telemetry/ia-experiment/summary${qs}` as any,
    method: "GET"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Claim Clusters API
// ─────────────────────────────────────────────────────────────────────────────

export const fetchJobClaimClusters = async (
  jobId: number
): Promise<WatchlistClusterSubscription[]> => {
  const response = await bgRequest<{ clusters: WatchlistClusterSubscription[] }>({
    path: `/api/v1/watchlists/${jobId}/clusters` as any,
    method: "GET"
  })
  return response.clusters
}

export const subscribeJobToCluster = async (
  jobId: number,
  clusterId: number
): Promise<void> => {
  return bgRequest<void>({
    path: `/api/v1/watchlists/${jobId}/clusters` as any,
    method: "POST",
    body: { cluster_id: clusterId }
  })
}

export const unsubscribeJobFromCluster = async (
  jobId: number,
  clusterId: number
): Promise<void> => {
  return bgRequest<void>({
    path: `/api/v1/watchlists/${jobId}/clusters/${clusterId}` as any,
    method: "DELETE"
  })
}

export const fetchClaimClusters = async (params?: {
  limit?: number
  offset?: number
  keyword?: string
  since?: string
  min_size?: number
  watchlisted?: boolean
}): Promise<ClaimCluster[]> => {
  const qs = buildQuery(params || {})
  return bgRequest<ClaimCluster[]>({
    path: `/api/v1/claims/clusters${qs}` as any,
    method: "GET"
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Template Validation & Preview
// ─────────────────────────────────────────────────────────────────────────────

export interface TemplateValidationError {
  line?: number | null
  column?: number | null
  message: string
}

export interface TemplateValidationResult {
  valid: boolean
  errors: TemplateValidationError[]
}

export interface TemplatePreviewResult {
  rendered: string
  context_keys: string[]
  warnings: string[]
}

export interface TemplateComposerSectionRequest {
  run_id: number
  block_id: string
  prompt: string
  input_scope?: "all_items" | "top_items" | "selected_items"
  style?: string
  length_target?: "short" | "medium" | "long"
}

export interface TemplateComposerSectionResult {
  block_id: string
  content: string
  warnings: string[]
  diagnostics: Record<string, unknown>
}

export interface TemplateComposerFlowSection {
  id: string
  content: string
}

export type TemplateComposerFlowCheckMode = "suggest_only" | "auto_apply"

export interface TemplateComposerFlowIssue {
  section_id?: string | null
  severity: "info" | "warning"
  message: string
}

export interface TemplateComposerFlowCheckResult {
  mode: TemplateComposerFlowCheckMode
  issues: TemplateComposerFlowIssue[]
  diff: string
  sections: TemplateComposerFlowSection[]
}

export const validateWatchlistTemplate = async (
  content: string,
  format: "md" | "html" = "md"
): Promise<TemplateValidationResult> => {
  return bgRequest<TemplateValidationResult>({
    path: "/api/v1/watchlists/templates/validate" as any,
    method: "POST",
    body: { content, format }
  })
}

export const previewWatchlistTemplate = async (
  content: string,
  runId: number,
  format: "md" | "html" = "md",
  signal?: AbortSignal
): Promise<TemplatePreviewResult> => {
  return bgRequest<TemplatePreviewResult>({
    path: "/api/v1/watchlists/templates/preview" as any,
    method: "POST",
    body: { content, format, run_id: runId },
    abortSignal: signal
  })
}

export const composeWatchlistTemplateSection = async (
  payload: TemplateComposerSectionRequest
): Promise<TemplateComposerSectionResult> => {
  return bgRequest<TemplateComposerSectionResult>({
    path: "/api/v1/watchlists/templates/compose/section" as any,
    method: "POST",
    body: payload
  })
}

export const flowCheckWatchlistTemplateSections = async (payload: {
  run_id: number
  mode: TemplateComposerFlowCheckMode
  sections: TemplateComposerFlowSection[]
}): Promise<TemplateComposerFlowCheckResult> => {
  return bgRequest<TemplateComposerFlowCheckResult>({
    path: "/api/v1/watchlists/templates/compose/flow-check" as any,
    method: "POST",
    body: payload
  })
}
