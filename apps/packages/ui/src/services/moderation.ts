/**
 * Moderation service - API client for moderation admin endpoints
 */

import { bgRequest } from "@/services/background-proxy"
import type { ApiSendResponse } from "@/services/api-send"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type ModerationAction = "block" | "redact" | "warn" | "pass"
export type ModerationReviewStatus =
  | "needs_review"
  | "approved"
  | "blocked"
  | "redacted"
  | "dismissed"
  | "escalated"
export type ModerationDecisionAction = "approve" | "block" | "redact" | "dismiss" | "escalate"
export type ModerationSeverity = "low" | "medium" | "high" | "critical"
export type ModerationReviewSort = "newest" | "oldest"

export interface ModerationOverrideRule {
  id: string
  pattern: string
  is_regex: boolean
  action: "block" | "warn"
  phase: "input" | "output" | "both"
}

export interface ModerationSettingsResponse {
  pii_enabled?: boolean | null
  categories_enabled?: string[] | null
  effective: {
    pii_enabled?: boolean
    categories_enabled?: string[]
  }
}

export interface ModerationSettingsUpdate {
  pii_enabled?: boolean | null
  categories_enabled?: string[] | null
  persist?: boolean
}

export interface ModerationUserOverride {
  enabled?: boolean
  input_enabled?: boolean
  output_enabled?: boolean
  input_action?: "block" | "redact" | "warn"
  output_action?: "block" | "redact" | "warn"
  redact_replacement?: string
  categories_enabled?: string[] | string
  rules?: ModerationOverrideRule[]
}

export interface ModerationUserOverridesResponse {
  overrides: Record<string, Record<string, any>>
}

export interface ModerationUserOverrideLookupResponse {
  exists: boolean
  override: Record<string, any>
}

export interface BlocklistManagedItem {
  id: number
  line: string
  pattern_type?: "literal" | "regex" | "comment" | "empty"
  action?: "block" | "redact" | "warn"
  replacement?: string | null
  categories?: string[]
  sample?: string | null
  ok?: boolean
  warning?: string | null
  error?: string | null
}

export interface BlocklistManagedResponse {
  version: string
  items: BlocklistManagedItem[]
}

export interface BlocklistAppendResponse {
  version: string
  index: number
  count: number
}

export interface BlocklistDeleteResponse {
  version: string
  count: number
}

export interface BlocklistLintItem {
  index: number
  line: string
  ok: boolean
  pattern_type?: "literal" | "regex" | "comment" | "empty"
  action?: "block" | "redact" | "warn"
  replacement?: string | null
  categories?: string[]
  error?: string | null
  warning?: string | null
  sample?: string | null
}

export interface BlocklistLintResponse {
  items: BlocklistLintItem[]
  valid_count: number
  invalid_count: number
}

export interface ModerationTestRequest {
  user_id?: string
  phase?: "input" | "output"
  text: string
}

export interface ModerationTestResponse {
  flagged: boolean
  action: ModerationAction
  sample?: string | null
  redacted_text?: string | null
  effective: Record<string, any>
  category?: string | null
}

export interface ModerationReviewMatch {
  rule_id?: string | null
  pattern_type?: "literal" | "regex" | "pii" | "category" | null
  category?: string | null
  action?: ModerationAction | null
  sample?: string | null
  confidence?: number | null
}

export interface ModerationReviewItem {
  id: string
  status: ModerationReviewStatus
  phase: "input" | "output"
  source_type?: string | null
  source_id?: string | null
  user_id?: string | null
  session_id?: string | null
  created_at: string
  updated_at?: string | null
  severity?: ModerationSeverity | null
  category?: string | null
  safe_fields: Record<string, boolean>
  excerpt: string
  context?: Record<string, any> | null
  effective_policy: Record<string, any>
  matches: ModerationReviewMatch[]
  recommended_action?: ModerationDecisionAction | null
  retention_expires_at?: string | null
  content_redacted_at?: string | null
  decision_history?: ModerationReviewDecisionHistoryEntry[]
}

export interface ModerationReviewDecisionHistoryEntry {
  id: string
  action: ModerationDecisionAction
  status: ModerationReviewStatus
  previous_status: ModerationReviewStatus
  actor_id: string
  reason?: string | null
  decided_at: string
  undo_eligible: boolean
  undo_expires_at?: string | null
  undone_at?: string | null
  redaction_state: "not_redacted" | "redacted"
}

export interface ModerationReviewListParams {
  status?: ModerationReviewStatus | ""
  category?: string
  severity?: ModerationSeverity | ""
  source_type?: string
  source_id?: string
  user_id?: string
  q?: string
  sort?: ModerationReviewSort
  limit?: number
  cursor?: string | null
}

export interface ModerationReviewListResponse {
  items: ModerationReviewItem[]
  next_cursor?: string | null
  total?: number | null
}

export interface ModerationReviewDecisionRequest {
  action: ModerationDecisionAction
  reason?: string
  actor_id?: string
}

export interface ModerationReviewDecision {
  id: string
  item_id: string
  action: ModerationDecisionAction
  status: ModerationReviewStatus
  previous_status: ModerationReviewStatus
  decided_by: string
  reason?: string | null
  decided_at: string
  undo_expires_at?: string | null
  undone_at?: string | null
  undo_token?: string | null
}

export interface ModerationReviewDecisionResponse {
  item: ModerationReviewItem
  decision: ModerationReviewDecision
  undo_token?: string | null
}

export interface ModerationReviewBulkDecisionRequest {
  item_ids: string[]
  action: ModerationDecisionAction
  reason?: string
}

export interface ModerationReviewBulkDecisionResult {
  item_id: string
  ok: boolean
  item?: ModerationReviewItem | null
  decision?: ModerationReviewDecision | null
  undo_token?: string | null
  error?: string | null
}

export interface ModerationReviewBulkDecisionResponse {
  results: ModerationReviewBulkDecisionResult[]
  ok_count: number
  error_count: number
}

export interface ModerationReviewAuditParams {
  item_id?: string
  decision_id?: string
  actor?: string
  action?: string
  date_from?: string
  date_to?: string
  limit?: number
  cursor?: string | null
}

export interface ModerationReviewAuditEvent {
  id: string
  item_id?: string | null
  decision_id?: string | null
  actor_id?: string | null
  action: string
  summary?: string | null
  created_at: string
  metadata: Record<string, any>
}

export interface ModerationReviewAuditResponse {
  events: ModerationReviewAuditEvent[]
  next_cursor?: string | null
}

function buildModerationReviewQuery(
  params: Record<string, string | number | null | undefined>,
  keys: string[]
): string {
  const query = new URLSearchParams()
  for (const key of keys) {
    const value = params[key]
    if (value === undefined || value === null || value === "") {
      continue
    }
    query.set(key, String(value))
  }
  const text = query.toString()
  return text ? `?${text}` : ""
}

export async function getModerationSettings(): Promise<ModerationSettingsResponse> {
  return await bgRequest<ModerationSettingsResponse>({
    path: "/api/v1/moderation/settings",
    method: "GET"
  })
}

export async function updateModerationSettings(
  body: ModerationSettingsUpdate
): Promise<ModerationSettingsResponse> {
  return await bgRequest<ModerationSettingsResponse>({
    path: "/api/v1/moderation/settings",
    method: "PUT",
    body
  })
}

export async function getEffectivePolicy(userId?: string): Promise<Record<string, any>> {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : ""
  return await bgRequest<Record<string, any>>({
    path: appendPathQuery("/api/v1/moderation/policy/effective", query),
    method: "GET"
  })
}

export async function reloadModeration(): Promise<{ status: string }> {
  return await bgRequest<{ status: string }>({
    path: "/api/v1/moderation/reload",
    method: "POST"
  })
}

export async function listUserOverrides(): Promise<ModerationUserOverridesResponse> {
  return await bgRequest<ModerationUserOverridesResponse>({
    path: "/api/v1/moderation/users",
    method: "GET"
  })
}

export async function getUserOverride(
  userId: string
): Promise<ModerationUserOverrideLookupResponse> {
  return await bgRequest<ModerationUserOverrideLookupResponse>({
    path: toAllowedPath(`/api/v1/moderation/users/${encodeURIComponent(userId)}`),
    method: "GET"
  })
}

export async function setUserOverride(
  userId: string,
  body: ModerationUserOverride
): Promise<Record<string, any>> {
  return await bgRequest<Record<string, any>>({
    path: toAllowedPath(`/api/v1/moderation/users/${encodeURIComponent(userId)}`),
    method: "PUT",
    body
  })
}

export async function deleteUserOverride(userId: string): Promise<{ status: string; persisted?: boolean } > {
  return await bgRequest<{ status: string; persisted?: boolean }>({
    path: toAllowedPath(`/api/v1/moderation/users/${encodeURIComponent(userId)}`),
    method: "DELETE"
  })
}

export async function getBlocklist(): Promise<string[]> {
  return await bgRequest<string[]>({
    path: "/api/v1/moderation/blocklist",
    method: "GET"
  })
}

export async function updateBlocklist(lines: string[]): Promise<{ status: string; count: number } > {
  return await bgRequest<{ status: string; count: number }>({
    path: "/api/v1/moderation/blocklist",
    method: "PUT",
    body: { lines }
  })
}

export async function getManagedBlocklist(): Promise<{
  data: BlocklistManagedResponse
  etag: string | null
}> {
  const resp = await bgRequest<ApiSendResponse<BlocklistManagedResponse>>({
    path: "/api/v1/moderation/blocklist/managed",
    method: "GET",
    returnResponse: true
  })
  if (!resp.ok) {
    throw new Error(resp.error || "Failed to load managed blocklist")
  }
  const etag = resp.headers?.etag || resp.headers?.ETag || null
  return { data: resp.data as BlocklistManagedResponse, etag }
}

export async function appendManagedBlocklist(
  version: string,
  line: string
): Promise<BlocklistAppendResponse> {
  return await bgRequest<BlocklistAppendResponse>({
    path: "/api/v1/moderation/blocklist/append",
    method: "POST",
    headers: { "If-Match": version },
    body: { line }
  })
}

export async function deleteManagedBlocklistItem(
  version: string,
  itemId: number
): Promise<BlocklistDeleteResponse> {
  return await bgRequest<BlocklistDeleteResponse>({
    path: toAllowedPath(`/api/v1/moderation/blocklist/${itemId}`),
    method: "DELETE",
    headers: { "If-Match": version }
  })
}

export async function lintBlocklist(payload: { line?: string; lines?: string[] }): Promise<BlocklistLintResponse> {
  return await bgRequest<BlocklistLintResponse>({
    path: "/api/v1/moderation/blocklist/lint",
    method: "POST",
    body: payload
  })
}

export async function testModeration(
  payload: ModerationTestRequest
): Promise<ModerationTestResponse> {
  return await bgRequest<ModerationTestResponse>({
    path: "/api/v1/moderation/test",
    method: "POST",
    body: payload
  })
}

export async function listModerationReviewItems(
  params: ModerationReviewListParams = {}
): Promise<ModerationReviewListResponse> {
  const query = buildModerationReviewQuery(params as Record<string, string | number | null | undefined>, [
    "status",
    "category",
    "severity",
    "source_type",
    "source_id",
    "user_id",
    "q",
    "sort",
    "limit",
    "cursor"
  ])
  return await bgRequest<ModerationReviewListResponse>({
    path: appendPathQuery("/api/v1/moderation/review/items", query),
    method: "GET"
  })
}

export async function getModerationReviewItem(itemId: string): Promise<ModerationReviewItem> {
  return await bgRequest<ModerationReviewItem>({
    path: toAllowedPath(`/api/v1/moderation/review/items/${encodeURIComponent(itemId)}`),
    method: "GET"
  })
}

export async function decideModerationReviewItem(
  itemId: string,
  body: ModerationReviewDecisionRequest
): Promise<ModerationReviewDecisionResponse> {
  return await bgRequest<ModerationReviewDecisionResponse>({
    path: toAllowedPath(`/api/v1/moderation/review/items/${encodeURIComponent(itemId)}/decision`),
    method: "POST",
    body
  })
}

export async function undoModerationReviewDecision(
  itemId: string,
  undoToken: string
): Promise<ModerationReviewItem> {
  return await bgRequest<ModerationReviewItem>({
    path: toAllowedPath(`/api/v1/moderation/review/items/${encodeURIComponent(itemId)}/undo`),
    method: "POST",
    body: { undo_token: undoToken }
  })
}

export async function bulkDecideModerationReviewItems(
  body: ModerationReviewBulkDecisionRequest
): Promise<ModerationReviewBulkDecisionResponse> {
  return await bgRequest<ModerationReviewBulkDecisionResponse>({
    path: "/api/v1/moderation/review/bulk-decision",
    method: "POST",
    body
  })
}

export async function listModerationReviewAudit(
  params: ModerationReviewAuditParams = {}
): Promise<ModerationReviewAuditResponse> {
  const query = buildModerationReviewQuery(params as Record<string, string | number | null | undefined>, [
    "item_id",
    "decision_id",
    "actor",
    "action",
    "date_from",
    "date_to",
    "limit",
    "cursor"
  ])
  return await bgRequest<ModerationReviewAuditResponse>({
    path: appendPathQuery("/api/v1/moderation/review/audit", query),
    method: "GET"
  })
}
