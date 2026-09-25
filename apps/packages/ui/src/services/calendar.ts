/**
 * Calendar API client.
 */

import { bgRequest } from "./background-proxy"
import { appendPathQuery, toAllowedPath } from "./tldw/path-utils"

export type CalendarItemKind = "event" | "todo"
export type CalendarRole = "owner" | "editor" | "commenter" | "viewer"
export type CalendarPrincipalType = "user" | "org_role"
export type CalendarSourceOwner = "tldw" | "provider" | "linked_projection"

export interface CalendarCreateRequest {
  name: string
  description?: string | null
  color?: string | null
  timezone?: string
  org_id?: number | null
  visibility?: string
  default_reminder_policy?: Record<string, unknown> | null
  rbac_policy_ref?: string | null
}

export interface CalendarResponse {
  id: number
  tenant_id: string
  owner_user_id: number
  org_id?: number | null
  name: string
  description?: string | null
  color?: string | null
  timezone: string
  visibility: string
  default_reminder_policy?: Record<string, unknown> | null
  rbac_policy_ref?: string | null
  archived_at?: string | null
  created_at: string
  updated_at: string
}

export interface CalendarListResponse {
  items: CalendarResponse[]
  total: number
}

export interface CalendarRecurrenceRequest {
  rrule?: string | null
  rdate?: string[] | null
  exdate?: string[] | null
  timezone?: string | null
}

export interface CalendarRecurrenceResponse {
  id: number
  calendar_item_id: number
  rrule?: string | null
  rdate?: string[] | null
  exdate?: string[] | null
  timezone?: string | null
  created_at: string
  updated_at: string
}

export interface CalendarItemCreateRequest {
  calendar_id: number
  kind: CalendarItemKind
  title: string
  description?: string | null
  location?: string | null
  start_at?: string | null
  end_at?: string | null
  due_at?: string | null
  timezone?: string | null
  all_day?: boolean
  status?: string
  local_tags?: string[] | null
  metadata?: Record<string, unknown> | null
  recurrence?: CalendarRecurrenceRequest | null
}

export interface CalendarItemUpdateRequest {
  kind?: CalendarItemKind
  title?: string
  description?: string | null
  location?: string | null
  start_at?: string | null
  end_at?: string | null
  due_at?: string | null
  timezone?: string | null
  all_day?: boolean
  status?: string
  local_tags?: string[] | null
  metadata?: Record<string, unknown> | null
  recurrence?: CalendarRecurrenceRequest | null
  source_owner?: CalendarSourceOwner
  provider_owned?: boolean
}

export interface CalendarLocalTagsUpdateRequest {
  tags: string[]
}

export type CalendarItemMutationContext =
  | {
      source_owner: CalendarSourceOwner | string
      provider_owned?: boolean
      read_only_reason?: string | null
    }
  | {
      provider_owned: boolean
      source_owner?: CalendarSourceOwner | string
      read_only_reason?: string | null
    }
  | {
      read_only_reason: string
      source_owner?: CalendarSourceOwner | string
      provider_owned?: boolean
    }

export type CalendarItemDeleteTarget = CalendarItemMutationContext & {
  id?: string | number | null
  calendar_item_id?: string | number | null
}

export interface CalendarItemResponse {
  id: number
  calendar_id: number
  kind: string
  source_owner: CalendarSourceOwner | string
  provider_owned: boolean
  title: string
  description?: string | null
  location?: string | null
  start_at?: string | null
  end_at?: string | null
  due_at?: string | null
  timezone?: string | null
  all_day: boolean
  status: string
  local_tags: string[]
  metadata: Record<string, unknown>
  external_binding_id?: number | null
  source_uid?: string | null
  source_etag?: string | null
  source_ctag?: string | null
  source_updated_at?: string | null
  copied_from_item_id?: number | null
  linked_projection_type?: string | null
  linked_projection_id?: string | null
  deleted_at?: string | null
  remote_deleted_at?: string | null
  created_at: string
  updated_at: string
  recurrence?: CalendarRecurrenceResponse | null
}

export interface CalendarItemDeleteResponse {
  deleted: boolean
}

export interface CalendarAgendaQuery {
  start_at: string
  end_at: string
  calendar_ids?: number[]
  include_scheduled_tasks?: boolean
}

export interface CalendarWeekQuery {
  week_start: string
  timezone?: string
  calendar_ids?: number[]
  include_scheduled_tasks?: boolean
}

export interface CalendarViewLinkResponse {
  target_type: string
  target_id: string
  label?: string | null
  url?: string | null
  metadata: Record<string, unknown>
}

export interface CalendarViewItemResponse {
  id: string
  title: string
  kind: CalendarItemKind | string
  source_owner: string
  start_at?: string | null
  end_at?: string | null
  due_at?: string | null
  calendar_id?: number | null
  calendar_item_id?: number | null
  description?: string | null
  location?: string | null
  all_day: boolean
  status?: string | null
  local_tags: string[]
  read_only_reason?: string | null
  recurrence_id?: number | null
  occurrence_index?: number | null
  link?: CalendarViewLinkResponse | null
  metadata: Record<string, unknown>
}

export interface CalendarViewResponse {
  start_at: string
  end_at: string
  items: CalendarViewItemResponse[]
  partial?: boolean
  warnings?: string[]
}

export interface CalendarAnnotationCreateRequest {
  body: string
  tags?: string[] | null
}

export interface CalendarAnnotationResponse {
  id: number
  calendar_item_id: number
  author_user_id: number
  body: string
  tags: string[]
  deleted_at?: string | null
  created_at: string
  updated_at: string
}

export interface CalendarLinkCreateRequest {
  target_type: string
  target_id: string
  label?: string | null
  url?: string | null
  metadata?: Record<string, unknown> | null
}

export interface CalendarLinkResponse {
  id: number
  calendar_item_id: number
  target_type: string
  target_id: string
  label?: string | null
  url?: string | null
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface CalendarItemCopyRequest {
  target_calendar_id?: number | null
  title?: string | null
}

export interface CalDavAccountCreateRequest {
  display_name: string
  server_url?: string
  username?: string
  password?: string
  token?: string
  secret_ref?: string | null
  account_metadata?: Record<string, unknown> | null
}

export interface CalDavAccountVerifyRequest {
  server_url?: string
  username?: string
  password?: string
  token?: string
}

export interface ExternalCalendarAccountResponse {
  id: number
  tenant_id?: string
  user_id?: number
  provider: string
  display_name: string
  secret_ref?: string | null
  account_metadata?: Record<string, unknown> | null
  status?: string
  revoked_at?: string | null
  deleted_at?: string | null
  created_at?: string
  updated_at?: string
}

export interface ExternalCalendarAccountListResponse {
  items: ExternalCalendarAccountResponse[]
  total: number
}

export interface CalDavAccountVerifyResponse {
  account_id: number
  verified: boolean
  status?: string
  error?: string | null
}

export interface CalDavAccountMutationResponse {
  revoked?: boolean
  deleted?: boolean
}

export interface ExternalCalendarDiscoveryResponse {
  items: Array<{
    remote_calendar_id: string
    remote_display_name?: string | null
    provider_capabilities?: Record<string, unknown> | null
  }>
}

export interface ExternalCalendarBindingCreateRequest {
  account_id: number
  calendar_id: number
  remote_calendar_id: string
  remote_display_name?: string | null
  sync_enabled?: boolean
  sync_interval_minutes?: number | null
  lookback_days?: number
  lookahead_days?: number
  provider_capabilities?: Record<string, unknown> | null
}

export interface ExternalCalendarBindingResponse {
  id: number
  account_id: number
  calendar_id: number
  remote_calendar_id: string
  remote_display_name?: string | null
  sync_enabled: boolean
  sync_interval_minutes?: number | null
  lookback_days: number
  lookahead_days: number
  provider_capabilities?: Record<string, unknown> | null
  sync_cursor?: string | null
  last_sync_at?: string | null
  next_scan_at?: string | null
  last_error?: string | null
  disabled_at?: string | null
  deleted_at?: string | null
  created_at: string
  updated_at: string
}

export interface ExternalCalendarBindingListResponse {
  items: ExternalCalendarBindingResponse[]
  total: number
}

export interface CalendarSyncTriggerResponse {
  binding_id: number
  queued: boolean
  status: "not_implemented" | string
  job_id?: number | null
  idempotency_key?: string | null
}

export interface CalendarSyncTriggerRequest {
  reason?: string
  window_start?: string | null
  window_end?: string | null
}

const CALENDAR_BASE = "/api/v1/calendar"
const SECRET_KEYS = new Set(["password", "token", "access_token", "refresh_token", "client_secret"])

const encodePathId = (id: string | number): string => {
  const normalized = String(id || "").trim()
  if (!normalized) {
    throw new Error("id is required")
  }
  return encodeURIComponent(normalized)
}

const buildCalendarViewQuery = (
  params: CalendarAgendaQuery | CalendarWeekQuery,
  required: string[]
): string => {
  for (const key of required) {
    if (!String((params as unknown as Record<string, unknown>)[key] || "").trim()) {
      throw new Error(`${required.join(" and ")} are required`)
    }
  }

  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) {
      continue
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        query.append(key, String(item))
      }
      continue
    }
    query.set(key, String(value))
  }
  return query.toString() ? `?${query.toString()}` : ""
}

const assertLocalItemMutation = (payload: CalendarItemUpdateRequest): void => {
  if (payload.source_owner === "provider" || payload.provider_owned === true) {
    throw new Error("Provider-owned calendar items are read-only")
  }
}

const assertDeletableItem = (context?: CalendarItemMutationContext): void => {
  if (!context) {
    throw new Error("calendar item mutation context is required")
  }
  const hasMutationContext =
    context.source_owner !== undefined ||
    context.provider_owned !== undefined ||
    context.read_only_reason !== undefined
  if (!hasMutationContext) {
    throw new Error("calendar item mutation context is required")
  }
  if (context.source_owner === "provider" || context.provider_owned === true) {
    throw new Error("Provider-owned calendar items are read-only")
  }
  if (context.source_owner === "linked_projection" || context.read_only_reason) {
    throw new Error("Read-only calendar items cannot be deleted")
  }
}

const resolveDeleteTargetId = (target: CalendarItemDeleteTarget): string | number => {
  assertDeletableItem(target)
  const id = target.calendar_item_id ?? target.id
  if (id === undefined || id === null || String(id).trim() === "") {
    throw new Error("calendar item id is required")
  }
  return id
}

const stripMutationHints = (payload: CalendarItemUpdateRequest): Omit<
  CalendarItemUpdateRequest,
  "source_owner" | "provider_owned"
> => {
  const { source_owner: _sourceOwner, provider_owned: _providerOwned, ...updates } = payload
  return updates
}

const withoutSecrets = <T>(payload: T): T => {
  if (Array.isArray(payload)) {
    return payload.map((item) => withoutSecrets(item)) as T
  }
  if (payload && typeof payload === "object") {
    return Object.fromEntries(
      Object.entries(payload as Record<string, unknown>)
        .filter(([key]) => !SECRET_KEYS.has(key))
        .map(([key, value]) => [key, withoutSecrets(value)])
    ) as T
  }
  return payload
}

export async function listCalendars(): Promise<CalendarListResponse> {
  return await bgRequest<CalendarListResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/calendars`),
    method: "GET"
  })
}

export async function createCalendar(payload: CalendarCreateRequest): Promise<CalendarResponse> {
  return await bgRequest<CalendarResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/calendars`),
    method: "POST",
    body: payload
  })
}

export async function createCalendarItem(
  payload: CalendarItemCreateRequest
): Promise<CalendarItemResponse> {
  return await bgRequest<CalendarItemResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items`),
    method: "POST",
    body: payload
  })
}

export async function updateCalendarItem(
  itemId: string | number,
  payload: CalendarItemUpdateRequest
): Promise<CalendarItemResponse> {
  assertLocalItemMutation(payload)
  return await bgRequest<CalendarItemResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}`),
    method: "PATCH",
    body: stripMutationHints(payload)
  })
}

export async function deleteCalendarItem(
  target: CalendarItemDeleteTarget
): Promise<CalendarItemDeleteResponse> {
  const itemId = resolveDeleteTargetId(target)
  return await bgRequest<CalendarItemDeleteResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}`),
    method: "DELETE"
  })
}

export async function getCalendarAgenda(query: CalendarAgendaQuery): Promise<CalendarViewResponse> {
  return await bgRequest<CalendarViewResponse>({
    path: appendPathQuery(
      toAllowedPath(`${CALENDAR_BASE}/views/agenda`),
      buildCalendarViewQuery(query, ["start_at", "end_at"])
    ),
    method: "GET"
  })
}

export async function getCalendarWeek(query: CalendarWeekQuery): Promise<CalendarViewResponse> {
  return await bgRequest<CalendarViewResponse>({
    path: appendPathQuery(
      toAllowedPath(`${CALENDAR_BASE}/views/week`),
      buildCalendarViewQuery(query, ["week_start"])
    ),
    method: "GET"
  })
}

export async function createCalendarAnnotation(
  itemId: string | number,
  payload: CalendarAnnotationCreateRequest
): Promise<CalendarAnnotationResponse> {
  return await bgRequest<CalendarAnnotationResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}/annotations`),
    method: "POST",
    body: payload
  })
}

export async function updateCalendarLocalTags(
  itemId: string | number,
  payload: CalendarLocalTagsUpdateRequest
): Promise<CalendarAnnotationResponse> {
  return await bgRequest<CalendarAnnotationResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}/local-tags`),
    method: "PUT",
    body: payload
  })
}

export async function createCalendarLink(
  itemId: string | number,
  payload: CalendarLinkCreateRequest
): Promise<CalendarLinkResponse> {
  return await bgRequest<CalendarLinkResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}/links`),
    method: "POST",
    body: payload
  })
}

export async function copyCalendarItemIntoTldw(
  itemId: string | number,
  payload: CalendarItemCopyRequest
): Promise<CalendarItemResponse> {
  return await bgRequest<CalendarItemResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/items/${encodePathId(itemId)}/copy`),
    method: "POST",
    body: withoutSecrets(payload)
  })
}

export async function listCalDavAccounts(): Promise<ExternalCalendarAccountListResponse> {
  return await bgRequest<ExternalCalendarAccountListResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts`),
    method: "GET"
  })
}

export async function createCalDavAccount(
  payload: CalDavAccountCreateRequest
): Promise<ExternalCalendarAccountResponse> {
  return await bgRequest<ExternalCalendarAccountResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts`),
    method: "POST",
    body: {
      provider: "caldav",
      ...payload
    }
  })
}

export async function verifyCalDavAccount(
  accountId: string | number,
  payload: CalDavAccountVerifyRequest
): Promise<CalDavAccountVerifyResponse> {
  return await bgRequest<CalDavAccountVerifyResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts/${encodePathId(accountId)}/verify`),
    method: "POST",
    body: payload
  })
}

export async function revokeCalDavAccount(
  accountId: string | number,
  _payload?: Record<string, unknown>
): Promise<CalDavAccountMutationResponse> {
  return await bgRequest<CalDavAccountMutationResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts/${encodePathId(accountId)}/revoke`),
    method: "POST"
  })
}

export async function deleteCalDavAccount(
  accountId: string | number,
  _payload?: Record<string, unknown>
): Promise<CalDavAccountMutationResponse> {
  return await bgRequest<CalDavAccountMutationResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts/${encodePathId(accountId)}`),
    method: "DELETE"
  })
}

export async function discoverExternalCalendars(
  accountId: string | number,
  _payload?: Record<string, unknown>
): Promise<ExternalCalendarDiscoveryResponse> {
  return await bgRequest<ExternalCalendarDiscoveryResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts/${encodePathId(accountId)}/discover`),
    method: "POST"
  })
}

export async function createExternalCalendarBinding(
  payload: ExternalCalendarBindingCreateRequest
): Promise<ExternalCalendarBindingResponse> {
  return await bgRequest<ExternalCalendarBindingResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/bindings`),
    method: "POST",
    body: withoutSecrets(payload)
  })
}

export async function listExternalCalendarBindings(
  accountId: string | number
): Promise<ExternalCalendarBindingListResponse> {
  return await bgRequest<ExternalCalendarBindingListResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/accounts/${encodePathId(accountId)}/bindings`),
    method: "GET"
  })
}

export async function triggerCalendarSync(
  bindingId: string | number,
  payload?: CalendarSyncTriggerRequest & Record<string, unknown>
): Promise<CalendarSyncTriggerResponse> {
  const body = payload ? withoutSecrets(payload) : undefined
  return await bgRequest<CalendarSyncTriggerResponse>({
    path: toAllowedPath(`${CALENDAR_BASE}/external/bindings/${encodePathId(bindingId)}/sync`),
    method: "POST",
    ...(body && Object.keys(body).length ? { body } : {})
  })
}
