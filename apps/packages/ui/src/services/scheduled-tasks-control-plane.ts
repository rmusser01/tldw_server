/**
 * Scheduled tasks control-plane API client.
 */

import { bgRequest } from "@/services/background-proxy"
import { toAllowedPath } from "@/services/tldw/path-utils"

export type ScheduledTaskPrimitive =
  | "reminder_task"
  | "watchlist_job"
  | "automation_definition"
export type ScheduledTaskEditMode = "native" | "external"
export type ReminderScheduleKind = "one_time" | "recurring"
export type ScheduledTaskAutomationFamily = "recurring_question" | "agent_task"
export type ScheduledTaskAutomationActionStatus =
  | "available"
  | "unavailable"
  | "planned"
  | "disabled"
export type ScheduledTaskAutomationFamilyAvailability =
  | "available"
  | "planned"
  | "unavailable"
  | "degraded"
export type ScheduledTaskPreviewMode = "create" | "update"
export type ScheduledTaskPreviewStatus = "valid" | "invalid" | "expired" | "consumed"
export type ScheduledTaskDefinitionLifecycle =
  | "configured"
  | "paused"
  | "archived"
  | "disabled"
export type ScheduledTaskDefinitionCreateLifecycle = "configured" | "paused"
export type ScheduledTaskDefinitionHealth =
  | "ready"
  | "execution_unavailable"
  | "capability_unavailable"
  | "needs_attention"
  | "permission_required"
export type ScheduledTaskDefinitionDisabledLockKind =
  | "none"
  | "admin"
  | "security"
  | "system"

export interface ScheduledTask {
  id: string
  primitive: ScheduledTaskPrimitive
  title: string
  description?: string | null
  status: string
  enabled: boolean
  schedule_summary?: string | null
  timezone?: string | null
  next_run_at?: string | null
  last_run_at?: string | null
  edit_mode: ScheduledTaskEditMode
  manage_url?: string | null
  source_ref: Record<string, unknown>
}

export interface ScheduledTaskListResponse {
  items: ScheduledTask[]
  total: number
  partial: boolean
  errors: string[]
}

export interface ScheduledTaskDeleteResponse {
  deleted: boolean
}

export interface ScheduledTaskActionCapability {
  status: ScheduledTaskAutomationActionStatus
  reason?: string | null
  required_permissions: string[]
}

export interface ScheduledTaskAutomationCapability {
  family: ScheduledTaskAutomationFamily
  family_availability: ScheduledTaskAutomationFamilyAvailability
  actions: Record<string, ScheduledTaskActionCapability>
  missing_dependencies: string[]
  related_capabilities: Record<string, unknown>
  reason?: string | null
  schema_version: string
}

export interface ScheduledTaskAutomationCapabilitiesResponse {
  items: ScheduledTaskAutomationCapability[]
}

export interface ScheduledTaskPreviewCreateRequest {
  mode?: ScheduledTaskPreviewMode
  family: ScheduledTaskAutomationFamily
  definition_id?: string | null
  definition_version?: number | null
  name?: string | null
  description?: string | null
  config?: Record<string, unknown>
  input?: Record<string, unknown>
  schedule?: Record<string, unknown>
  visibility_policy?: Record<string, unknown>
  notification_policy?: Record<string, unknown>
  approval_policy?: Record<string, unknown>
}

export interface ScheduledTaskPreviewResponse {
  id: string
  owner_id?: string | null
  mode: ScheduledTaskPreviewMode
  family: ScheduledTaskAutomationFamily
  definition_id?: string | null
  definition_version?: number | null
  status: ScheduledTaskPreviewStatus
  payload_hash?: string | null
  normalized_config: Record<string, unknown>
  validation_errors: Record<string, unknown>[]
  warnings: Record<string, unknown>[]
  risk_class?: string | null
  visibility_policy: Record<string, unknown>
  schedule_preview: Record<string, unknown>
  redaction_policy: Record<string, unknown>
  expires_at?: string | null
  created_by?: string | null
  created_at?: string | null
  consumed_at?: string | null
  created_definition_id?: string | null
}

export interface ScheduledTaskPreviewListResponse {
  items: ScheduledTaskPreviewResponse[]
  total: number
  limit: number
  offset: number
  has_more: boolean
  next_offset?: number | null
}

export interface ScheduledTaskDefinitionCreateRequest {
  preview_id: string
  initial_lifecycle?: ScheduledTaskDefinitionCreateLifecycle
}

export interface ScheduledTaskDefinitionUpdateRequest {
  preview_id: string
}

export interface ScheduledTaskDefinitionResponse {
  id: string
  owner_id?: string | null
  version: number
  family: ScheduledTaskAutomationFamily
  name: string
  description?: string | null
  lifecycle: ScheduledTaskDefinitionLifecycle
  health: ScheduledTaskDefinitionHealth
  disabled_lock_kind?: ScheduledTaskDefinitionDisabledLockKind | null
  disabled_reason?: string | null
  schedule: Record<string, unknown>
  input: Record<string, unknown>
  config: Record<string, unknown>
  visibility_policy: Record<string, unknown>
  notification_policy: Record<string, unknown>
  approval_policy: Record<string, unknown>
  preview_id?: string | null
  created_by?: string | null
  updated_by?: string | null
  created_at?: string | null
  updated_at?: string | null
  archived_at?: string | null
}

export interface ScheduledTaskDefinitionListResponse {
  items: ScheduledTaskDefinitionResponse[]
  total: number
  limit: number
  offset: number
  has_more: boolean
  next_offset?: number | null
}

export interface ScheduledTaskDuplicateRequest {
  name?: string | null
  description?: string | null
}

export interface ScheduledTaskAuditEventResponse {
  id: string
  definition_id: string
  event_type: string
  actor?: string | null
  summary?: string | null
  before?: Record<string, unknown> | null
  after?: Record<string, unknown> | null
  created_at?: string | null
  request_id?: string | null
  idempotency_key?: string | null
}

export interface ScheduledTaskAuditListResponse {
  items: ScheduledTaskAuditEventResponse[]
  total: number
  limit: number
  offset: number
  has_more: boolean
  next_offset?: number | null
}

export interface ScheduledTaskErrorEnvelope {
  code: string
  message: string
  details: Record<string, unknown>
  field_errors: Record<string, unknown>[]
  retryable: boolean
  correlation_id?: string | null
}

export interface CreateScheduledTaskReminderPayload {
  title: string
  body?: string | null
  schedule_kind: ReminderScheduleKind
  run_at?: string | null
  cron?: string | null
  timezone?: string | null
  link_type?: string | null
  link_id?: string | null
  link_url?: string | null
  enabled?: boolean
}

export interface UpdateScheduledTaskReminderPayload {
  title?: string
  body?: string | null
  schedule_kind?: ReminderScheduleKind
  run_at?: string | null
  cron?: string | null
  timezone?: string | null
  link_type?: string | null
  link_id?: string | null
  link_url?: string | null
  enabled?: boolean
}

export interface ScheduledTaskPreviewListParams {
  limit?: number
  offset?: number
  family?: ScheduledTaskAutomationFamily | string | null
  mode?: ScheduledTaskPreviewMode | string | null
  status?: ScheduledTaskPreviewStatus | string | null
  definition_id?: string | null
  expired?: boolean | null
}

export interface ScheduledTaskDefinitionListParams {
  limit?: number
  offset?: number
  family?: ScheduledTaskAutomationFamily | string | null
  lifecycle?: ScheduledTaskDefinitionLifecycle | string | null
  health?: ScheduledTaskDefinitionHealth | string | null
  visibility_policy?: string | null
  q?: string | null
  created_from?: string | null
  created_to?: string | null
}

export interface ScheduledTaskAuditListParams {
  limit?: number
  offset?: number
  event_type?: string | null
  actor?: string | null
  created_from?: string | null
  created_to?: string | null
  idempotency_key?: string | null
  request_id?: string | null
}

export interface ScheduledTaskMutationOptions {
  idempotencyKey?: string | null
}

const withIdempotency = (
  key?: string | null
): Record<string, string> | undefined => {
  const trimmed = typeof key === "string" ? key.trim() : ""
  return trimmed ? { "Idempotency-Key": trimmed } : undefined
}

const buildQuery = (params?: Record<string, unknown>): string => {
  if (!params) return ""

  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null) return
    query.set(key, String(value))
  })

  const queryString = query.toString()
  return queryString ? `?${queryString}` : ""
}

const normalizeReminderTaskMutationId = (taskId: string): string => {
  const normalized = String(taskId || "").trim()
  if (!normalized) {
    throw new Error("taskId is required")
  }
  if (normalized.startsWith("reminder_task:")) {
    return normalized.slice("reminder_task:".length)
  }
  if (normalized.includes(":")) {
    throw new Error("Reminder mutations require a reminder_task id")
  }
  return normalized
}

const normalizeAutomationDefinitionMutationId = (definitionId: string): string => {
  const normalized = String(definitionId || "").trim()
  if (!normalized) {
    throw new Error("definitionId is required")
  }
  if (normalized.startsWith("automation_definition:")) {
    return normalized.slice("automation_definition:".length)
  }
  if (normalized.includes(":")) {
    throw new Error("Definition mutations require an automation_definition id")
  }
  return normalized
}

const assertReminderUpdatePayload = (payload: Record<string, unknown>): void => {
  if (payload.title === null) {
    throw new Error("title cannot be null")
  }
  if (payload.schedule_kind === null) {
    throw new Error("schedule_kind cannot be null")
  }
  if (payload.enabled === null) {
    throw new Error("enabled cannot be null")
  }
}

export async function listScheduledTasks(): Promise<ScheduledTaskListResponse> {
  return await bgRequest<ScheduledTaskListResponse>({
    path: "/api/v1/scheduled-tasks",
    method: "GET"
  })
}

export async function getScheduledTask(taskId: string): Promise<ScheduledTask> {
  return await bgRequest<ScheduledTask>({
    path: toAllowedPath(`/api/v1/scheduled-tasks/${encodeURIComponent(taskId)}`),
    method: "GET"
  })
}

export async function createScheduledTaskReminder(
  payload: CreateScheduledTaskReminderPayload
): Promise<ScheduledTask> {
  return await bgRequest<ScheduledTask>({
    path: "/api/v1/scheduled-tasks/reminders",
    method: "POST",
    body: payload
  })
}

export async function updateScheduledTaskReminder(
  taskId: string,
  payload: UpdateScheduledTaskReminderPayload
): Promise<ScheduledTask> {
  assertReminderUpdatePayload(payload as Record<string, unknown>)
  return await bgRequest<ScheduledTask>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/reminders/${encodeURIComponent(normalizeReminderTaskMutationId(taskId))}`
    ),
    method: "PATCH",
    body: payload
  })
}

export async function deleteScheduledTaskReminder(taskId: string): Promise<ScheduledTaskDeleteResponse> {
  return await bgRequest<ScheduledTaskDeleteResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/reminders/${encodeURIComponent(normalizeReminderTaskMutationId(taskId))}`
    ),
    method: "DELETE"
  })
}

export async function getScheduledTaskCapabilities(): Promise<ScheduledTaskAutomationCapabilitiesResponse> {
  return await bgRequest<ScheduledTaskAutomationCapabilitiesResponse>({
    path: toAllowedPath("/api/v1/scheduled-tasks/capabilities"),
    method: "GET"
  })
}

export async function createScheduledTaskPreview(
  payload: ScheduledTaskPreviewCreateRequest,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskPreviewResponse> {
  return await bgRequest<ScheduledTaskPreviewResponse>({
    path: toAllowedPath("/api/v1/scheduled-tasks/previews"),
    method: "POST",
    body: payload,
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function listScheduledTaskPreviews(
  params?: ScheduledTaskPreviewListParams
): Promise<ScheduledTaskPreviewListResponse> {
  return await bgRequest<ScheduledTaskPreviewListResponse>({
    path: toAllowedPath(`/api/v1/scheduled-tasks/previews${buildQuery(params)}`),
    method: "GET"
  })
}

export async function getScheduledTaskPreview(
  previewId: string
): Promise<ScheduledTaskPreviewResponse> {
  return await bgRequest<ScheduledTaskPreviewResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/previews/${encodeURIComponent(previewId)}`
    ),
    method: "GET"
  })
}

export async function createScheduledTaskDefinition(
  payload: ScheduledTaskDefinitionCreateRequest,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath("/api/v1/scheduled-tasks/definitions"),
    method: "POST",
    body: payload,
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function listScheduledTaskDefinitions(
  params?: ScheduledTaskDefinitionListParams
): Promise<ScheduledTaskDefinitionListResponse> {
  return await bgRequest<ScheduledTaskDefinitionListResponse>({
    path: toAllowedPath(`/api/v1/scheduled-tasks/definitions${buildQuery(params)}`),
    method: "GET"
  })
}

export async function getScheduledTaskDefinition(
  definitionId: string
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(normalizedDefinitionId)}`
    ),
    method: "GET"
  })
}

export async function updateScheduledTaskDefinition(
  definitionId: string,
  payload: ScheduledTaskDefinitionUpdateRequest,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(normalizedDefinitionId)}`
    ),
    method: "PATCH",
    body: payload,
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function pauseScheduledTaskDefinition(
  definitionId: string,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(
        normalizedDefinitionId
      )}/pause`
    ),
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function resumeScheduledTaskDefinition(
  definitionId: string,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(
        normalizedDefinitionId
      )}/resume`
    ),
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function archiveScheduledTaskDefinition(
  definitionId: string,
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(
        normalizedDefinitionId
      )}/archive`
    ),
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function duplicateScheduledTaskDefinition(
  definitionId: string,
  payload: ScheduledTaskDuplicateRequest = {},
  options?: ScheduledTaskMutationOptions
): Promise<ScheduledTaskDefinitionResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskDefinitionResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(
        normalizedDefinitionId
      )}/duplicate`
    ),
    method: "POST",
    body: payload,
    headers: withIdempotency(options?.idempotencyKey)
  })
}

export async function listScheduledTaskDefinitionAudit(
  definitionId: string,
  params?: ScheduledTaskAuditListParams
): Promise<ScheduledTaskAuditListResponse> {
  const normalizedDefinitionId = normalizeAutomationDefinitionMutationId(definitionId)
  return await bgRequest<ScheduledTaskAuditListResponse>({
    path: toAllowedPath(
      `/api/v1/scheduled-tasks/definitions/${encodeURIComponent(
        normalizedDefinitionId
      )}/audit${buildQuery(params)}`
    ),
    method: "GET"
  })
}
