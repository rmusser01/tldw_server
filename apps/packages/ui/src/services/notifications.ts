import { bgRequest, bgStream } from "@/services/background-proxy"

export type NotificationSeverity = "info" | "warning" | "error"

export type NotificationItem = {
  id: number
  user_id?: string | number | null
  kind: string
  title: string
  message: string
  severity: NotificationSeverity | string
  link_type?: string | null
  link_id?: string | null
  link_url?: string | null
  source_task_id?: string | null
  source_task_run_id?: string | null
  source_job_id?: string | number | null
  source_domain?: string | null
  source_job_type?: string | null
  created_at: string
  read_at?: string | null
  dismissed_at?: string | null
  snooze_until?: string | null
}

export type NotificationsListResponse = {
  items: NotificationItem[]
  total: number
}

export type NotificationsUnreadCountResponse = {
  unread_count: number
}

export type NotificationSnoozeResponse = {
  task_id: string
  run_at: string
}

export type NotificationCancelSnoozeResponse = {
  cancelled: boolean
  deleted_tasks: number
}

export type NotificationPreferences = {
  user_id: string
  reminder_enabled: boolean
  job_completed_enabled: boolean
  job_failed_enabled: boolean
  updated_at: string
}

export type NotificationPreferencesUpdate = {
  reminder_enabled?: boolean
  job_completed_enabled?: boolean
  job_failed_enabled?: boolean
}

export type NotificationStreamEvent = {
  event: string
  id?: number
  payload?: unknown
}

export type SubscribeNotificationsOptions = {
  after?: number
  reconnectDelayMs?: number
  onEvent: (event: NotificationStreamEvent) => void
  onError?: (error: unknown) => void
}

const DEFAULT_RECONNECT_DELAY_MS = 1200

export type NotificationStreamReader = (
  signal: AbortSignal,
  cursor: number,
  onEvent: (event: NotificationStreamEvent) => void
) => Promise<number>

export const buildNotificationsQuery = (params?: Record<string, unknown>): string => {
  if (!params) return ""
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value == null) return
    query.set(key, String(value))
  })
  const qs = query.toString()
  return qs ? `?${qs}` : ""
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

const toNumber = (value: unknown): number | undefined => {
  const next = typeof value === "number" ? value : Number(value)
  return Number.isFinite(next) ? next : undefined
}

const inferNotificationEventName = (payload: Record<string, unknown>): string => {
  if ("reason" in payload && "min_event_id" in payload && "latest_event_id" in payload) {
    return "reset_required"
  }
  if (
    "count" in payload &&
    ("from_event_id" in payload || "to_event_id" in payload)
  ) {
    return "notifications_coalesced"
  }
  if (
    "notification_id" in payload ||
    "event_id" in payload ||
    "kind" in payload ||
    "title" in payload ||
    "message" in payload
  ) {
    return "notification"
  }
  if (Object.keys(payload).length === 0) {
    return "heartbeat"
  }
  return "message"
}

export const parseNotificationStreamEvent = (line: string): NotificationStreamEvent | null => {
  // `bgStream` strips SSE framing and yields one payload line at a time.
  // This helper therefore expects single-line payloads, not raw multi-line SSE.
  const trimmed = line.trim()
  if (!trimmed || trimmed.startsWith(":")) return null

  let payload: unknown = trimmed

  if (typeof payload === "string" && payload.length > 0) {
    try {
      payload = JSON.parse(payload)
    } catch {
      // Keep the raw string payload if it is not JSON.
    }
  }

  if (isRecord(payload)) {
    const eventName =
      typeof payload.event === "string" && payload.event.trim().length > 0
        ? payload.event.trim()
        : inferNotificationEventName(payload)
    const eventPayload =
      Object.prototype.hasOwnProperty.call(payload, "payload")
        ? payload.payload
        : payload
    const eventId =
      toNumber(payload.id) ??
      toNumber(payload.event_id) ??
      toNumber(payload.notification_id) ??
      toNumber(payload.to_event_id) ??
      toNumber(payload.latest_event_id)
    return {
      event: eventName,
      ...(typeof eventId === "number" ? { id: eventId } : {}),
      payload: eventPayload
    }
  }

  return {
    event: "message",
    payload
  }
}

const readNotificationsStream = async (
  signal: AbortSignal,
  after: number,
  onEvent: (event: NotificationStreamEvent) => void
): Promise<number> => {
  const headers: Record<string, string> = {}
  if (after > 0) {
    headers["Last-Event-ID"] = String(after)
  }

  let cursor = after
  try {
    for await (const line of bgStream({
      path: `/api/v1/notifications/stream${buildNotificationsQuery({ after })}` as any,
      method: "GET",
      headers,
      abortSignal: signal
    })) {
      const event = parseNotificationStreamEvent(line)
      if (!event) continue
      onEvent(event)
      if (typeof event.id === "number" && Number.isFinite(event.id) && event.id > cursor) {
        cursor = event.id
      }
    }
    return cursor
  } catch (error) {
    if (error && typeof error === "object") {
      ;(error as { cursor?: number }).cursor = cursor
    }
    throw error
  }
}

export function createNotificationStreamSubscription(options: SubscribeNotificationsOptions & {
  readStream: NotificationStreamReader
}): () => void {
  const controller = new AbortController()
  let cursor = Math.max(0, options.after ?? 0)
  const reconnectDelayMs = Math.max(250, options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS)

  const delay = async (ms: number): Promise<void> => {
    await new Promise<void>((resolve) => {
      let timeoutId: ReturnType<typeof setTimeout> | null = null
      const onAbort = () => {
        if (timeoutId !== null) {
          clearTimeout(timeoutId)
          timeoutId = null
        }
        controller.signal.removeEventListener("abort", onAbort)
        resolve()
      }
      timeoutId = setTimeout(() => {
        controller.signal.removeEventListener("abort", onAbort)
        resolve()
      }, ms)
      if (controller.signal.aborted) {
        onAbort()
        return
      }
      controller.signal.addEventListener("abort", onAbort, { once: true })
    })
  }

  const run = async (): Promise<void> => {
    while (!controller.signal.aborted) {
      try {
        cursor = await options.readStream(controller.signal, cursor, options.onEvent)
        if (controller.signal.aborted) break
        await delay(reconnectDelayMs)
      } catch (error) {
        if (controller.signal.aborted) break
        const nextCursor =
          error && typeof error === "object" ? (error as { cursor?: unknown }).cursor : undefined
        if (typeof nextCursor === "number" && Number.isFinite(nextCursor) && nextCursor > cursor) {
          cursor = nextCursor
        }
        options.onError?.(error)
        await delay(reconnectDelayMs)
      }
    }
  }

  void run()
  return () => controller.abort()
}

export async function listNotifications(params?: {
  limit?: number
  offset?: number
  include_archived?: boolean
  only_snoozed?: boolean
}): Promise<NotificationsListResponse> {
  return bgRequest<NotificationsListResponse>({
    path: `/api/v1/notifications${buildNotificationsQuery(params || {})}` as any,
    method: "GET"
  })
}

export async function getUnreadCount(): Promise<NotificationsUnreadCountResponse> {
  return bgRequest<NotificationsUnreadCountResponse>({
    path: "/api/v1/notifications/unread-count" as any,
    method: "GET"
  })
}

export async function markNotificationsRead(ids: number[]): Promise<{ updated: number }> {
  return bgRequest<{ updated: number }>({
    path: "/api/v1/notifications/mark-read" as any,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { ids }
  })
}

export async function dismissNotification(notificationId: number): Promise<{ dismissed: boolean }> {
  return bgRequest<{ dismissed: boolean }>({
    path: `/api/v1/notifications/${notificationId}/dismiss` as any,
    method: "POST",
    headers: { "Content-Type": "application/json" }
  })
}

export async function cancelNotificationSnooze(
  notificationId: number
): Promise<NotificationCancelSnoozeResponse> {
  return bgRequest<NotificationCancelSnoozeResponse>({
    path: `/api/v1/notifications/${notificationId}/snooze` as any,
    method: "DELETE"
  })
}

export async function snoozeNotification(
  notificationId: number,
  minutes: number
): Promise<NotificationSnoozeResponse> {
  return bgRequest<NotificationSnoozeResponse>({
    path: `/api/v1/notifications/${notificationId}/snooze` as any,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { minutes }
  })
}

export async function getNotificationPreferences(): Promise<NotificationPreferences> {
  return bgRequest<NotificationPreferences>({
    path: "/api/v1/notifications/preferences" as any,
    method: "GET"
  })
}

export async function updateNotificationPreferences(
  update: NotificationPreferencesUpdate
): Promise<NotificationPreferences> {
  return bgRequest<NotificationPreferences>({
    path: "/api/v1/notifications/preferences" as any,
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: update
  })
}

export function subscribeNotificationsStream(
  options: SubscribeNotificationsOptions
): () => void {
  return createNotificationStreamSubscription({
    ...options,
    readStream: readNotificationsStream
  })
}
