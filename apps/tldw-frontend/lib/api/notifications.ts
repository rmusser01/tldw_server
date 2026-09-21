import {
  buildAuthHeaders,
  getApiBaseUrl,
  apiClient,
  ApiError,
  hasExplicitAuthHeaders
} from "@web/lib/api"
import { getApiBearer, getEffectiveStoredTldwConfig } from "@web/lib/authStorage"
import { streamStructuredSSE } from "@web/lib/sse"
import {
  buildNotificationsQuery as buildNotificationsQueryShared,
  createNotificationStreamSubscription as createNotificationStreamSubscriptionShared
} from "@/services/notifications"
import type {
  NotificationCancelSnoozeResponse,
  NotificationSnoozeResponse,
  NotificationStreamEvent,
  NotificationsListResponse,
  NotificationsUnreadCountResponse,
  SubscribeNotificationsOptions
} from "@/services/notifications"

export type {
  NotificationCancelSnoozeResponse,
  NotificationItem,
  NotificationSeverity,
  NotificationSnoozeResponse,
  NotificationStreamEvent,
  NotificationsListResponse,
  NotificationsUnreadCountResponse,
  SubscribeNotificationsOptions
} from "@/services/notifications"

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

const shouldUseCookieCredentials = (headers: Record<string, string>): boolean => {
  return !headers.Authorization && !headers["X-API-KEY"]
}

/** The WebUI endpoint is fixed; a different configured target must never supply its credentials. */
function assertNotificationTarget(): void {
  const config = getEffectiveStoredTldwConfig()
  if (config?.authMode === "multi-user" && config.authSource !== "cookie-session" && !getApiBearer()) {
    throw new ApiError("Sign in to access notifications.", { status: 401 })
  }
  if (!config?.serverUrl || config.authSource === "cookie-session") return
  try {
    const target = new URL(config.serverUrl)
    const base = new URL(getApiBaseUrl(), window.location.origin)
    const serverPath = base.pathname.replace(/\/api\/[^/]+\/?$/, "").replace(/\/+$/, "")
    if (target.origin === base.origin &&
      target.pathname.replace(/\/+$/, "") === serverPath &&
      !target.search && !target.hash) return
  } catch {
    // An invalid or changed target cannot authorize a request to this endpoint.
  }
  throw new ApiError("Notification access is unavailable for the selected server.", { status: 412 })
}

async function readNotificationsStream(
  signal: AbortSignal,
  after: number,
  onEvent: (event: NotificationStreamEvent) => void,
  onOpen: () => void
): Promise<number> {
  assertNotificationTarget()
  const baseUrl = getApiBaseUrl().replace(/\/$/, "")
  const url = `${baseUrl}/notifications/stream${buildNotificationsQueryShared({ after })}`
  const headers = buildAuthHeaders("GET")
  if (after > 0) {
    headers["Last-Event-ID"] = String(after)
  }
  const useCookieCredentials = shouldUseCookieCredentials(headers)

  let cursor = after
  try {
    await streamStructuredSSE(
      url,
      {
        method: "GET",
        headers,
        credentials: useCookieCredentials ? "include" : "omit",
        signal
      },
      (event) => {
        onEvent(event)
        if (typeof event.id === "number" && Number.isFinite(event.id) && event.id > cursor) {
          cursor = event.id
        }
      },
      undefined,
      onOpen
    )
    return cursor
  } catch (error) {
    if (error && typeof error === "object") {
      ;(error as { cursor?: number }).cursor = cursor
    }
    throw error
  }
}

export function subscribeNotificationsStream(options: SubscribeNotificationsOptions): () => void {
  return createNotificationStreamSubscriptionShared({
    ...options,
    readStream: readNotificationsStream
  })
}

export async function listNotifications(params?: {
  limit?: number
  offset?: number
  include_archived?: boolean
  only_snoozed?: boolean
  signal?: AbortSignal
}): Promise<NotificationsListResponse> {
  assertNotificationTarget()
  return apiClient.get<NotificationsListResponse>(
    `/notifications${buildNotificationsQueryShared({
      limit: params?.limit ?? 100,
      offset: params?.offset ?? 0,
      include_archived: params?.include_archived ?? false,
      only_snoozed: params?.only_snoozed
    })}`,
    { signal: params?.signal, withCredentials: !hasExplicitAuthHeaders() }
  )
}

export async function getUnreadCount(options?: {
  signal?: AbortSignal
}): Promise<NotificationsUnreadCountResponse> {
  assertNotificationTarget()
  return apiClient.get<NotificationsUnreadCountResponse>("/notifications/unread-count", {
    signal: options?.signal,
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function markNotificationsRead(ids: number[]): Promise<{ updated: number }> {
  assertNotificationTarget()
  return apiClient.post<{ updated: number }>("/notifications/mark-read", { ids }, {
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function dismissNotification(notificationId: number): Promise<{ dismissed: boolean }> {
  assertNotificationTarget()
  return apiClient.post<{ dismissed: boolean }>(
    `/notifications/${notificationId}/dismiss`,
    undefined,
    {
      withCredentials: !hasExplicitAuthHeaders()
    }
  )
}

export async function cancelNotificationSnooze(
  notificationId: number
): Promise<NotificationCancelSnoozeResponse> {
  assertNotificationTarget()
  return apiClient.delete<NotificationCancelSnoozeResponse>(
    `/notifications/${notificationId}/snooze`,
    {
      withCredentials: !hasExplicitAuthHeaders()
    }
  )
}

export async function snoozeNotification(
  notificationId: number,
  minutes: number
): Promise<NotificationSnoozeResponse> {
  assertNotificationTarget()
  return apiClient.post<NotificationSnoozeResponse>(`/notifications/${notificationId}/snooze`, {
    minutes
  }, {
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function getNotificationPreferences(): Promise<NotificationPreferences> {
  assertNotificationTarget()
  const headers = buildAuthHeaders("GET")
  return apiClient.get<NotificationPreferences>("/notifications/preferences", {
    headers,
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function updateNotificationPreferences(
  update: NotificationPreferencesUpdate
): Promise<NotificationPreferences> {
  assertNotificationTarget()
  const headers = buildAuthHeaders("PATCH")
  return apiClient.patch<NotificationPreferences>("/notifications/preferences", update, {
    headers,
    withCredentials: !hasExplicitAuthHeaders()
  })
}
