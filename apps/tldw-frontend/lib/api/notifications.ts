import {
  buildAuthHeaders,
  getApiBaseUrl,
  apiClient,
  hasExplicitAuthHeaders
} from "@web/lib/api"
import { streamStructuredSSE } from "@web/lib/sse"
import {
  buildNotificationsQuery as buildNotificationsQueryShared,
  createNotificationStreamSubscription as createNotificationStreamSubscriptionShared
} from "@/services/notifications"
import type {
  NotificationCancelSnoozeResponse,
  NotificationItem,
  NotificationSeverity,
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

async function readNotificationsStream(
  signal: AbortSignal,
  after: number,
  onEvent: (event: NotificationStreamEvent) => void
): Promise<number> {
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
      }
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
}): Promise<NotificationsListResponse> {
  return apiClient.get<NotificationsListResponse>(
    `/notifications${buildNotificationsQueryShared({
      limit: params?.limit ?? 100,
      offset: params?.offset ?? 0,
      include_archived: params?.include_archived ?? false,
      only_snoozed: params?.only_snoozed
    })}`,
    { withCredentials: !hasExplicitAuthHeaders() }
  )
}

export async function getUnreadCount(): Promise<NotificationsUnreadCountResponse> {
  return apiClient.get<NotificationsUnreadCountResponse>("/notifications/unread-count", {
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function markNotificationsRead(ids: number[]): Promise<{ updated: number }> {
  return apiClient.post<{ updated: number }>("/notifications/mark-read", { ids }, {
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function dismissNotification(notificationId: number): Promise<{ dismissed: boolean }> {
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
  return apiClient.post<NotificationSnoozeResponse>(`/notifications/${notificationId}/snooze`, {
    minutes
  }, {
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function getNotificationPreferences(): Promise<NotificationPreferences> {
  const headers = buildAuthHeaders("GET")
  return apiClient.get<NotificationPreferences>("/notifications/preferences", {
    headers,
    withCredentials: !hasExplicitAuthHeaders()
  })
}

export async function updateNotificationPreferences(
  update: NotificationPreferencesUpdate
): Promise<NotificationPreferences> {
  const headers = buildAuthHeaders("PATCH")
  return apiClient.patch<NotificationPreferences>("/notifications/preferences", update, {
    headers,
    withCredentials: !hasExplicitAuthHeaders()
  })
}
