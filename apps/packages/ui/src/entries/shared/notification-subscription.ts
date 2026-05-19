/**
 * Background notification subscription for the browser extension.
 *
 * Subscribes to the server's SSE notification stream and:
 * 1. Updates a storage-backed unread count for the UI bell icon
 * 2. Shows Chrome system notifications for new items
 *
 * Designed to run in the background service worker context.
 */

import { subscribeNotificationsStream, getUnreadCount } from "@/services/notifications"
import type { NotificationStreamEvent } from "@/services/notifications"
import { notify } from "@/services/background-helpers"
import { createSafeStorage } from "@/utils/safe-storage"
import { toUnreadCount } from "@/utils/notifications"

const UNREAD_COUNT_KEY = "tldw:notifications:unreadCount"
const SUBSCRIPTION_ACTIVE_KEY = "tldw:notifications:subscriptionActive"

let unsubscribe: (() => void) | null = null
let startInFlight: Promise<void> | null = null

/**
 * Start listening for notifications from the server.
 * Safe to call multiple times — only one subscription is active at a time.
 */
export async function startNotificationSubscription(): Promise<void> {
  if (unsubscribe) return // Already subscribed
  if (startInFlight) return startInFlight

  startInFlight = (async () => {
    const storage = createSafeStorage({ area: "local" })
    let unreadCountWrite = Promise.resolve()

    // Fetch initial unread count
    try {
      const { unread_count } = await getUnreadCount()
      await storage.set(UNREAD_COUNT_KEY, unread_count)
    } catch (error) {
      // Server may not be reachable yet
      console.debug("[background] Failed to fetch initial unread count:", error)
    }

    // Subscribe to SSE stream
    try {
      unsubscribe = subscribeNotificationsStream({
        onEvent: async (event: NotificationStreamEvent) => {
          if (event.event === "notification") {
            // Show Chrome system notification
            const payload = event.payload as {
              title?: string
              message?: string
              severity?: string
            } | null
            if (payload?.title) {
              notify(payload.title, payload.message || "")
            }

            unreadCountWrite = unreadCountWrite
              .catch(() => undefined)
              .then(async () => {
                const current = toUnreadCount(await storage.get<number>(UNREAD_COUNT_KEY))
                await storage.set(UNREAD_COUNT_KEY, current + 1)
              })
              .catch((error) => {
                console.debug(
                  "[background] Failed to update unread count from notification event:",
                  error
                )
              })
            await unreadCountWrite
          }

          if (event.event === "notifications_coalesced") {
            // Batch arrived — refresh full count
            try {
              const { unread_count } = await getUnreadCount()
              await storage.set(UNREAD_COUNT_KEY, unread_count)
            } catch (error) {
              console.debug(
                "[background] Failed to refresh unread count after coalesced notifications:",
                error
              )
            }
          }
        },
        onError: () => {
          // Stream will auto-reconnect (handled by subscribeNotificationsStream)
        },
      })

      await storage.set(SUBSCRIPTION_ACTIVE_KEY, true)
    } catch (error) {
      await storage.set(SUBSCRIPTION_ACTIVE_KEY, false)
      // Server not available — will retry on next init
      console.debug("[background] Failed to start notification subscription:", error)
    }
  })()

  try {
    await startInFlight
  } finally {
    startInFlight = null
  }
}

/**
 * Stop the notification subscription.
 */
export function stopNotificationSubscription(): void {
  if (unsubscribe) {
    unsubscribe()
    unsubscribe = null
  }

  const storage = createSafeStorage({ area: "local" })
  void storage.set(SUBSCRIPTION_ACTIVE_KEY, false)
}

/**
 * Get the current unread count from storage (for UI components).
 */
export async function getStoredUnreadCount(): Promise<number> {
  const storage = createSafeStorage({ area: "local" })
  return toUnreadCount(await storage.get<number>(UNREAD_COUNT_KEY))
}

/**
 * Reset the unread count (e.g., when user opens notifications page).
 */
export async function resetStoredUnreadCount(): Promise<void> {
  const storage = createSafeStorage({ area: "local" })
  await storage.set(UNREAD_COUNT_KEY, 0)
}
