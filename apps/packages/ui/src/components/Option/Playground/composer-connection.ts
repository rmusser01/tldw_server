import type { NotificationInstance } from "antd/es/notification/interface"

import { deriveConnectionUxState, type ConnectionState } from "@/types/connection"

type Translate = (key: string, defaultValue: string) => string

/**
 * True while the app is still running its first server check and nothing has
 * failed yet (CC-03). The connection store starts in SEARCHING before any
 * check has finished, so treating "not connected" as "offline" told returning
 * users they were offline for the first moments of every visit. A finished
 * check records `lastCheckedAt` (failures also bump `consecutiveFailures`), so
 * a retry after a failure is not counted as the first check.
 */
export const isAwaitingFirstConnectionCheck = (
  state: Partial<ConnectionState> | null | undefined
): boolean => {
  if (!state || state.isConnected) return false
  if (state.lastCheckedAt != null) return false
  if ((state.consecutiveFailures ?? 0) > 0) return false
  return deriveConnectionUxState(state as ConnectionState) === "testing"
}

const SEND_NEEDS_CONNECTION_NOTICE_KEY = "playground-send-needs-connection"

/**
 * Tell the user why a send did nothing while the server is not connected.
 * Drafts are never queued for later (owner decision D7), so the message says
 * the draft was not sent; one notice key keeps repeated presses from stacking.
 */
export const notifySendNeedsConnection = (
  notificationApi: Pick<NotificationInstance, "info">,
  t: Translate,
  connectionPending: boolean
) => {
  notificationApi.info({
    key: SEND_NEEDS_CONNECTION_NOTICE_KEY,
    message: connectionPending
      ? t(
          "playground:composer.connectionPendingSend",
          "Still connecting to your tldw server. Your draft has not been sent."
        )
      : t(
          "playground:composer.connectionRequired",
          "Connect to your tldw server before sending. Your draft has not been queued."
        )
  })
}
