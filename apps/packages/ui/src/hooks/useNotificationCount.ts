/**
 * Hook to read the notification unread count from extension storage.
 * The count is maintained by the background notification subscription.
 *
 * Returns 0 when running outside the extension context.
 */

import { useStorage } from "@plasmohq/storage/hook"

import { toUnreadCount } from "@/utils/notifications"
import { safeStorageSerde } from "@/utils/safe-storage"

const UNREAD_COUNT_KEY = "tldw:notifications:unreadCount"

export function useNotificationCount(): number {
  const [count] = useStorage<number>(
    {
      key: UNREAD_COUNT_KEY,
      area: "local",
      serde: safeStorageSerde
    },
    toUnreadCount
  )

  return toUnreadCount(count)
}
