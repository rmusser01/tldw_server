/** Read the active scoped notification unread count from extension storage. */

import { useRef } from "react"
import { useStorage } from "@plasmohq/storage/hook"

import { toUnreadCount } from "@/utils/notifications"
import { safeStorageSerde } from "@/utils/safe-storage"
import { notificationRecordKeyForConfig } from "@/services/notification-runtime-scope"

const ACTIVE_SCOPE_KEY = "tldw:notifications:activeScope"
const CONFIG_KEY = "tldwConfig"

type NotificationRecord = {
  state: "connecting" | "active" | "degraded" | "auth-required" | "unavailable"
  unreadCount: number
  updatedAt: number
}

const storageOptions = (key: string) => ({
  key,
  area: "local" as const,
  serde: safeStorageSerde
})

export function useNotificationCount(): number {
  const [config] = useStorage<unknown>(storageOptions(CONFIG_KEY), (value) => value)
  const expectedScope = notificationRecordKeyForConfig(config)
  const [activeScope] = useStorage<string | null>(storageOptions(ACTIVE_SCOPE_KEY), (value) =>
    typeof value === "string" && value ? value : null
  )
  const [record] = useStorage<NotificationRecord | undefined>(
    storageOptions(activeScope || ACTIVE_SCOPE_KEY),
    (value) => (value && typeof value === "object" ? (value as NotificationRecord) : undefined)
  )
  const previousScope = useRef(activeScope)
  const scopeChanged = previousScope.current !== activeScope
  previousScope.current = activeScope

  if (!expectedScope || activeScope !== expectedScope || scopeChanged) return 0
  return toUnreadCount(record?.unreadCount)
}
