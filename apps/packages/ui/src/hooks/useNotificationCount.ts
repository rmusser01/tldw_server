/** Read the active scoped notification unread count from extension storage. */

import { useEffect, useMemo, useState } from "react"
import { useStorage } from "@plasmohq/storage/hook"

import { toUnreadCount } from "@/utils/notifications"
import { createSafeStorage } from "@/utils/safe-storage"
import { notificationRecordKeyForConfig } from "@/services/notification-runtime-scope"

const ACTIVE_SCOPE_KEY = "tldw:notifications:activeScope"
const CONFIG_KEY = "tldwConfig"
const storage = createSafeStorage({ area: "local" })

export function useNotificationCount(): number {
  const [config] = useStorage<unknown>({ key: CONFIG_KEY, instance: storage })
  const expectedScope = notificationRecordKeyForConfig(config)
  const [activeScope] = useStorage<unknown>({ key: ACTIVE_SCOPE_KEY, instance: storage })
  const scope = expectedScope && activeScope === expectedScope ? expectedScope : null
  // A fresh subscription identity also protects an A -> B -> A switch from
  // reusing A's earlier snapshot while its new read is still pending.
  const subscription = useMemo(() => ({ scope }), [scope])
  const [snapshot, setSnapshot] = useState<{
    subscription: typeof subscription
    count: number
  }>()

  useEffect(() => {
    const key = subscription.scope
    if (!key) return
    let cancelled = false
    let watchUpdated = false
    const apply = (value: unknown) => {
      if (cancelled) return
      const unreadCount = value && typeof value === "object" && "unreadCount" in value
        ? value.unreadCount
        : undefined
      setSnapshot({ subscription, count: toUnreadCount(unreadCount) })
    }
    const callbacks = {
      [key]: (change: { newValue?: unknown }) => {
        watchUpdated = true
        apply(change.newValue)
      }
    }
    storage.watch(callbacks)
    void storage.get<unknown>(key).then(
      (value) => { if (!watchUpdated) apply(value) },
      () => { if (!watchUpdated) apply(undefined) }
    )
    return () => {
      cancelled = true
      storage.unwatch(callbacks)
    }
  }, [subscription])

  return snapshot?.subscription === subscription ? snapshot.count : 0
}
