import { createSafeStorage } from "@/utils/safe-storage"
import React from "react"

const casualPreferenceKey = "playgroundComposerCasualAdvancedControlsOpen"

/** Existing presentation preference, with session-only recovery on storage failure. */
export function useCasualComposerDisclosure(
  preferenceKey: "playgroundComposerCasualAdvancedControlsOpen" | "playgroundComposerAdvancedControlsOpen" = casualPreferenceKey
) {
  const storage = React.useMemo(() => createSafeStorage(), [])
  const [open, setOpen] = React.useState(false)
  const [status, setStatus] = React.useState<
    "pending" | "ready" | "unavailable"
  >("pending")
  const statusRef = React.useRef(status)
  const mountedRef = React.useRef(false)

  React.useEffect(() => {
    mountedRef.current = true
    let current = true
    let watchedValueArrived = false
    const listeners = {
      [preferenceKey]: (change: { newValue?: unknown }) => {
        if (!current) return
        watchedValueArrived = true
        setOpen(change.newValue === true)
        statusRef.current = "ready"
        setStatus("ready")
      }
    }
    storage.watch(listeners)
    void storage
      .get<boolean>(preferenceKey)
      .then((value) => {
        if (!current || watchedValueArrived) return
        setOpen(value === true)
        statusRef.current = "ready"
        setStatus("ready")
      })
      .catch(() => {
        if (!current || watchedValueArrived) return
        statusRef.current = "unavailable"
        setStatus("unavailable")
      })
    return () => {
      current = false
      mountedRef.current = false
      storage.unwatch(listeners)
    }
  }, [storage, preferenceKey])

  const setExplicitOpen = React.useCallback(
    async (next: boolean) => {
      if (statusRef.current === "pending") return
      setOpen(next)
      // An unread preference must never be overwritten by session recovery.
      if (statusRef.current !== "ready") return
      try {
        await storage.set(preferenceKey, next)
      } catch {
        if (!mountedRef.current) return
        statusRef.current = "unavailable"
        setStatus("unavailable")
      }
    },
    [storage, preferenceKey]
  )

  return {
    open,
    setExplicitOpen,
    loading: status === "pending",
    unavailable: status === "unavailable"
  }
}
