import React from "react"

import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import {
  consumePendingQuickIngestOpen,
  createQuickIngestSessionSeedFromOpenDetail,
  retainQuickIngestOpenRequest,
  type QuickIngestOpenDetail,
  type QuickIngestPendingOpenOptions,
} from "@/utils/quick-ingest-open"

type UseSidepanelQuickIngestOpenOptions = {
  focusTriggerRef: React.RefObject<HTMLElement>
  setAutoProcessQueued: React.Dispatch<React.SetStateAction<boolean>>
  setIngestOpen: React.Dispatch<React.SetStateAction<boolean>>
}

export const useSidepanelQuickIngestOpen = ({
  focusTriggerRef,
  setAutoProcessQueued,
  setIngestOpen,
}: UseSidepanelQuickIngestOpenOptions) => {
  const [sessionHydrated, setSessionHydrated] = React.useState(
    () => useQuickIngestSessionStore.persist?.hasHydrated?.() ?? true
  )
  const mountedRef = React.useRef(true)
  const createDraftSession = useQuickIngestSessionStore(
    (state) => state.createDraftSession
  )
  const upsertSession = useQuickIngestSessionStore(
    (state) => state.upsertSession
  )
  const showSession = useQuickIngestSessionStore((state) => state.showSession)

  const performOpen = React.useCallback(
    (
      options?: QuickIngestPendingOpenOptions,
      detail?: QuickIngestOpenDetail
    ) => {
      const { autoProcessQueued = false, focusTrigger = true } = options || {}
      const seed = createQuickIngestSessionSeedFromOpenDetail(detail)
      setAutoProcessQueued(autoProcessQueued)
      if (useQuickIngestSessionStore.getState().session) {
        if (seed) {
          upsertSession(seed)
        }
        showSession()
      } else {
        createDraftSession(seed ?? undefined)
      }
      setIngestOpen(true)
      if (focusTrigger) {
        requestAnimationFrame(() => focusTriggerRef.current?.focus())
      }
    },
    [
      createDraftSession,
      focusTriggerRef,
      setAutoProcessQueued,
      setIngestOpen,
      showSession,
      upsertSession,
    ]
  )

  const consumePendingOpen = React.useCallback(() => {
    const pending = consumePendingQuickIngestOpen("normal")
    if (!pending) return false
    performOpen(pending.options, pending.detail)
    return true
  }, [performOpen])

  const rehydrateSession = React.useCallback(async () => {
    const persistApi = useQuickIngestSessionStore.persist
    if (!persistApi) return
    try {
      await persistApi.rehydrate?.()
    } catch {
      return
    }
    if (mountedRef.current) {
      const hydrated = persistApi.hasHydrated?.() ?? true
      setSessionHydrated(hydrated)
      if (hydrated) {
        consumePendingOpen()
      }
    }
  }, [consumePendingOpen])

  React.useEffect(() => {
    mountedRef.current = true
    const persistApi = useQuickIngestSessionStore.persist
    if (!persistApi) {
      setSessionHydrated(true)
      return () => {
        mountedRef.current = false
      }
    }

    const syncHydrationState = () => {
      if (mountedRef.current) {
        setSessionHydrated(persistApi.hasHydrated?.() ?? true)
      }
    }
    syncHydrationState()
    const unsubscribeHydrate = persistApi.onHydrate?.(() => {
      if (mountedRef.current) {
        setSessionHydrated(false)
      }
    })
    const unsubscribeFinishHydration =
      persistApi.onFinishHydration?.(syncHydrationState)

    return () => {
      mountedRef.current = false
      unsubscribeHydrate?.()
      unsubscribeFinishHydration?.()
    }
  }, [])

  React.useEffect(() => {
    if (sessionHydrated) {
      consumePendingOpen()
    }
  }, [consumePendingOpen, sessionHydrated])

  return React.useCallback(
    (detail?: QuickIngestOpenDetail) => {
      const pending = retainQuickIngestOpenRequest("normal", detail)
      const persistApi = useQuickIngestSessionStore.persist
      const hydrated = persistApi?.hasHydrated?.() ?? sessionHydrated
      if (!hydrated) {
        void rehydrateSession()
        return
      }
      consumePendingQuickIngestOpen("normal")
      performOpen(pending?.options, pending?.detail ?? detail)
    }, [performOpen, rehydrateSession, sessionHydrated]
  )
}
