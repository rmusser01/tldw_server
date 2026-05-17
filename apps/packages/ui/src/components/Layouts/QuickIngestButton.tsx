import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { UploadCloud } from "lucide-react"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { createEventHost } from "@/utils/create-event-host"
import {
  consumePendingQuickIngestOpen,
  createQuickIngestSessionSeedFromOpenDetail,
  rememberQuickIngestOpenRequest,
  type QuickIngestOpenDetail,
  type QuickIngestPendingOpenOptions,
} from "@/utils/quick-ingest-open"

const QuickIngestModal = lazy(() =>
  import("../Common/QuickIngestWizardModal").then((m) => ({
    default: m.QuickIngestWizardModal
  }))
)

const classNames = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(" ")

interface QuickIngestButtonProps {
  /** Additional CSS classes */
  className?: string
}

type QuickIngestOpenOptions = QuickIngestPendingOpenOptions

type QuickIngestEventsOptions = {
  focusTriggerRef?: React.RefObject<HTMLElement>
}

export const useQuickIngestEvents = (options?: QuickIngestEventsOptions) => {
  const focusTriggerRef = options?.focusTriggerRef
  const [quickIngestAutoProcessQueued, setQuickIngestAutoProcessQueued] =
    useState(false)
  const [quickIngestSessionHydrated, setQuickIngestSessionHydrated] = useState(
    () => useQuickIngestSessionStore.persist?.hasHydrated?.() ?? true
  )
  const quickIngestReadyRef = useRef(false)
  const pendingQuickIngestIntroRef = useRef(false)
  const { session, createDraftSession, upsertSession, showSession, hideSession } =
    useQuickIngestSessionStore((s) => ({
      session: s.session,
      createDraftSession: s.createDraftSession,
      upsertSession: s.upsertSession,
      showSession: s.showSession,
      hideSession: s.hideSession,
    }))
  const quickIngestOpen = session?.visibility === "visible"
  const hasQuickIngestSession = Boolean(session)

  const rehydrateQuickIngestSession = useCallback(async () => {
    const persistApi = useQuickIngestSessionStore.persist
    if (!persistApi) {
      return
    }
    if (persistApi.hasHydrated?.()) {
      setQuickIngestSessionHydrated(true)
      return
    }
    await persistApi.rehydrate?.()
    setQuickIngestSessionHydrated(persistApi.hasHydrated?.() ?? true)
  }, [])

  const performOpenQuickIngest = useCallback(
    (options?: QuickIngestOpenOptions, detail?: QuickIngestOpenDetail) => {
      const { autoProcessQueued = false, focusTrigger = true } = options || {}
      const seed = createQuickIngestSessionSeedFromOpenDetail(detail)
      setQuickIngestAutoProcessQueued(autoProcessQueued)
      const currentSession = useQuickIngestSessionStore.getState().session
      if (currentSession) {
        if (seed) {
          upsertSession(seed)
        }
        showSession()
      } else {
        createDraftSession(seed ?? undefined)
      }
      if (focusTrigger && focusTriggerRef?.current) {
        requestAnimationFrame(() => {
          focusTriggerRef.current?.focus()
        })
      }
    },
    [createDraftSession, focusTriggerRef, showSession, upsertSession]
  )

  const performOpenQuickIngestIntro = useCallback(
    (options?: QuickIngestOpenOptions, detail?: QuickIngestOpenDetail) => {
      performOpenQuickIngest({ ...options, focusTrigger: false }, detail)
      if (quickIngestReadyRef.current) {
        window.dispatchEvent(new CustomEvent("tldw:quick-ingest-force-intro"))
      } else {
        pendingQuickIngestIntroRef.current = true
      }
    },
    [performOpenQuickIngest]
  )

  const consumePendingOpenRequest = useCallback(() => {
    const pending = consumePendingQuickIngestOpen()
    if (!pending) {
      return false
    }
    if (pending.mode === "intro") {
      performOpenQuickIngestIntro(pending.options, pending.detail)
      return true
    }
    performOpenQuickIngest(pending.options, pending.detail)
    return true
  }, [performOpenQuickIngest, performOpenQuickIngestIntro])

  const openQuickIngest = useCallback(
    (nextOptions?: QuickIngestOpenOptions, detail?: QuickIngestOpenDetail) => {
      if (!quickIngestSessionHydrated) {
        rememberQuickIngestOpenRequest("normal", detail, nextOptions)
        void rehydrateQuickIngestSession()
        return
      }
      performOpenQuickIngest(nextOptions, detail)
    },
    [performOpenQuickIngest, quickIngestSessionHydrated, rehydrateQuickIngestSession]
  )

  const closeQuickIngest = useCallback(
    (options?: { focusTrigger?: boolean }) => {
      hideSession()
      setQuickIngestAutoProcessQueued(false)
      if ((options?.focusTrigger ?? true) && focusTriggerRef?.current) {
        requestAnimationFrame(() => {
          focusTriggerRef.current?.focus()
        })
      }
    },
    [focusTriggerRef, hideSession]
  )

  // Global event listeners for opening quick ingest
  useEffect(() => {
    const handler = (event: Event) => {
      openQuickIngest(
        undefined,
        (event as CustomEvent<QuickIngestOpenDetail>).detail
      )
    }
    window.addEventListener("tldw:open-quick-ingest", handler)
    return () => {
      window.removeEventListener("tldw:open-quick-ingest", handler)
    }
  }, [openQuickIngest])

  useEffect(() => {
    const persistApi = useQuickIngestSessionStore.persist
    if (!persistApi) {
      return
    }

    const syncHydrationState = () => {
      setQuickIngestSessionHydrated(persistApi.hasHydrated?.() ?? true)
    }

    syncHydrationState()
    const unsubscribeHydrate = persistApi.onHydrate?.(() => {
      setQuickIngestSessionHydrated(false)
    })
    const unsubscribeFinishHydration = persistApi.onFinishHydration?.(() => {
      setQuickIngestSessionHydrated(true)
    })

    if (!(persistApi.hasHydrated?.() ?? true)) {
      const rehydrateResult = persistApi.rehydrate?.()
      if (rehydrateResult && typeof rehydrateResult.then === "function") {
        void rehydrateResult.then(syncHydrationState)
      } else {
        syncHydrationState()
      }
    }

    return () => {
      unsubscribeHydrate?.()
      unsubscribeFinishHydration?.()
    }
  }, [])

  useEffect(() => {
    if (!quickIngestSessionHydrated) {
      return
    }
    consumePendingOpenRequest()
  }, [consumePendingOpenRequest, quickIngestSessionHydrated])

  useEffect(() => {
    const markQuickIngestReady = () => {
      quickIngestReadyRef.current = true
      if (pendingQuickIngestIntroRef.current) {
        pendingQuickIngestIntroRef.current = false
        window.dispatchEvent(new CustomEvent("tldw:quick-ingest-force-intro"))
      }
    }
    window.addEventListener("tldw:quick-ingest-ready", markQuickIngestReady)
    return () => {
      window.removeEventListener(
        "tldw:quick-ingest-ready",
        markQuickIngestReady
      )
    }
  }, [])

  useEffect(() => {
    const handler = () => {
      if (!quickIngestSessionHydrated) {
        rememberQuickIngestOpenRequest("intro", undefined, {
          focusTrigger: false,
        })
        void rehydrateQuickIngestSession()
        return
      }
      performOpenQuickIngestIntro({ focusTrigger: false })
    }
    window.addEventListener("tldw:open-quick-ingest-intro", handler)
    return () => {
      window.removeEventListener("tldw:open-quick-ingest-intro", handler)
    }
  }, [
    performOpenQuickIngestIntro,
    quickIngestSessionHydrated,
    rehydrateQuickIngestSession,
  ])

  return {
    quickIngestOpen,
    hasQuickIngestSession,
    quickIngestAutoProcessQueued,
    openQuickIngest,
    closeQuickIngest
  }
}

/**
 * Quick ingest button with badge for queued items and modal.
 * Extracted from Header.tsx for better maintainability.
 */
export function QuickIngestButton({ className }: QuickIngestButtonProps) {
  const { t } = useTranslation(["option", "playground", "quickIngest"])
  const quickIngestBtnRef = useRef<HTMLButtonElement>(null)
  const {
    quickIngestOpen,
    hasQuickIngestSession,
    quickIngestAutoProcessQueued,
    openQuickIngest,
    closeQuickIngest
  } = useQuickIngestEvents({ focusTriggerRef: quickIngestBtnRef })
  const { quickIngestSession, quickIngestSessionSummary } =
    useQuickIngestSessionStore((s) => ({
      quickIngestSession: s.session,
      quickIngestSessionSummary: s.triggerSummary,
    }))

  const { queuedQuickIngestCount, quickIngestHadFailure } = useQuickIngestStore(
    (s) => ({
      queuedQuickIngestCount: s.queuedCount,
      quickIngestHadFailure: s.hadRecentFailure,
    })
  )

  const sessionBadgeCount = quickIngestSessionSummary.count
  const visibleBadgeCount =
    sessionBadgeCount > 0 ? sessionBadgeCount : queuedQuickIngestCount
  const hasQueuedQuickIngest = visibleBadgeCount > 0
  const shouldShowProcessQueuedCta =
    quickIngestSession?.lifecycle === "draft" &&
    (quickIngestSession.badge.queueCount > 0 || queuedQuickIngestCount > 0)

  const quickIngestAriaLabel = React.useMemo(() => {
    const base = t("option:header.quickIngest", "Quick Ingest")
    if (!hasQueuedQuickIngest) {
      return base
    }

    if (quickIngestSessionSummary.label) {
      return t(
        "option:header.quickIngestSessionAria",
        "{{label}} - {{summary}} - click to reopen current ingest session",
        {
          label: base,
          summary: quickIngestSessionSummary.label,
        }
      )
    }

    const queuedText = t(
      "option:header.quickIngestQueuedAria",
      "{{label}} - {{count}} items queued - click to review and process",
      {
        label: base,
        count: visibleBadgeCount,
      }
    )

    if (quickIngestHadFailure) {
      const failureHint = t(
        "quickIngest:healthAriaHint",
        "Recent runs failed - open Health & diagnostics from the header for more details."
      )
      return `${queuedText} ${failureHint}`
    }

    return queuedText
  }, [
    hasQueuedQuickIngest,
    quickIngestHadFailure,
    quickIngestSessionSummary.label,
    t,
    visibleBadgeCount,
  ])

  return (
    <>
      <div className={`flex items-center gap-3 ${className || ""}`}>
        <button
          type="button"
          ref={quickIngestBtnRef}
          onClick={() => openQuickIngest()}
          data-testid="open-quick-ingest"
          aria-label={quickIngestAriaLabel}
          title={
            t(
              "option:header.quickIngestTooltip",
              "Import URLs, documents, and media to your knowledge base"
            ) as string
          }
          className={classNames(
            "relative inline-flex min-w-[180px] items-center justify-center gap-2 rounded-full border border-transparent px-4 py-2 text-sm font-medium transition hover:border-border hover:bg-surface focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus",
            "text-text-muted"
          )}
          data-has-queued-ingest={hasQueuedQuickIngest ? "true" : "false"}
          aria-disabled={false}
        >
          <UploadCloud className="h-3 w-3" aria-hidden="true" />
          <span>{t("option:header.quickIngest", "Quick Ingest")}</span>
          {hasQueuedQuickIngest && (
            <span className="absolute -top-1 -right-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-semibold text-white">
              {visibleBadgeCount > 9 ? "9+" : visibleBadgeCount}
            </span>
          )}
        </button>

        {shouldShowProcessQueuedCta && (
          <button
            type="button"
            data-testid="process-queued-ingest-header"
            onClick={() =>
              openQuickIngest({
                autoProcessQueued: true,
                focusTrigger: false,
              })
            }
            className="inline-flex items-center rounded-full border border-transparent px-2 py-1 text-xs text-primary hover:text-primaryStrong"
            title={t(
              "quickIngest:processQueuedItemsShort",
              "Process queued items"
            )}
          >
            {t(
              "quickIngest:processQueuedItemsShort",
              "Process queued items"
            )}
          </button>
        )}
      </div>

      <Suspense fallback={null}>
        <QuickIngestModal
          open={quickIngestOpen}
          autoProcessQueued={quickIngestAutoProcessQueued}
          onClose={closeQuickIngest}
        />
      </Suspense>
    </>
  )
}

export const QuickIngestModalHost = createEventHost({
  useEvents: useQuickIngestEvents,
  isActive: ({ quickIngestOpen, hasQuickIngestSession }) =>
    quickIngestOpen || hasQuickIngestSession,
  render: ({
    quickIngestOpen,
    quickIngestAutoProcessQueued,
    closeQuickIngest,
  }) => (
    <Suspense fallback={null}>
      <QuickIngestModal
        open={quickIngestOpen}
        autoProcessQueued={quickIngestAutoProcessQueued}
        onClose={() => closeQuickIngest({ focusTrigger: false })}
      />
    </Suspense>
  )
})

export default QuickIngestButton
