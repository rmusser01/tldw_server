import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { UploadCloud } from "lucide-react"
import { useStorage } from "@plasmohq/storage/hook"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import {
  DEFAULT_PRESET,
  resolvePresetMap,
  type PresetMap,
} from "@/components/Common/QuickIngest/presets"
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
  const [storedPresetConfigs, , presetStorageMeta] = useStorage<PresetMap>(
    "quickIngestPresetConfigs",
    resolvePresetMap()
  )
  const resolvedPresetMap = React.useMemo(
    () => resolvePresetMap(storedPresetConfigs),
    [storedPresetConfigs]
  )
  const [capturedPresetMap, setCapturedPresetMap] = useState<PresetMap>(() =>
    resolvePresetMap()
  )
  const capturedPresetMapRef = useRef(capturedPresetMap)
  const [openRevision, setOpenRevision] = useState(0)
  const [preparedSessionId, setPreparedSessionId] = useState<string | null>(null)
  const preparedSessionIdRef = useRef<string | null>(null)
  const [quickIngestAutoProcessQueued, setQuickIngestAutoProcessQueued] =
    useState(false)
  const [quickIngestSessionHydrated, setQuickIngestSessionHydrated] = useState(
    () => useQuickIngestSessionStore.persist?.hasHydrated?.() ?? true
  )
  const quickIngestModalReadyRef = useRef(false)
  const pendingQuickIngestIntroRef = useRef(false)
  const session = useQuickIngestSessionStore((s) => s.session)
  const createDraftSession = useQuickIngestSessionStore((s) => s.createDraftSession)
  const upsertSession = useQuickIngestSessionStore((s) => s.upsertSession)
  const showSession = useQuickIngestSessionStore((s) => s.showSession)
  const hideSession = useQuickIngestSessionStore((s) => s.hideSession)
  const replaceWithNewDraft = useQuickIngestSessionStore(
    (s) => s.replaceWithNewDraft
  )
  const quickIngestOpen = session?.visibility === "visible"
  const hasQuickIngestSession = Boolean(session)
  const storageAndSessionReady =
    quickIngestSessionHydrated && !presetStorageMeta.isLoading
  const quickIngestReady =
    storageAndSessionReady &&
    (!quickIngestOpen || preparedSessionId === session?.id)

  const capturePresetSnapshot = useCallback((incrementRevision = true) => {
    if (!incrementRevision) {
      return capturedPresetMapRef.current
    }
    capturedPresetMapRef.current = resolvedPresetMap
    setCapturedPresetMap(resolvedPresetMap)
    setOpenRevision((revision) => revision + 1)
    return resolvedPresetMap
  }, [resolvedPresetMap])

  const buildNamedPresetSeed = useCallback(
    (presetMap: PresetMap, preset = DEFAULT_PRESET) => ({
      selectedPreset: preset,
      customBasePreset: preset,
      presetConfig: presetMap[preset],
      customOptions: {},
    }),
    []
  )

  const prepareExistingSession = useCallback(
    (presetMap: PresetMap) => {
      const currentSession = useQuickIngestSessionStore.getState().session
      if (
        currentSession?.lifecycle === "draft" &&
        currentSession.selectedPreset !== "custom" &&
        !currentSession.firstSourceAddMode
      ) {
        upsertSession(
          buildNamedPresetSeed(presetMap, currentSession.selectedPreset)
        )
      }
      return currentSession
    },
    [buildNamedPresetSeed, upsertSession]
  )

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
      const currentSession = useQuickIngestSessionStore.getState().session
      const shouldRebaseCurrent = Boolean(
        currentSession?.lifecycle === "draft" &&
        currentSession.selectedPreset !== "custom" &&
        !currentSession.firstSourceAddMode &&
        !seed?.firstSourceAddMode
      )
      const shouldSeedNamedDraft = !currentSession && !seed?.firstSourceAddMode
      const shouldRemountSeededDraft = Boolean(
        currentSession?.lifecycle === "draft" && seed
      )
      const presetMap = capturePresetSnapshot(
        shouldRebaseCurrent || shouldSeedNamedDraft || shouldRemountSeededDraft
      )
      setQuickIngestAutoProcessQueued(autoProcessQueued)
      if (currentSession) {
        preparedSessionIdRef.current = currentSession.id
        setPreparedSessionId(currentSession.id)
        if (seed && currentSession.lifecycle === "draft") {
          upsertSession({
            ...(shouldRebaseCurrent && currentSession.selectedPreset !== "custom"
              ? buildNamedPresetSeed(presetMap, currentSession.selectedPreset)
              : {}),
            ...seed,
          })
        } else {
          // Processing and terminal sessions keep their active snapshot; a new
          // open detail applies only to drafts.
          prepareExistingSession(presetMap)
        }
        showSession()
      } else {
        const nextSession = createDraftSession(
          seed?.firstSourceAddMode
            ? seed
            : {
                ...buildNamedPresetSeed(presetMap),
                ...(seed ?? {}),
              }
        )
        preparedSessionIdRef.current = nextSession.id
        setPreparedSessionId(nextSession.id)
      }
      if (focusTrigger && focusTriggerRef?.current) {
        requestAnimationFrame(() => {
          focusTriggerRef.current?.focus()
        })
      }
    },
    [
      buildNamedPresetSeed,
      capturePresetSnapshot,
      createDraftSession,
      focusTriggerRef,
      prepareExistingSession,
      showSession,
      upsertSession,
    ]
  )

  const performOpenQuickIngestIntro = useCallback(
    (options?: QuickIngestOpenOptions, detail?: QuickIngestOpenDetail) => {
      performOpenQuickIngest({ ...options, focusTrigger: false }, detail)
      if (quickIngestModalReadyRef.current) {
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
      if (!storageAndSessionReady) {
        rememberQuickIngestOpenRequest("normal", detail, nextOptions)
        if (!quickIngestSessionHydrated) {
          void rehydrateQuickIngestSession()
        }
        return
      }
      performOpenQuickIngest(nextOptions, detail)
    },
    [
      performOpenQuickIngest,
      quickIngestSessionHydrated,
      rehydrateQuickIngestSession,
      storageAndSessionReady,
    ]
  )

  const createNewDraft = useCallback(() => {
    const presetMap = capturePresetSnapshot()
    const nextSession = replaceWithNewDraft(buildNamedPresetSeed(presetMap))
    setQuickIngestAutoProcessQueued(false)
    preparedSessionIdRef.current = nextSession.id
    setPreparedSessionId(nextSession.id)
    return nextSession
  }, [buildNamedPresetSeed, capturePresetSnapshot, replaceWithNewDraft])

  const closeQuickIngest = useCallback(
    (options?: { focusTrigger?: boolean }) => {
      hideSession()
      setQuickIngestAutoProcessQueued(false)
      preparedSessionIdRef.current = null
      setPreparedSessionId(null)
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
      const pending = consumePendingQuickIngestOpen()
      openQuickIngest(
        pending?.mode === "normal" ? pending.options : undefined,
        pending?.mode === "normal"
          ? pending.detail
          : (event as CustomEvent<QuickIngestOpenDetail>).detail
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
    if (!storageAndSessionReady) {
      return
    }
    consumePendingOpenRequest()
  }, [consumePendingOpenRequest, storageAndSessionReady])

  useEffect(() => {
    if (
      !storageAndSessionReady ||
      !session ||
      session.visibility !== "visible" ||
      preparedSessionIdRef.current === session.id
    ) {
      return
    }
    preparedSessionIdRef.current = session.id
    setPreparedSessionId(session.id)
    const shouldRebase =
      session.lifecycle === "draft" &&
      session.selectedPreset !== "custom" &&
      !session.firstSourceAddMode
    const presetMap = capturePresetSnapshot(shouldRebase)
    prepareExistingSession(presetMap)
  }, [
    capturePresetSnapshot,
    prepareExistingSession,
    session,
    storageAndSessionReady,
  ])

  useEffect(() => {
    const markQuickIngestReady = () => {
      quickIngestModalReadyRef.current = true
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
    const handler = (event: Event) => {
      const pending = consumePendingQuickIngestOpen()
      const detail =
        pending?.mode === "intro"
          ? pending.detail
          : (event as CustomEvent<QuickIngestOpenDetail>).detail
      const openOptions =
        pending?.mode === "intro"
          ? { ...pending.options, focusTrigger: false }
          : { focusTrigger: false }
      if (!storageAndSessionReady) {
        rememberQuickIngestOpenRequest("intro", detail, openOptions)
        if (!quickIngestSessionHydrated) {
          void rehydrateQuickIngestSession()
        }
        return
      }
      performOpenQuickIngestIntro(openOptions, detail)
    }
    window.addEventListener("tldw:open-quick-ingest-intro", handler)
    return () => {
      window.removeEventListener("tldw:open-quick-ingest-intro", handler)
    }
  }, [
    performOpenQuickIngestIntro,
    quickIngestSessionHydrated,
    rehydrateQuickIngestSession,
    storageAndSessionReady,
  ])

  return {
    quickIngestOpen,
    quickIngestReady,
    hasQuickIngestSession,
    quickIngestAutoProcessQueued,
    presetMap: capturedPresetMap,
    openRevision,
    createNewDraft,
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
    quickIngestReady,
    quickIngestAutoProcessQueued,
    presetMap,
    openRevision,
    createNewDraft,
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

      {quickIngestReady ? (
        <Suspense fallback={null}>
          <QuickIngestModal
            open={quickIngestOpen}
            autoProcessQueued={quickIngestAutoProcessQueued}
            presetMap={presetMap}
            openRevision={openRevision}
            createNewDraft={createNewDraft}
            onClose={closeQuickIngest}
          />
        </Suspense>
      ) : null}
    </>
  )
}

export const QuickIngestModalHost = createEventHost({
  useEvents: useQuickIngestEvents,
  isActive: ({ quickIngestOpen, quickIngestReady, hasQuickIngestSession }) =>
    quickIngestReady && (quickIngestOpen || hasQuickIngestSession),
  render: ({
    quickIngestOpen,
    quickIngestAutoProcessQueued,
    presetMap,
    openRevision,
    createNewDraft,
    closeQuickIngest,
  }) => (
    <Suspense fallback={null}>
      <QuickIngestModal
        open={quickIngestOpen}
        autoProcessQueued={quickIngestAutoProcessQueued}
        presetMap={presetMap}
        openRevision={openRevision}
        createNewDraft={createNewDraft}
        onClose={() => closeQuickIngest({ focusTrigger: false })}
      />
    </Suspense>
  )
})

export default QuickIngestButton
