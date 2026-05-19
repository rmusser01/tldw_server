import { createWithEqualityFn } from "zustand/traditional"
import { useQuickIngestSessionStore } from "./quick-ingest-session"

const normalizeCount = (value: number) =>
  Number.isFinite(value) && value > 0 ? Math.floor(value) : 0

const syncBadgeStateFromSession = () => {
  const sessionStore = useQuickIngestSessionStore.getState()
  const nextQueuedCount = sessionStore.session?.badge.queueCount ?? 0
  const nextHadRecentFailure =
    Boolean(sessionStore.session?.badge.hasRecentFailure) ||
    Boolean(sessionStore.triggerSummary.hadFailure)
  useQuickIngestStore.setState((state) =>
    state.queuedCount === nextQueuedCount &&
    state.hadRecentFailure === nextHadRecentFailure
      ? state
      : {
          ...state,
          queuedCount: nextQueuedCount,
          hadRecentFailure: nextHadRecentFailure
        }
  )
}

const upsertSessionBadge = (patch: {
  queueCount?: number
  hadRecentFailure?: boolean
}) => {
  const sessionStore = useQuickIngestSessionStore.getState()
  if (
    !sessionStore.session &&
    patch.queueCount !== undefined &&
    patch.queueCount <= 0 &&
    patch.hadRecentFailure !== true
  ) {
    return
  }
  if (
    !sessionStore.session &&
    patch.queueCount === undefined &&
    patch.hadRecentFailure !== true
  ) {
    return
  }
  const current = sessionStore.session ?? sessionStore.createDraftSession()
  sessionStore.upsertSession({
    badge: {
      queueCount: patch.queueCount ?? current.badge.queueCount,
      hasRecentFailure:
        typeof patch.hadRecentFailure === "boolean"
          ? patch.hadRecentFailure
          : current.badge.hasRecentFailure
    }
  })
}

export type QuickIngestLastRunStatus =
  | "idle"
  | "success"
  | "error"
  | "cancelled"

export type QuickIngestLastRunSummary = {
  status: QuickIngestLastRunStatus
  attemptedAt: number | null
  completedAt: number | null
  totalCount: number
  successCount: number
  failedCount: number
  cancelledCount: number
  firstMediaId: string | null
  primarySourceLabel: string | null
  errorMessage: string | null
}

export const createInitialQuickIngestLastRunSummary =
  (): QuickIngestLastRunSummary => ({
    status: "idle",
    attemptedAt: null,
    completedAt: null,
    totalCount: 0,
    successCount: 0,
    failedCount: 0,
    cancelledCount: 0,
    firstMediaId: null,
    primarySourceLabel: null,
    errorMessage: null
  })

type QuickIngestStore = {
  /**
   * Number of items queued in Quick Ingest (used for UI badges).
   */
  queuedCount: number
  setQueuedCount: (count: number) => void
  clearQueued: () => void
  /**
   * Whether the most recent Quick Ingest processing attempt
   * failed due to a server or network error. Used to enrich
   * header/sidepanel tooltips and ARIA labels.
   */
  hadRecentFailure: boolean
  markFailure: () => void
  clearFailure: () => void
  /**
   * Summary of the most recent ingest run to support
   * onboarding next-step recommendations.
   */
  lastRunSummary: QuickIngestLastRunSummary
  recordRunSuccess: (payload: {
    totalCount: number
    successCount: number
    failedCount: number
    firstMediaId?: string | number | null
    primarySourceLabel?: string | null
  }) => void
  recordRunFailure: (payload?: {
    totalCount?: number
    failedCount?: number
    errorMessage?: string | null
  }) => void
  recordRunCancelled: (payload?: {
    totalCount?: number
    successCount?: number
    failedCount?: number
    cancelledCount?: number
    errorMessage?: string | null
  }) => void
  resetLastRunSummary: () => void
  /**
   * Recently ingested documents (most recent first, max 20).
   * Stores {id, type} tuples so DocumentPickerModal can pass type info
   * to handleOpen and route to the correct viewer.
   */
  recentlyIngestedDocs: Array<{ id: number; type: string; title?: string }>
  addRecentlyIngestedDoc: (doc: { id: number; type: string; title?: string }) => void
  addRecentlyIngestedDocs: (docs: Array<{ id: number; type: string; title?: string }>) => void
  clearRecentlyIngestedDocs: () => void
}

export const useQuickIngestStore = createWithEqualityFn<QuickIngestStore>((set) => ({
  queuedCount: 0,
  hadRecentFailure: false,
  lastRunSummary: createInitialQuickIngestLastRunSummary(),
  setQueuedCount: (count) =>
    {
      const nextCount = count > 0 ? count : 0
      set({
        queuedCount: nextCount
      })
      upsertSessionBadge({ queueCount: nextCount })
    },
  clearQueued: () =>
    {
      set({
        queuedCount: 0
      })
      const session = useQuickIngestSessionStore.getState().session
      if (session) {
        upsertSessionBadge({ queueCount: 0 })
      }
    },
  markFailure: () =>
    {
      set({
        hadRecentFailure: true
      })
      upsertSessionBadge({ hadRecentFailure: true })
    },
  clearFailure: () =>
    {
      set({
        hadRecentFailure: false
      })
      const session = useQuickIngestSessionStore.getState().session
      if (session) {
        upsertSessionBadge({ hadRecentFailure: false })
      }
    },
  recordRunSuccess: ({
    totalCount,
    successCount,
    failedCount,
    firstMediaId,
    primarySourceLabel
  }) => {
    const now = Date.now()
    const nextTotal = normalizeCount(totalCount)
    const nextSuccess = normalizeCount(successCount)
    const nextFailed = normalizeCount(failedCount)
    set({
      lastRunSummary: {
        status: "success",
        attemptedAt: now,
        completedAt: now,
        totalCount: nextTotal,
        successCount: nextSuccess,
        failedCount: nextFailed,
        cancelledCount: 0,
        firstMediaId:
          firstMediaId === null || typeof firstMediaId === "undefined"
            ? null
            : String(firstMediaId),
        primarySourceLabel:
          typeof primarySourceLabel === "string" && primarySourceLabel.trim()
            ? primarySourceLabel.trim()
            : null,
        errorMessage: null
      }
    })
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "completed",
      currentStep: 5,
      completedAt: now,
      badge: {
        queueCount: 0,
        hasRecentFailure: false
      },
      resultSummary: {
        status: "success",
        attemptedAt: now,
        completedAt: now,
        totalCount: nextTotal,
        successCount: nextSuccess,
        failedCount: nextFailed,
        cancelledCount: 0,
        firstMediaId:
          firstMediaId === null || typeof firstMediaId === "undefined"
            ? null
            : String(firstMediaId),
        primarySourceLabel:
          typeof primarySourceLabel === "string" && primarySourceLabel.trim()
            ? primarySourceLabel.trim()
            : null,
        errorMessage: null
      }
    })
  },
  recordRunFailure: (payload) => {
    const now = Date.now()
    const totalCount = normalizeCount(payload?.totalCount ?? 0)
    const failedCount = normalizeCount(
      payload?.failedCount ?? (totalCount > 0 ? totalCount : 1)
    )
    set({
      lastRunSummary: {
        status: "error",
        attemptedAt: now,
        completedAt: now,
        totalCount,
        successCount: 0,
        failedCount,
        cancelledCount: 0,
        firstMediaId: null,
        primarySourceLabel: null,
        errorMessage: payload?.errorMessage?.trim() || null
      }
    })
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "partial_failure",
      currentStep: 5,
      completedAt: now,
      badge: {
        queueCount: 0,
        hasRecentFailure: true
      },
      resultSummary: {
        status: "error",
        attemptedAt: now,
        completedAt: now,
        totalCount,
        successCount: 0,
        failedCount,
        cancelledCount: 0,
        firstMediaId: null,
        primarySourceLabel: null,
        errorMessage: payload?.errorMessage?.trim() || null
      }
    })
  },
  recordRunCancelled: (payload) => {
    const now = Date.now()
    const totalCount = normalizeCount(payload?.totalCount ?? 0)
    const successCount = normalizeCount(payload?.successCount ?? 0)
    const failedCount = normalizeCount(payload?.failedCount ?? 0)
    const cancelledCount = normalizeCount(
      payload?.cancelledCount ??
        Math.max(totalCount - successCount - failedCount, 1)
    )
    set({
      lastRunSummary: {
        status: "cancelled",
        attemptedAt: now,
        completedAt: now,
        totalCount,
        successCount,
        failedCount,
        cancelledCount,
        firstMediaId: null,
        primarySourceLabel: null,
        errorMessage: payload?.errorMessage?.trim() || null
      }
    })
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "cancelled",
      currentStep: 5,
      completedAt: now,
      badge: {
        queueCount: 0,
        hasRecentFailure: false
      },
      resultSummary: {
        status: "cancelled",
        attemptedAt: now,
        completedAt: now,
        totalCount,
        successCount,
        failedCount,
        cancelledCount,
        firstMediaId: null,
        primarySourceLabel: null,
        errorMessage: payload?.errorMessage?.trim() || null
      }
    })
  },
  resetLastRunSummary: () =>
    set({
      lastRunSummary: createInitialQuickIngestLastRunSummary()
    }),
  recentlyIngestedDocs: [] as Array<{ id: number; type: string; title?: string }>,
  addRecentlyIngestedDoc: (doc) =>
    set((s) => ({
      recentlyIngestedDocs: [doc, ...s.recentlyIngestedDocs.filter((x) => x.id !== doc.id)].slice(0, 20),
    })),
  addRecentlyIngestedDocs: (newDocs) =>
    set((s) => {
      const ids = new Set(newDocs.map((d) => d.id))
      const filteredExisting = s.recentlyIngestedDocs.filter((d) => !ids.has(d.id))
      return {
        recentlyIngestedDocs: [...newDocs, ...filteredExisting].slice(0, 20),
      }
    }),
  clearRecentlyIngestedDocs: () => set({ recentlyIngestedDocs: [] }),
}))

if (typeof window !== "undefined") {
  syncBadgeStateFromSession()
  const unsubscribeQuickIngestSessionSync = useQuickIngestSessionStore.subscribe(
    () => {
      syncBadgeStateFromSession()
    }
  )

  // Keep the sync subscription alive for the lifetime of the module.
  void unsubscribeQuickIngestSessionSync
}

if (typeof window !== "undefined") {
  // Expose for Playwright tests and debugging only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useQuickIngestStore = useQuickIngestStore
}
