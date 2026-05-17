import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

import type {
  IngestPreset,
  PresetConfig,
  QueueItemValidation,
  WizardProcessingState,
  WizardResultItem,
  WizardStep,
  DetectedMediaType,
  ConferenceBatchMetadata,
  ConferenceItemMetadataOverride,
  PlaylistQueueMetadata,
} from "@/components/Common/QuickIngest/types"
import { DEFAULT_PRESET, DEFAULT_PRESETS } from "@/components/Common/QuickIngest/presets"
import type { QuickIngestOpenDetail } from "@/utils/quick-ingest-open"

const STORAGE_KEY = "tldw-quick-ingest-session"

export type QuickIngestSessionLifecycle =
  | "draft"
  | "processing"
  | "completed"
  | "partial_failure"
  | "cancelled"
  | "interrupted"

export type PersistedQuickIngestTracking = {
  mode: "webui-direct" | "extension-runtime" | "unknown"
  sessionId?: string
  batchId?: string
  batchIds?: string[]
  collectionId?: string
  plannedItemIds?: string[]
  jobIds?: number[]
  submittedItemIds?: string[]
  /** @deprecated use submittedItemIds */
  itemIds?: string[]
  jobIdToItemId?: Record<string, string>
  jobIdToCollectionItemId?: Record<string, string>
  durableMode?: "durable_collection" | "degraded" | "unknown"
  startedAt?: number
}

export type PersistedWizardQueueItem = {
  id: string
  kind?: string
  fileName?: string
  name?: string
  key?: string
  size?: number
  type?: string
  lastModified?: number
  url?: string
  detectedType: DetectedMediaType
  icon: string
  fileSize: number
  mimeType?: string
  validation: QueueItemValidation
  playlist?: PlaylistQueueMetadata
  conferenceOverride?: ConferenceItemMetadataOverride
  fileStub?: {
    key?: string
    instanceId?: string
    lastModified?: number
  }
}

export type QuickIngestTriggerSummary = {
  count: number
  label: string | null
  hadFailure: boolean
}

export type QuickIngestSessionBadge = {
  queueCount: number
  hasRecentFailure: boolean
}

export type QuickIngestSessionResultSummary = {
  status: "idle" | "success" | "error" | "cancelled"
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

export type QuickIngestSessionRecord = {
  id: string
  visibility: "visible" | "hidden"
  lifecycle: QuickIngestSessionLifecycle
  currentStep: WizardStep
  queueItems: PersistedWizardQueueItem[]
  selectedPreset: IngestPreset
  customBasePreset: Exclude<IngestPreset, "custom">
  presetConfig: PresetConfig
  customOptions: Partial<PresetConfig>
  processingState: WizardProcessingState
  results: WizardResultItem[]
  openDetail?: QuickIngestOpenDetail | null
  conferenceBatchMetadata?: ConferenceBatchMetadata | null
  badge: QuickIngestSessionBadge
  resultSummary: QuickIngestSessionResultSummary
  tracking?: PersistedQuickIngestTracking
  errorMessage?: string | null
  createdAt: number
  updatedAt: number
  completedAt?: number | null
}

type QuickIngestSessionPersistedState = {
  session: QuickIngestSessionRecord | null
}

const isCustomBasePreset = (
  value: unknown
): value is Exclude<IngestPreset, "custom"> =>
  typeof value === "string" && value in DEFAULT_PRESETS

type QuickIngestSessionState = QuickIngestSessionPersistedState & {
  triggerSummary: QuickIngestTriggerSummary
  createDraftSession: (
    seed?: Partial<QuickIngestSessionRecord>
  ) => QuickIngestSessionRecord
  upsertSession: (next: Partial<QuickIngestSessionRecord>) => void
  showSession: () => void
  hideSession: () => void
  markProcessingTracking: (tracking: PersistedQuickIngestTracking) => void
  markInterrupted: (reason?: string) => void
  clearSession: () => void
  replaceWithNewDraft: (
    seed?: Partial<QuickIngestSessionRecord>
  ) => QuickIngestSessionRecord
}

const INITIAL_PROCESSING_STATE: WizardProcessingState = {
  status: "idle",
  perItemProgress: [],
  elapsed: 0,
  estimatedRemaining: 0,
}

const INITIAL_RESULT_SUMMARY: QuickIngestSessionResultSummary = {
  status: "idle",
  attemptedAt: null,
  completedAt: null,
  totalCount: 0,
  successCount: 0,
  failedCount: 0,
  cancelledCount: 0,
  firstMediaId: null,
  primarySourceLabel: null,
  errorMessage: null,
}

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
})

const createSessionStorage = (): StateStorage => {
  if (typeof window === "undefined") {
    return createMemoryStorage()
  }
  return {
    getItem: (name: string): string | null => {
      try {
        return window.sessionStorage.getItem(name)
      } catch {
        return null
      }
    },
    setItem: (name: string, value: string): void => {
      try {
        const parsed = JSON.parse(value) as {
          state?: QuickIngestSessionPersistedState
        }
        if (!parsed?.state?.session) {
          window.sessionStorage.removeItem(name)
          return
        }
        window.sessionStorage.setItem(name, value)
      } catch {
        // Ignore storage write failures.
      }
    },
    removeItem: (name: string): void => {
      try {
        window.sessionStorage.removeItem(name)
      } catch {
        // Ignore storage removal failures.
      }
    },
  }
}

const generateSessionId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `qi-session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

const normalizeStringIds = (values?: unknown[]): string[] =>
  Array.from(
    new Set(
      Array.isArray(values)
        ? values
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        : []
    )
  )

const sanitizeTracking = (
  tracking?: PersistedQuickIngestTracking
): PersistedQuickIngestTracking | undefined => {
  if (!tracking) return undefined
  const batchIds = normalizeStringIds([
    ...(Array.isArray(tracking.batchIds) ? tracking.batchIds : []),
    tracking.batchId,
  ])
  const jobIds = Array.isArray(tracking.jobIds)
    ? tracking.jobIds
        .map((jobId) => Number(jobId))
        .filter((jobId) => Number.isFinite(jobId) && jobId > 0)
        .map((jobId) => Math.trunc(jobId))
    : undefined
  const submittedItemIds = normalizeStringIds([
    ...(Array.isArray(tracking.submittedItemIds)
      ? tracking.submittedItemIds
      : []),
    ...(Array.isArray(tracking.itemIds) ? tracking.itemIds : []),
  ])
  const plannedItemIds = normalizeStringIds(
    Array.isArray(tracking.plannedItemIds) ? tracking.plannedItemIds : []
  )
  const jobIdToItemIdEntries = Object.entries(tracking.jobIdToItemId || {})
    .map(([jobId, itemId]) => [String(jobId || "").trim(), String(itemId || "").trim()] as const)
    .filter(([jobId, itemId]) => jobId && itemId)
  const jobIdToCollectionItemIdEntries = Object.entries(
    tracking.jobIdToCollectionItemId || {}
  )
    .map(([jobId, itemId]) => [String(jobId || "").trim(), String(itemId || "").trim()] as const)
    .filter(([jobId, itemId]) => jobId && itemId)
  const normalizedMode =
    tracking.mode === "webui-direct" ||
    tracking.mode === "extension-runtime" ||
    tracking.mode === "unknown"
      ? tracking.mode
      : "unknown"

  return {
    mode: normalizedMode,
    sessionId: tracking.sessionId?.trim() || undefined,
    batchId:
      tracking.batchId?.trim() ||
      (batchIds.length > 0 ? batchIds[batchIds.length - 1] : undefined),
    batchIds: batchIds.length > 0 ? batchIds : undefined,
    collectionId: tracking.collectionId?.trim() || undefined,
    plannedItemIds: plannedItemIds.length > 0 ? plannedItemIds : undefined,
    jobIds: jobIds && jobIds.length > 0 ? Array.from(new Set(jobIds)) : undefined,
    submittedItemIds:
      submittedItemIds.length > 0 ? submittedItemIds : undefined,
    itemIds: submittedItemIds.length > 0 ? submittedItemIds : undefined,
    jobIdToItemId:
      jobIdToItemIdEntries.length > 0
        ? Object.fromEntries(jobIdToItemIdEntries)
        : undefined,
    jobIdToCollectionItemId:
      jobIdToCollectionItemIdEntries.length > 0
        ? Object.fromEntries(jobIdToCollectionItemIdEntries)
        : undefined,
    durableMode:
      tracking.durableMode === "durable_collection" ||
      tracking.durableMode === "degraded" ||
      tracking.durableMode === "unknown"
        ? tracking.durableMode
        : undefined,
    startedAt:
      typeof tracking.startedAt === "number" && Number.isFinite(tracking.startedAt)
        ? tracking.startedAt
        : undefined,
  }
}

const mergeTracking = (
  current?: PersistedQuickIngestTracking,
  incoming?: PersistedQuickIngestTracking
): PersistedQuickIngestTracking | undefined => {
  const base = sanitizeTracking(current)
  const next = sanitizeTracking(incoming)

  if (!base && !next) return undefined
  if (!base) return next
  if (!next) return base
  if (base.sessionId && next.sessionId && base.sessionId !== next.sessionId) {
    return next
  }

  return sanitizeTracking({
    mode: next.mode !== "unknown" ? next.mode : base.mode,
    sessionId: next.sessionId || base.sessionId,
    batchId: next.batchId || base.batchId,
    batchIds: [...(base.batchIds || []), ...(next.batchIds || [])],
    collectionId: next.collectionId || base.collectionId,
    plannedItemIds: [
      ...(base.plannedItemIds || []),
      ...(next.plannedItemIds || []),
    ],
    jobIds: [...(base.jobIds || []), ...(next.jobIds || [])],
    submittedItemIds: [
      ...(base.submittedItemIds || base.itemIds || []),
      ...(next.submittedItemIds || next.itemIds || []),
    ],
    jobIdToItemId: {
      ...(base.jobIdToItemId || {}),
      ...(next.jobIdToItemId || {}),
    },
    jobIdToCollectionItemId: {
      ...(base.jobIdToCollectionItemId || {}),
      ...(next.jobIdToCollectionItemId || {}),
    },
    durableMode: next.durableMode || base.durableMode,
    startedAt: base.startedAt || next.startedAt,
  })
}

const sanitizeQueueItems = (
  queueItems?: PersistedWizardQueueItem[]
): PersistedWizardQueueItem[] => {
  if (!Array.isArray(queueItems)) return []

  return queueItems.map((item) => {
    const next: PersistedWizardQueueItem = {
      id: String(item?.id || generateSessionId()),
      kind: typeof item?.kind === "string" ? item.kind : undefined,
      fileName:
        typeof item?.fileName === "string" ? item.fileName : undefined,
      name: typeof item?.name === "string" ? item.name : undefined,
      key: typeof item?.key === "string" ? item.key : undefined,
      size:
        typeof item?.size === "number" && Number.isFinite(item.size)
          ? item.size
          : undefined,
      type: typeof item?.type === "string" ? item.type : undefined,
      lastModified:
        typeof item?.lastModified === "number" &&
        Number.isFinite(item.lastModified)
          ? item.lastModified
          : undefined,
      url: typeof item?.url === "string" ? item.url : undefined,
      detectedType: item?.detectedType || "unknown",
      icon: item?.icon || "File",
      fileSize:
        typeof item?.fileSize === "number" && Number.isFinite(item.fileSize)
          ? item.fileSize
          : typeof item?.size === "number" && Number.isFinite(item.size)
            ? item.size
            : 0,
      mimeType:
        typeof item?.mimeType === "string"
          ? item.mimeType
          : typeof item?.type === "string"
            ? item.type
            : undefined,
      validation: item?.validation || { valid: true },
      playlist: item?.playlist,
      conferenceOverride: item?.conferenceOverride,
    }

    if (item?.fileStub) {
      next.fileStub = {
        key: item.fileStub.key,
        instanceId: item.fileStub.instanceId,
        lastModified:
          typeof item.fileStub.lastModified === "number" &&
          Number.isFinite(item.fileStub.lastModified)
            ? item.fileStub.lastModified
            : undefined,
      }
    }

    return next
  })
}

const countTerminalFailures = (session: QuickIngestSessionRecord): number => {
  if (session.lifecycle === "partial_failure" || session.lifecycle === "interrupted") {
    return Math.max(
      1,
      session.resultSummary.failedCount ||
        session.results.filter((item) => item.status === "error").length
    )
  }
  return session.resultSummary.failedCount ||
    session.results.filter((item) => item.status === "error").length
}

const buildTriggerSummary = (
  session: QuickIngestSessionRecord | null
): QuickIngestTriggerSummary => {
  if (!session) {
    return {
      count: 0,
      label: null,
      hadFailure: false,
    }
  }

  const queueCount = session.queueItems.length
  const resultCount = session.results.length
  const progressCount =
    session.processingState.perItemProgress.length || queueCount || resultCount
  const failureCount = countTerminalFailures(session)
  const badgeCount = session.badge.queueCount

  switch (session.lifecycle) {
    case "draft":
      return {
        count: badgeCount || queueCount,
        label: badgeCount || queueCount ? `${badgeCount || queueCount} queued` : null,
        hadFailure: session.badge.hasRecentFailure,
      }
    case "processing":
      return {
        count: progressCount,
        label: progressCount > 0 ? `${progressCount} processing` : "Processing",
        hadFailure: false,
      }
    case "completed":
      return {
        count: resultCount || queueCount,
        label: `${resultCount || queueCount} completed`,
        hadFailure: false,
      }
    case "partial_failure":
      return {
        count: failureCount || resultCount || queueCount,
        label: `${failureCount || resultCount || queueCount} failed`,
        hadFailure: true,
      }
    case "cancelled":
      return {
        count: resultCount || queueCount,
        label: `${resultCount || queueCount} cancelled`,
        hadFailure: false,
      }
    case "interrupted":
      return {
        count: failureCount || progressCount,
        label: "Ingest interrupted",
        hadFailure: true,
      }
    default:
      return {
        count: 0,
        label: null,
        hadFailure: false,
      }
  }
}

const sanitizeSession = (
  session: QuickIngestSessionRecord | null
): QuickIngestSessionRecord | null => {
  if (!session) return null

  const createdAt =
    typeof session.createdAt === "number" && Number.isFinite(session.createdAt)
      ? session.createdAt
      : Date.now()
  const updatedAt =
    typeof session.updatedAt === "number" && Number.isFinite(session.updatedAt)
      ? session.updatedAt
      : createdAt

  const customBasePreset = isCustomBasePreset(session.customBasePreset)
    ? session.customBasePreset
    : DEFAULT_PRESET

  return {
    id: session.id || generateSessionId(),
    visibility: session.visibility === "hidden" ? "hidden" : "visible",
    lifecycle: session.lifecycle || "draft",
    currentStep: session.currentStep || 1,
    queueItems: sanitizeQueueItems(session.queueItems),
    selectedPreset: session.selectedPreset || DEFAULT_PRESET,
    customBasePreset,
    presetConfig: session.presetConfig || DEFAULT_PRESETS[DEFAULT_PRESET],
    customOptions: session.customOptions || {},
    processingState: session.processingState || { ...INITIAL_PROCESSING_STATE },
    results: Array.isArray(session.results) ? session.results : [],
    openDetail:
      session.openDetail && typeof session.openDetail === "object"
        ? session.openDetail
        : null,
    conferenceBatchMetadata: session.conferenceBatchMetadata ?? null,
    badge: {
      queueCount: Math.max(
        0,
        normalizeCountLike(session.badge?.queueCount) ?? sanitizeQueueItems(session.queueItems).length
      ),
      hasRecentFailure: Boolean(session.badge?.hasRecentFailure),
    },
    resultSummary: {
      ...INITIAL_RESULT_SUMMARY,
      ...(session.resultSummary || {}),
    },
    tracking: sanitizeTracking(session.tracking),
    errorMessage: session.errorMessage || null,
    createdAt,
    updatedAt,
    completedAt:
      typeof session.completedAt === "number" && Number.isFinite(session.completedAt)
        ? session.completedAt
        : null,
  }
}

const normalizeCountLike = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null
  }
  return Math.floor(value)
}

const buildPersistedState = (
  session: QuickIngestSessionRecord | null
): QuickIngestSessionPersistedState => ({
  session: sanitizeSession(session),
})

export const createEmptyQuickIngestSession = (): QuickIngestSessionRecord => {
  const now = Date.now()
  return {
    id: generateSessionId(),
    visibility: "visible",
    lifecycle: "draft",
    currentStep: 1,
    queueItems: [],
    selectedPreset: DEFAULT_PRESET,
    customBasePreset: DEFAULT_PRESET,
    presetConfig: DEFAULT_PRESETS[DEFAULT_PRESET],
    customOptions: {},
    processingState: { ...INITIAL_PROCESSING_STATE },
    results: [],
    openDetail: null,
    conferenceBatchMetadata: null,
    badge: {
      queueCount: 0,
      hasRecentFailure: false,
    },
    resultSummary: { ...INITIAL_RESULT_SUMMARY },
    tracking: undefined,
    errorMessage: null,
    createdAt: now,
    updatedAt: now,
    completedAt: null,
  }
}

const createInitialState = (): QuickIngestSessionPersistedState & {
  triggerSummary: QuickIngestTriggerSummary
} => ({
  session: null,
  triggerSummary: buildTriggerSummary(null),
})

const withSessionUpdate = (
  set: (
    partial:
      | QuickIngestSessionState
      | Partial<QuickIngestSessionState>
      | ((state: QuickIngestSessionState) => QuickIngestSessionState | Partial<QuickIngestSessionState>)
  ) => void,
  resolver: (current: QuickIngestSessionRecord | null) => QuickIngestSessionRecord | null
) => {
  set((state) => {
    const session = sanitizeSession(resolver(state.session))
    return {
      session,
      triggerSummary: buildTriggerSummary(session),
    }
  })
}

export const createQuickIngestSessionStore = () =>
  createWithEqualityFn<QuickIngestSessionState>()(
    persist(
      (set, get) => ({
        ...createInitialState(),
        createDraftSession: (seed) => {
          const next = sanitizeSession({
            ...createEmptyQuickIngestSession(),
            ...(seed || {}),
            updatedAt: Date.now(),
          })
          set({
            session: next,
            triggerSummary: buildTriggerSummary(next),
          })
          return next as QuickIngestSessionRecord
        },
        upsertSession: (next) =>
          withSessionUpdate(set, (current) => {
            const base = current || createEmptyQuickIngestSession()
            return {
              ...base,
              ...next,
              badge: {
                ...base.badge,
                ...(next.badge || {}),
              },
              resultSummary: {
                ...base.resultSummary,
                ...(next.resultSummary || {}),
              },
              tracking:
                next.tracking === undefined
                  ? base.tracking
                  : mergeTracking(base.tracking, next.tracking),
              updatedAt: Date.now(),
            }
          }),
        showSession: () =>
          withSessionUpdate(set, (current) => {
            if (!current) {
              return createEmptyQuickIngestSession()
            }
            return {
              ...current,
              visibility: "visible",
              updatedAt: Date.now(),
            }
          }),
        hideSession: () =>
          withSessionUpdate(set, (current) => {
            if (!current) return current
            return {
              ...current,
              visibility: "hidden",
              updatedAt: Date.now(),
            }
          }),
        markProcessingTracking: (tracking) =>
          withSessionUpdate(set, (current) => {
            const base = current || createEmptyQuickIngestSession()
            return {
              ...base,
              lifecycle: "processing",
              tracking: mergeTracking(base.tracking, {
                ...tracking,
                startedAt:
                  tracking.startedAt || base.tracking?.startedAt || Date.now(),
              }),
              updatedAt: Date.now(),
            }
          }),
        markInterrupted: (reason) =>
          withSessionUpdate(set, (current) => {
            if (!current) return current
            return {
              ...current,
              lifecycle: "interrupted",
              badge: {
                ...current.badge,
                hasRecentFailure: true,
              },
              resultSummary: {
                ...current.resultSummary,
                status: "error",
                errorMessage: reason || "Quick ingest was interrupted.",
              },
              errorMessage: reason || "Quick ingest was interrupted.",
              updatedAt: Date.now(),
            }
          }),
        clearSession: () =>
          {
            set({
              session: null,
              triggerSummary: buildTriggerSummary(null),
            })
            createSessionStorage().removeItem(STORAGE_KEY)
          },
        replaceWithNewDraft: (seed) => {
          get().clearSession()
          return get().createDraftSession(seed)
        },
      }),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => createSessionStorage()),
        partialize: (state) => buildPersistedState(state.session),
        merge: (persistedState, currentState) => {
          const nextSession = sanitizeSession(
            (persistedState as QuickIngestSessionPersistedState | undefined)?.session ||
              null
          )
          return {
            ...currentState,
            session: nextSession,
            triggerSummary: buildTriggerSummary(nextSession),
          }
        },
      }
    )
  )

export const useQuickIngestSessionStore = createQuickIngestSessionStore()

if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // Expose for debugging/tests only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useQuickIngestSessionStore = useQuickIngestSessionStore
}
