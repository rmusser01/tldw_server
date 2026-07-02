import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

export type FeedbackThumb = "up" | "down" | null

export type FeedbackDetail = {
  rating?: number | null
  issues?: string[]
  notes?: string
  submittedAt?: number | null
}

export type SourceFeedbackEntry = {
  thumb?: FeedbackThumb
  submittedAt?: number | null
}

export type FeedbackEntry = {
  thumb?: FeedbackThumb
  thumbSubmittedAt?: number | null
  detail?: FeedbackDetail
  detailIdempotencyKeys?: {
    report?: string
    relevance?: string
  }
  sourceFeedback?: Record<string, SourceFeedbackEntry>
}

type FeedbackState = {
  entries: Record<string, FeedbackEntry>
  setThumb: (messageKey: string, thumb: FeedbackThumb) => void
  setThumbSubmittedAt: (messageKey: string, submittedAt: number) => void
  setDetail: (messageKey: string, detail: FeedbackDetail) => void
  setDetailSubmittedAt: (messageKey: string, submittedAt: number) => void
  setDetailIdempotencyKey: (
    messageKey: string,
    kind: "report" | "relevance",
    key: string
  ) => void
  clearDetailIdempotencyKey: (
    messageKey: string,
    kind: "report" | "relevance"
  ) => void
  setSourceThumb: (
    messageKey: string,
    sourceKey: string,
    thumb: FeedbackThumb
  ) => void
  setSourceSubmittedAt: (
    messageKey: string,
    sourceKey: string,
    submittedAt: number
  ) => void
}

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

const updateEntry = (
  entries: Record<string, FeedbackEntry>,
  key: string,
  updater: (entry: FeedbackEntry) => FeedbackEntry
) => {
  const current = entries[key] || {}
  return {
    ...entries,
    [key]: updater(current)
  }
}

export const useFeedbackStore = createWithEqualityFn<FeedbackState>()(
  persist(
    (set) => ({
      entries: {},
      setThumb: (messageKey, thumb) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            thumb
          }))
        })),
      setThumbSubmittedAt: (messageKey, submittedAt) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            thumbSubmittedAt: submittedAt
          }))
        })),
      setDetail: (messageKey, detail) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            detail: {
              ...(entry.detail || {}),
              ...detail
            }
          }))
        })),
      setDetailSubmittedAt: (messageKey, submittedAt) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            detail: {
              ...(entry.detail || {}),
              submittedAt
            }
          }))
        })),
      setDetailIdempotencyKey: (messageKey, kind, key) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            detailIdempotencyKeys: {
              ...(entry.detailIdempotencyKeys || {}),
              [kind]: key
            }
          }))
        })),
      clearDetailIdempotencyKey: (messageKey, kind) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => {
            const nextKeys = { ...(entry.detailIdempotencyKeys || {}) }
            delete nextKeys[kind]
            return {
              ...entry,
              detailIdempotencyKeys: nextKeys
            }
          })
        })),
      setSourceThumb: (messageKey, sourceKey, thumb) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            sourceFeedback: {
              ...(entry.sourceFeedback || {}),
              [sourceKey]: {
                ...(entry.sourceFeedback?.[sourceKey] || {}),
                thumb
              }
            }
          }))
        })),
      setSourceSubmittedAt: (messageKey, sourceKey, submittedAt) =>
        set((state) => ({
          entries: updateEntry(state.entries, messageKey, (entry) => ({
            ...entry,
            sourceFeedback: {
              ...(entry.sourceFeedback || {}),
              [sourceKey]: {
                ...(entry.sourceFeedback?.[sourceKey] || {}),
                submittedAt
              }
            }
          }))
        }))
    }),
    {
      name: "tldw-feedback-store",
      // Baseline version so future shape changes can migrate instead of discarding
      // persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
      version: 1,
      migrate: (persisted) => persisted as any,
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? localStorage : createMemoryStorage()
      ),
      partialize: (state) => ({ entries: state.entries })
    }
  )
)

if (
  typeof window !== "undefined" &&
  process.env.NODE_ENV !== "production"
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useFeedbackStore = useFeedbackStore
}
