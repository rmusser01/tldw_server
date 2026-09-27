import type { HistorySelectionReference } from "@/hooks/chat/useHistorySelection"
import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"
import type { QueuedRequest } from "@/utils/chat-request-queue"
import type { AssistantSelection } from "@/types/assistant-selection"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

const STORAGE_KEY = "tldw-playground-session"
const STALE_THRESHOLD_MS = 24 * 60 * 60 * 1000 // 24 hours
type SyncStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">

const createMemoryStorage = (): SyncStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

const bestEffortStorage = (getStorage: () => Storage): SyncStorage => {
  let storage: Storage
  try {
    storage = getStorage()
  } catch {
    return createMemoryStorage()
  }
  return {
    getItem: (key) => {
      try { return storage.getItem(key) } catch { return null }
    },
    setItem: (key, value) => {
      try {
        storage.setItem(key, value)
      } catch {
        // A rejected replacement must not leave an older restore target pinned.
        try { storage.removeItem(key) } catch { /* Storage may be blocked entirely. */ }
      }
    },
    removeItem: (key) => {
      try { storage.removeItem(key) } catch { /* Preserve the other backing store. */ }
    }
  }
}

const createBrowserStorage = (): StateStorage => {
  const tabStorage = bestEffortStorage(() => sessionStorage)
  const durableStorage = bestEffortStorage(() => localStorage)
  return {
    getItem: (key) => {
      const current = tabStorage.getItem(key)
      if (current !== null) return current
      // New tabs keep last-session recovery, then own their restore target.
      const previous = durableStorage.getItem(key)
      if (previous !== null) tabStorage.setItem(key, previous)
      return previous
    },
    setItem: (key, value) => {
      tabStorage.setItem(key, value)
      durableStorage.setItem(key, value)
    },
    removeItem: (key) => {
      tabStorage.removeItem(key)
      durableStorage.removeItem(key)
    }
  }
}

export interface PlaygroundSessionData {
  // Core identifier (used to restore messages from Dexie)
  historySelectionReference?: HistorySelectionReference | null
  historyId: string | null
  serverChatId: string | null
  trackedAssistantSelection: AssistantSelection | null
  trackedAssistantKind: "character" | "persona" | null
  trackedAssistantId: string | null
  trackedCharacterId: string | null
  trackedAssistantDisplayName: string | null
  trackedAssistantAvatarUrl: string | null
  serverChatPersonaMemoryMode: "read_only" | "read_write" | null
  scopeKey: string | null

  // Settings NOT already persisted elsewhere (selectedModel uses useStorage)
  chatMode: "normal" | "rag" | "vision"
  webSearch: boolean
  compareMode: boolean
  compareSelectedModels: string[]

  // RAG settings (when chatMode === "rag")
  fileRetrievalEnabled: boolean
  ragMediaIds: number[] | null
  ragSearchMode: "hybrid" | "vector" | "fts"
  ragTopK: number | null
  ragEnableGeneration: boolean
  ragEnableCitations: boolean
  queuedMessages: QueuedRequest[]

  // Metadata
  lastUpdated: number
}

interface PlaygroundSessionState extends PlaygroundSessionData {
  restoreRevision: number
  sourceSelectionRevision: number
  // Actions
  saveSession: (data: Partial<PlaygroundSessionData>) => void
  clearSession: () => void
  cancelPendingRestore: () => void
  markSourceSelectionIntent: () => void
  isSessionStale: () => boolean
  isSessionValid: (expectedScopeKey?: string | null) => boolean
}

const initialState: PlaygroundSessionData = {
  historySelectionReference: null,
  historyId: null,
  serverChatId: null,
  trackedAssistantSelection: null,
  trackedAssistantKind: null,
  trackedAssistantId: null,
  trackedCharacterId: null,
  trackedAssistantDisplayName: null,
  trackedAssistantAvatarUrl: null,
  serverChatPersonaMemoryMode: null,
  scopeKey: null,
  chatMode: "normal",
  webSearch: false,
  compareMode: false,
  compareSelectedModels: [],
  fileRetrievalEnabled: false,
  ragMediaIds: null,
  ragSearchMode: "hybrid",
  ragTopK: null,
  ragEnableGeneration: true,
  ragEnableCitations: true,
  queuedMessages: [],
  lastUpdated: 0
}

export const usePlaygroundSessionStore = createWithEqualityFn<PlaygroundSessionState>()(
  persist(
    (set, get) => ({
      ...initialState,
      restoreRevision: 0,
      sourceSelectionRevision: 0,

      saveSession: (data) =>
        set((state) => ({
          ...state,
          ...data,
          lastUpdated: Date.now()
        })),

      clearSession: () =>
        set((state) => ({
          ...initialState,
          lastUpdated: 0,
          restoreRevision: state.restoreRevision + 1
        })),

      cancelPendingRestore: () =>
        set((state) => ({ restoreRevision: state.restoreRevision + 1 })),

      markSourceSelectionIntent: () =>
        set((state) => ({ sourceSelectionRevision: state.sourceSelectionRevision + 1 })),

      isSessionStale: () => {
        const { lastUpdated } = get()
        if (lastUpdated === 0) return true
        return Date.now() - lastUpdated > STALE_THRESHOLD_MS
      },

      isSessionValid: (expectedScopeKey) => {
        const {
          historyId,
          serverChatId,
          scopeKey,
          queuedMessages,
          lastUpdated
        } = get()
        // Session is valid if we have a conversation or queued work and it's not stale.
        const hasConversationOrQueue =
          historyId !== null ||
          serverChatId !== null ||
          queuedMessages.length > 0
        const isNotStale = lastUpdated > 0 && Date.now() - lastUpdated <= STALE_THRESHOLD_MS
        const matchesScope =
          typeof expectedScopeKey === "undefined" || expectedScopeKey === null
            ? true
            : scopeKey === expectedScopeKey
        return hasConversationOrQueue && isNotStale && matchesScope
      }
    }),
    {
      name: STORAGE_KEY,
      // Baseline version so future shape changes can migrate instead of discarding
      // persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
      version: 1,
      migrate: (persisted) => persisted as any,
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? createBrowserStorage() : createMemoryStorage()
      ),
      partialize: (state) => ({
        historySelectionReference: state.historySelectionReference,
        historyId: state.historyId,
        serverChatId: state.serverChatId,
        trackedAssistantSelection: state.trackedAssistantSelection,
        trackedAssistantKind: state.trackedAssistantKind,
        trackedAssistantId: state.trackedAssistantId,
        trackedCharacterId: state.trackedCharacterId,
        trackedAssistantDisplayName: state.trackedAssistantDisplayName,
        trackedAssistantAvatarUrl: state.trackedAssistantAvatarUrl,
        serverChatPersonaMemoryMode: state.serverChatPersonaMemoryMode,
        scopeKey: state.scopeKey,
        chatMode: state.chatMode,
        webSearch: state.webSearch,
        compareMode: state.compareMode,
        compareSelectedModels: state.compareSelectedModels,
        fileRetrievalEnabled: state.fileRetrievalEnabled,
        ragMediaIds: state.ragMediaIds,
        ragSearchMode: state.ragSearchMode,
        ragTopK: state.ragTopK,
        ragEnableGeneration: state.ragEnableGeneration,
        ragEnableCitations: state.ragEnableCitations,
        queuedMessages: state.queuedMessages,
        lastUpdated: state.lastUpdated
      })
    }
  )
)

const stopWatchingAccount = watchChatAccountChanges((invalidated) => {
  if (invalidated) usePlaygroundSessionStore.getState().clearSession()
})
const hot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
hot?.dispose(stopWatchingAccount)

if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_usePlaygroundSessionStore = usePlaygroundSessionStore
}
