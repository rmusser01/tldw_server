import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"
import type { QueuedRequest } from "@/utils/chat-request-queue"
import type { AssistantSelection } from "@/types/assistant-selection"

const STORAGE_KEY = "tldw-playground-session"
const STALE_THRESHOLD_MS = 24 * 60 * 60 * 1000 // 24 hours

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

export interface PlaygroundSessionData {
  // Core identifier (used to restore messages from Dexie)
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
  // Actions
  saveSession: (data: Partial<PlaygroundSessionData>) => void
  clearSession: () => void
  isSessionStale: () => boolean
  isSessionValid: (expectedScopeKey?: string | null) => boolean
}

const initialState: PlaygroundSessionData = {
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

      saveSession: (data) =>
        set((state) => ({
          ...state,
          ...data,
          lastUpdated: Date.now()
        })),

      clearSession: () => set({ ...initialState, lastUpdated: 0 }),

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
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? localStorage : createMemoryStorage()
      ),
      partialize: (state) => ({
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

if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_usePlaygroundSessionStore = usePlaygroundSessionStore
}
