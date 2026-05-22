import { useCallback, useEffect, useRef, useState } from "react"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { restoreQueuedRequests } from "@/utils/chat-request-queue"
import {
  formatToChatHistory,
  formatToMessage,
  getFullChatData,
  getPromptById
} from "@/db/dexie/helpers"
import { useStoreChatModelSettings } from "@/store/model"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { useConnectionState } from "@/hooks/useConnectionState"

const DEBOUNCE_MS = 1000

/**
 * Hook to persist and restore playground session state.
 *
 * - Automatically saves session state (debounced) when relevant state changes
 * - Provides restoreSession() to restore from persisted state on mount
 * - Clears session when user starts a new chat
 */
export function usePlaygroundSessionPersistence() {
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isRestoringRef = useRef(false)
  const { serverUrl, lastConfigUpdatedAt } = useConnectionState()

  // Session store
  const sessionStore = usePlaygroundSessionStore()
  const saveSession = usePlaygroundSessionStore((s) => s.saveSession)
  const clearSession = usePlaygroundSessionStore((s) => s.clearSession)
  const isSessionValid = usePlaygroundSessionStore((s) => s.isSessionValid)
  const [currentScopeKey, setCurrentScopeKey] = useState<string | null>(null)
  const [sessionScopeReady, setSessionScopeReady] = useState(false)

  // Main message option store
  const {
    historyId,
    serverChatId,
    chatMode,
    webSearch,
    compareMode,
    compareSelectedModels,
    ragMediaIds,
    ragSearchMode,
    ragTopK,
    ragEnableGeneration,
    ragEnableCitations,
    queuedMessages,
    temporaryChat,
    setHistoryId,
    setServerChatId,
    setChatMode,
    setWebSearch,
    setCompareMode,
    setCompareSelectedModels,
    setRagMediaIds,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setQueuedMessages,
    setHistory,
    setMessages,
    setSelectedSystemPrompt
  } = useStoreMessageOption(
    (state) => ({
      historyId: state.historyId,
      serverChatId: state.serverChatId,
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
      temporaryChat: state.temporaryChat,
      setHistoryId: state.setHistoryId,
      setServerChatId: state.setServerChatId,
      setChatMode: state.setChatMode,
      setWebSearch: state.setWebSearch,
      setCompareMode: state.setCompareMode,
      setCompareSelectedModels: state.setCompareSelectedModels,
      setRagMediaIds: state.setRagMediaIds,
      setRagSearchMode: state.setRagSearchMode,
      setRagTopK: state.setRagTopK,
      setRagEnableGeneration: state.setRagEnableGeneration,
      setRagEnableCitations: state.setRagEnableCitations,
      setQueuedMessages: state.setQueuedMessages,
      setHistory: state.setHistory,
      setMessages: state.setMessages,
      setSelectedSystemPrompt: state.setSelectedSystemPrompt
    }),
    shallow
  )

  const { setSystemPrompt } = useStoreChatModelSettings()

  const resolveCurrentScopeKey = useCallback(async (): Promise<string> => {
    const config = await tldwClient.getConfig().catch(() => null)
    return buildChatSurfaceScopeKeyFromConfig(config)
  }, [])

  useEffect(() => {
    let cancelled = false
    setSessionScopeReady(false)

    const syncScope = async () => {
      const nextScopeKey = await resolveCurrentScopeKey()
      if (cancelled) return
      setCurrentScopeKey(nextScopeKey)
      setSessionScopeReady(true)
    }

    void syncScope()

    return () => {
      cancelled = true
    }
  }, [lastConfigUpdatedAt, resolveCurrentScopeKey, serverUrl])

  const buildPersistableSessionSnapshot = useCallback(() => {
    // Don't save while a restore is replaying into the stores.
    if (isRestoringRef.current) return null

    // Allow queue-only restores even before a history/server chat id exists.
    if (temporaryChat && queuedMessages.length === 0) return null
    if (!historyId && !serverChatId && queuedMessages.length === 0) return null

    return {
      historyId,
      serverChatId,
      chatMode,
      webSearch,
      compareMode,
      compareSelectedModels,
      ragMediaIds,
      ragSearchMode,
      ragTopK,
      ragEnableGeneration,
      ragEnableCitations,
      queuedMessages
    }
  }, [
    historyId,
    serverChatId,
    chatMode,
    webSearch,
    compareMode,
    compareSelectedModels,
    ragMediaIds,
    ragSearchMode,
    ragTopK,
    ragEnableGeneration,
    ragEnableCitations,
    queuedMessages,
    temporaryChat
  ])

  const latestSessionSnapshotRef = useRef<ReturnType<
    typeof buildPersistableSessionSnapshot
  >>(null)

  useEffect(() => {
    latestSessionSnapshotRef.current = buildPersistableSessionSnapshot()
  }, [buildPersistableSessionSnapshot])

  // Debounced save
  const saveCurrentSession = useCallback(() => {
    const snapshot = buildPersistableSessionSnapshot()
    if (!snapshot) return

    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }

    saveTimerRef.current = setTimeout(() => {
      saveTimerRef.current = null
      void resolveCurrentScopeKey().then((scopeKey) => {
        saveSession({
          ...snapshot,
          scopeKey
        })
      })
    }, DEBOUNCE_MS)
  }, [buildPersistableSessionSnapshot, resolveCurrentScopeKey, saveSession])

  const flushPendingSessionSave = useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    // Prefer a fresh snapshot over the ref to avoid saving stale state on unmount.
    const snapshot = buildPersistableSessionSnapshot() ?? latestSessionSnapshotRef.current
    if (!snapshot) return
    void resolveCurrentScopeKey().then((scopeKey) => {
      saveSession({
        ...snapshot,
        scopeKey
      })
    })
  }, [buildPersistableSessionSnapshot, resolveCurrentScopeKey, saveSession])

  const flushPendingSessionSaveRef = useRef(flushPendingSessionSave)

  useEffect(() => {
    flushPendingSessionSaveRef.current = flushPendingSessionSave
  }, [flushPendingSessionSave])

  // Auto-save when state changes
  useEffect(() => {
    saveCurrentSession()
    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
        saveTimerRef.current = null
      }
    }
  }, [saveCurrentSession])

  // Flush the latest session state when leaving the chat page.
  useEffect(() => {
    return () => {
      flushPendingSessionSaveRef.current()
    }
  }, [])

  // Restore session from persisted state
  const restoreSession = useCallback(async (): Promise<boolean> => {
    const scopeKey = await resolveCurrentScopeKey()
    if (!isSessionValid(scopeKey)) {
      clearSession()
      return false
    }

    const savedHistoryId = sessionStore.historyId
    const savedServerChatId = sessionStore.serverChatId
    const savedQueue = sessionStore.queuedMessages ?? []
    if (!savedHistoryId && !savedServerChatId && savedQueue.length === 0) {
      return false
    }

    isRestoringRef.current = true

    try {
      if (savedHistoryId) {
        // Restore messages from Dexie
        const chatData = await getFullChatData(savedHistoryId)
        if (!chatData) {
          // History was deleted, clear session
          clearSession()
          return false
        }

        // Restore messages and history
        setHistoryId(savedHistoryId)
        setHistory(formatToChatHistory(chatData.messages))
        setMessages(formatToMessage(chatData.messages))

        // Restore system prompt if present
        const lastUsedPrompt = (chatData.historyInfo as any)?.last_used_prompt
        if (lastUsedPrompt?.prompt_id) {
          const prompt = await getPromptById(lastUsedPrompt.prompt_id)
          if (prompt) {
            setSelectedSystemPrompt(lastUsedPrompt.prompt_id)
            setSystemPrompt(prompt.content)
          }
        } else if (lastUsedPrompt?.prompt_content) {
          setSystemPrompt(lastUsedPrompt.prompt_content)
        }
      }

      // Restore settings from session store
      if (savedServerChatId) {
        if (!savedHistoryId) {
          setHistoryId(null)
          setHistory([])
          setMessages([])
        }
        setServerChatId(savedServerChatId)
      }
      setChatMode(sessionStore.chatMode)
      setWebSearch(sessionStore.webSearch)
      setCompareMode(sessionStore.compareMode)
      if (sessionStore.compareSelectedModels.length > 0) {
        setCompareSelectedModels(sessionStore.compareSelectedModels)
      }

      // Restore RAG settings
      if (sessionStore.ragMediaIds) {
        setRagMediaIds(sessionStore.ragMediaIds)
      }
      setRagSearchMode(sessionStore.ragSearchMode)
      if (sessionStore.ragTopK !== null) {
        setRagTopK(sessionStore.ragTopK)
      }
      setRagEnableGeneration(sessionStore.ragEnableGeneration)
      setRagEnableCitations(sessionStore.ragEnableCitations)
      setQueuedMessages(restoreQueuedRequests(savedQueue))

      return true
    } catch (error) {
      console.warn("Failed to restore session:", error)
      clearSession()
      return false
    } finally {
      isRestoringRef.current = false
    }
  }, [
    isSessionValid,
    sessionStore,
    clearSession,
    resolveCurrentScopeKey,
    setHistoryId,
    setServerChatId,
    setHistory,
    setMessages,
    setSelectedSystemPrompt,
    setSystemPrompt,
    setChatMode,
    setWebSearch,
    setCompareMode,
    setCompareSelectedModels,
    setRagMediaIds,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setQueuedMessages
  ])

  // Clear persisted session (call when user starts new chat)
  const clearPersistedSession = useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    clearSession()
  }, [clearSession])

  return {
    restoreSession,
    clearPersistedSession,
    sessionScopeReady,
    hasPersistedSession:
      sessionScopeReady && isSessionValid(currentScopeKey),
    persistedHistoryId:
      sessionScopeReady && isSessionValid(currentScopeKey)
        ? sessionStore.historyId ?? null
        : null,
    persistedServerChatId:
      sessionScopeReady && isSessionValid(currentScopeKey)
        ? sessionStore.serverChatId ?? null
        : null
  }
}
