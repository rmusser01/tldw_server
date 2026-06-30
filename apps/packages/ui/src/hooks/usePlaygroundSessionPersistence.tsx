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
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import {
  effectiveAssistantStateToSelection,
  resolveEffectiveAssistantState
} from "@/hooks/chat/effective-assistant-state"
import {
  characterToAssistantSelection,
  getAssistantSelectionMode,
  normalizeAssistantSelection
} from "@/types/assistant-selection"
import {
  SELECTED_ASSISTANT_STORAGE_KEY,
  parseSelectedAssistantValue,
  selectedAssistantStorage
} from "@/utils/selected-assistant-storage"
import {
  SELECTED_CHARACTER_STORAGE_KEY,
  parseSelectedCharacterValue,
  selectedCharacterStorage
} from "@/utils/selected-character-storage"

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
  const initialRestoreSettledRef = useRef(false)
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
    serverChatAssistantKind,
    serverChatAssistantId,
    serverChatCharacterId,
    serverChatPersonaMemoryMode,
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
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    setServerChatMetaLoaded,
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
      serverChatAssistantKind: state.serverChatAssistantKind,
      serverChatAssistantId: state.serverChatAssistantId,
      serverChatCharacterId: state.serverChatCharacterId,
      serverChatPersonaMemoryMode: state.serverChatPersonaMemoryMode,
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
      setServerChatCharacterId: state.setServerChatCharacterId,
      setServerChatAssistantKind: state.setServerChatAssistantKind,
      setServerChatAssistantId: state.setServerChatAssistantId,
      setServerChatPersonaMemoryMode: state.setServerChatPersonaMemoryMode,
      setServerChatMetaLoaded: state.setServerChatMetaLoaded,
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
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)

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

    const trackedAssistantSelection =
      serverChatId == null
        ? null
        : (() => {
            const effectiveAssistantState = resolveEffectiveAssistantState({
              tracked: {
                assistantKind: serverChatAssistantKind,
                assistantId: serverChatAssistantId,
                characterId: serverChatCharacterId
              },
              draftSelection: selectedAssistant
            })
            const resolvedSelection =
              effectiveAssistantStateToSelection(effectiveAssistantState)
            return getAssistantSelectionMode(resolvedSelection) === "tracked"
              ? resolvedSelection
              : null
          })()
    const trackedAssistantKind =
      trackedAssistantSelection?.kind ?? serverChatAssistantKind ?? null
    const trackedAssistantId =
      trackedAssistantSelection?.id ??
      (serverChatAssistantId != null ? String(serverChatAssistantId) : null)
    const trackedCharacterId =
      trackedAssistantSelection?.kind === "character"
        ? trackedAssistantSelection.id
        : serverChatCharacterId != null
          ? String(serverChatCharacterId)
          : null
    const trackedAssistantDisplayName =
      trackedAssistantSelection?.name ??
      (serverChatId != null &&
      trackedAssistantKind != null &&
      trackedAssistantId != null
        ? resolveEffectiveAssistantState({
            tracked: {
              assistantKind: trackedAssistantKind,
              assistantId: trackedAssistantId,
              characterId: trackedCharacterId
            },
            draftSelection: selectedAssistant
          }).displayName
        : null) ??
      null
    const trackedAssistantAvatarUrl =
      trackedAssistantSelection?.avatar_url ??
      (serverChatId != null &&
      trackedAssistantKind != null &&
      trackedAssistantId != null
        ? resolveEffectiveAssistantState({
            tracked: {
              assistantKind: trackedAssistantKind,
              assistantId: trackedAssistantId,
              characterId: trackedCharacterId
            },
            draftSelection: selectedAssistant
          }).avatarUrl
        : null) ??
      null

    return {
      historyId,
      serverChatId,
      trackedAssistantSelection,
      trackedAssistantKind,
      trackedAssistantId,
      trackedCharacterId,
      trackedAssistantDisplayName,
      trackedAssistantAvatarUrl,
      serverChatPersonaMemoryMode,
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
    serverChatAssistantKind,
    serverChatAssistantId,
    serverChatCharacterId,
    serverChatPersonaMemoryMode,
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
    selectedAssistant
  ])

  const latestSessionSnapshotRef = useRef<ReturnType<
    typeof buildPersistableSessionSnapshot
  >>(null)
  const lastImmediateSaveKeyRef = useRef<string | null>(null)

  const enrichTrackedAssistantSnapshot = useCallback(
    async (
      snapshot: ReturnType<typeof buildPersistableSessionSnapshot>
    ): Promise<ReturnType<typeof buildPersistableSessionSnapshot>> => {
      if (
        !snapshot ||
        !snapshot.serverChatId ||
        !snapshot.trackedAssistantKind ||
        !snapshot.trackedAssistantId
      ) {
        return snapshot
      }

      const displayNameNeedsEnrichment =
        !snapshot.trackedAssistantDisplayName ||
        snapshot.trackedAssistantDisplayName === "Assistant" ||
        snapshot.trackedAssistantDisplayName === "Persona"

      if (
        snapshot.trackedAssistantSelection &&
        getAssistantSelectionMode(snapshot.trackedAssistantSelection) === "tracked" &&
        !displayNameNeedsEnrichment
      ) {
        return snapshot
      }

      if (displayNameNeedsEnrichment) {
        const currentSession = usePlaygroundSessionStore.getState()
        const storedSelection = normalizeAssistantSelection(
          currentSession.trackedAssistantSelection
        )
        const storedKind =
          storedSelection?.kind ?? currentSession.trackedAssistantKind
        const storedId =
          storedSelection?.id ?? currentSession.trackedAssistantId
        const storedDisplayName =
          storedSelection?.name ?? currentSession.trackedAssistantDisplayName
        const storedAvatarUrl =
          storedSelection?.avatar_url ??
          currentSession.trackedAssistantAvatarUrl

        if (
          storedKind === snapshot.trackedAssistantKind &&
          storedId === snapshot.trackedAssistantId &&
          storedDisplayName &&
          storedDisplayName !== "Assistant" &&
          storedDisplayName !== "Persona"
        ) {
          return {
            ...snapshot,
            trackedAssistantSelection:
              storedSelection ?? snapshot.trackedAssistantSelection,
            trackedAssistantDisplayName: storedDisplayName,
            trackedAssistantAvatarUrl:
              storedAvatarUrl ?? snapshot.trackedAssistantAvatarUrl
          }
        }
      }

      try {
        const storedAssistant = normalizeAssistantSelection(
          parseSelectedAssistantValue(
            await selectedAssistantStorage.get<unknown>(
              SELECTED_ASSISTANT_STORAGE_KEY
            )
          )
        )
        if (
          storedAssistant &&
          getAssistantSelectionMode(storedAssistant) === "tracked" &&
          storedAssistant.kind === snapshot.trackedAssistantKind &&
          storedAssistant.id === snapshot.trackedAssistantId
        ) {
          return {
            ...snapshot,
            trackedAssistantSelection: storedAssistant,
            trackedAssistantDisplayName:
              storedAssistant.name ?? snapshot.trackedAssistantDisplayName,
            trackedAssistantAvatarUrl:
              storedAssistant.avatar_url ?? snapshot.trackedAssistantAvatarUrl
          }
        }
      } catch {
        // ignore tracked assistant storage hydration failures during session save
      }

      if (snapshot.trackedAssistantKind === "character") {
        try {
          const legacyCharacterSelection = characterToAssistantSelection(
            parseSelectedCharacterValue<Record<string, unknown>>(
              await selectedCharacterStorage.get<unknown>(
                SELECTED_CHARACTER_STORAGE_KEY
              )
            )
          )
          if (
            legacyCharacterSelection &&
            legacyCharacterSelection.id === snapshot.trackedAssistantId
          ) {
            return {
              ...snapshot,
              trackedAssistantSelection: legacyCharacterSelection,
              trackedAssistantDisplayName:
                legacyCharacterSelection.name ?? snapshot.trackedAssistantDisplayName,
              trackedAssistantAvatarUrl:
                legacyCharacterSelection.avatar_url ??
                snapshot.trackedAssistantAvatarUrl
            }
          }
        } catch {
          // ignore legacy character storage hydration failures during session save
        }
      }

      return snapshot
    },
    []
  )

  useEffect(() => {
    latestSessionSnapshotRef.current = buildPersistableSessionSnapshot()
  }, [buildPersistableSessionSnapshot])

  useEffect(() => {
    if (!initialRestoreSettledRef.current) return
    if (!sessionScopeReady || !currentScopeKey) return

    const snapshot = buildPersistableSessionSnapshot()
    if (!snapshot) return

    const immediateSaveKey = JSON.stringify({
      historyId: snapshot.historyId,
      serverChatId: snapshot.serverChatId,
      trackedAssistantKind: snapshot.trackedAssistantKind,
      trackedAssistantId: snapshot.trackedAssistantId
    })

    if (immediateSaveKey === lastImmediateSaveKeyRef.current) {
      return
    }

    lastImmediateSaveKeyRef.current = immediateSaveKey

    void enrichTrackedAssistantSnapshot(snapshot).then((enrichedSnapshot) => {
      saveSession({
        ...enrichedSnapshot,
        scopeKey: currentScopeKey
      })
    })
  }, [
    buildPersistableSessionSnapshot,
    currentScopeKey,
    enrichTrackedAssistantSnapshot,
    saveSession,
    sessionScopeReady
  ])

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
      void Promise.all([
        resolveCurrentScopeKey(),
        enrichTrackedAssistantSnapshot(snapshot)
      ]).then(([scopeKey, enrichedSnapshot]) => {
        saveSession({
          ...enrichedSnapshot,
          scopeKey
        })
      })
    }, DEBOUNCE_MS)
  }, [
    buildPersistableSessionSnapshot,
    enrichTrackedAssistantSnapshot,
    resolveCurrentScopeKey,
    saveSession
  ])

  const flushPendingSessionSave = useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    // Prefer a fresh snapshot over the ref to avoid saving stale state on unmount.
    const snapshot = buildPersistableSessionSnapshot() ?? latestSessionSnapshotRef.current
    if (!snapshot) return
    void Promise.all([
      resolveCurrentScopeKey(),
      enrichTrackedAssistantSnapshot(snapshot)
    ]).then(([scopeKey, enrichedSnapshot]) => {
      saveSession({
        ...enrichedSnapshot,
        scopeKey
      })
    })
  }, [
    buildPersistableSessionSnapshot,
    enrichTrackedAssistantSnapshot,
    resolveCurrentScopeKey,
    saveSession
  ])

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
      initialRestoreSettledRef.current = true
      return false
    }

    const savedHistoryId = sessionStore.historyId
    const savedServerChatId = sessionStore.serverChatId
    const savedTrackedAssistantSelection =
      sessionStore.trackedAssistantSelection ?? null
    const savedTrackedAssistantKind = sessionStore.trackedAssistantKind ?? null
    const savedTrackedAssistantId = sessionStore.trackedAssistantId ?? null
    const savedTrackedCharacterId = sessionStore.trackedCharacterId ?? null
    const savedTrackedAssistantDisplayName =
      sessionStore.trackedAssistantDisplayName ?? null
    const savedTrackedAssistantAvatarUrl =
      sessionStore.trackedAssistantAvatarUrl ?? null
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
        if (
          savedTrackedAssistantSelection &&
          getAssistantSelectionMode(savedTrackedAssistantSelection) === "tracked"
        ) {
          await setSelectedAssistant(savedTrackedAssistantSelection)
        } else if (savedTrackedAssistantKind && savedTrackedAssistantId) {
          const reconstructedSelection = normalizeAssistantSelection({
            kind: savedTrackedAssistantKind,
            id: savedTrackedAssistantId,
            name:
              savedTrackedAssistantDisplayName ??
              (savedTrackedAssistantKind === "persona" ? "Persona" : "Assistant"),
            avatar_url: savedTrackedAssistantAvatarUrl,
            metadata: {
              selectionMode: "tracked"
            }
          })
          if (reconstructedSelection) {
            await setSelectedAssistant(reconstructedSelection)
          }
        }
        if (savedTrackedAssistantKind && savedTrackedAssistantId) {
          if (savedTrackedAssistantKind === "character") {
            setServerChatCharacterId(savedTrackedCharacterId ?? savedTrackedAssistantId)
            setServerChatAssistantKind("character")
            setServerChatAssistantId(savedTrackedAssistantId)
            setServerChatPersonaMemoryMode(null)
          } else {
            setServerChatCharacterId(null)
            setServerChatAssistantKind("persona")
            setServerChatAssistantId(savedTrackedAssistantId)
            setServerChatPersonaMemoryMode(
              sessionStore.serverChatPersonaMemoryMode ?? "read_only"
            )
          }
          setServerChatMetaLoaded(true)
        }
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
      initialRestoreSettledRef.current = true
    }
  }, [
    isSessionValid,
    sessionStore,
    clearSession,
    resolveCurrentScopeKey,
    setHistoryId,
    setServerChatId,
    setServerChatAssistantId,
    setServerChatAssistantKind,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    setServerChatPersonaMemoryMode,
    setHistory,
    setMessages,
    setSelectedAssistant,
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
    initialRestoreSettledRef.current = true
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
