import { restoreReadableLocalComparison } from "@/hooks/useLoadLocalConversation"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { chatDraftOwnerKey, useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { captureChatDraftLoad, chatDraftAddressForReceipt } from "@/hooks/chat/chat-draft-transition"
import { chatDraftAddressKey, isChatDraftHandle } from "@/utils/chat-draft-address"
import { hasCurrentServerChatSelection, usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { restoreQueuedRequests } from "@/utils/chat-request-queue"
import {
  formatToChatHistory,
  formatToMessage,
  getFullChatData,
  getSessionFiles,
  getPromptById
} from "@/db/dexie/helpers"
import { useStoreChatModelSettings } from "@/store/model"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { useConnectionState } from "@/hooks/useConnectionState"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import {
  effectiveAssistantStateToSelection,
  resolveEffectiveAssistantState
} from "@/hooks/chat/effective-assistant-state"
import {
  getAssistantSelectionMode,
  normalizeAssistantSelection
} from "@/types/assistant-selection"

const DEBOUNCE_MS = 1000

const selectSessionOptions = (state: ReturnType<typeof useStoreMessageOption.getState>) => ({
  chatDraftAddress: state.chatDraftAddress,
  chatDraftRevision: state.chatDraftRevision,
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
  fileRetrievalEnabled: state.fileRetrievalEnabled,
  ragMediaIds: state.ragMediaIds,
  ragSearchMode: state.ragSearchMode,
  ragTopK: state.ragTopK,
  ragEnableGeneration: state.ragEnableGeneration,
  ragEnableCitations: state.ragEnableCitations,
  queuedMessages: state.queuedMessages,
  temporaryChat: state.temporaryChat,
  temporaryHistory: state.temporaryChat ? state.history : undefined,
  temporaryMessages: state.temporaryChat ? state.messages : undefined,
  setHistoryId: state.setHistoryId,
  setServerChatId: state.setServerChatId,
  setServerChatTitle: state.setServerChatTitle,
  setServerChatCharacterId: state.setServerChatCharacterId,
  setServerChatAssistantKind: state.setServerChatAssistantKind,
  setServerChatAssistantId: state.setServerChatAssistantId,
  setServerChatPersonaMemoryMode: state.setServerChatPersonaMemoryMode,
  setServerChatMetaLoaded: state.setServerChatMetaLoaded,
  setChatMode: state.setChatMode,
  setWebSearch: state.setWebSearch,
  setCompareMode: state.setCompareMode,
  setCompareSelectedModels: state.setCompareSelectedModels,
  setFileRetrievalEnabled: state.setFileRetrievalEnabled,
  setRagMediaIds: state.setRagMediaIds,
  setRagSearchMode: state.setRagSearchMode,
  setRagTopK: state.setRagTopK,
  setRagEnableGeneration: state.setRagEnableGeneration,
  setRagEnableCitations: state.setRagEnableCitations,
  setQueuedMessages: state.setQueuedMessages,
  setHistory: state.setHistory,
  setMessages: state.setMessages,
  setSelectedSystemPrompt: state.setSelectedSystemPrompt
})

export type PlaygroundSessionRestoreOutcome =
  | "restored"
  | "not-restored"
  | "cancelled"

/**
 * Hook to persist and restore playground session state.
 *
 * - Automatically saves session state (debounced) when relevant state changes
 * - Provides restoreSession() to restore from persisted state on mount
 * - Clears session when user starts a new chat
 */
export function usePlaygroundSessionPersistence() {
  const historySelection = useHistorySelectionContext()
  const selectionRef = useRef(historySelection)
  selectionRef.current = historySelection
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isRestoringRef = useRef<{ isCurrent: () => boolean } | null>(null)
  const initialRestoreSettledRef = useRef(false)
  const { serverUrl, lastConfigUpdatedAt } = useConnectionState()

  // Session store
  const sessionStore = usePlaygroundSessionStore()
  const { ownerKey: draftOwnerKey, isCurrent: isCurrentDraftOwner } = useChatDraftOwner(() => {})
  const saveSession = usePlaygroundSessionStore((s) => s.saveSession)
  const clearSession = usePlaygroundSessionStore((s) => s.clearSession)
  const isSessionValid = usePlaygroundSessionStore((s) => s.isSessionValid)
  const [currentScopeKey, setCurrentScopeKey] = useState<string | null>(null)
  const [connectionScopeReady, setSessionScopeReady] = useState(false)
  const persistenceRevision = sessionStore.restoreRevision
  const isCurrentPersistence = useCallback(
    () => usePlaygroundSessionStore.getState().restoreRevision === persistenceRevision,
    [persistenceRevision]
  )

  // Main message option store
  const {
    chatDraftAddress,
    chatDraftRevision,
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
    fileRetrievalEnabled,
    ragMediaIds,
    ragSearchMode,
    ragTopK,
    ragEnableGeneration,
    ragEnableCitations,
    queuedMessages,
    temporaryChat,
    temporaryHistory,
    temporaryMessages,
    setHistoryId,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    setServerChatMetaLoaded,
    setChatMode,
    setWebSearch,
    setCompareMode,
    setCompareSelectedModels,
    setFileRetrievalEnabled,
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
    selectSessionOptions,
    shallow
  )

  const { setSystemPrompt } = useStoreChatModelSettings()
  const [selectedAssistant, setSelectedAssistant, assistantMeta] = useSelectedAssistant(null)
  const needsDraftOwner = chatDraftAddress !== null || isChatDraftHandle(sessionStore.chatDraftHandle)
  const sessionScopeReady = connectionScopeReady && !assistantMeta?.isLoading &&
    (!needsDraftOwner || draftOwnerKey !== null)
  const readOwnedSelection = assistantMeta?.readSelection
  // Form handoffs can arrive before initial hydration starts or while its DB
  // read is pending, even with identical values. Observe accepted intent, not
  // value equality; this baseline is used only for the initial saved target.
  const initialSourceSelectionRef = useRef({
    restoreRevision: sessionStore.restoreRevision,
    revision: sessionStore.sourceSelectionRevision,
    scopeKey: sessionStore.scopeKey,
    historyId: sessionStore.historyId,
    serverChatId: sessionStore.serverChatId
  })

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

  const hasInitialRestoreTarget = useCallback(() => {
    if (hasCurrentServerChatSelection(useStoreMessageOption.getState().serverChatId)) return false
    if (isSessionValid(currentScopeKey)) return true
    try { return Boolean(selectionRef.current?.getStoredReference()) }
    catch { return true }
  }, [currentScopeKey, isSessionValid])

  const buildPersistableSessionSnapshot = useCallback(() => {
    if (!isCurrentPersistence()) return null
    // Don't save while a restore is replaying into the stores.
    if (isRestoringRef.current?.isCurrent()) return null
    if (!initialRestoreSettledRef.current && hasInitialRestoreTarget()) return null

    const live = useStoreMessageOption.getState()
    if (live.chatDraftAddress !== chatDraftAddress || live.chatDraftRevision !== chatDraftRevision ||
        live.historyId !== historyId || live.serverChatId !== serverChatId) return null
    let chatDraftHandle = null
    if (chatDraftAddress) {
      if (!draftOwnerKey || !isCurrentDraftOwner()) return null
      // The composer's automatic fresh allocation must not replace a cold restore target.
      if (chatDraftAddress.kind === "fresh") {
        if (
          (historyId !== null && !(temporaryChat && historyId === "temp")) ||
          serverChatId ||
          chatDraftAddress.temporary !== temporaryChat ||
          (temporaryChat &&
            (live.history !== temporaryHistory || live.messages !== temporaryMessages))
        ) return null
      } else {
        const selected = historySelection?.getCurrent()
        const qualified = selected?.owner && selected.view
          ? chatDraftAddressForReceipt({ owner: selected.owner, view: selected.view }) : null
        if (temporaryChat || !qualified || chatDraftAddressKey(qualified) !== chatDraftAddressKey(chatDraftAddress)) return null
        if (chatDraftAddress.kind === "local"
          ? historyId !== chatDraftAddress.id || serverChatId !== null
          : serverChatId !== chatDraftAddress.id || selected?.owner?.kind !== "native" ||
            chatDraftOwnerKey(selected.owner.request_scope) !== draftOwnerKey) return null
      }
      chatDraftHandle = { address: chatDraftAddress, ownerHash: draftOwnerKey }
    } else {
      // Retain legacy queue/session compatibility until a qualified address exists.
      if (temporaryChat && queuedMessages.length === 0) return null
      if (!historyId && !serverChatId && queuedMessages.length === 0) return null
    }

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
      chatDraftHandle,
      temporaryTranscript: chatDraftAddress?.kind === "fresh" && temporaryChat
        ? { history: temporaryHistory!, messages: temporaryMessages! } : null,
      historySelectionReference: chatDraftAddress?.kind === "fresh" ? null : historySelection?.getReference() ?? null,
      // Native authority is the validated server address; a mirror is rediscovered on load.
      historyId: chatDraftAddress?.kind === "native" || temporaryChat ? null : historyId,
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
      fileRetrievalEnabled,
      ragMediaIds,
      ragSearchMode,
      ragTopK,
      ragEnableGeneration,
      ragEnableCitations,
      queuedMessages: chatDraftAddress && temporaryChat ? [] : queuedMessages,
      ...(chatDraftAddress?.kind === "fresh" ? {
        trackedAssistantSelection: null, trackedAssistantKind: null, trackedAssistantId: null,
        trackedCharacterId: null, trackedAssistantDisplayName: null, trackedAssistantAvatarUrl: null,
        serverChatPersonaMemoryMode: null
      } : {})
    }
  }, [
    chatDraftAddress,
    chatDraftRevision,
    temporaryHistory,
    temporaryMessages,
    draftOwnerKey,
    isCurrentDraftOwner,
    historySelection,
    hasInitialRestoreTarget,
    isCurrentPersistence,
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
    fileRetrievalEnabled,
    ragMediaIds,
    ragSearchMode,
    ragTopK,
    ragEnableGeneration,
    ragEnableCitations,
    queuedMessages,
    temporaryChat,
    selectedAssistant
  ])

  const lastImmediateSaveKeyRef = useRef<string | null>(null)
  const currentSnapshotBuilderRef = useRef(buildPersistableSessionSnapshot)
  useLayoutEffect(() => { currentSnapshotBuilderRef.current = buildPersistableSessionSnapshot }, [buildPersistableSessionSnapshot])

  const captureSaveFence = useCallback(() => {
    const source = useStoreMessageOption.getState()
    const inputs = selectSessionOptions(source)
    const builder = buildPersistableSessionSnapshot
    const ownerCurrent = isCurrentDraftOwner
    return () => {
      const current = useStoreMessageOption.getState()
      return isCurrentPersistence() && !isRestoringRef.current?.isCurrent() &&
        (initialRestoreSettledRef.current || !hasInitialRestoreTarget()) &&
        currentSnapshotBuilderRef.current === builder && shallow(inputs, selectSessionOptions(current)) &&
        (!source.chatDraftAddress || ownerCurrent())
    }
  }, [buildPersistableSessionSnapshot, isCurrentDraftOwner, hasInitialRestoreTarget, isCurrentPersistence])

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
        const storedAssistant = await readOwnedSelection?.()
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
        // Keep canonical metadata when owned selection storage is unavailable.
      }

      return snapshot
    },
    [readOwnedSelection]
  )

  useEffect(() => {
    if (!sessionScopeReady) return
    if (hasInitialRestoreTarget() && sessionStore.restoreRevision === initialSourceSelectionRef.current.restoreRevision) return

    // A fresh or invalid session has nothing to replay. Treat that initial
    // restore lifecycle as settled so the first server chat can be persisted
    // immediately, including when the user reloads before the debounce fires.
    initialRestoreSettledRef.current = true
  }, [hasInitialRestoreTarget, sessionScopeReady, sessionStore.restoreRevision])

  useEffect(() => {
    if (!initialRestoreSettledRef.current) return
    if (!sessionScopeReady || !currentScopeKey) return

    const snapshot = buildPersistableSessionSnapshot()
    if (!snapshot) return

    const immediateSaveKey = JSON.stringify({
      chatDraftHandle: snapshot.chatDraftHandle,
      historyId: snapshot.historyId,
      serverChatId: snapshot.serverChatId,
      trackedAssistantKind: snapshot.trackedAssistantKind,
      trackedAssistantId: snapshot.trackedAssistantId
    })

    if (immediateSaveKey === lastImmediateSaveKeyRef.current) {
      return
    }

    lastImmediateSaveKeyRef.current = immediateSaveKey
    const isCurrentSave = captureSaveFence()
    if (snapshot.chatDraftHandle && isCurrentSave()) saveSession({ ...snapshot, scopeKey: currentScopeKey })
    void enrichTrackedAssistantSnapshot(snapshot).then((enrichedSnapshot) => {
      if (!isCurrentSave() || (snapshot.chatDraftHandle && enrichedSnapshot === snapshot)) return
      saveSession({
        ...enrichedSnapshot,
        scopeKey: currentScopeKey
      })
    })
  }, [
    captureSaveFence,
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
    const isCurrentSave = captureSaveFence()

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
        if (!isCurrentSave() || (currentScopeKey && scopeKey !== currentScopeKey)) return
        saveSession({
          ...enrichedSnapshot,
          scopeKey
        })
      })
    }, DEBOUNCE_MS)
  }, [
    captureSaveFence,
    buildPersistableSessionSnapshot,
    enrichTrackedAssistantSnapshot,
    currentScopeKey,
    resolveCurrentScopeKey,
    saveSession
  ])

  const flushPendingSessionSave = useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    if (!isCurrentPersistence()) return
    const snapshot = buildPersistableSessionSnapshot()
    if (!snapshot) return
    const isCurrentSave = captureSaveFence()
    // Layout cleanup runs before the verified owner's passive lifetime ends.
    if (snapshot.chatDraftHandle && currentScopeKey && isCurrentSave()) {
      saveSession({ ...snapshot, scopeKey: currentScopeKey })
      return
    }
    void Promise.all([
      resolveCurrentScopeKey(),
      enrichTrackedAssistantSnapshot(snapshot)
    ]).then(([scopeKey, enrichedSnapshot]) => {
      if (!isCurrentSave() || (currentScopeKey && scopeKey !== currentScopeKey)) return
      saveSession({
        ...enrichedSnapshot,
        scopeKey
      })
    })
  }, [
    captureSaveFence,
    buildPersistableSessionSnapshot,
    enrichTrackedAssistantSnapshot,
    currentScopeKey,
    isCurrentPersistence,
    resolveCurrentScopeKey,
    saveSession
  ])

  const flushPendingSessionSaveRef = useRef(flushPendingSessionSave)

  useLayoutEffect(() => {
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
  useLayoutEffect(() => {
    return () => {
      flushPendingSessionSaveRef.current()
    }
  }, [])

  useEffect(() => {
    const flush = () => flushPendingSessionSaveRef.current()
    window.addEventListener("pagehide", flush)
    return () => window.removeEventListener("pagehide", flush)
  }, [])

  // Restore session from persisted state
  const restoreSessionState = useCallback(async (): Promise<PlaygroundSessionRestoreOutcome> => {
    if (hasCurrentServerChatSelection(useStoreMessageOption.getState().serverChatId)) return "cancelled"
    // Account changes require an explicit reopen, not a fresh automatic lease.
    if (selectionRef.current?.canAutomaticallyLoad?.() === false) return "cancelled"
    if (!sessionScopeReady) return "cancelled"
    selectionRef.current?.beginLoad()
    let selectionCurrent = selectionRef.current?.fence() || (() => true)
    const initialSelection = initialSourceSelectionRef.current
    const selectionBeforeRestore =
      !initialRestoreSettledRef.current &&
      initialSelection.scopeKey === sessionStore.scopeKey &&
      initialSelection.historyId === sessionStore.historyId &&
      initialSelection.serverChatId === sessionStore.serverChatId
        ? initialSelection.revision
        : usePlaygroundSessionStore.getState().sourceSelectionRevision
    const restoreRevision =
      usePlaygroundSessionStore.getState().restoreRevision
    let snapshot: ServicePromptSnapshot | undefined
    // Opening advances the controller fence; pass only the independent caller
    // lease into the owner, then capture the controller's new fence afterward.
    const scopeCurrent = () =>
      !snapshot?.scopeSignal.aborted && !snapshot?.scopeInvalidatedSignal.aborted &&
      usePlaygroundSessionStore.getState().restoreRevision === restoreRevision &&
      (!isChatDraftHandle(sessionStore.chatDraftHandle) || isCurrentDraftOwner())
    const acceptRestoredDraft = captureChatDraftLoad("switch", scopeCurrent)
    const restoreLeaseValid = acceptRestoredDraft.isCurrent
    const isCurrentRestore = () => restoreLeaseValid() && selectionCurrent()
    const scopeKey = await resolveCurrentScopeKey()
    if (!isCurrentRestore()) {
      return "cancelled"
    }
    const savedHandle = isChatDraftHandle(sessionStore.chatDraftHandle) ? sessionStore.chatDraftHandle : null
    if (savedHandle && (savedHandle.ownerHash !== draftOwnerKey || !isSessionValid(scopeKey))) return "cancelled"
    const restoreSettings = () => {
      if (usePlaygroundSessionStore.getState().sourceSelectionRevision === selectionBeforeRestore) {
        setChatMode(sessionStore.chatMode)
        setFileRetrievalEnabled(sessionStore.fileRetrievalEnabled === true)
        if (sessionStore.ragMediaIds) setRagMediaIds(sessionStore.ragMediaIds)
      }
      setWebSearch(sessionStore.webSearch)
      const restoredOwner = selectionRef.current?.getCurrent().owner
      if (restoredOwner?.kind !== "local" || restoredOwner.conversation_id !== sessionStore.historyId) {
        setCompareMode(sessionStore.compareMode)
        if (sessionStore.compareSelectedModels.length > 0) setCompareSelectedModels(sessionStore.compareSelectedModels)
      }
      setRagSearchMode(sessionStore.ragSearchMode)
      if (sessionStore.ragTopK !== null) setRagTopK(sessionStore.ragTopK)
      setRagEnableGeneration(sessionStore.ragEnableGeneration)
      setRagEnableCitations(sessionStore.ragEnableCitations)
      setQueuedMessages(restoreQueuedRequests(savedHandle?.address.kind === "fresh" && savedHandle.address.temporary ? [] : sessionStore.queuedMessages ?? []))
    }
    if (savedHandle?.address.kind === "fresh") {
      if (usePlaygroundSessionStore.getState().sourceSelectionRevision !== selectionBeforeRestore) {
        initialRestoreSettledRef.current = true
        usePlaygroundSessionStore.getState().cancelPendingRestore()
        return "cancelled"
      }
      const transcript =
        savedHandle.address.temporary &&
        !sessionStore.serverChatId &&
        (sessionStore.historyId === null || sessionStore.historyId === "temp")
          ? sessionStore.temporaryTranscript
          : null
      if (
        transcript &&
        (!Array.isArray(transcript.history) ||
          !Array.isArray(transcript.messages) ||
          !transcript.history.every(
            (row) =>
              row &&
              ["user", "assistant", "system"].includes(row.role) &&
              typeof row.content === "string"
          ) ||
          !transcript.messages.every(
            (row) =>
              row &&
              typeof row.isBot === "boolean" &&
              typeof row.name === "string" &&
              typeof row.message === "string"
          ))
      )
        return "cancelled"
      const source = useStoreMessageOption.getState()
      if (!isCurrentRestore() || !source.setChatDraftAddress(savedHandle.address, {
        kind: "switch", expectedRevision: source.chatDraftRevision, sourceAddress: source.chatDraftAddress
      })) return "cancelled"
      source.setTemporaryChat(savedHandle.address.temporary)
      selectionRef.current?.reset()
      setHistoryId(null)
      setServerChatId(null)
      setServerChatTitle(null)
      setServerChatAssistantKind(null)
      setServerChatAssistantId(null)
      setServerChatCharacterId(null)
      setServerChatPersonaMemoryMode(null)
      setServerChatMetaLoaded(false)
      setHistory(transcript?.history ?? [])
      setMessages(transcript?.messages ?? [])
      restoreSettings()
      return "restored"
    }
    const publishRestoredDraft = () => {
      const selected = selectionRef.current?.getCurrent()
      if (!selected?.owner || !selected.view || selected.owner.kind === "unavailable") return true
      const receipt = { owner: selected.owner, view: selected.view }
      const address = chatDraftAddressForReceipt(receipt)
      if (!address || !isCurrentRestore()) return false
      if (savedHandle?.address.id === address.id && chatDraftAddressKey(savedHandle.address) !== chatDraftAddressKey(address)) return false
      return acceptRestoredDraft(receipt)
    }
    const tabReference = selectionRef.current?.getStoredReference()
    const sharedReference = sessionStore.historySelectionReference
    const matchesShared = tabReference && sharedReference &&
      tabReference.profile_id === sharedReference.profile_id &&
      tabReference.owner_key === sharedReference.owner_key &&
      tabReference.conversation_id === sharedReference.conversation_id
    const canReuseSharedSession = matchesShared && isSessionValid(scopeKey) &&
      (tabReference.owner_kind === "local"
        ? sessionStore.historyId === tabReference.conversation_id && !sessionStore.serverChatId
        : sessionStore.serverChatId === tabReference.conversation_id)
    if (tabReference) {
      const selected = selectionRef.current!
      const loaded = await selected.loadConversation(tabReference.owner_kind === "local"
        ? { historyId: tabReference.conversation_id, isCurrent: restoreLeaseValid }
        : { serverChatId: tabReference.conversation_id, scope: savedHandle?.address.kind === "native" && savedHandle.address.id === tabReference.conversation_id ? savedHandle.address.scope : undefined, isCurrent: restoreLeaseValid }, tabReference)
      if (!loaded || usePlaygroundSessionStore.getState().restoreRevision !== restoreRevision) return "cancelled"
      selectionCurrent = selected.fence()
      if (!publishRestoredDraft()) return "cancelled"
      // A failed owner validation stays visible; never substitute the other tab's session.
      if (!selected.getCurrent().capture) return "restored"
      setHistoryId(tabReference.owner_kind === "local" ? tabReference.conversation_id : null)
      setServerChatId(tabReference.owner_kind === "native" ? tabReference.conversation_id : null)
      if (!canReuseSharedSession) {
        setQueuedMessages([])
        setServerChatAssistantKind(null)
        setServerChatAssistantId(null)
        setServerChatCharacterId(null)
        setServerChatPersonaMemoryMode(null)
        setServerChatMetaLoaded(false)
        await setSelectedAssistant(null)
        if (!isCurrentRestore()) return "cancelled"
        setSystemPrompt("")
        useStoreMessageOption.getState().setContextFiles([])
        if (tabReference.owner_kind === "local") {
          const chatData = await getFullChatData(tabReference.conversation_id)
          if (!isCurrentRestore()) return "cancelled"
          if (chatData) {
            const prompt = chatData.historyInfo.last_used_prompt
            setSystemPrompt(prompt?.prompt_content || "")
            const files = await getSessionFiles(tabReference.conversation_id)
            if (!isCurrentRestore()) return "cancelled"
            useStoreMessageOption.getState().setContextFiles(files)
            if (chatData.historyInfo.model_id) useStoreMessageOption.getState().setSelectedModel(chatData.historyInfo.model_id)
            if (selected.getCurrent().capture?.status !== "captured") {
              if (selected.getCurrent().error === "unsupported_comparison_history") {
                await restoreReadableLocalComparison(
                  selected,
                  (display) => {
                    useStoreMessageOption.setState({
                      history: display.history,
                      messages: display.messages
                    })
                  },
                  chatData
                )

                if (!isCurrentRestore()) return "cancelled"
              } else {
                setHistory(formatToChatHistory(chatData.messages))
                setMessages(formatToMessage(chatData.messages))
              }
            }
          }
        }
        return "restored"
      }
    }
    if (!tabReference && !isSessionValid(scopeKey)) {
      clearSession()
      return "not-restored"
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
      return "not-restored"
    }

    try {
      let cachedServerTitle: string | null = null
      if (savedHistoryId) {
        try {
          snapshot = await loadServicePromptSnapshot([])
        } catch (error) {
          // An unavailable identity service does not prove the saved selection
          // is invalid. Keep it and its queue for a later verified restore.
          if (isCurrentRestore()) console.warn("Saved Chat owner verification unavailable; restore deferred:", error)
          return "cancelled"
        }
        if (!isCurrentRestore()) return "cancelled"
        // Restore messages from Dexie
        const chatData = await getFullChatData(savedHistoryId)
        if (!isCurrentRestore()) return "cancelled"
        if (!chatData) {
          if (tabReference) return "restored"
          // History was deleted, clear session
          clearSession()
          return "not-restored"
        }
        // The selected-history controller validates stamped records before
        // capture and preserves explicit profile-owned and unbound read-only paths.
        if (!selectionRef.current && chatData.historyInfo.server_scope_key !== serverChatMirrorOwnerKey(snapshot)) {
          clearSession()
          return "not-restored"
        }
        if (chatData.historyInfo.server_chat_id === savedServerChatId) {
          cachedServerTitle = chatData.historyInfo.title || null
        }

        if (selectionRef.current && !tabReference) {
          if (!await selectionRef.current.loadConversation({ historyId: savedHistoryId, serverChatId: savedServerChatId, scope: savedHandle?.address.kind === "native" ? savedHandle.address.scope : undefined, isCurrent: restoreLeaseValid }, tabReference || sessionStore.historySelectionReference)) return "cancelled"
          if (usePlaygroundSessionStore.getState().restoreRevision !== restoreRevision) return "cancelled"
          selectionCurrent = selectionRef.current.fence()
          if (!publishRestoredDraft()) return "cancelled"
          const current = selectionRef.current.getCurrent()
          if (!current.capture && current.error) {
            if (current.error === "unbound_server_mirror") {
              // Preserve the locator for explicit authorized binding. An automatic
              // restore cannot prove which account owns an unstamped transcript.
              setHistoryId(savedHistoryId)
              setServerChatId(null)
              setHistory([])
              setMessages([])
            }
            return "restored"
          }
        }
        // Restore messages and history
        setHistoryId(savedHistoryId)
        if (selectionRef.current?.getCurrent().capture?.status !== "captured") {
          setHistory(formatToChatHistory(chatData.messages))
          setMessages(formatToMessage(chatData.messages))
        }

        // Restore system prompt if present
        const lastUsedPrompt = (chatData.historyInfo as any)?.last_used_prompt
        if (lastUsedPrompt?.prompt_id) {
          const prompt = await getPromptById(lastUsedPrompt.prompt_id)
          if (!isCurrentRestore()) return "cancelled"
          if (prompt) {
            setSelectedSystemPrompt(lastUsedPrompt.prompt_id)
            setSystemPrompt(prompt.content)
          }
        } else if (lastUsedPrompt?.prompt_content) {
          setSystemPrompt(lastUsedPrompt.prompt_content)
        }
      }

      // Restore settings from session store
      if (savedServerChatId && selectionRef.current?.getCurrent().error !== "unbound_server_mirror") {
        if (!savedHistoryId && selectionRef.current && !tabReference) {
          if (!await selectionRef.current.loadConversation({ serverChatId: savedServerChatId, scope: savedHandle?.address.kind === "native" ? savedHandle.address.scope : undefined, isCurrent: restoreLeaseValid }, tabReference || sessionStore.historySelectionReference)) return "cancelled"
          if (usePlaygroundSessionStore.getState().restoreRevision !== restoreRevision) return "cancelled"
          selectionCurrent = selectionRef.current.fence()
          if (!publishRestoredDraft()) return "cancelled"
          if (!selectionRef.current.getCurrent().capture && selectionRef.current.getCurrent().error) return "restored"
        }
        if (!savedHistoryId) {
          setHistoryId(null)
          if (selectionRef.current?.getCurrent().capture?.status !== "captured") {
            setHistory([])
            setMessages([])
          }
        }
        setServerChatId(savedServerChatId)
        setServerChatTitle(cachedServerTitle)
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
          // Cached assistant identity omits canonical title/version metadata.
          // Publish it before awaiting storage so a server refresh stays authoritative.
          setServerChatMetaLoaded(false)
        }
        if (
          savedTrackedAssistantSelection &&
          getAssistantSelectionMode(savedTrackedAssistantSelection) === "tracked"
        ) {
          // Canonical identity belongs to this restored conversation. Mirror
          // its card fields without selecting tracked mode in another tab.
          const mirroredSelection = {
            ...savedTrackedAssistantSelection,
            metadata: { ...savedTrackedAssistantSelection.metadata }
          }
          delete mirroredSelection.metadata.selectionMode
          await setSelectedAssistant(mirroredSelection, {
            isCurrent: isCurrentRestore
          })
          if (!isCurrentRestore()) return "cancelled"
        } else if (savedTrackedAssistantKind && savedTrackedAssistantId) {
          const reconstructedSelection = normalizeAssistantSelection({
            kind: savedTrackedAssistantKind,
            id: savedTrackedAssistantId,
            name:
              savedTrackedAssistantDisplayName ??
              (savedTrackedAssistantKind === "persona" ? "Persona" : "Assistant"),
            avatar_url: savedTrackedAssistantAvatarUrl
          })
          if (reconstructedSelection) {
            await setSelectedAssistant(reconstructedSelection, {
              isCurrent: isCurrentRestore
            })
            if (!isCurrentRestore()) return "cancelled"
          }
        }
      }
      restoreSettings()

      return "restored"
    } catch (error) {
      if (!isCurrentRestore()) return "cancelled"
      console.warn("Failed to restore session:", error)
      clearSession()
      return "not-restored"
    } finally {
      snapshot?.release()
    }
  }, [
    draftOwnerKey,
    isCurrentDraftOwner,
    sessionScopeReady,
    isSessionValid,
    sessionStore,
    clearSession,
    resolveCurrentScopeKey,
    setHistoryId,
    setServerChatId,
    setServerChatTitle,
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
    setFileRetrievalEnabled,
    setRagMediaIds,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setQueuedMessages
  ])

  const restoreSession = useCallback(async (): Promise<PlaygroundSessionRestoreOutcome> => {
    const restore = { isCurrent: isCurrentPersistence }
    isRestoringRef.current = restore
    try {
      const result = await restoreSessionState()
      if (restore.isCurrent() && result !== "cancelled") initialRestoreSettledRef.current = true
      return result
    } finally {
      if (isRestoringRef.current === restore) isRestoringRef.current = null
    }
  }, [isCurrentPersistence, restoreSessionState])

  // Clear persisted session (call when user starts new chat)
  const clearPersistedSession = useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    initialRestoreSettledRef.current = true
    selectionRef.current?.reset()
    clearSession()
  }, [clearSession])

  return {
    restoreSession,
    clearPersistedSession,
    sessionScopeReady,
    hasPersistedSession:
      sessionScopeReady && isSessionValid(currentScopeKey),
    hasPersistedDraftHandle:
      sessionScopeReady && isSessionValid(currentScopeKey) && isChatDraftHandle(sessionStore.chatDraftHandle),
    hasPersistedFreshDraft:
      sessionScopeReady && isSessionValid(currentScopeKey) && isChatDraftHandle(sessionStore.chatDraftHandle) && sessionStore.chatDraftHandle.address.kind === "fresh",
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
