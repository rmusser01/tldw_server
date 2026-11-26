import { useLayoutEffect, useRef, useCallback, useState } from "react"
import { useStoreMessageOption } from "@/store/option"
import type { Knowledge, State } from "@/store/option"
import { useConnectionStore } from "@/store/connection"
import {
  saveHistory,
  saveMessage,
  getAllHistoriesByDocId,
  getFullChatData,
  formatToMessage,
  formatToChatHistory,
  generateID
} from "@/db/dexie/helpers"
import type { HistoryInfo } from "@/db/dexie/types"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { captureActorSceneSendForTarget } from "@/hooks/chat/actor-scene-source"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { runChatPersistenceTransaction } from "@/db/dexie/chat-persistence-transaction"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"

type DocumentChatSession = Pick<
  State,
  | "messages"
  | "history"
  | "historyId"
  | "isFirstMessage"
  | "serverChatId"
  | "serverChatTitle"
  | "serverChatCharacterId"
  | "serverChatAssistantKind"
  | "serverChatAssistantId"
  | "serverChatPersonaMemoryMode"
  | "serverChatMetaLoaded"
  | "serverChatLoadState"
  | "serverChatLoadError"
  | "serverChatRoleplaySetup"
  | "serverChatState"
  | "serverChatVersion"
  | "serverChatTopic"
  | "serverChatClusterId"
  | "serverChatSource"
  | "serverChatExternalRef"
  | "selectedKnowledge"
  | "replyTarget"
  | "actionInfo"
>

const createEmptySession = (): DocumentChatSession => ({
  messages: [],
  history: [],
  historyId: null,
  isFirstMessage: true,
  serverChatId: null,
  serverChatTitle: null,
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
  serverChatMetaLoaded: false,
  serverChatLoadState: "idle",
  serverChatLoadError: null,
  serverChatRoleplaySetup: null,
  serverChatState: null,
  serverChatVersion: null,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  selectedKnowledge: null,
  replyTarget: null,
  actionInfo: null
})

type Conversation = {
  key: string
  historyId: string | null
  title: string | null
  input: string
  knowledge: Knowledge | null
  session?: DocumentChatSession
  pending?: Promise<void>
}

type DocumentConversations = {
  loaded: boolean
  tabs: Conversation[]
  selected: Conversation | null
}

export type DocumentChatCache = Map<number, DocumentConversations>

type Selection = {
  mediaId: number
  document: DocumentConversations
  tab: Conversation | null
  status: "loading" | "ready" | "error"
}

const makeConversation = (history?: HistoryInfo): Conversation => ({
  key: history?.id ?? `draft-${generateID()}`,
  historyId: history?.id ?? null,
  title: history?.title ?? null,
  input: "",
  knowledge: null
})

const ownsHistory = (history: HistoryInfo | undefined, docId: string, requestScope: ServicePromptSnapshot["requestScope"]) =>
  history?.doc_id === docId && history.server_scope_key === serverChatMirrorOwnerKey({ requestScope })

// The send pipeline usually saves the turn itself. Only recover unsaved rows.
async function persistDocumentSession(
  mediaId: number,
  session: DocumentChatSession,
  operation: QuickIngestOperation
): Promise<HistoryInfo | null> {
  const { requestScope } = operation
  const messages = session.messages.filter(msg => !msg.id || msg.id.startsWith("temp_"))
  if (!messages.length || session.historyId === "temp" || !operation.isCurrent()) return null
  const docId = documentKnowledgeId(mediaId)
  let owner: ServicePromptSnapshot | undefined

  try {
    owner = await loadServicePromptSnapshot([], { requestScope, signal: operation.signal })
    if (!operation.isCurrent() || owner.scopeSignal.aborted || owner.scopeInvalidatedSignal.aborted ||
        serverChatMirrorOwnerKey(owner) !== serverChatMirrorOwnerKey(operation)) return null
    return await runChatPersistenceTransaction(owner.scopeSignal, async () => {
      const history = session.historyId
        ? await new PageAssistDatabase().getHistoryInfo(session.historyId)
        : await saveHistory(
          session.messages[0]?.message?.slice(0, 50) || "Document Chat",
          true,
          "web-ui",
          docId,
          session.serverChatId ?? undefined,
          requestScope
        )
      if (!history || !ownsHistory(history, docId, requestScope) || !operation.isCurrent()) return null
      for (const msg of messages) {
        await saveMessage({
          history_id: history.id,
          name: msg.name || "user",
          role: msg.role,
          content: msg.message,
          images: msg.images || [],
          source: msg.sources,
          documents: msg.documents
        })
      }

      return history
    })
  } catch (error) {
    console.error("Failed to persist document chat session:", error)
    return null
  } finally {
    owner?.release()
  }
}

const documentKnowledgeId = (mediaId: number) => `document:${mediaId}`

const buildDocumentKnowledge = (mediaId: number): Knowledge => ({
  id: documentKnowledgeId(mediaId),
  title: `Document ${mediaId}`
})

const isDocumentKnowledge = (
  knowledge: Knowledge | null,
  mediaId: number | null
) => {
  if (!knowledge || mediaId === null) return false
  return knowledge.id === documentKnowledgeId(mediaId)
}

const isMediaDbOnlySources = (sources: string[] | null | undefined) =>
  Array.isArray(sources) && sources.length === 1 && sources[0] === "media_db"

/**
 * Hook that manages document-scoped chat state by automatically setting
 * `ragMediaIds` based on the active document.
 *
 * When a mediaId is provided, this hook:
 * - Sets ragMediaIds to [mediaId] to scope RAG queries to this document
 * - Sets ragSources to ["media_db"] to ensure queries go to the media database
 * - Cleans up on unmount or when mediaId changes
 *
 * @param mediaId - The active document's media ID (null when no document is open)
 */
export function useDocumentChat(
  mediaId: number | null,
  operation: QuickIngestOperation,
  workspaceCache?: DocumentChatCache
) {
  const { requestScope, isCurrent: accountIsCurrent } = operation
  const historyDocumentId = mediaId === null ? undefined : documentKnowledgeId(mediaId)
  const [cache] = useState<DocumentChatCache>(() => workspaceCache ?? new Map())
  const [view, setView] = useState<{ selection: Selection } | null>(null)
  const selectionRef = useRef<Selection | null>(null)
  const previousRagSourcesRef = useRef<string[] | null>(null)

  // RAG state from the store
  const setRagMediaIds = useStoreMessageOption((s) => s.setRagMediaIds)
  const setRagSources = useStoreMessageOption((s) => s.setRagSources)
  const ragMediaIds = useStoreMessageOption((s) => s.ragMediaIds)
  const ragSources = useStoreMessageOption((s) => s.ragSources)
  const ragSearchMode = useStoreMessageOption((s) => s.ragSearchMode)
  const ragTopK = useStoreMessageOption((s) => s.ragTopK)

  // Chat state
  const messages = useStoreMessageOption((s) => s.messages)
  const setMessages = useStoreMessageOption((s) => s.setMessages)
  const streaming = useStoreMessageOption((s) => s.streaming)
  const isProcessing = useStoreMessageOption((s) => s.isProcessing)
  const selectedKnowledge = useStoreMessageOption((s) => s.selectedKnowledge)
  const setSelectedKnowledge = useStoreMessageOption(
    (s) => s.setSelectedKnowledge
  )

  // Connection state
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  const snapshotSession = useCallback((): DocumentChatSession => {
    const state = useStoreMessageOption.getState()
    return {
      messages: state.messages,
      history: state.history,
      historyId: state.historyId,
      isFirstMessage: state.isFirstMessage,
      serverChatId: state.serverChatId,
      serverChatTitle: state.serverChatTitle,
      serverChatCharacterId: state.serverChatCharacterId,
      serverChatAssistantKind: state.serverChatAssistantKind,
      serverChatAssistantId: state.serverChatAssistantId,
      serverChatPersonaMemoryMode: state.serverChatPersonaMemoryMode,
      serverChatMetaLoaded: state.serverChatMetaLoaded,
      serverChatLoadState: state.serverChatLoadState,
      serverChatLoadError: state.serverChatLoadError,
      serverChatRoleplaySetup: state.serverChatRoleplaySetup,
      serverChatState: state.serverChatState,
      serverChatVersion: state.serverChatVersion,
      serverChatTopic: state.serverChatTopic,
      serverChatClusterId: state.serverChatClusterId,
      serverChatSource: state.serverChatSource,
      serverChatExternalRef: state.serverChatExternalRef,
      selectedKnowledge: state.selectedKnowledge,
      replyTarget: state.replyTarget,
      actionInfo: state.actionInfo
    }
  }, [])

  const applySession = useCallback((session: DocumentChatSession) => {
    usePlaygroundSessionStore.getState().markSourceSelectionIntent()
    useStoreMessageOption.setState({ ...session, streaming: false, isProcessing: false })
  }, [])

  const ensureDocumentSources = useCallback(() => {
    const { ragSources } = useStoreMessageOption.getState()
    if (isMediaDbOnlySources(ragSources)) {
      return
    }
    if (!previousRagSourcesRef.current) {
      previousRagSourcesRef.current = ragSources
    }
    setRagSources(["media_db"])
  }, [setRagSources])

  const restoreRagSources = useCallback(() => {
    const previous = previousRagSourcesRef.current
    if (!previous) return
    const current = useStoreMessageOption.getState().ragSources
    if (isMediaDbOnlySources(current)) {
      setRagSources(previous)
    }
    previousRagSourcesRef.current = null
  }, [setRagSources])

  const ragEnabled = isDocumentKnowledge(selectedKnowledge, mediaId)

  const setRagEnabled = useCallback(
    (enabled: boolean) => {
      if (mediaId === null || selectionRef.current?.status !== "ready") return
      if (enabled) {
        setSelectedKnowledge(buildDocumentKnowledge(mediaId))
        ensureDocumentSources()
      } else {
        setSelectedKnowledge(null)
        restoreRagSources()
      }
    },
    [ensureDocumentSources, mediaId, restoreRagSources, setSelectedKnowledge]
  )

  const publish = useCallback((selection: Selection | null) => {
    if (selectionRef.current === selection) setView(selection && { selection })
  }, [])

  const stashConversation = useCallback(() => {
    const selected = selectionRef.current
    if (selected?.status !== "ready" || !selected.tab || !accountIsCurrent()) return

    const { tab, mediaId: id, document } = selected
    const session = snapshotSession()
    // Reload saved or stopped turns from their persisted rows, without the stream buffer.
    tab.knowledge = session.selectedKnowledge
    if (!tab.historyId && !tab.pending) tab.session = session
    if (tab.pending || session.historyId === "temp" ||
        !session.messages.some(msg => !msg.id || msg.id.startsWith("temp_"))) return
    tab.pending = persistDocumentSession(id, session, operation).then(history => {
      if (!history || !accountIsCurrent() || cache.get(id) !== document) return
      tab.historyId = history.id
      tab.title = history.title
      publish(selectionRef.current)
    }).finally(() => { tab.pending = undefined })
  }, [accountIsCurrent, cache, operation, publish, snapshotSession])

  const openConversation = useCallback((id: number | null, tab?: Conversation) => {
    stashConversation()
    if (id === null) {
      selectionRef.current = null
      publish(null)
      applySession(createEmptySession())
      setRagMediaIds(null)
      restoreRagSources()
      return
    }

    let document = cache.get(id)
    if (!document) {
      document = { loaded: false, tabs: [], selected: null }
      cache.set(id, document)
    }
    const collection = document
    if (tab) collection.selected = tab
    const selection: Selection = { mediaId: id, document: collection, tab: collection.selected, status: "loading" }
    selectionRef.current = selection
    publish(selection)
    applySession(createEmptySession())
    setRagMediaIds([id])
    restoreRagSources()

    const docId = documentKnowledgeId(id)
    const revision = usePlaygroundSessionStore.getState().restoreRevision
    let owner: ServicePromptSnapshot | undefined
    const isCurrent = () => selectionRef.current === selection && accountIsCurrent() &&
      revision === usePlaygroundSessionStore.getState().restoreRevision
    const verifyOwner = async () => {
      owner ??= await loadServicePromptSnapshot([], { requestScope, signal: operation.signal })
      if (owner.scopeSignal.aborted || owner.scopeInvalidatedSignal.aborted ||
          serverChatMirrorOwnerKey(owner) !== serverChatMirrorOwnerKey(operation)) {
        throw new Error("Document chat account changed")
      }
    }

    void (async () => {
      try {
        if (!collection.loaded) {
          await verifyOwner()
          if (!isCurrent()) return
          const rows = await getAllHistoriesByDocId(docId, requestScope)
          if (!isCurrent()) return
          const ownedTabs = rows.map(row => makeConversation(row))
          collection.tabs.unshift(...ownedTabs)
          collection.selected ??= ownedTabs[0] ?? makeConversation()
          if (!collection.tabs.includes(collection.selected)) collection.tabs.push(collection.selected)
          collection.loaded = true
          selection.tab = collection.selected
          publish(selection)
        }

        const selected = selection.tab
        if (!selected) throw new Error("Document conversation is missing")
        if (selected.pending) await selected.pending
        if (!isCurrent()) return

        let session = { ...(selected.session ?? createEmptySession()), selectedKnowledge: selected.knowledge }
        if (selected.historyId) {
          await verifyOwner()
          if (!isCurrent()) return
          const chat = await getFullChatData(selected.historyId)
          if (!isCurrent()) return
          if (!chat || chat.historyInfo.id !== selected.historyId || !ownsHistory(chat.historyInfo, docId, requestScope)) {
            throw new Error("Document conversation is unavailable")
          }
          const rows = [...new Map(chat.messages.map(row => [row.serverMessageId || row.id, row])).values()]
          const messageIds = rows.map(row => row.id)
          session = {
            ...createEmptySession(),
            messages: formatToMessage(rows, messageIds),
            history: formatToChatHistory(rows, messageIds),
            historyId: selected.historyId,
            isFirstMessage: rows.length === 0,
            serverChatId: chat.historyInfo.server_chat_id ?? null,
            serverChatTitle: chat.historyInfo.title,
            selectedKnowledge: selected.knowledge
          }
        }
        if (owner?.scopeSignal.aborted || owner?.scopeInvalidatedSignal.aborted) return

        applySession(session)
        if (isDocumentKnowledge(session.selectedKnowledge, id)) ensureDocumentSources()
        selection.status = "ready"
        publish(selection)
      } catch (error) {
        if (!isCurrent()) return
        console.error("Failed to load document chat:", error)
        selection.status = "error"
        publish(selection)
      } finally {
        owner?.release()
      }
    })()
  }, [accountIsCurrent, applySession, cache, ensureDocumentSources, operation, publish, requestScope,
    restoreRagSources, setRagMediaIds, stashConversation])

  useLayoutEffect(() => {
    const baseline = snapshotSession()
    const { ragMediaIds, ragSources } = useStoreMessageOption.getState()
    let invalidated = false
    const stopWatching = watchChatAccountChanges(changed => {
      if (!changed) return
      invalidated = true
      const id = selectionRef.current?.mediaId ?? null
      selectionRef.current = null
      cache.clear()
      openConversation(accountIsCurrent() ? id : null)
    })

    return () => {
      stopWatching()
      stashConversation()
      selectionRef.current = null
      if (!invalidated && accountIsCurrent()) {
        applySession(baseline)
        setRagMediaIds(ragMediaIds)
        setRagSources(ragSources)
      } else {
        applySession(createEmptySession())
        setRagMediaIds(null)
        restoreRagSources()
      }
      previousRagSourcesRef.current = null
    }
  }, [accountIsCurrent, applySession, cache, openConversation, restoreRagSources,
    setRagMediaIds, setRagSources, snapshotSession, stashConversation])

  useLayoutEffect(() => {
    openConversation(mediaId)
  }, [mediaId, openConversation])

  const selectConversation = useCallback((key: string) => {
    const current = selectionRef.current
    const tab = current?.document.tabs.find(item => item.key === key)
    if (current && tab && tab !== current.tab) openConversation(current.mediaId, tab)
  }, [openConversation])

  const newConversation = useCallback(() => {
    const current = selectionRef.current
    if (!current?.document.loaded) return
    const tab = makeConversation()
    current.document.tabs.push(tab)
    openConversation(current.mediaId, tab)
  }, [openConversation])

  const retryConversation = useCallback(() => {
    const current = selectionRef.current
    if (current) openConversation(current.mediaId)
  }, [openConversation])

  const active = view?.selection.mediaId === mediaId && accountIsCurrent() ? view.selection : null
  const activeTab = active?.tab
  const document = active?.document
  const isChatReady = isServerAvailable && active?.status === "ready"

  const setInputValue = useCallback((input: string | ((current: string) => string)) => {
    if (!active?.tab || selectionRef.current !== active || !accountIsCurrent()) return
    active.tab.input = typeof input === "function" ? input(active.tab.input) : input
    publish(active)
  }, [accountIsCurrent, active, publish])

  const onHistorySaved = useCallback(async (historyId: string) => {
    if (!activeTab || !document || !historyDocumentId || mediaId === null || activeTab.historyId === historyId) return
    if (!accountIsCurrent() || cache.get(mediaId) !== document) return
    try {
      const history = await new PageAssistDatabase().getHistoryInfo(historyId)
      if (!history || !ownsHistory(history, historyDocumentId, requestScope) ||
          !accountIsCurrent() || cache.get(mediaId) !== document) return
      activeTab.historyId = history.id
      activeTab.title = history.title
      activeTab.session = undefined
      publish(selectionRef.current)
    } catch (error) {
      console.error("Failed to identify saved document chat:", error)
    }
  }, [accountIsCurrent, activeTab, cache, document, historyDocumentId, mediaId, publish, requestScope])

  const trackTurn = useCallback(<T,>(send: () => Promise<T>) => {
    if (!activeTab) return send()
    if (!activeTab.historyId) activeTab.session = snapshotSession()
    const turn = send()
    const settled = turn.then(() => {}, () => {}).finally(() => {
      if (activeTab.pending === settled) activeTab.pending = undefined
    })
    activeTab.pending = settled
    return turn
  }, [activeTab, snapshotSession])

  const capturePrivateScene = useCallback((commandIsCurrent: () => boolean) => {
    if (!isChatReady || !active || !historyDocumentId || !accountIsCurrent()) return null
    const isCurrent = () => selectionRef.current === active && accountIsCurrent() && commandIsCurrent()
    if (!isCurrent()) return null

    const source = captureChatDraftSource(isCurrent)
    // The workspace transcript owns this private scene independently of main Chat history.
    const target = captureActorSettingsTarget({
      address: {
        kind: "fresh",
        id: historyDocumentId,
        scope: { type: "global" },
        temporary: useStoreMessageOption.getState().temporaryChat
      },
      requestScope,
      accountIsCurrent,
      isCurrent: source.isCurrent
    })
    return captureActorSceneSendForTarget(target, target?.account, source.isCurrent, isCurrent)
  }, [accountIsCurrent, active, historyDocumentId, isChatReady, requestScope])

  // Check if document is currently scoped for RAG
  const isDocumentScoped =
    ragMediaIds !== null &&
    ragMediaIds.length === 1 &&
    ragMediaIds[0] === mediaId

  return {
    // Current state
    messages,
    streaming,
    isProcessing,
    isDocumentScoped,
    isChatReady,
    historyDocumentId,
    isServerAvailable,
    ragEnabled,
    conversations: document?.tabs.map(tab => ({ key: tab.key, title: tab.title })) ?? [],
    activeConversationKey: activeTab?.key ?? null,
    conversationStatus: active?.status ?? "loading",
    inputValue: activeTab?.input ?? "",

    // RAG configuration
    ragMediaIds,
    ragSources,
    ragSearchMode,
    ragTopK,

    // Actions
    capturePrivateScene,
    newConversation,
    retryConversation,
    selectConversation,
    onHistorySaved,
    trackTurn,
    setInputValue,
    setRagEnabled,
    setMessages
  }
}

export type UseDocumentChatReturn = ReturnType<typeof useDocumentChat>
