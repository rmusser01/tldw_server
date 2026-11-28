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
  formatSelectedHistory,
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
import { tldwClient } from "@/services/tldw"
import { hasDocumentChatSync } from "@/services/tldw/server-capabilities"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import type { DocumentChatPage, DocumentChatTarget } from "@/services/tldw/domains/media"
import {
  createLinkedDocumentChat, readPendingDocumentChats, readPendingDocumentDraft, settleDocumentChatCreation
} from "./document-chat-creation"
import { DocumentChatReviewRequired, rememberDocumentChatMedia, verifyDocumentChatImport } from "@/db/dexie/document-chat-import"
import { linkExistingDocumentChat, saveBrowserDocumentChat } from "./document-chat-save"

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
  serverChatId?: string
  pendingFirstMessage?: boolean
  browserOnly?: boolean
  recoveryPending?: boolean
  recoveryChatId?: string
  recoveredInput?: string
  initialDraft?: boolean
  mediaUuid?: string
  mirrorId?: string
  browserCopyId?: string
  saveNeedsReview?: boolean
}

type DocumentConversations = {
  loaded: boolean
  tabs: Conversation[]
  selected: Conversation | null
  mediaUuid?: string
  ownerKey?: string
  sync?: "supported" | "unsupported"
  syncIssue?: "unavailable" | "denied"
}

export type DocumentChatCache = Map<number, DocumentConversations>

type Selection = {
  mediaId: number
  document: DocumentConversations
  tab: Conversation | null
  status: "loading" | "ready" | "error"
  controller: AbortController
  refreshRevision?: number
}

const makeConversation = (history?: HistoryInfo): Conversation => ({
  key: history?.id ?? `draft-${generateID()}`,
  historyId: history?.id ?? null,
  title: history?.title ?? null,
  mediaUuid: history?.document_media_uuid,
  mirrorId: history?.server_chat_id ?? history?.document_chat_import?.chatId,
  input: "",
  knowledge: null
})

const ownsHistory = (history: HistoryInfo | undefined, docId: string, requestScope: ServicePromptSnapshot["requestScope"]) =>
  history?.doc_id === docId && history.server_scope_key === serverChatMirrorOwnerKey({ requestScope })

// The send pipeline usually saves the turn itself. Only recover unsaved rows.
async function persistDocumentSession(
  mediaId: number,
  session: DocumentChatSession,
  operation: QuickIngestOperation,
  mediaUuid?: string
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
          requestScope,
          ...(mediaUuid ? [mediaUuid] as const : [])
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
  workspaceCache?: DocumentChatCache,
  documentTarget?: DocumentChatTarget
) {
  const { requestScope, isCurrent: accountIsCurrent } = operation
  const historyDocumentId = mediaId === null ? undefined : documentKnowledgeId(mediaId)
  const [cache] = useState<DocumentChatCache>(() => workspaceCache ?? new Map())
  const [view, setView] = useState<{ selection: Selection } | null>(null)
  const [saveFeedback, setSaveFeedback] = useState<{ tab: Conversation; message: string } | null>(null)
  const selectionRef = useRef<Selection | null>(null)
  const previousRagSourcesRef = useRef<string[] | null>(null)
  const nativeHistory = useHistorySelection()
  const { loadConversation: loadNative, getCurrent: getNative, reset: resetNative } = nativeHistory
  const ownerKey = serverChatMirrorOwnerKey(operation)

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
      const tab = selectionRef.current.tab
      const knowledge = enabled ? buildDocumentKnowledge(mediaId) : null
      if (tab) tab.knowledge = knowledge
      setSelectedKnowledge(knowledge)
      if (enabled) {
        ensureDocumentSources()
      } else {
        restoreRagSources()
      }
    },
    [ensureDocumentSources, mediaId, restoreRagSources, setSelectedKnowledge]
  )

  const publish = useCallback((selection: Selection | null) => {
    if (selectionRef.current === selection) setView(selection && { selection })
  }, [])

  const applyNativeChat = useCallback((tab: Conversation) => {
    const capture = getNative().capture
    const display = capture?.status === "captured" ? formatSelectedHistory(capture) : { messages: [], history: [] }
    if (tab.recoveredInput && display.messages[0]?.role === "user" && display.messages[0].message === tab.recoveredInput) {
      if (tab.input === tab.recoveredInput) tab.input = ""
      tab.recoveredInput = undefined
    }
    applySession({
      ...createEmptySession(), ...display, isFirstMessage: display.messages.length === 0,
      serverChatId: tab.serverChatId ?? null, serverChatTitle: tab.title,
      serverChatMetaLoaded: true, serverChatLoadState: "loaded", selectedKnowledge: tab.knowledge
    })
  }, [applySession, getNative])

  const openLinkedHistory = useCallback(async (
    serverChatId: string, target: DocumentChatTarget, isCurrent: () => boolean,
  ) => {
    await loadNative({ serverChatId, documentTarget: target, documentRequestScope: requestScope, isCurrent })
    const { status, capture } = getNative()
    return isCurrent() && status === "ready" && capture?.status === "captured" &&
      capture.snapshot.conversation_id === serverChatId
  }, [getNative, loadNative, requestScope])

  const refreshServerChats = useCallback(async (selection: Selection) => {
    const { document: collection, controller } = selection
    if (!documentTarget || selection.mediaId !== documentTarget.mediaId || collection.sync === "unsupported") return
    const revision = selection.refreshRevision = (selection.refreshRevision ?? 0) + 1
    const current = () => selectionRef.current === selection && !controller.signal.aborted &&
      selection.refreshRevision === revision && accountIsCurrent()
    const options = { signal: AbortSignal.any([controller.signal, operation.signal]), requestScope }
    try {
      if (collection.sync === undefined) {
        const spec = await tldwClient.getOpenAPISpec(options)
        if (!current()) return
        if (!spec?.paths) throw new Error("Document chat capability unavailable")
        collection.sync = hasDocumentChatSync(spec) ? "supported" : "unsupported"
      }
      if (collection.sync !== "supported") return
      const rows = new Map<string, DocumentChatPage["items"][number]>()
      let offset = 0
      for (;;) {
        const page = await tldwClient.listDocumentChats(documentTarget, { ...options, offset })
        if (!current()) return
        page.items.forEach(row => rows.set(row.id, row))
        if (!page.has_more || !page.items.length) break
        offset += page.items.length
      }
      collection.syncIssue = undefined
      for (const tab of collection.tabs) {
        if (!tab.historyId || !tab.mirrorId || tab.mediaUuid !== documentTarget.mediaUuid || tab.pending) continue
        try {
          const details = await getFullChatData(tab.historyId)
          if (!current()) return
          if (!details || !ownsHistory(details.historyInfo, documentKnowledgeId(selection.mediaId), requestScope)) continue
          const reservation = details.historyInfo.document_chat_import
          const saved = reservation?.attached && rows.has(reservation.chatId)
            ? { id: reservation.chatId, fullySaved: await verifyDocumentChatImport(
              tab.historyId, documentTarget, requestScope, reservation,
            ) }
            : await linkExistingDocumentChat(details.historyInfo, details.messages, documentTarget, options)
          if (!current()) return
          if (!saved) continue
          if (!rows.has(saved.id)) rows.set(saved.id, { id: saved.id, title: tab.title, last_modified: "" })
          if (saved.fullySaved) {
            if (selection.tab === tab && selection.status === "ready") {
              const ready = await openLinkedHistory(saved.id, documentTarget, () =>
                selectionRef.current === selection && accountIsCurrent())
              if (!current()) return
              if (!ready) {
                rows.delete(saved.id)
                continue
              }
            }
            tab.browserCopyId = tab.historyId
            tab.historyId = null
            tab.serverChatId = saved.id
            if (selection.tab === tab && selection.status === "ready") applyNativeChat(tab)
          }
        } catch {
          // A legacy copy stays available in its verified browser scope on a failed reconciliation.
        }
      }
      const known = new Map(collection.tabs.flatMap(tab => {
        const id = tab.serverChatId ?? tab.recoveryChatId
        return id ? [[id, tab] as const] : []
      }))
      const linked = [...rows.values()].map(row => {
        const tab = known.get(row.id) ?? makeConversation()
        tab.serverChatId = row.id
        tab.title = row.title
        tab.pendingFirstMessage = false
        if (tab.recoveryPending) {
          void settleDocumentChatCreation(documentTarget, options, tab.key, row.id).catch(() => {})
          tab.recoveryPending = false
        }
        return tab
      })
      const browser = collection.tabs.filter(tab => !tab.serverChatId || tab.pendingFirstMessage)
      for (const tab of collection.tabs) {
        if (tab.serverChatId && !rows.has(tab.serverChatId) && tab.browserCopyId) {
          tab.historyId = tab.browserCopyId
          tab.serverChatId = undefined
          browser.push(tab)
        }
      }
      collection.tabs = [...linked, ...browser]
      if (linked.length && selection.status === "loading" && collection.selected?.initialDraft &&
          !collection.selected.input && !collection.selected.historyId && !collection.selected.pending) {
        collection.tabs = collection.tabs.filter(tab => tab !== collection.selected)
        collection.selected = linked[0]
        selection.tab = linked[0]
      }
      if (selection.tab?.serverChatId && !collection.tabs.includes(selection.tab)) {
        resetNative()
        applySession(createEmptySession())
        selection.status = "error"
      }
    } catch (error) {
      if (!current()) return
      console.warn("Document conversations refresh failed:", error)
      const status = (error as { status?: number } | null)?.status
      collection.syncIssue = status === 403 || status === 404 ? "denied" : "unavailable"
      if (collection.syncIssue === "denied") {
        collection.tabs = collection.tabs.filter(tab => !tab.serverChatId)
        if (selection.tab?.serverChatId) {
          resetNative()
          applySession(createEmptySession())
          selection.status = "error"
        }
      }
    }
    if (current()) publish(selection)
  }, [accountIsCurrent, applyNativeChat, applySession, documentTarget, openLinkedHistory, operation.signal, publish, requestScope, resetNative])

  const stashConversation = useCallback(() => {
    const selected = selectionRef.current
    if (selected?.status !== "ready" || !selected.tab || !accountIsCurrent()) return

    const { tab, mediaId: id, document } = selected
    const session = snapshotSession()
    // Reload saved or stopped turns from their persisted rows, without the stream buffer.
    tab.knowledge = session.selectedKnowledge
    if (tab.serverChatId) {
      if (!tab.pending) tab.session = session
      return
    }
    if (!tab.historyId && !tab.pending) tab.session = session
    if (tab.pending || session.historyId === "temp" ||
        !session.messages.some(msg => !msg.id || msg.id.startsWith("temp_"))) return
    tab.pending = persistDocumentSession(id, session, operation, selected.document.mediaUuid).then(history => {
      if (!history || !accountIsCurrent() || cache.get(id) !== document) return
      tab.historyId = history.id
      tab.title = history.title
      publish(selectionRef.current)
    }).finally(() => { tab.pending = undefined })
  }, [accountIsCurrent, cache, operation, publish, snapshotSession])

  const openConversation = useCallback((id: number | null, tab?: Conversation) => {
    stashConversation()
    selectionRef.current?.controller.abort()
    resetNative()
    if (id === null) {
      selectionRef.current = null
      publish(null)
      applySession(createEmptySession())
      setRagMediaIds(null)
      restoreRagSources()
      return
    }

    let document = cache.get(id)
    if (!document || (document.ownerKey && document.ownerKey !== ownerKey) ||
        (document.mediaUuid && documentTarget && document.mediaUuid !== documentTarget.mediaUuid)) {
      document = { loaded: false, tabs: [], selected: null, ownerKey }
      cache.set(id, document)
    }
    if (documentTarget?.mediaId === id) {
      document.mediaUuid = documentTarget.mediaUuid
      document.tabs = document.tabs.filter(tab => !tab.mediaUuid || tab.mediaUuid === documentTarget.mediaUuid)
      if (document.selected && !document.tabs.includes(document.selected)) document.selected = null
    }
    if (tab && !document.tabs.includes(tab)) tab = undefined
    const collection = document
    if (tab) collection.selected = tab
    const selection: Selection = {
      mediaId: id, document: collection, tab: collection.selected, status: "loading", controller: new AbortController()
    }
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
          const ownedTabs = rows.filter(row => !collection.mediaUuid ||
            !row.document_media_uuid || row.document_media_uuid === collection.mediaUuid).map(row => makeConversation(row))
          collection.tabs.unshift(...ownedTabs)
          collection.loaded = true
        }
        // Media metadata can arrive after browser histories have already loaded.
        if (documentTarget) {
          const options = { requestScope }
          for (const pending of readPendingDocumentChats(documentTarget, options)) {
            if (collection.tabs.some(tab => tab.key === pending.key)) continue
            const input = await readPendingDocumentDraft(pending.key, options)
            if (!isCurrent()) return
            const recovered = {
              ...makeConversation(), key: pending.key, input, recoveryPending: true,
              recoveredInput: input, recoveryChatId: pending.receipt?.id,
              title: pending.receipt?.title ?? "Unconfirmed chat"
            }
            collection.tabs.push(recovered)
            const selected = collection.selected
            if (input && selected?.initialDraft && !selected.input && !selected.historyId && !selected.pending) {
              collection.tabs = collection.tabs.filter(tab => tab !== selected)
              collection.selected = recovered
              selection.tab = recovered
            }
          }
        }
        if (!tab) await refreshServerChats(selection)
        if (!isCurrent()) return
        if (selection.status === "error") return
        collection.selected ??= collection.tabs[0] ?? { ...makeConversation(), initialDraft: true }
        if (!collection.tabs.includes(collection.selected)) collection.tabs.push(collection.selected)
        selection.tab = collection.selected
        publish(selection)

        const selected = selection.tab
        if (!selected) throw new Error("Document conversation is missing")
        if (selected.pending) await selected.pending
        if (!isCurrent()) return

        let session = { ...(selected.session ?? createEmptySession()), selectedKnowledge: selected.knowledge }
        if (selected.serverChatId && documentTarget) {
          if (collection.syncIssue) {
            if (collection.syncIssue === "unavailable" && selected.session) applySession(selected.session)
            throw new Error("Document chat unavailable")
          }
          const ready = await openLinkedHistory(selected.serverChatId, documentTarget, isCurrent)
          if (!isCurrent()) return
          if (!ready) {
            const error = getNative().error
            if (error === "document_chat_unavailable") {
              collection.tabs = collection.tabs.filter(tab => tab !== selected)
              selected.session = undefined
            } else if (selected.session) {
              applySession(selected.session)
            }
            throw new Error(error || "Document chat unavailable")
          }
          applyNativeChat(selected)
        } else if (selected.historyId) {
          await verifyOwner()
          if (!isCurrent()) return
          const chat = await getFullChatData(selected.historyId)
          if (!isCurrent()) return
          if (!chat || chat.historyInfo.id !== selected.historyId || !ownsHistory(chat.historyInfo, docId, requestScope) ||
              (collection.mediaUuid && chat.historyInfo.document_media_uuid &&
                collection.mediaUuid !== chat.historyInfo.document_media_uuid)) {
            throw new Error("Document conversation is unavailable")
          }
          // IndexedDB returns these rows by message ID. The explicit projection
          // below preserves its input order, so restore the saved chronology first.
          const rows = [...new Map(chat.messages.map(row => [row.serverMessageId || row.id, row])).values()]
            .sort((a, b) => a.createdAt - b.createdAt)
          const messageIds = rows.map(row => row.id)
          session = {
            ...createEmptySession(),
            messages: formatToMessage(rows, messageIds),
            history: formatToChatHistory(rows, messageIds),
            historyId: selected.historyId,
            isFirstMessage: rows.length === 0,
            serverChatId: collection.sync === "supported" ? null : chat.historyInfo.server_chat_id ?? null,
            serverChatTitle: chat.historyInfo.title,
            selectedKnowledge: selected.knowledge
          }
        }
        if (owner?.scopeSignal.aborted || owner?.scopeInvalidatedSignal.aborted) return

        if (!selected.serverChatId) applySession(session)
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
    restoreRagSources, setRagMediaIds, stashConversation, documentTarget, getNative, openLinkedHistory, ownerKey,
    refreshServerChats, resetNative, applyNativeChat])

  useLayoutEffect(() => {
    const baseline = snapshotSession()
    const { ragMediaIds, ragSources } = useStoreMessageOption.getState()
    let invalidated = false
    const stopWatching = watchChatAccountChanges(changed => {
      if (!changed) return
      invalidated = true
      const id = selectionRef.current?.mediaId ?? null
      selectionRef.current?.controller.abort()
      selectionRef.current = null
      cache.clear()
      openConversation(accountIsCurrent() ? id : null)
    })

    return () => {
      stopWatching()
      stashConversation()
      selectionRef.current?.controller.abort()
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

  useLayoutEffect(() => {
    const refresh = () => {
      const selection = selectionRef.current
      if (selection?.status === "ready" && window.document.visibilityState !== "hidden") {
        void refreshServerChats(selection)
      }
    }
    window.addEventListener("focus", refresh)
    window.document.addEventListener("visibilitychange", refresh)
    return () => {
      window.removeEventListener("focus", refresh)
      window.document.removeEventListener("visibilitychange", refresh)
    }
  }, [refreshServerChats])

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
  const isChatReady = isServerAvailable && active?.status === "ready" && document?.syncIssue !== "denied" &&
    (!activeTab?.serverChatId || !document?.syncIssue)

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
      if (document.mediaUuid && !history.document_media_uuid && await rememberDocumentChatMedia(
        history.id, history.server_chat_id ?? null, { mediaId, mediaUuid: document.mediaUuid },
        requestScope, operation.signal
      )) {
        activeTab.mediaUuid = document.mediaUuid
      }
      if (!accountIsCurrent() || cache.get(mediaId) !== document) return
      activeTab.historyId = history.id
      activeTab.mediaUuid ??= history.document_media_uuid
      activeTab.title = history.title
      activeTab.session = undefined
      publish(selectionRef.current)
    } catch (error) {
      console.error("Failed to identify saved document chat:", error)
    }
  }, [accountIsCurrent, activeTab, cache, document, historyDocumentId, mediaId, operation.signal, publish, requestScope])

  const prepareLinkedSend = useCallback(async (question: string, signal: AbortSignal) => {
    if (!active || !activeTab || !documentTarget || activeTab.historyId ||
        activeTab.browserOnly || useStoreMessageOption.getState().temporaryChat) return undefined
    const current = () => selectionRef.current === active && accountIsCurrent() && !signal.aborted
    if (!current() || document?.syncIssue) throw new Error("Document chat is unavailable. Retry when the server is reachable.")
    if (document?.sync !== "supported") return undefined
    if (activeTab.serverChatId) return { chatId: activeTab.serverChatId, firstInputAlreadySaved: false }
    const receipt = await createLinkedDocumentChat({
      key: activeTab.key, question, target: documentTarget, isCurrent: current,
      options: { requestScope, signal: AbortSignal.any([signal, active.controller.signal, operation.signal]) }
    })
    if (!current()) return undefined
    const ready = await openLinkedHistory(receipt.id, documentTarget, () =>
      selectionRef.current === active && accountIsCurrent())
    if (!ready || !current()) throw new Error("Could not open the saved conversation. Your draft is still available.")
    activeTab.serverChatId = receipt.id
    activeTab.title = receipt.title
    activeTab.pendingFirstMessage = true
    activeTab.recoveryPending = true
    activeTab.recoveredInput ??= question
    const capture = getNative().capture
    const firstInputAlreadySaved = activeTab.recoveredInput === question && capture?.status === "captured" &&
      capture.rows[0]?.role === "user" && capture.selected_content[0]?.message === question
    applyNativeChat(activeTab)
    publish(active)
    return { chatId: receipt.id, firstInputAlreadySaved }
  }, [accountIsCurrent, active, activeTab, applyNativeChat, document, documentTarget, getNative, openLinkedHistory,
    operation.signal, publish, requestScope])

  const setBrowserOnly = useCallback((browserOnly: boolean) => {
    if (!active || !activeTab || selectionRef.current !== active || activeTab.serverChatId || activeTab.historyId ||
        activeTab.recoveryPending || activeTab.pending || !accountIsCurrent()) return
    activeTab.browserOnly = browserOnly
    publish(active)
  }, [accountIsCurrent, active, activeTab, publish])

  const refreshAfterSend = useCallback(async (reload = false) => {
    if (!active || !activeTab?.serverChatId || !documentTarget || selectionRef.current !== active || !accountIsCurrent()) return
    const current = () => selectionRef.current === active && accountIsCurrent()
    if (reload) {
      const ready = await openLinkedHistory(activeTab.serverChatId, documentTarget, current)
      if (!current()) return
      if (!ready) {
        const denied = getNative().error === "document_chat_unavailable"
        if (denied) {
          active.document.tabs = active.document.tabs.filter(tab => tab !== activeTab)
          activeTab.session = undefined
          applySession(createEmptySession())
        }
        active.status = "error"
        publish(active)
        if (denied) await refreshServerChats(active)
        return
      }
    }
    applyNativeChat(activeTab)
    await refreshServerChats(active)
  }, [accountIsCurrent, active, activeTab, applyNativeChat, applySession, documentTarget, getNative, openLinkedHistory, publish,
    refreshServerChats])

  const saveToServer = useCallback(async (confirmedLegacy = false) => {
    if (!active || !activeTab?.historyId || !documentTarget || document?.sync !== "supported" ||
        document.syncIssue || activeTab.pending) return
    const tab = activeTab
    const historyId = tab.historyId!
    const current = () => selectionRef.current === active && accountIsCurrent()
    const options = { requestScope, signal: AbortSignal.any([active.controller.signal, operation.signal]) }
    setSaveFeedback(null)
    const work = (async () => {
      try {
        const details = await getFullChatData(historyId)
        if (!current() || !details || !ownsHistory(details.historyInfo, documentKnowledgeId(active.mediaId), requestScope)) return
        if (!details.historyInfo.document_media_uuid && !confirmedLegacy) {
          throw new Error("Confirm this document before saving the conversation.")
        }
        const capture = getNative().capture
        const interpretation = capture?.snapshot.interpretation_status
        const reviewedOrder = tab.saveNeedsReview && capture?.snapshot.conversation_id === historyId &&
          interpretation?.kind === "legacy_linear_v1" ? interpretation.ordered_path_ids : undefined
        const saved = details.historyInfo.server_chat_id
          ? await linkExistingDocumentChat(details.historyInfo, details.messages, documentTarget, options, confirmedLegacy)
          : await saveBrowserDocumentChat(historyId, documentTarget, options, confirmedLegacy, reviewedOrder)
        if (!current()) return
        if (!saved) throw new Error("The server copy could not be verified. This conversation is still on this browser.")
        tab.mediaUuid = documentTarget.mediaUuid
        tab.mirrorId = saved.id
        tab.saveNeedsReview = false
        if (saved.fullySaved) {
          tab.browserCopyId = historyId
          tab.historyId = null
          tab.serverChatId = saved.id
          document.tabs = document.tabs.filter(other => other === tab || other.serverChatId !== saved.id)
        } else {
          setSaveFeedback({ tab, message: "Saved a server copy. This browser still has additional chat history." })
        }
        await refreshServerChats(active)
        if (current() && saved.fullySaved) {
          tab.pending = undefined
          openConversation(active.mediaId, tab)
        }
      } catch (error) {
        if (!current()) return
        if (error instanceof DocumentChatReviewRequired) {
          tab.saveNeedsReview = true
          await loadNative({ historyId, documentRequestScope: requestScope, isCurrent: current })
          if (!current()) return
        }
        const apiError = error as { detail?: { error_code?: string }; details?: { detail?: { error_code?: string } } } | null
        const code = (apiError?.details?.detail ?? apiError?.detail)?.error_code
        const message = code === "sync_chat_import_unsupported"
          ? "Saving to the server is unavailable with this account’s Sync v2 profile. Your browser copy is still available."
          : error instanceof Error ? error.message : "Could not save this conversation. Retry to keep the same save."
        setSaveFeedback({ tab, message })
      }
    })()
    tab.pending = work
    publish(active)
    await work
    if (tab.pending === work) tab.pending = undefined
    if (current()) publish(active)
  }, [accountIsCurrent, active, activeTab, document, documentTarget, getNative, loadNative, openConversation,
    operation.signal, publish, refreshServerChats, requestScope])

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
    canShowTranscript: Boolean(active && active.status !== "loading" &&
      (!activeTab?.serverChatId || document?.syncIssue !== "denied")),
    historyDocumentId,
    isServerAvailable,
    ragEnabled,
    conversations: document?.tabs.map(tab => ({ key: tab.key, title: tab.title, serverChatId: tab.serverChatId })) ?? [],
    nativeHistory,
    linkedServerChatId: activeTab?.serverChatId,
    syncSupported: document?.sync === "supported",
    canChooseSaveMode: document?.sync === "supported" && !activeTab?.historyId && !activeTab?.serverChatId &&
      !activeTab?.recoveryPending,
    browserOnly: Boolean(activeTab?.browserOnly),
    syncIssue: document?.syncIssue,
    canSaveToServer: document?.sync === "supported" && Boolean(activeTab?.historyId) && !document.syncIssue,
    saveNeedsConfirmation: Boolean(activeTab?.historyId && !activeTab.mediaUuid),
    saveNeedsReview: Boolean(activeTab?.saveNeedsReview),
    saveFeedback: saveFeedback && saveFeedback.tab === activeTab ? saveFeedback.message : null,
    savingToServer: Boolean(activeTab?.historyId && activeTab.pending),
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
    setBrowserOnly,
    prepareLinkedSend,
    refreshAfterSend,
    saveToServer,
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
