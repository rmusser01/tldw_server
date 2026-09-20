import { useEffect, useRef, useCallback, useState } from "react"
import { useStoreMessageOption } from "@/store/option"
import type { Knowledge, State, Message as StoreMessage } from "@/store/option"
import { useConnectionStore } from "@/store/connection"
import {
  saveHistory,
  saveMessage,
  getHistoryByDocId,
  getFullChatData
} from "@/db/dexie/helpers"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
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
  | "serverChatMetaLoaded"
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
  serverChatMetaLoaded: false,
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

/**
 * Save a document chat session to IndexedDB for persistence across page refreshes.
 * Creates a new history entry if one doesn't exist, otherwise updates existing.
 */
async function persistDocumentSession(
  mediaId: number,
  session: DocumentChatSession
): Promise<string | null> {
  // Don't persist empty sessions
  if (session.messages.length === 0) {
    return session.historyId
  }

  const docId = `document:${mediaId}`
  const revision = usePlaygroundSessionStore.getState().restoreRevision
  let owner: ServicePromptSnapshot | undefined

  try {
    owner = await loadServicePromptSnapshot([])
    if (revision !== usePlaygroundSessionStore.getState().restoreRevision || owner.scopeSignal.aborted || owner.scopeInvalidatedSignal.aborted) return null
    const capturedOwner = owner
    return await runChatPersistenceTransaction(owner.scopeSignal, async () => {
      // Check if we already have a history ID for this session
      let historyId = session.historyId
      if (historyId) {
        const existing = await new PageAssistDatabase().getHistoryInfo(historyId)
        if (existing?.server_scope_key !== serverChatMirrorOwnerKey(capturedOwner)) return null
      }

      if (!historyId) {
        // Check if there's an existing history for this document
        const existingHistory = await getHistoryByDocId(docId, capturedOwner.requestScope)
        if (existingHistory) {
          historyId = existingHistory.id
        } else {
          // Create new history entry
          const title = session.messages[0]?.message?.slice(0, 50) || "Document Chat"
          const newHistory = await saveHistory(
            title,
            true, // is_rag
            "web-ui",
            docId,
            session.serverChatId ?? undefined,
            capturedOwner.requestScope
          )
          historyId = newHistory.id
        }
      }

      // Save each message that doesn't have a persisted flag
      for (const msg of session.messages) {
        // Skip if message is already persisted (check by looking for a generated ID pattern)
        if (msg.id && !msg.id.startsWith("temp_")) {
          continue
        }

        await saveMessage({
          history_id: historyId,
          name: msg.name || "user",
          role: msg.role,
          content: msg.message,
          images: msg.images || [],
          source: msg.sources,
          documents: msg.documents
        })
      }

      return historyId
    })
  } catch (error) {
    console.error("Failed to persist document chat session:", error)
    return null
  } finally {
    owner?.release()
  }
}

/**
 * Load a document chat session from IndexedDB.
 */
async function loadDocumentSession(
  mediaId: number
): Promise<DocumentChatSession | null> {
  const docId = `document:${mediaId}`
  let owner: ServicePromptSnapshot | undefined

  try {
    owner = await loadServicePromptSnapshot([])
    const history = await getHistoryByDocId(docId, owner.requestScope)
    if (!history) {
      return null
    }

    // Load full chat data including messages
    const chatData = await getFullChatData(history.id)
    if (!chatData || owner.scopeSignal.aborted || owner.scopeInvalidatedSignal.aborted) {
      return null
    }

    const { historyInfo, messages } = chatData

    // Convert DB messages to store message format
    const storeMessages: StoreMessage[] = messages.map((msg) => ({
      id: msg.id,
      role: msg.role as "user" | "assistant" | "system",
      message: msg.content,
      name: msg.name,
      images: msg.images || [],
      sources: msg.sources || [],
      documents: msg.documents,
      isBot: msg.role === "assistant",
      modelName: msg.modelName,
      modelImage: msg.modelImage
    }))
    const chatHistory = storeMessages.map((msg) => ({
      role: msg.role || "user",
      content: msg.message,
      image: msg.images?.[0],
      messageType: msg.messageType
    }))

    return {
      messages: storeMessages,
      history: chatHistory,
      historyId: historyInfo.id,
      isFirstMessage: storeMessages.length === 0,
      serverChatId: historyInfo.server_chat_id ?? null,
      serverChatTitle: historyInfo.title ?? null,
      serverChatCharacterId: null,
      serverChatMetaLoaded: false,
      serverChatState: null,
      serverChatVersion: null,
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      selectedKnowledge: null,
      replyTarget: null,
      actionInfo: null
    }
  } catch (error) {
    console.error("Failed to load document chat session:", error)
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
export function useDocumentChat(mediaId: number | null) {
  const [accountRevision, setAccountRevision] = useState(0)
  const previousMediaIdRef = useRef<number | null>(null)
  const activeMediaIdRef = useRef<number | null>(mediaId)
  const loadGenerationRef = useRef(0)
  const sessionsRef = useRef<Map<number, DocumentChatSession>>(new Map())
  const baselineSessionRef = useRef<DocumentChatSession | null>(null)
  const baselineRagSettingsRef = useRef<{
    ragMediaIds: number[] | null
    ragSources: string[]
  } | null>(null)
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
      serverChatMetaLoaded: state.serverChatMetaLoaded,
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
    useStoreMessageOption.setState({
      messages: session.messages,
      history: session.history,
      historyId: session.historyId,
      isFirstMessage: session.isFirstMessage,
      serverChatId: session.serverChatId,
      serverChatTitle: session.serverChatTitle,
      serverChatCharacterId: session.serverChatCharacterId,
      serverChatMetaLoaded: session.serverChatMetaLoaded,
      serverChatState: session.serverChatState,
      serverChatVersion: session.serverChatVersion,
      serverChatTopic: session.serverChatTopic,
      serverChatClusterId: session.serverChatClusterId,
      serverChatSource: session.serverChatSource,
      serverChatExternalRef: session.serverChatExternalRef,
      selectedKnowledge: session.selectedKnowledge,
      replyTarget: session.replyTarget,
      actionInfo: session.actionInfo
    })
  }, [])

  useEffect(() => watchChatAccountChanges(invalidated => {
    if (!invalidated) return
    sessionsRef.current.clear()
    baselineSessionRef.current = null
    previousMediaIdRef.current = null
    applySession(createEmptySession())
    setAccountRevision(revision => revision + 1)
  }), [applySession])

  const ensureDocumentSources = useCallback(() => {
    if (isMediaDbOnlySources(ragSources)) {
      return
    }
    if (!previousRagSourcesRef.current) {
      previousRagSourcesRef.current = ragSources
    }
    setRagSources(["media_db"])
  }, [ragSources, setRagSources])

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
      if (mediaId === null) return
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

  useEffect(() => {
    activeMediaIdRef.current = mediaId
  }, [mediaId])

  useEffect(() => {
    if (!baselineSessionRef.current) {
      baselineSessionRef.current = snapshotSession()
      const state = useStoreMessageOption.getState()
      baselineRagSettingsRef.current = {
        ragMediaIds: state.ragMediaIds,
        ragSources: state.ragSources
      }
    }
  }, [snapshotSession])

  // Update RAG scope when mediaId changes
  useEffect(() => {
    if (mediaId === previousMediaIdRef.current) {
      return
    }

    const previousMediaId = previousMediaIdRef.current
    const loadGeneration = ++loadGenerationRef.current

    // Save previous session to memory and persist to IndexedDB
    if (previousMediaId !== null) {
      const session = snapshotSession()
      const revision = usePlaygroundSessionStore.getState().restoreRevision
      sessionsRef.current.set(previousMediaId, session)
      // Persist to IndexedDB in background (don't await)
      persistDocumentSession(previousMediaId, session).then((historyId) => {
        if (historyId && revision === usePlaygroundSessionStore.getState().restoreRevision) {
          const savedSession = sessionsRef.current.get(previousMediaId)
          if (savedSession === session) {
            sessionsRef.current.set(previousMediaId, {
              ...savedSession,
              historyId
            })
          }
        }
      })
    }

    previousMediaIdRef.current = mediaId

    if (mediaId !== null) {
      // Try to get from memory first
      let session = sessionsRef.current.get(mediaId)

      if (session) {
        // Use cached in-memory session
        applySession(session)
        setRagMediaIds([mediaId])
        if (isDocumentKnowledge(session.selectedKnowledge, mediaId)) {
          ensureDocumentSources()
        } else {
          restoreRagSources()
        }
      } else {
        // Try to load from IndexedDB
        const restoreRevision = usePlaygroundSessionStore.getState().restoreRevision
        loadDocumentSession(mediaId).then((loadedSession) => {
          // Only apply if we're still on the same document
          if (loadGeneration === loadGenerationRef.current && activeMediaIdRef.current === mediaId && restoreRevision === usePlaygroundSessionStore.getState().restoreRevision) {
            const sessionToUse = loadedSession ?? createEmptySession()
            sessionsRef.current.set(mediaId, sessionToUse)
            applySession(sessionToUse)
            // Scope RAG to this specific document
            setRagMediaIds([mediaId])
            if (isDocumentKnowledge(sessionToUse.selectedKnowledge, mediaId)) {
              ensureDocumentSources()
            } else {
              restoreRagSources()
            }
          }
        })

        // Apply empty session immediately while loading
        applySession(createEmptySession())
        setRagMediaIds([mediaId])
        restoreRagSources()
      }
    } else {
      // Clear document-specific RAG scope
      applySession(createEmptySession())
      setRagMediaIds(null)
      restoreRagSources()
    }
  }, [
    accountRevision,
    applySession,
    ensureDocumentSources,
    mediaId,
    restoreRagSources,
    setRagMediaIds,
    snapshotSession
  ])

  // Clean up on unmount
  useEffect(() => {
    return () => {
      loadGenerationRef.current += 1
      const currentMediaId = activeMediaIdRef.current
      if (currentMediaId !== null) {
        const session = snapshotSession()
        // Persist to IndexedDB (fire-and-forget since we're unmounting)
        persistDocumentSession(currentMediaId, session)
      }
      // StrictMode replays effects after cleanup; the replacement mount must
      // start a fresh read instead of retaining the cancelled load's marker.
      previousMediaIdRef.current = null
      if (baselineSessionRef.current) {
        applySession(baselineSessionRef.current)
      }
      const baselineRag = baselineRagSettingsRef.current
      if (baselineRag) {
        setRagMediaIds(baselineRag.ragMediaIds)
        setRagSources(baselineRag.ragSources)
      } else {
        // Reset RAG scope when leaving document workspace
        setRagMediaIds(null)
        restoreRagSources()
      }
      previousRagSourcesRef.current = null
    }
  }, [
    applySession,
    restoreRagSources,
    setRagMediaIds,
    setRagSources,
    snapshotSession
  ])

  // Clear messages for new document session
  const clearDocumentChat = useCallback(() => {
    const currentMediaId = activeMediaIdRef.current
    const existingSession = snapshotSession()
    const clearedSession: DocumentChatSession = {
      ...existingSession,
      messages: [],
      history: [],
      historyId: null,
      isFirstMessage: true,
      serverChatId: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatMetaLoaded: false,
      serverChatState: null,
      serverChatVersion: null,
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      replyTarget: null,
      actionInfo: null
    }
    if (currentMediaId !== null) {
      sessionsRef.current.set(currentMediaId, clearedSession)
    }
    applySession(clearedSession)
  }, [applySession, snapshotSession])

  // Check if chat is ready for use
  const isChatReady = isServerAvailable && mediaId !== null

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
    isServerAvailable,
    ragEnabled,

    // RAG configuration
    ragMediaIds,
    ragSources,
    ragSearchMode,
    ragTopK,

    // Actions
    clearDocumentChat,
    setRagEnabled,
    setMessages
  }
}

export type UseDocumentChatReturn = ReturnType<typeof useDocumentChat>
