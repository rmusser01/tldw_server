import React, { useEffect, useRef, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Empty, Skeleton, message, Modal, Input, Button } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useShallow } from "zustand/react/shallow"
import { FolderPlus } from "lucide-react"

import { useConnectionState } from "@/hooks/useConnectionState"
import { isRecoverableAuthConfigError } from "@/services/auth-errors"
import { createServicePromptScopeChangedError, isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { FolderTree } from "@/components/Folders"
import { useFolderStore, type FolderOperation } from "@/store/folder"
import { useConnectionStore } from "@/store/connection"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreMessageOption } from "@/store/option"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { cn } from "@/libs/utils"

interface FolderChatListProps {
  className?: string
  onConversationSelected?: () => void
  owner?: FolderOperation & { key: string }
  verificationState?: "pending" | "ready" | "error"
  onRetryVerification?: () => void
  readOnly?: boolean
}

const MISSING_CHAT_BATCH_SIZE = 5
const isGlobalChat = (chat: ServerChatSummary, id: string) =>
  String(chat.id) === id && (!chat.scope_type || chat.scope_type === "global") && !chat.workspace_id

export function FolderChatList({ className, onConversationSelected, owner,
  verificationState = "pending", onRetryVerification, readOnly = false }: FolderChatListProps) {
  const { t } = useTranslation(["common"])
  const { isConnected } = useConnectionState()
  const checkConnection = useConnectionStore((state) => state.checkOnce)
  const { restoreRevision, sourceSelectionRevision } = usePlaygroundSessionStore(useShallow(state => ({
    restoreRevision: state.restoreRevision, sourceSelectionRevision: state.sourceSelectionRevision
  })))
  const draftRevision = useStoreMessageOption(state => state.chatDraftRevision)
  const temporaryChat = useStoreMessageOption(state => state.temporaryChat)
  const readOnlyNow = useRef(readOnly)
  readOnlyNow.current = readOnly
  const lifetime = useRef<{ operation: FolderOperation; controller: AbortController } | null>(null)
  const selectionRequest = useRef<AbortController | null>(null)
  const createDialog = useRef<object | null>(null)
  const createRequest = useRef<object | null>(null)
  const [newFolderModalOpen, setNewFolderModalOpen] = useState(false)
  const [newFolderName, setNewFolderName] = useState("")
  const [isCreating, setIsCreating] = useState(false)
  const selectServerChat = useSelectServerChat()
  const folderData = useFolderStore(useShallow(state => ({
      folders: state.folders, keywords: state.keywords, folderKeywordLinks: state.folderKeywordLinks,
      conversationKeywordLinks: state.conversationKeywordLinks, isLoading: state.isLoading, error: state.error,
      folderApiAvailable: state.folderApiAvailable, serverDataScope: state.serverDataScope,
      refreshFromServer: state.refreshFromServer, createFolder: state.createFolder, getConversationsForFolder: state.getConversationsForFolder
    })))
  const { folders, isLoading, error, folderApiAvailable, serverDataScope, refreshFromServer, createFolder } = folderData
  const operation = useMemo<FolderOperation | null>(() => {
    if (!owner || verificationState !== "ready" || !isConnected) return null
    const captured: FolderOperation = {
      requestScope: owner.requestScope,
      isCurrent: () => lifetime.current?.operation === captured && !lifetime.current.controller.signal.aborted &&
        owner.isCurrent() && useConnectionStore.getState().state.isConnected &&
        restoreRevision === usePlaygroundSessionStore.getState().restoreRevision &&
        sourceSelectionRevision === usePlaygroundSessionStore.getState().sourceSelectionRevision &&
        draftRevision === useStoreMessageOption.getState().chatDraftRevision &&
        readOnly === readOnlyNow.current
    }
    return captured
  }, [owner, verificationState, restoreRevision, sourceSelectionRevision, draftRevision, readOnly, isConnected])

  React.useLayoutEffect(() => {
    const controller = new AbortController()
    lifetime.current = operation ? { operation, controller } : null
    createDialog.current = null; createRequest.current = null
    setNewFolderModalOpen(false); setNewFolderName(""); setIsCreating(false)
    return () => {
      controller.abort()
      selectionRequest.current?.abort(); selectionRequest.current = null
      if (lifetime.current?.controller === controller) lifetime.current = null
      createDialog.current = null; createRequest.current = null
      if (operation) useFolderStore.getState().releaseServerData(operation)
    }
  }, [operation])

  const refresh = React.useCallback(async () => {
    if (!operation?.isCurrent()) return
    const result = await refreshFromServer(operation)
    if (result === "scope-changed" && operation.isCurrent()) onRetryVerification?.()
  }, [operation, refreshFromServer, onRetryVerification])
  useEffect(() => { void refresh() }, [refresh])

  const hasFreshSnapshot = React.useCallback(() => {
    const state = useFolderStore.getState()
    return !!operation?.isCurrent() && state.serverDataScope === operation &&
      !state.isLoading && !state.error && state.folderApiAvailable === true
  }, [operation])
  const ready = serverDataScope === operation && !!operation?.isCurrent() &&
    !isLoading && !error && folderApiAvailable === true
  const canManage = ready && !readOnly && !temporaryChat
  const folderConversationIds = useMemo(() => !ready ? [] : Array.from(new Set(
    folderData.folders.filter(folder => !folder.deleted).flatMap(folder => folderData.getConversationsForFolder(folder.id))
  )).sort((a, b) => a.localeCompare(b)), [ready, folderData])

  const titleQuery = useQuery({
    queryKey: ["folderConversationMissingChats", owner?.key, restoreRevision, sourceSelectionRevision,
      draftRevision, readOnly, folderConversationIds],
    enabled: ready && folderConversationIds.length > 0,
    queryFn: async ({ signal: querySignal }) => {
      const viewSignal = lifetime.current?.operation === operation ? lifetime.current.controller.signal : null
      if (!operation || !viewSignal) throw createServicePromptScopeChangedError()
      const signal = AbortSignal.any([querySignal, viewSignal])
      const assertCurrent = () => {
        signal.throwIfAborted()
        if (!hasFreshSnapshot()) throw createServicePromptScopeChangedError()
      }
      const chats: ServerChatSummary[] = []
      const unavailable: string[] = []
      for (let index = 0; index < folderConversationIds.length; index += MISSING_CHAT_BATCH_SIZE) {
        assertCurrent()
        const batch = await Promise.all(folderConversationIds.slice(index, index + MISSING_CHAT_BATCH_SIZE).map(async id => {
          assertCurrent()
          try {
            const chat = await tldwClient.getChat(id, { requestScope: operation.requestScope, scope: { type: "global" }, signal })
            assertCurrent()
            if (isGlobalChat(chat, id)) return chat
            unavailable.push(id)
          } catch (error) {
            signal.throwIfAborted()
            if (isRequestConfigScopeChangedError(error) || (error as { status?: unknown })?.status === 412) {
              if (useFolderStore.getState().serverDataScope === operation) {
                useFolderStore.getState().releaseServerData(operation)
                onRetryVerification?.()
              }
              throw error
            }
            assertCurrent()
            if (isRecoverableAuthConfigError(error)) throw error
            unavailable.push(id)
          }
          return null
        }))
        assertCurrent()
        chats.push(...batch.filter((chat): chat is ServerChatSummary => chat !== null))
      }
      return { operation, chats, unavailable }
    },
    staleTime: 0,
    gcTime: 0,
    retry: false
  })
  const currentTitles = ready && titleQuery.data?.operation === operation ? titleQuery.data : null
  const folderTreeConversations = useMemo(() => {
    const byId = new Map(currentTitles?.chats.map(chat => [String(chat.id), chat]) || [])
    const counts = new Map<string, number>()
    byId.forEach(chat => { if (chat.title) counts.set(chat.title, (counts.get(chat.title) || 0) + 1) })
    return folderConversationIds.map(id => {
      const title = byId.get(id)?.title
      return { id, title: title ? ((counts.get(title) || 0) > 1 ? `${title} · ${id}` : title)
        : t("common:chatSidebar.folderConversationFallback", { defaultValue: "Conversation {{id}}", id }) }
    })
  }, [folderConversationIds, currentTitles, t])

  const loadServerChatById = async (id: string) => {
    if (!operation || !hasFreshSnapshot() || !folderConversationIds.includes(id)) return
    selectionRequest.current?.abort()
    const controller = new AbortController(); selectionRequest.current = controller
    const viewSignal = lifetime.current!.controller.signal
    const signal = AbortSignal.any([controller.signal, viewSignal])
    const current = () => selectionRequest.current === controller && !signal.aborted && hasFreshSnapshot()
    try {
      const chat = await tldwClient.getChat(id, { requestScope: operation.requestScope, scope: { type: "global" }, signal })
      if (!current()) return
      if (!isGlobalChat(chat, id)) throw new Error("Conversation is unavailable in this history view.")
      selectServerChat(chat)
      onConversationSelected?.()
    } catch (error) {
      if (!current()) return
      if (isRequestConfigScopeChangedError(error) || (error as { status?: unknown })?.status === 412) {
        useFolderStore.getState().releaseServerData(operation)
        onRetryVerification?.()
      }
      message.error(t("common:error.friendlyGenericSummary"))
    } finally {
      if (selectionRequest.current === controller) selectionRequest.current = null
    }
  }
  const closeCreate = () => {
    createDialog.current = null; createRequest.current = null
    setNewFolderModalOpen(false); setNewFolderName(""); setIsCreating(false)
  }
  const handleCreateFolder = async () => {
    const token = createDialog.current
    const name = newFolderName.trim()
    if (!token || createRequest.current === token || !name || !operation || !hasFreshSnapshot() ||
      readOnlyNow.current || useStoreMessageOption.getState().temporaryChat) return
    const current = () => createDialog.current === token && operation.isCurrent() &&
      !readOnlyNow.current && !useStoreMessageOption.getState().temporaryChat
    createRequest.current = token; setIsCreating(true)
    try {
      const created = await createFolder(name, undefined, operation)
      if (!current()) return
      if (!created) {
        if (useFolderStore.getState().serverDataScope !== operation) onRetryVerification?.()
        message.error(t("common:error.createFolder"))
        return
      }
      closeCreate()
      message.success(t("common:chatSidebar.folderCreated", "Folder created successfully"))
    } catch {
      if (current()) message.error(t("common:error.createFolder"))
    } finally {
      if (createRequest.current === token) { createRequest.current = null; setIsCreating(false) }
    }
  }
  const organizationFailed = serverDataScope === operation && !!operation?.isCurrent() && !isLoading &&
    (error || folderApiAvailable === false)

  if (!isConnected) return <div className={cn("py-8 text-center", className)}>
    <Empty description={t("common:chatSidebar.foldersRequireConnection")} />
    <Button onClick={() => void checkConnection().catch(() => null)}>{t("common:retry")}</Button>
  </div>
  if (!owner || verificationState !== "ready" || !operation?.isCurrent()) return <div className={cn("py-8 text-center", className)}>
    {verificationState === "error" ? <><p role="alert">{t("common:chatSidebar.folderAccountError", "Unable to verify this account.")}</p>
      <Button onClick={onRetryVerification}>{t("common:retry")}</Button></> : <Skeleton active paragraph={{ rows: 4 }} />}
  </div>

  return <div className={cn("flex flex-col", className)}>
    {organizationFailed || (!ready && !isLoading) ? <div className="space-y-2 px-3 py-4">
      <p role="alert">{folderApiAvailable === false ? t("common:chatSidebar.foldersUnavailable", "Folders are unavailable on this server.")
        : t("common:chatSidebar.folderLoadError", "Unable to load folders for this account.")}</p>
      <Button onClick={refresh}>{t("common:retry")}</Button>
    </div> : null}
    {isLoading && serverDataScope === operation ? <Skeleton active paragraph={{ rows: 4 }} /> : null}
    {ready ? <>
      <div className="flex items-center justify-end px-3 py-2">
        <button type="button" disabled={!canManage} onClick={() => {
          if (!hasFreshSnapshot() || readOnlyNow.current || useStoreMessageOption.getState().temporaryChat) return
          createDialog.current = {}; createRequest.current = null; setIsCreating(false)
          setNewFolderModalOpen(true)
        }} className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-subtle hover:bg-surface hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)] disabled:opacity-50">
          <FolderPlus className="size-3.5" /><span>{t("common:newFolder")}</span>
        </button>
      </div>
      {readOnly || temporaryChat ? <p className="px-3 pb-2 text-xs text-text-subtle">{t("common:chatSidebar.folderReadOnly", "Temporary chat can preview history. Switch to History to organize it.")}</p> : null}
      {titleQuery.isFetching ? <p role="status" className="px-3 pb-2 text-xs text-text-subtle">{t("common:chatSidebar.folderTitlesLoading", "Loading conversation details…")}</p> : null}
      {titleQuery.isError || (currentTitles && currentTitles.unavailable.length > 0) ? <div className="space-y-2 px-3 pb-3">
        <p role="status" className="text-xs text-text-subtle">{t("common:chatSidebar.folderTitlesPartial", "Some conversation details are unavailable. Folder links remain visible.")}</p>
        <Button size="small" onClick={() => { if (hasFreshSnapshot()) void titleQuery.refetch() }}>{t("common:retry")}</Button>
      </div> : null}
      {folders.some(folder => !folder.deleted) ? <FolderTree operation={operation}
        onConversationSelect={id => { void loadServerChatById(id) }} conversations={folderTreeConversations} showConversations />
        : <Empty description={t("common:chatSidebar.foldersEmpty", "No folders yet. Create a folder to organize your chats.")} className="py-8" />}
    </> : null}
    <Modal open={newFolderModalOpen} onCancel={closeCreate} onOk={handleCreateFolder}
      okText={t("common:create")} cancelText={t("common:cancel")}
      okButtonProps={{ loading: isCreating, disabled: !newFolderName.trim() || !canManage }} title={t("common:newFolder")}>
      <Input aria-label={t("common:folderName")} placeholder={t("common:folderName")} value={newFolderName}
        onChange={event => setNewFolderName(event.target.value)} onPressEnter={handleCreateFolder} autoFocus />
      {!ready && !isCreating ? <div className="mt-3 space-y-2"><p role="alert">{t("common:chatSidebar.folderLoadError", "Unable to load folders for this account.")}</p>
        <Button onClick={refresh}>{t("common:retry")}</Button></div> : null}
    </Modal>
  </div>
}

export default FolderChatList
