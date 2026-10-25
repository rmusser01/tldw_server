import React from "react"
import { useTranslation } from "react-i18next"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Empty, Skeleton, Input, Modal, Select, message } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { FolderPlus, RotateCcw, Tag, Trash2 } from "lucide-react"
import { browser } from "wxt/browser"

import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { useConnectionStore } from "@/store/connection"
import type { VisibleServerChats } from "@/db/dexie/chat"
import { useConnectionState } from "@/hooks/useConnectionState"
import {
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
  useServerChatHistory,
  type ServerChatHistoryItem
} from "@/hooks/useServerChatHistory"
import { useSetting } from "@/hooks/useSetting"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { useBulkChatOperations } from "@/hooks/useBulkChatOperations"
import { useStoreMessageOption } from "@/store/option"
import { useFolderStore } from "@/store/folder"
import {
  shouldEnableOptionalResource,
  useChatSurfaceCoordinatorStore
} from "@/store/chat-surface-coordinator"
import { shallow } from "zustand/shallow"
import {
  tldwClient,
  type ConversationState,
  type ServerChatSummary
} from "@/services/tldw/TldwApiClient"
import { updatePageTitle } from "@/utils/update-page-title"
import { cn } from "@/libs/utils"
import { normalizeConversationState } from "@/utils/conversation-state"
import { isDictionaryVersionConflictError } from "@/components/Option/Dictionaries/listUtils"
import { useDataTablesStore } from "@/store/data-tables"
import { queueDataTablesPrefill } from "@/utils/data-tables-prefill"
import { startCreateTableFromChat } from "@/utils/data-tables-create-flow"
import {
  SIDEBAR_SERVER_CHAT_FILTER_SETTING,
  type SidebarServerChatFilterValue
} from "@/services/settings/ui-settings"
import { ServerChatRow } from "./ServerChatRow"
import { removeLocalCopyOfServerChat } from "@/components/Sidepanel/Chat/tab-conversation-actions"

const BulkFolderPickerModal = React.lazy(
  () => import("@/components/Sidepanel/Chat/BulkFolderPickerModal")
)
const BulkTagPickerModal = React.lazy(
  () => import("@/components/Sidepanel/Chat/BulkTagPickerModal")
)

const CHAT_HISTORY_PAGE_SIZE = 25
type BulkConfirmAction = "trash" | "hard_delete"

interface ServerChatListProps {
  searchQuery: string
  className?: string
  selectionMode?: boolean
  /** Temporary mode: saved chats open read-only, without pin or management actions. */
  readOnly?: boolean
  onConversationSelected?: () => void
  onVisibleChatsChange?: (receipt: VisibleServerChats | null) => void
}

type ServerHistoryAction = {
  serverScopeKey: string
  requestScope: ServicePromptRequestScope
  isCurrent: () => boolean
}

type UpdateChatRequestPayload = {
  action: ServerHistoryAction
  chatId: string
  data:
    | { title?: string }
    | { topic_label?: string | null }
    | { state?: ConversationState }
  expectedVersion?: number | null
}

export function ServerChatList({
  searchQuery,
  className,
  selectionMode: selectionModeProp,
  readOnly = false,
  onConversationSelected,
  onVisibleChatsChange
}: ServerChatListProps) {
  const { t } = useTranslation([
    "common",
    "sidepanel",
    "option",
    "playground",
    "dataTables"
  ])
  const { isConnected } = useConnectionState()
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()
  const [pinnedChatIds, setPinnedChatIds] = useStorage<string[]>(
    "tldw:server-chat-pins",
    []
  )
  const [chatTypeFilter, setChatTypeFilter] = useSetting(
    SIDEBAR_SERVER_CHAT_FILTER_SETTING
  )
  const isTrashView = chatTypeFilter === "trash"

  const {
    serverChatId,
    setServerChatTitle,
    setServerChatState,
    setServerChatVersion,
    setServerChatTopic
  } = useStoreMessageOption(
    (state) => ({
      serverChatId: state.serverChatId,
      setServerChatTitle: state.setServerChatTitle,
      setServerChatState: state.setServerChatState,
      setServerChatVersion: state.setServerChatVersion,
      setServerChatTopic: state.setServerChatTopic
    }),
    shallow
  )
  const selectServerChat = useSelectServerChat()
  const clearChat = useClearChat()
  const { resetWizard, addSource, setWizardStep } = useDataTablesStore(
    (state) => ({
      resetWizard: state.resetWizard,
      addSource: state.addSource,
      setWizardStep: state.setWizardStep
    }),
    shallow
  )
  const [openMenuFor, setOpenMenuFor] = React.useState<string | null>(null)
  const [renamingChat, setRenamingChat] =
    React.useState<ServerChatHistoryItem | null>(null)
  const [renameValue, setRenameValue] = React.useState("")
  const [renameError, setRenameError] = React.useState<string | null>(null)
  const [editingTopicChat, setEditingTopicChat] =
    React.useState<ServerChatHistoryItem | null>(null)
  const [topicValue, setTopicValue] = React.useState("")
  const restoreRevision = usePlaygroundSessionStore(
    (state) => state.restoreRevision
  )
  const sourceSelectionRevision = usePlaygroundSessionStore(
    (state) => state.sourceSelectionRevision
  )
  const chatDraftRevision = useStoreMessageOption(
    (state) => state.chatDraftRevision
  )
  const selectionMode = selectionModeProp ?? false
  const [selectedChatIds, setSelectedChatIds] = React.useState<string[]>([])
  const [currentPage, setCurrentPage] = React.useState(1)
  const [pageJumpValue, setPageJumpValue] = React.useState("1")
  const selectedChatIdSet = React.useMemo(
    () => new Set(selectedChatIds),
    [selectedChatIds]
  )
  const [bulkFolderPickerOpen, setBulkFolderPickerOpen] = React.useState(false)
  const [bulkTagPickerOpen, setBulkTagPickerOpen] = React.useState(false)
  const [bulkDeleteConfirmOpen, setBulkDeleteConfirmOpen] =
    React.useState(false)
  const [bulkConfirmAction, setBulkConfirmAction] =
    React.useState<BulkConfirmAction>("trash")
  const [isBulkDeleting, setIsBulkDeleting] = React.useState(false)
  const openSettingsTimeoutRef = React.useRef<number | null>(null)
  const markPanelEngaged = useChatSurfaceCoordinatorStore(
    (state) => state.markPanelEngaged
  )
  const serverHistoryOverviewEnabled = useChatSurfaceCoordinatorStore((state) =>
    shouldEnableOptionalResource(state, "server-history")
  )
  const hasSearchQuery = searchQuery.trim().length > 0
  const {
    ownerKey: historyOwnerKey,
    serverScopeKey: historyServerScopeKey,
    requestScope: historyRequestScope,
    isCurrent: historyOwnerCurrent,
    verificationState: historyVerificationState,
    retryVerification: retryHistoryVerification
  } = useChatDraftOwner(() => {
    setOpenMenuFor(null)
    setRenamingChat(null)
    setEditingTopicChat(null)
    setSelectedChatIds([])
    setBulkFolderPickerOpen(false)
    setBulkTagPickerOpen(false)
    setBulkDeleteConfirmOpen(false)
  })
  const ownerReady =
    historyVerificationState === "ready" &&
    !!historyOwnerKey &&
    !!historyRequestScope &&
    historyOwnerCurrent()
  const historyOwner = React.useMemo(
    () =>
      ownerReady
        ? {
            key: historyOwnerKey!,
            requestScope: historyRequestScope!,
            isCurrent: historyOwnerCurrent
          }
        : undefined,
    [ownerReady, historyOwnerKey, historyRequestScope, historyOwnerCurrent]
  )

  const openExtensionUrl = React.useCallback(
    (path: `/options.html${string}` | `/sidepanel.html${string}`) => {
      try {
        if (browser?.runtime?.getURL) {
          const url = browser.runtime.getURL(path)
          if (browser.tabs?.create) {
            browser.tabs.create({ url })
          } else {
            window.open(url, "_blank")
          }
          return
        }
      } catch (err) {
        console.debug(
          "[ServerChatList] openExtensionUrl browser API unavailable:",
          err
        )
      }

      try {
        if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
          const url = chrome.runtime.getURL(path)
          window.open(url, "_blank")
          return
        }
        if (
          typeof chrome !== "undefined" &&
          chrome.runtime?.openOptionsPage &&
          path.includes("/options.html")
        ) {
          chrome.runtime.openOptionsPage()
          return
        }
      } catch (err) {
        console.debug(
          "[ServerChatList] openExtensionUrl chrome API unavailable:",
          err
        )
      }

      let fallbackUrl: string = path
      try {
        fallbackUrl = new URL(path, window.location.origin).toString()
      } catch (err) {
        console.warn(
          "[ServerChatList] openExtensionUrl failed to build fallback URL:",
          err
        )
      }
      console.warn(
        "[ServerChatList] openExtensionUrl runtime API unavailable; falling back to",
        fallbackUrl
      )
      window.open(fallbackUrl, "_blank")
    },
    []
  )

  const { folderApiAvailable } = useFolderStore(
    (state) => ({
      folderApiAvailable: state.serverDataScope?.requestScope === historyRequestScope &&
        state.serverDataScope.isCurrent() ? state.folderApiAvailable : null
    }),
    shallow
  )

  const updateChatRequest = React.useCallback(
    async (payload: UpdateChatRequestPayload): Promise<ServerChatSummary> => {
      if (!payload.action.isCurrent())
        throw new Error("server_history_action_expired")
      return tldwClient.updateChat(payload.chatId, payload.data, {
        expectedVersion: payload.expectedVersion ?? undefined,
        requestScope: payload.action.requestScope,
        scope: { type: "global" }
      })
    },
    []
  )

  const invalidateServerChatHistory = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
  }, [queryClient])

  const { mutate: updateChatMetadata } = useMutation<
    ServerChatSummary,
    Error,
    UpdateChatRequestPayload
  >({
    mutationKey: ["updateChatMetadata"],
    mutationFn: updateChatRequest,
    onSettled: (_data, _error, payload) => {
      if (payload.action.isCurrent()) invalidateServerChatHistory()
    }
  })
  const { mutate: renameChat, isPending: renameLoading } = useMutation<
    ServerChatSummary,
    Error,
    UpdateChatRequestPayload
  >({
    mutationKey: ["renameChat"],
    mutationFn: updateChatRequest,
    onSettled: (_data, _error, payload) => {
      if (payload.action.isCurrent()) invalidateServerChatHistory()
    }
  })
  const { mutate: updateChatTopic, isPending: topicLoading } = useMutation<
    ServerChatSummary,
    Error,
    UpdateChatRequestPayload
  >({
    mutationKey: ["updateChatTopic"],
    mutationFn: updateChatRequest,
    onSettled: (_data, _error, payload) => {
      if (payload.action.isCurrent()) invalidateServerChatHistory()
    }
  })

  const {
    data: serverChatData,
    total: serverChatTotal = 0,
    sidebarRefreshState,
    hasUsableData,
    isShowingStaleData,
    isLoading,
    isServerPagedResult = false,
    status,
    isFetching,
    refetch
  } = useServerChatHistory(searchQuery, {
    deletedOnly: isTrashView,
    enabled: ownerReady && (hasSearchQuery || serverHistoryOverviewEnabled),
    owner: historyOwner,
    scope: { type: "global" },
    mode: hasSearchQuery ? "search" : "overview",
    page: currentPage,
    limit: SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
    filterMode: chatTypeFilter
  })
  const serverChats = React.useMemo(
    () =>
      ownerReady
        ? (serverChatData || []).filter(
            (chat) =>
              (!chat.scope_type || chat.scope_type === "global") &&
              !chat.workspace_id
          )
        : [],
    [ownerReady, serverChatData]
  )
  const hasForeignScope =
    ownerReady && serverChats.length !== (serverChatData || []).length
  const freshResults =
    ownerReady &&
    isConnected &&
    status === "success" &&
    sidebarRefreshState === "ready" &&
    !isFetching &&
    !isShowingStaleData

  React.useEffect(() => {
    if (hasSearchQuery) {
      markPanelEngaged("server-history")
    }
  }, [hasSearchQuery, markPanelEngaged])
  const getChatVersionConflictMessage = React.useCallback(
    () =>
      t("common:chatChangedRetryFailed", {
        defaultValue: "This chat changed on the server. Refresh and try again."
      }),
    [t]
  )
  const resolveChatMutationErrorMessage = React.useCallback(
    (error: unknown, fallbackKey: string, fallbackDefaultValue: string) =>
      isDictionaryVersionConflictError(error)
        ? getChatVersionConflictMessage()
        : t(fallbackKey, {
            defaultValue: fallbackDefaultValue
          }),
    [getChatVersionConflictMessage, t]
  )
  const filteredChatsByType = React.useMemo(() => {
    if (chatTypeFilter === "trash") return serverChats
    if (chatTypeFilter === "all") return serverChats
    const isCharacterChat = (chat: ServerChatHistoryItem) => {
      const characterId = chat.character_id
      if (characterId == null) return false
      if (typeof characterId === "string") return characterId.trim().length > 0
      return true
    }
    if (chatTypeFilter === "character") {
      return serverChats.filter((chat) => isCharacterChat(chat))
    }
    return serverChats.filter((chat) => !isCharacterChat(chat))
  }, [chatTypeFilter, serverChats])
  const pinnedChatSet = React.useMemo(
    () => (isTrashView ? new Set<string>() : new Set(pinnedChatIds || [])),
    [isTrashView, pinnedChatIds]
  )
  const pinnedChats = React.useMemo(
    () =>
      isTrashView
        ? []
        : filteredChatsByType.filter((chat) => pinnedChatSet.has(chat.id)),
    [filteredChatsByType, isTrashView, pinnedChatSet]
  )
  const unpinnedChats = React.useMemo(
    () =>
      isTrashView
        ? filteredChatsByType
        : filteredChatsByType.filter((chat) => !pinnedChatSet.has(chat.id)),
    [filteredChatsByType, isTrashView, pinnedChatSet]
  )
  const orderedChats = React.useMemo(
    () => [...pinnedChats, ...unpinnedChats],
    [pinnedChats, unpinnedChats]
  )
  const totalChatCount = isServerPagedResult
    ? serverChatTotal
    : orderedChats.length
  const totalPages = Math.max(
    1,
    Math.ceil(totalChatCount / CHAT_HISTORY_PAGE_SIZE)
  )
  const currentPageSafe = currentPage
  const pageStartIndex = (currentPageSafe - 1) * CHAT_HISTORY_PAGE_SIZE
  const pagedChats = React.useMemo(
    () =>
      isServerPagedResult
        ? orderedChats
        : orderedChats.slice(
            pageStartIndex,
            pageStartIndex + CHAT_HISTORY_PAGE_SIZE
          ),
    [isServerPagedResult, orderedChats, pageStartIndex]
  )
  const pagedPinnedChats = React.useMemo(
    () => pagedChats.filter((chat) => pinnedChatSet.has(chat.id)),
    [pagedChats, pinnedChatSet]
  )
  const pagedUnpinnedChats = React.useMemo(
    () => pagedChats.filter((chat) => !pinnedChatSet.has(chat.id)),
    [pagedChats, pinnedChatSet]
  )
  const pageStartNumber = pagedChats.length === 0 ? 0 : pageStartIndex + 1
  const pageEndNumber = pageStartIndex + pagedChats.length
  const canLoadNextPage =
    currentPageSafe < totalPages ||
    (isServerPagedResult &&
      freshResults &&
      (serverChatData || []).length === CHAT_HISTORY_PAGE_SIZE)
  const hasMultiplePages =
    currentPageSafe > 1 || totalPages > 1 || canLoadNextPage
  const visibleChatIds = React.useMemo(
    () => Array.from(new Set(pagedChats.map((chat) => chat.id))),
    [pagedChats]
  )
  const currentPageAndFilter = React.useRef({
    page: currentPageSafe,
    filter: chatTypeFilter
  })
  currentPageAndFilter.current = {
    page: currentPageSafe,
    filter: chatTypeFilter
  }
  const visibleReceiptRef = React.useRef<VisibleServerChats | null>(null)
  const visibleReceipt = React.useMemo<VisibleServerChats | null>(() => {
    if (!freshResults || !historyServerScopeKey) return null
    const receipt: VisibleServerChats = {
      serverScopeKey: historyServerScopeKey,
      query: searchQuery.trim().toLowerCase(),
      ids: new Set(visibleChatIds),
      isCurrent: () =>
        visibleReceiptRef.current === receipt &&
        currentPageAndFilter.current.page === currentPageSafe &&
        currentPageAndFilter.current.filter === chatTypeFilter &&
        historyOwnerCurrent() &&
        useConnectionStore.getState().state.isConnected
    }
    return receipt
  }, [
    freshResults,
    historyServerScopeKey,
    historyOwnerCurrent,
    searchQuery,
    visibleChatIds,
    currentPageSafe,
    chatTypeFilter
  ])
  visibleReceiptRef.current = visibleReceipt
  React.useLayoutEffect(() => {
    visibleReceiptRef.current = visibleReceipt
    onVisibleChatsChange?.(visibleReceipt)
    return () => {
      visibleReceiptRef.current = null
    }
  }, [onVisibleChatsChange, visibleReceipt])
  const readOnlyNow = React.useRef(readOnly)
  readOnlyNow.current = readOnly
  const managementAction = React.useRef<ServerHistoryAction | null>(null)
  const captureAction = React.useCallback(
    (ids: string[]): ServerHistoryAction | null => {
      const receipt = visibleReceiptRef.current
      if (
        readOnlyNow.current ||
        !historyRequestScope ||
        !receipt?.isCurrent() ||
        ids.length === 0 ||
        ids.some((id) => !receipt.ids.has(id))
      )
        return null
      const session = usePlaygroundSessionStore.getState()
      const draftRevision = useStoreMessageOption.getState().chatDraftRevision
      return {
        serverScopeKey: receipt.serverScopeKey,
        requestScope: historyRequestScope,
        isCurrent: () =>
          !readOnlyNow.current &&
          receipt.isCurrent() &&
          session.restoreRevision ===
            usePlaygroundSessionStore.getState().restoreRevision &&
          session.sourceSelectionRevision ===
            usePlaygroundSessionStore.getState().sourceSelectionRevision &&
          draftRevision === useStoreMessageOption.getState().chatDraftRevision
      }
    },
    [historyRequestScope]
  )
  React.useLayoutEffect(() => {
    managementAction.current = null
    setOpenMenuFor(null)
    setRenamingChat(null)
    setEditingTopicChat(null)
    setSelectedChatIds([])
    setBulkFolderPickerOpen(false)
    setBulkTagPickerOpen(false)
    setBulkDeleteConfirmOpen(false)
    setIsBulkDeleting(false)
  }, [
    visibleReceipt,
    readOnly,
    restoreRevision,
    sourceSelectionRevision,
    chatDraftRevision
  ])
  const visibleChatIdSet = React.useMemo(
    () => new Set(visibleChatIds),
    [visibleChatIds]
  )
  const chatById = React.useMemo(
    () => new Map(filteredChatsByType.map((chat) => [chat.id, chat])),
    [filteredChatsByType]
  )
  const selectedChats = React.useMemo(
    () =>
      selectedChatIds
        .map((id) => chatById.get(id))
        .filter(Boolean) as ServerChatHistoryItem[],
    [selectedChatIds, chatById]
  )
  const selectedConversationIds = React.useMemo(
    () => selectedChats.map((chat) => chat.id),
    [selectedChats]
  )
  const deleteServerChat = React.useCallback(
    async (
      chat: ServerChatHistoryItem,
      action: ServerHistoryAction,
      hardDelete = false
    ) => {
      if (!action.isCurrent()) throw new Error("server_history_action_expired")
      await tldwClient.deleteChat(chat.id, {
        requestScope: action.requestScope,
        scope: { type: "global" },
        expectedVersion: chat.version ?? undefined,
        ...(hardDelete ? { hardDelete: true } : {})
      })
      if (!action.isCurrent()) return
      const removed = await removeLocalCopyOfServerChat(
        chat.id,
        action.serverScopeKey,
        {
          scope: { type: "global" },
          validate_lease: action.isCurrent
        }
      )
      if (!removed && action.isCurrent())
        message.warning({
          key: "chat-device-copy-removal",
          content: t(
            "common:chatSidebar.deletedDeviceCopyRetained",
            "Removed from server history. A cached copy remains on this device because it could not be removed."
          )
        })
    },
    [t]
  )

  const { openBulkFolderPicker, openBulkTagPicker, applyBulkDelete } =
    useBulkChatOperations({
      selectedConversationIds,
      folderApiAvailable,
      deleteConversation: async (conversationId) => {
        const action = managementAction.current
        if (!action?.isCurrent())
          throw new Error("server_history_action_expired")
        const chat = chatById.get(conversationId)
        if (!chat) throw new Error("server_history_action_expired")
        await deleteServerChat(chat, action)
      },
      t,
      setBulkFolderPickerOpen,
      setBulkTagPickerOpen
    })

  React.useEffect(() => {
    setCurrentPage(1)
  }, [searchQuery, chatTypeFilter])

  React.useEffect(() => {
    setPageJumpValue(String(currentPageSafe))
  }, [currentPageSafe])

  const togglePinned = React.useCallback(
    (chatId: string) => {
      if (!captureAction([chatId])) return
      setPinnedChatIds((prev) => {
        const current = prev || []
        if (current.includes(chatId)) {
          return current.filter((id) => id !== chatId)
        }
        return [...current, chatId]
      })
    },
    [captureAction, setPinnedChatIds]
  )

  React.useEffect(() => {
    if (!selectionMode) {
      setSelectedChatIds((prev) => (prev.length === 0 ? prev : []))
      return
    }
    setSelectedChatIds((prev) => {
      const next = prev.filter((id) => visibleChatIdSet.has(id))
      if (
        next.length === prev.length &&
        next.every((id, idx) => id === prev[idx])
      ) {
        return prev
      }
      return next
    })
  }, [selectionMode, visibleChatIdSet])

  const handleCreateTable = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      const isOptionsPage =
        typeof window !== "undefined" &&
        window.location.pathname.endsWith("options.html")
      const navigateFn = (
        window as Window & { __tldwNavigate?: (path: string) => void }
      ).__tldwNavigate

      try {
        await startCreateTableFromChat(
          {
            id: chat.id,
            title: chat.title,
            topic_label: chat.topic_label
          },
          {
            isOptionsPage,
            navigate: navigateFn,
            resetWizard,
            addSource,
            setWizardStep,
            queuePrefill: queueDataTablesPrefill,
            openOptionsPage: () =>
              openExtensionUrl("/options.html#/data-tables")
          }
        )
      } catch (error) {
        if (!action.isCurrent()) return
        console.error("[ServerChatList] Failed to start table creation", {
          error,
          chatId: chat.id
        })
        resetWizard()
        message.error(
          t("dataTables:createFromChatError", {
            defaultValue: "Failed to start table creation."
          })
        )
      }
    },
    [addSource, captureAction, openExtensionUrl, resetWizard, setWizardStep, t]
  )

  React.useEffect(() => {
    if (selectionMode) {
      setOpenMenuFor(null)
      return
    }
    setBulkFolderPickerOpen(false)
    setBulkTagPickerOpen(false)
    setBulkDeleteConfirmOpen(false)
    setBulkConfirmAction("trash")
  }, [selectionMode])

  const toggleChatSelected = React.useCallback(
    (chatId: string) => {
      if (!captureAction([chatId])) return
      setSelectedChatIds((prev) => {
        const next = new Set(prev)
        if (next.has(chatId)) {
          next.delete(chatId)
        } else {
          next.add(chatId)
        }
        return Array.from(next)
      })
    },
    [captureAction]
  )

  const handleSelectAllVisible = React.useCallback(() => {
    if (!captureAction(visibleChatIds)) return
    setSelectedChatIds(visibleChatIds)
  }, [captureAction, visibleChatIds])

  const clearSelection = React.useCallback(() => {
    setSelectedChatIds([])
  }, [])

  const handlePreviousPage = React.useCallback(() => {
    setCurrentPage((prev) => Math.max(1, prev - 1))
  }, [])

  const handleNextPage = React.useCallback(() => {
    if (canLoadNextPage) setCurrentPage((page) => page + 1)
  }, [canLoadNextPage])

  const handleJumpToPage = React.useCallback(() => {
    const nextPage = Number.parseInt(pageJumpValue, 10)
    if (!Number.isFinite(nextPage)) {
      setPageJumpValue(String(currentPageSafe))
      return
    }
    const clamped = Math.min(totalPages, Math.max(1, nextPage))
    setCurrentPage(clamped)
  }, [currentPageSafe, pageJumpValue, totalPages])

  const handleRenameSubmit = () => {
    if (renameLoading) return
    if (!renamingChat) return
    const action = managementAction.current
    if (!action?.isCurrent()) return

    const newTitle = renameValue.trim()
    if (!newTitle) {
      setRenameError(
        t("common:renameChatEmptyError", {
          defaultValue: "Title cannot be empty."
        })
      )
      return
    }

    setRenameError(null)
    renameChat(
      {
        action,
        chatId: renamingChat.id,
        data: { title: newTitle },
        expectedVersion: renamingChat.version ?? null
      },
      {
        onSuccess: (updated) => {
          if (!action.isCurrent()) return
          const resolvedTitle = updated?.title || newTitle
          if (serverChatId === renamingChat.id) {
            setServerChatTitle(resolvedTitle)
            setServerChatVersion(updated?.version ?? null)
            updatePageTitle(resolvedTitle)
          }
          setRenamingChat(null)
          setRenameValue("")
        },
        onError: (error) => {
          if (!action.isCurrent()) return
          message.error(
            resolveChatMutationErrorMessage(
              error,
              "common:renameChatError",
              "Failed to rename chat."
            )
          )
        }
      }
    )
  }

  const handleTopicSubmit = () => {
    if (topicLoading) return
    if (!editingTopicChat) return
    const action = managementAction.current
    if (!action?.isCurrent()) return

    const nextTopic = topicValue.trim()
    updateChatTopic(
      {
        action,
        chatId: editingTopicChat.id,
        data: { topic_label: nextTopic || null },
        expectedVersion: editingTopicChat.version ?? null
      },
      {
        onSuccess: (updated) => {
          if (!action.isCurrent()) return
          const resolvedTopic = updated?.topic_label ?? (nextTopic || null)
          if (serverChatId === editingTopicChat.id) {
            setServerChatTopic(resolvedTopic)
            setServerChatVersion(updated?.version ?? null)
          }
          setEditingTopicChat(null)
          setTopicValue("")
        },
        onError: (error) => {
          if (!action.isCurrent()) return
          message.error(
            resolveChatMutationErrorMessage(
              error,
              "common:updateChatTopicError",
              "Failed to update chat topic."
            )
          )
        }
      }
    )
  }

  const handleUpdateState = React.useCallback(
    (chat: ServerChatHistoryItem, nextState: ConversationState) => {
      const action = captureAction([chat.id])
      if (!action) return
      updateChatMetadata(
        {
          action,
          chatId: chat.id,
          data: { state: nextState },
          expectedVersion: chat.version ?? null
        },
        {
          onSuccess: (updated) => {
            if (!action.isCurrent()) return
            const resolvedState = normalizeConversationState(
              updated?.state ?? nextState
            )
            if (serverChatId === chat.id) {
              setServerChatState(resolvedState)
              setServerChatVersion(updated?.version ?? null)
            }
          },
          onError: (error) => {
            if (!action.isCurrent()) return
            message.error(
              resolveChatMutationErrorMessage(
                error,
                "common:updateChatStateError",
                "Failed to update chat status."
              )
            )
          }
        }
      )
    },
    [
      captureAction,
      serverChatId,
      setServerChatState,
      setServerChatVersion,
      t,
      updateChatMetadata
    ]
  )

  const handleMoveChatToTrash = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      const ok = await confirmDanger({
        title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
        content: t("common:deleteHistoryConfirmation", {
          defaultValue: "Move this chat to trash?"
        }),
        okText: t("common:moveToTrash", { defaultValue: "Move to trash" }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" })
      })
      if (!ok || !action.isCurrent()) return

      try {
        await deleteServerChat(chat, action)
        if (!action.isCurrent()) return
        setPinnedChatIds((prev) => (prev || []).filter((id) => id !== chat.id))
        queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
        setOpenMenuFor(null)
        if (serverChatId === chat.id) {
          clearChat()
        }
      } catch (err) {
        if (!action.isCurrent()) return
        console.error("[ServerChatList] Failed to delete chat", {
          error: err,
          chatId: chat.id
        })
        message.error(
          resolveChatMutationErrorMessage(
            err,
            "common:deleteChatError",
            "Failed to move chat to trash."
          )
        )
      }
    },
    [
      captureAction,
      clearChat,
      deleteServerChat,
      confirmDanger,
      queryClient,
      resolveChatMutationErrorMessage,
      serverChatId,
      setPinnedChatIds
    ]
  )

  const handleRestoreChat = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      try {
        await tldwClient.restoreChat(chat.id, {
          requestScope: action.requestScope,
          scope: { type: "global" },
          expectedVersion: chat.version ?? undefined
        })
        if (!action.isCurrent()) return
        queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
        message.success(
          t("common:chatRestored", {
            defaultValue: "Chat restored."
          })
        )
      } catch (err) {
        if (!action.isCurrent()) return
        console.error("[ServerChatList] Failed to restore chat", {
          error: err,
          chatId: chat.id
        })
        message.error(
          resolveChatMutationErrorMessage(
            err,
            "common:restoreChatError",
            "Failed to restore chat."
          )
        )
      }
    },
    [captureAction, queryClient, resolveChatMutationErrorMessage]
  )

  const handleHardDeleteChat = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      const ok = await confirmDanger({
        title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
        content: t("common:deleteHistoryPermanentlyConfirmation", {
          defaultValue: "Delete this chat permanently? This cannot be undone."
        }),
        okText: t("common:deletePermanently", {
          defaultValue: "Delete permanently"
        }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" })
      })
      if (!ok || !action.isCurrent()) return

      try {
        await deleteServerChat(chat, action, true)
        if (!action.isCurrent()) return
        setPinnedChatIds((prev) => (prev || []).filter((id) => id !== chat.id))
        queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
        setOpenMenuFor(null)
        if (serverChatId === chat.id) {
          clearChat()
        }
      } catch (err) {
        if (!action.isCurrent()) return
        console.error("[ServerChatList] Failed to permanently delete chat", {
          error: err,
          chatId: chat.id
        })
        message.error(
          resolveChatMutationErrorMessage(
            err,
            "common:deleteChatError",
            "Failed to delete chat permanently."
          )
        )
      }
    },
    [
      captureAction,
      clearChat,
      deleteServerChat,
      confirmDanger,
      queryClient,
      resolveChatMutationErrorMessage,
      serverChatId,
      setPinnedChatIds
    ]
  )

  const handleOpenSettings = React.useCallback(
    (chat: ServerChatHistoryItem) => {
      if (!captureAction([chat.id])) return
      const receipt = visibleReceiptRef.current
      if (serverChatId !== chat.id) {
        if (selectServerChat(chat) === false) return
      }
      if (typeof window === "undefined") return
      if (openSettingsTimeoutRef.current !== null) {
        window.clearTimeout(openSettingsTimeoutRef.current)
      }
      openSettingsTimeoutRef.current = window.setTimeout(() => {
        if (
          !receipt?.isCurrent() ||
          useStoreMessageOption.getState().serverChatId !== chat.id
        )
          return
        window.dispatchEvent(new CustomEvent("tldw:open-model-settings"))
        openSettingsTimeoutRef.current = null
      }, 0)
    },
    [captureAction, selectServerChat, serverChatId]
  )

  React.useEffect(() => {
    return () => {
      if (openSettingsTimeoutRef.current !== null) {
        window.clearTimeout(openSettingsTimeoutRef.current)
        openSettingsTimeoutRef.current = null
      }
    }
  }, [])

  const handleRenameChat = React.useCallback(
    (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      managementAction.current = action
      setRenamingChat(chat)
      setRenameValue(chat.title || "")
      setRenameError(null)
    },
    [captureAction]
  )

  const handleEditTopic = React.useCallback(
    (chat: ServerChatHistoryItem) => {
      const action = captureAction([chat.id])
      if (!action) return
      managementAction.current = action
      setEditingTopicChat(chat)
      setTopicValue(chat.topic_label || "")
    },
    [captureAction]
  )

  const handleRowClick = React.useCallback(
    (chat: ServerChatHistoryItem) => {
      if (
        !freshResults ||
        !historyOwnerCurrent() ||
        !useConnectionStore.getState().state.isConnected ||
        !visibleChatIdSet.has(chat.id)
      )
        return
      if (selectionMode) {
        toggleChatSelected(chat.id)
        return
      }
      if (isTrashView) return
      if (selectServerChat(chat) === false) return
      onConversationSelected?.()
    },
    [
      freshResults,
      historyOwnerCurrent,
      visibleChatIdSet,
      isTrashView,
      onConversationSelected,
      selectionMode,
      selectServerChat,
      toggleChatSelected
    ]
  )

  const selectionPropsForChat = React.useCallback(
    (chatId: string) =>
      selectionMode
        ? {
            selectionMode: true as const,
            isSelected: selectedChatIdSet.has(chatId),
            onToggleSelected: toggleChatSelected
          }
        : { selectionMode: false as const },
    [selectionMode, selectedChatIdSet, toggleChatSelected]
  )

  const openBulkDeleteConfirm = React.useCallback(() => {
    const action = captureAction(selectedConversationIds)
    if (!action) return
    managementAction.current = action
    setBulkConfirmAction("trash")
    setBulkDeleteConfirmOpen(true)
  }, [captureAction, selectedConversationIds])

  const openBulkHardDeleteConfirm = React.useCallback(() => {
    const action = captureAction(selectedConversationIds)
    if (!action) return
    managementAction.current = action
    setBulkConfirmAction("hard_delete")
    setBulkDeleteConfirmOpen(true)
  }, [captureAction, selectedConversationIds])

  const handleBulkFolderPickerClose = React.useCallback(() => {
    setBulkFolderPickerOpen(false)
  }, [])

  const handleBulkTagPickerClose = React.useCallback(() => {
    setBulkTagPickerOpen(false)
  }, [])

  const handleBulkDeleteConfirmClose = React.useCallback(() => {
    setBulkDeleteConfirmOpen(false)
  }, [])

  const handleBulkRestore = React.useCallback(async () => {
    const action = captureAction(selectedConversationIds)
    if (!action) return

    setIsBulkDeleting(true)
    try {
      const results = await Promise.allSettled(
        selectedChats.map((chat) =>
          tldwClient.restoreChat(chat.id, {
            requestScope: action.requestScope,
            scope: { type: "global" },
            expectedVersion: chat.version ?? undefined
          })
        )
      )
      if (!action.isCurrent()) return
      const restoredConversationIds = new Set<string>()
      const failedConversationIds = new Set<string>()
      results.forEach((result, index) => {
        const chat = selectedChats[index]
        if (!chat) return
        if (result.status === "fulfilled") {
          restoredConversationIds.add(chat.id)
        } else {
          failedConversationIds.add(chat.id)
        }
      })

      const failureCount = failedConversationIds.size
      if (failureCount === 0) {
        message.success(
          t("sidepanel:multiSelect.restoreSuccess", {
            defaultValue: "Chats restored."
          })
        )
      } else if (failureCount === selectedChats.length) {
        message.error(
          t("sidepanel:multiSelect.restoreFailed", {
            defaultValue: "Unable to restore selected chats."
          })
        )
      } else {
        message.error(
          t("sidepanel:multiSelect.restorePartial", {
            defaultValue: "Some chats could not be restored."
          })
        )
      }

      if (restoredConversationIds.size > 0) {
        queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
      }
      setSelectedChatIds(
        selectedChats
          .map((chat) => chat.id)
          .filter((id) => failedConversationIds.has(id))
      )
    } catch (error) {
      if (!action.isCurrent()) return
      console.error("[ServerChatList] Failed to bulk restore chats", {
        error,
        selectedCount: selectedChats.length
      })
      message.error(
        t("sidepanel:multiSelect.restoreFailed", {
          defaultValue: "Unable to restore selected chats."
        })
      )
    } finally {
      if (action.isCurrent()) setIsBulkDeleting(false)
    }
  }, [captureAction, queryClient, selectedChats, selectedConversationIds, t])

  const handleBulkDelete = React.useCallback(async () => {
    const action = managementAction.current
    if (selectedChats.length === 0 || !action?.isCurrent()) return

    setIsBulkDeleting(true)
    try {
      if (bulkConfirmAction === "hard_delete") {
        const results = await Promise.allSettled(
          selectedChats.map((chat) =>
            deleteServerChat(chat, action, true)
          )
        )
        if (!action.isCurrent()) return
        const deletedConversationIds = new Set<string>()
        const failedConversationIds = new Set<string>()
        results.forEach((result, index) => {
          const chat = selectedChats[index]
          if (!chat) return
          if (result.status === "fulfilled") {
            deletedConversationIds.add(chat.id)
          } else {
            failedConversationIds.add(chat.id)
          }
        })

        const failureCount = failedConversationIds.size
        if (failureCount === 0) {
          message.success(
            t("sidepanel:multiSelect.deleteSuccess", {
              defaultValue: "Chats deleted permanently."
            })
          )
        } else if (failureCount === selectedChats.length) {
          message.error(
            t("sidepanel:multiSelect.deleteFailed", {
              defaultValue: "Unable to delete selected chats."
            })
          )
        } else {
          message.error(
            t("sidepanel:multiSelect.deletePartial", {
              defaultValue: "Some chats could not be deleted."
            })
          )
        }

        setPinnedChatIds((prev) =>
          (prev || []).filter((id) => !deletedConversationIds.has(id))
        )
        const selectedIds = selectedChats.map((chat) => chat.id)
        setSelectedChatIds(
          selectedIds.filter((id) => failedConversationIds.has(id))
        )
        if (serverChatId && deletedConversationIds.has(serverChatId)) {
          clearChat()
        }
      } else {
        const result = await applyBulkDelete(action.isCurrent)
        if (!result || !action.isCurrent()) return
        const { deletedConversationIds, failedConversationIds } = result
        setPinnedChatIds((prev) =>
          (prev || []).filter((id) => !deletedConversationIds.has(id))
        )
        setSelectedChatIds(
          selectedConversationIds.filter((id) => failedConversationIds.has(id))
        )
        if (serverChatId && deletedConversationIds.has(serverChatId)) {
          clearChat()
        }
      }
      queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
      setBulkDeleteConfirmOpen(false)
    } catch (error) {
      if (!action.isCurrent()) return
      console.error("[ServerChatList] Failed to bulk delete chats", {
        error,
        selectedCount: selectedChats.length
      })
      message.error(
        t("sidepanel:multiSelect.deleteFailed", {
          defaultValue: "Unable to delete selected chats."
        })
      )
    } finally {
      if (action.isCurrent()) setIsBulkDeleting(false)
    }
  }, [
    applyBulkDelete,
    bulkConfirmAction,
    clearChat,
    deleteServerChat,
    queryClient,
    selectedChats,
    selectedConversationIds,
    serverChatId,
    setPinnedChatIds,
    t
  ])

  const filterControl = (
    <div className="px-2 pt-1">
      <label id="chat-sidebar-type-filter-label" className="sr-only">
        {t("common:chatSidebar.filter.label", {
          defaultValue: "Chat type filter"
        })}
      </label>
      <Select<SidebarServerChatFilterValue>
        value={chatTypeFilter}
        onChange={(value) => {
          void setChatTypeFilter(value)
        }}
        size="small"
        className="w-full"
        aria-labelledby="chat-sidebar-type-filter-label"
        options={[
          {
            value: "all",
            label: t("common:chatSidebar.filter.allChats", {
              defaultValue: "All Chats"
            })
          },
          {
            value: "character",
            label: t("common:chatSidebar.filter.characterChats", {
              defaultValue: "Character Chats"
            })
          },
          {
            value: "non_character",
            label: t("common:chatSidebar.filter.nonCharacterChats", {
              defaultValue: "Non-Character Chats"
            })
          },
          {
            value: "trash",
            label: t("common:chatSidebar.filter.trash", {
              defaultValue: "Trash"
            })
          }
        ]}
      />
    </div>
  )
  const recoveryActions = (
    retry = () => {
      void refetch()
    }
  ) => (
    <div className="mt-2 flex flex-wrap gap-2">
      <button
        type="button"
        onClick={retry}
        className="rounded-md border border-border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-focus">
        {t("common:retry", { defaultValue: "Retry" })}
      </button>
      {freshResults && canLoadNextPage && (
        <button
          type="button"
          onClick={handleNextPage}
          className="rounded-md border border-border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-focus">
          {t("common:next", { defaultValue: "Next" })}
        </button>
      )}
      {currentPageSafe > 1 && (
        <button
          type="button"
          onClick={handlePreviousPage}
          className="rounded-md border border-border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-focus">
          {t("common:previous", { defaultValue: "Previous" })}
        </button>
      )}
    </div>
  )
  if (!ownerReady)
    return (
      <div
        className={cn("p-3 text-sm", className)}
        role={historyVerificationState === "error" ? "alert" : "status"}>
        {filterControl}
        {historyVerificationState === "error"
          ? t("common:chatSidebar.discovery.verifyError", {
              defaultValue: "Could not verify the account for Server history."
            })
          : t("common:chatSidebar.discovery.verifying", {
              defaultValue: "Verifying the account for Server history…"
            })}
        {historyVerificationState === "error" &&
          recoveryActions(retryHistoryVerification)}
      </div>
    )
  if (!isConnected)
    return (
      <div
        role="status"
        className={cn("p-3 text-sm text-text-muted", className)}>
        {filterControl}
        {t("common:chatSidebar.discovery.offline", {
          defaultValue:
            "Server history is offline. Owned cached copies remain available in On this device."
        })}
      </div>
    )
  if (!hasSearchQuery && !serverHistoryOverviewEnabled)
    return (
      <div className={cn("p-3 text-sm", className)}>
        {filterControl}
        {t("common:chatSidebar.loadServerChatsHint", {
          defaultValue:
            "Load server conversations on demand to keep the chat page lighter."
        })}
        <button
          type="button"
          data-testid="server-history-engage-button"
          onClick={() => markPanelEngaged("server-history")}
          className="rounded-md border border-border px-3 py-2 text-sm">
          {t("common:chatSidebar.loadServerChats", {
            defaultValue: "Load conversations"
          })}
        </button>
      </div>
    )
  if (isLoading && !hasUsableData)
    return (
      <div
        role="status"
        aria-label={t("common:loading.title", "Loading…")}
        className={className}>
        {filterControl}
        <Skeleton active paragraph={{ rows: 4 }} />
      </div>
    )
  if (
    (sidebarRefreshState === "hard-error" ||
      sidebarRefreshState === "recoverable-error") &&
    serverChats.length === 0
  )
    return (
      <div role="alert" className={cn("p-3 text-sm", className)}>
        {filterControl}
        {t("common:chatSidebar.discovery.refreshError", {
          defaultValue:
            "Could not refresh Server history. Try again; cached copies remain in On this device."
        })}
        {recoveryActions()}
      </div>
    )
  if (orderedChats.length === 0)
    return (
      <div className={cn("p-3 text-sm", className)}>
        {filterControl}
        <p role="status">
          {hasSearchQuery
            ? t("common:chatSidebar.discovery.noServerMatches", {
                defaultValue: "No matching chats on this Server page."
              })
            : isTrashView
              ? t("common:chatSidebar.trashEmpty", {
                  defaultValue: "Trash is empty."
                })
              : t("common:chatSidebar.noServerChats", {
                  defaultValue: "No server chats yet"
                })}
        </p>
        {hasForeignScope && (
          <p role="alert">
            {t("common:chatSidebar.discovery.foreignScope", {
              defaultValue:
                "Some Server results were excluded because they belong to a different chat scope."
            })}
          </p>
        )}
        {(currentPageSafe > 1 || canLoadNextPage) && recoveryActions()}
      </div>
    )

  return (
    <div
      className={cn("flex flex-col gap-2", className)}
      aria-busy={isFetching}>
      {!freshResults && !isFetching && (
        <div role="alert" className="p-3 text-sm">
          {t("common:chatSidebar.discovery.refreshError", {
            defaultValue:
              "Could not refresh Server history. Try again; cached copies remain in On this device."
          })}
          {recoveryActions()}
        </div>
      )}
      {hasForeignScope && (
        <p role="alert" className="p-3 text-sm">
          {t("common:chatSidebar.discovery.foreignScope", {
            defaultValue:
              "Some Server results were excluded because they belong to a different chat scope."
          })}
        </p>
      )}
      {renamingChat && (
        <Modal
          title={t("common:renameChat", { defaultValue: "Rename chat" })}
          open
          destroyOnHidden
          onCancel={() => {
            setRenamingChat(null)
            setRenameValue("")
            setRenameError(null)
          }}
          onOk={handleRenameSubmit}
          confirmLoading={renameLoading}
          okButtonProps={{
            disabled: renameLoading || !renameValue.trim()
          }}>
          <Input
            autoFocus
            value={renameValue}
            onChange={(e) => {
              setRenameValue(e.target.value)
              if (renameError) {
                setRenameError(null)
              }
            }}
            onPressEnter={handleRenameSubmit}
            status={renameError ? "error" : undefined}
            disabled={renameLoading}
          />
          {renameError && (
            <div className="mt-1 text-xs text-danger">{renameError}</div>
          )}
        </Modal>
      )}
      {editingTopicChat && (
        <Modal
          title={t("playground:composer.topicPlaceholder", {
            defaultValue: "Topic label (optional)"
          })}
          open
          destroyOnHidden
          onCancel={() => {
            setEditingTopicChat(null)
            setTopicValue("")
          }}
          onOk={handleTopicSubmit}
          confirmLoading={topicLoading}
          okButtonProps={{ disabled: topicLoading }}>
          <Input
            autoFocus
            value={topicValue}
            onChange={(e) => setTopicValue(e.target.value)}
            onPressEnter={handleTopicSubmit}
            placeholder={t("playground:composer.topicPlaceholder", {
              defaultValue: "Topic label (optional)"
            })}
            disabled={topicLoading}
          />
        </Modal>
      )}
      {filterControl}
      {sidebarRefreshState === "recoverable-error" && isShowingStaleData && (
        <div className="mx-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-text-subtle">
          {t("common:serverChatsShowingStaleData", {
            defaultValue:
              "Showing saved chats from the last successful refresh. Some recent changes may be missing."
          })}
        </div>
      )}
      {selectionMode && (
        <div className="sticky top-0 z-10 border-b border-border bg-surface2 px-2 py-2">
          <div className="flex items-center justify-between text-xs text-text-subtle">
            <span>
              {t("sidepanel:multiSelect.count", {
                defaultValue: "{{count}} selected",
                count: selectedChatIds.length
              })}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSelectAllVisible}
                className="text-text-subtle hover:text-text"
                disabled={
                  readOnly || !freshResults || visibleChatIds.length === 0
                }>
                {t("sidepanel:multiSelect.selectAll", "Select all")}
              </button>
              <button
                type="button"
                onClick={clearSelection}
                className="text-text-subtle hover:text-text"
                disabled={
                  readOnly || !freshResults || selectedChatIds.length === 0
                }>
                {t("sidepanel:multiSelect.clear", "Clear")}
              </button>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {!isTrashView && (
              <>
                <button
                  type="button"
                  onClick={() => {
                    const action = captureAction(selectedConversationIds)
                    if (action) {
                      managementAction.current = action
                      openBulkFolderPicker()
                    }
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
                  disabled={
                    readOnly || !freshResults || selectedChatIds.length === 0
                  }>
                  <FolderPlus className="size-3.5" />
                  {t("sidepanel:multiSelect.addToFolder", "Add to folders")}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const action = captureAction(selectedConversationIds)
                    if (action) {
                      managementAction.current = action
                      openBulkTagPicker()
                    }
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
                  disabled={
                    readOnly || !freshResults || selectedChatIds.length === 0
                  }>
                  <Tag className="size-3.5" />
                  {t("sidepanel:multiSelect.addTags", "Add tags")}
                </button>
                <button
                  type="button"
                  onClick={openBulkDeleteConfirm}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-danger hover:bg-surface2"
                  disabled={
                    readOnly || !freshResults || selectedChatIds.length === 0
                  }>
                  <Trash2 className="size-3.5" />
                  {t("common:moveToTrash", "Move to trash")}
                </button>
              </>
            )}
            {isTrashView && (
              <>
                <button
                  type="button"
                  onClick={handleBulkRestore}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
                  disabled={
                    readOnly ||
                    !freshResults ||
                    selectedChatIds.length === 0 ||
                    isBulkDeleting
                  }>
                  <RotateCcw className="size-3.5" />
                  {t("common:restore", "Restore")}
                </button>
                <button
                  type="button"
                  onClick={openBulkHardDeleteConfirm}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-danger hover:bg-surface2"
                  disabled={
                    readOnly || !freshResults || selectedChatIds.length === 0
                  }>
                  <Trash2 className="size-3.5" />
                  {t("common:deletePermanently", "Delete permanently")}
                </button>
              </>
            )}
          </div>
        </div>
      )}
      {orderedChats.length === 0 ? (
        <div className="px-2 py-8">
          <Empty
            description={t("common:chatSidebar.noChatsForFilter", {
              defaultValue:
                chatTypeFilter === "character"
                  ? "No character chats found."
                  : chatTypeFilter === "non_character"
                    ? "No non-character chats found."
                    : chatTypeFilter === "trash"
                      ? "Trash is empty."
                      : "No chats found."
            })}
          />
        </div>
      ) : (
        <>
          {pagedPinnedChats.length > 0 && (
            <div className="mx-2 flex flex-col gap-0.5">
              <div className="px-2 py-1 text-xs font-semibold text-text-muted">
                {t("common:pinned", { defaultValue: "Pinned" })}
              </div>
              {pagedPinnedChats.map((chat) => (
                <ServerChatRow
                  key={chat.id}
                  chat={chat}
                  showReference={pagedChats.some(
                    (other) =>
                      other.id !== chat.id && other.title === chat.title
                  )}
                  isTrashView={isTrashView}
                  isPinned={pinnedChatSet.has(chat.id)}
                  isActive={serverChatId === chat.id}
                  readOnly={readOnly || !freshResults}
                  disabled={!freshResults}
                  openMenuFor={openMenuFor}
                  setOpenMenuFor={setOpenMenuFor}
                  onSelectChat={handleRowClick}
                  onTogglePinned={togglePinned}
                  onOpenSettings={handleOpenSettings}
                  onRenameChat={handleRenameChat}
                  onCreateTable={handleCreateTable}
                  onEditTopic={handleEditTopic}
                  onDeleteChat={
                    isTrashView ? handleHardDeleteChat : handleMoveChatToTrash
                  }
                  onRestoreChat={handleRestoreChat}
                  onUpdateState={handleUpdateState}
                  {...selectionPropsForChat(chat.id)}
                  t={t}
                />
              ))}
            </div>
          )}
          {pagedUnpinnedChats.length > 0 && (
            <div
              className={cn(
                "mx-2 flex flex-col gap-0.5",
                pagedPinnedChats.length > 0 && "mt-3"
              )}>
              {pagedUnpinnedChats.map((chat) => (
                <ServerChatRow
                  key={chat.id}
                  chat={chat}
                  showReference={pagedChats.some(
                    (other) =>
                      other.id !== chat.id && other.title === chat.title
                  )}
                  isTrashView={isTrashView}
                  isPinned={pinnedChatSet.has(chat.id)}
                  isActive={serverChatId === chat.id}
                  readOnly={readOnly || !freshResults}
                  disabled={!freshResults}
                  openMenuFor={openMenuFor}
                  setOpenMenuFor={setOpenMenuFor}
                  onSelectChat={handleRowClick}
                  onTogglePinned={togglePinned}
                  onOpenSettings={handleOpenSettings}
                  onRenameChat={handleRenameChat}
                  onCreateTable={handleCreateTable}
                  onEditTopic={handleEditTopic}
                  onDeleteChat={
                    isTrashView ? handleHardDeleteChat : handleMoveChatToTrash
                  }
                  onRestoreChat={handleRestoreChat}
                  onUpdateState={handleUpdateState}
                  {...selectionPropsForChat(chat.id)}
                  t={t}
                />
              ))}
            </div>
          )}
          {hasMultiplePages && (
            <div className="mt-3 border-t border-border px-2 pt-3">
              <div className="mb-2 text-xs text-text-subtle">
                {t("common:chatSidebar.paginationRange", {
                  defaultValue: "Showing {{start}}-{{end}} of {{total}} chats",
                  start: pageStartNumber,
                  end: pageEndNumber,
                  total: totalChatCount
                })}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={handlePreviousPage}
                  disabled={currentPageSafe <= 1}
                  className="rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50">
                  {t("common:previous", { defaultValue: "Previous" })}
                </button>
                <button
                  type="button"
                  onClick={handleNextPage}
                  disabled={!canLoadNextPage || isFetching}
                  className="rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50">
                  {t("common:next", { defaultValue: "Next" })}
                </button>
                <span className="text-xs text-text-subtle">
                  {t("common:chatSidebar.discovery.page", {
                    defaultValue: "Page {{page}}",
                    page: currentPageSafe
                  })}
                </span>
                <Input
                  value={pageJumpValue}
                  onChange={(event) => setPageJumpValue(event.target.value)}
                  onPressEnter={handleJumpToPage}
                  size="small"
                  className="w-16"
                  inputMode="numeric"
                  aria-label={t("common:chatSidebar.pageNumber", {
                    defaultValue: "Page number"
                  })}
                />
                <button
                  type="button"
                  onClick={handleJumpToPage}
                  className="rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2">
                  {t("common:go", { defaultValue: "Go" })}
                </button>
              </div>
            </div>
          )}
        </>
      )}
      <React.Suspense fallback={null}>
        {bulkFolderPickerOpen && managementAction.current?.isCurrent() && (
          <BulkFolderPickerModal
            open={bulkFolderPickerOpen}
            operation={managementAction.current}
            conversationIds={selectedConversationIds}
            onClose={handleBulkFolderPickerClose}
            onSuccess={clearSelection}
          />
        )}
        {bulkTagPickerOpen && managementAction.current?.isCurrent() && (
          <BulkTagPickerModal
            open={bulkTagPickerOpen}
            operation={managementAction.current}
            conversationIds={selectedConversationIds}
            onClose={handleBulkTagPickerClose}
            onSuccess={clearSelection}
          />
        )}
      </React.Suspense>
      <Modal
        open={bulkDeleteConfirmOpen && !!managementAction.current?.isCurrent()}
        onCancel={handleBulkDeleteConfirmClose}
        onOk={handleBulkDelete}
        okText={
          bulkConfirmAction === "hard_delete"
            ? t("common:deletePermanently", "Delete permanently")
            : t("common:moveToTrash", "Move to trash")
        }
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true, loading: isBulkDeleting }}
        title={
          bulkConfirmAction === "hard_delete"
            ? t(
                "sidepanel:multiSelect.deletePermanentConfirmTitle",
                "Delete selected chats permanently?"
              )
            : t(
                "sidepanel:multiSelect.trashConfirmTitle",
                "Move selected chats to trash?"
              )
        }
        destroyOnHidden>
        <p>
          {bulkConfirmAction === "hard_delete"
            ? t(
                "sidepanel:multiSelect.deletePermanentConfirmBody",
                "This will permanently delete the selected chats and cannot be undone."
              )
            : t(
                "sidepanel:multiSelect.trashConfirmBody",
                "This will move the selected chats to trash."
              )}
        </p>
      </Modal>
    </div>
  )
}

export default ServerChatList
