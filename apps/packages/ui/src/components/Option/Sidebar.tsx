import {
  useMutation,
  useQueryClient,
  useInfiniteQuery,
  useQuery
} from "@tanstack/react-query"
import {
  Empty,
  Skeleton,
  Dropdown,
  type MenuProps,
  Tooltip,
  Input,
  Modal,
  message,
  Button
} from "antd"
import {
  PencilIcon,
  Trash2,
  MoreVertical,
  PinIcon,
  PinOffIcon,
  BotIcon,
  SearchIcon,
  Trash2Icon,
  Loader2,
  ChevronDown,
  GitBranch,
  MessageSquare,
  FolderIcon,
  Plus
} from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { lastUsedChatModelEnabled } from "@/services/model-settings"
import { useDebounce } from "@/hooks/useDebounce"
import React, { useState, useEffect } from "react"
import { FolderTree, FolderToolbar } from "@/components/Folders"
import { useFolderStore, useFolderViewMode, useFolderActions } from "@/store/folder"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { useStorage } from "@plasmohq/storage/hook"
import {
  deleteByHistoryId,
  deleteHistoriesByDateRange,
  formatToChatHistory,
  updateHistory,
  pinHistory,
  formatToMessage,
  getSessionFiles,
  getPromptById,
  getHistoriesWithMetadata
} from "@/db/dexie/helpers"
import { isDatabaseClosedError } from "@/utils/ff-error"
import { updatePageTitle } from "@/utils/update-page-title"
import { promptInput } from "@/components/Common/prompt-input"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { IconButton } from "../Common/IconButton"
import { useServerChatHistory, type ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import { useConnectionState } from "@/hooks/useConnectionState"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { shallow } from "zustand/shallow"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { ModeToggle } from "@/components/Sidepanel/Chat/ModeToggle"

const FolderPicker = React.lazy(() =>
  import("@/components/Folders/FolderPicker").then((m) => ({ default: m.FolderPicker }))
)

const FOLDER_CONVERSATION_BATCH_SIZE = 10

type Props = {
  onClose: () => void
  isOpen: boolean
}

export const Sidebar = ({ onClose, isOpen }: Props) => {
  const { t } = useTranslation(["option", "common"])
  const client = useQueryClient()
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState("")
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!isOpen) return
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 60_000)
    return () => clearInterval(timer)
  }, [isOpen])
  const debouncedSearchQuery = useDebounce(searchQuery, 300)
  const [deleteGroup, setDeleteGroup] = useState<string | null>(null)
  const [dexiePrivateWindowError, setDexiePrivateWindowError] = useState(false)
  const [openMenuFor, setOpenMenuFor] = useState<string | null>(null)
  const showLocalChats = false
  const [pinnedServerChatIds, setPinnedServerChatIds] = useStorage<string[]>(
    "tldw:server-chat-pins",
    []
  )
  const confirmDanger = useConfirmDanger()
  const { isConnected } = useConnectionState()

  // Folder system state
  const viewMode = useFolderViewMode()
  const folderKeywordLinks = useFolderStore((s) => s.folderKeywordLinks)
  const conversationKeywordLinks = useFolderStore((s) => s.conversationKeywordLinks)
  const { refreshFromServer, addConversationToFolder, removeConversationFromFolder } =
    useFolderActions()
  const [folderPickerOpen, setFolderPickerOpen] = useState(false)
  const [folderPickerChatId, setFolderPickerChatId] = useState<string | null>(null)
  const folderRefreshInFlightRef = React.useRef<Promise<void> | null>(null)
  // Load folders when the sidebar is open and folder view is active.
  useEffect(() => {
    if (isConnected && isOpen && viewMode === "folders") {
      if (folderRefreshInFlightRef.current) return
      const refreshPromise = refreshFromServer().finally(() => {
        if (folderRefreshInFlightRef.current === refreshPromise) {
          folderRefreshInFlightRef.current = null
        }
      })
      folderRefreshInFlightRef.current = refreshPromise
    }
  }, [isConnected, isOpen, viewMode, refreshFromServer])

  const clearChat = useClearChat()
  const selectServerChat = useSelectServerChat()
  const { setSystemPrompt } = useStoreChatModelSettings()
  const { historyId, setHistory, setHistoryId, setMessages } = useChatBaseState(
    useStoreMessageOption
  )
  const {
    serverChatId,
    setServerChatId,
    setSelectedModel,
    setSelectedSystemPrompt,
    setContextFiles,
    temporaryChat
  } = useStoreMessageOption(
    (state) => ({
      serverChatId: state.serverChatId,
      setServerChatId: state.setServerChatId,
      setSelectedModel: state.setSelectedModel,
      setSelectedSystemPrompt: state.setSelectedSystemPrompt,
      setContextFiles: state.setContextFiles,
      temporaryChat: state.temporaryChat
    }),
    shallow
  )
  const {
    data: serverChatData,
    status: serverStatus,
    isLoading: isServerLoading
  } = useServerChatHistory(debouncedSearchQuery)
  const serverChats = serverChatData || []
  const pinnedServerChatSet = React.useMemo(
    () => new Set(pinnedServerChatIds || []),
    [pinnedServerChatIds]
  )
  const pinnedServerChats = serverChats.filter((chat) =>
    pinnedServerChatSet.has(chat.id)
  )
  const unpinnedServerChats = serverChats.filter(
    (chat) => !pinnedServerChatSet.has(chat.id)
  )
  const serverChatById = React.useMemo(
    () => new Map(serverChats.map((chat) => [chat.id, chat])),
    [serverChats]
  )

  // Using infinite query for pagination
  const {
    data: chatHistoriesData,
    status,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading
  } = useInfiniteQuery({
    queryKey: ["fetchChatHistory", debouncedSearchQuery],
    queryFn: async ({ pageParam = 1 }) => {
      try {
        const db = new PageAssistDatabase()
        const result = await db.getChatHistoriesPaginated(
          pageParam,
          debouncedSearchQuery || undefined
        )

        // If searching, don't group by date - just return all results in a single group
        if (debouncedSearchQuery) {
          console.log("Search results:", result.histories)
          return {
            groups:
              result.histories.length > 0
                ? [{ label: "searchResults", items: result.histories }]
                : [],
            hasMore: result.hasMore,
            totalCount: result.totalCount
          }
        }

        // Group the histories by date only when not searching
        const now = new Date()
        const today = new Date(now.setHours(0, 0, 0, 0))
        const yesterday = new Date(today)
        yesterday.setDate(yesterday.getDate() - 1)
        const lastWeek = new Date(today)
        lastWeek.setDate(lastWeek.getDate() - 7)

        const pinnedItems = result.histories.filter((item) => item.is_pinned)
        const todayItems = result.histories.filter(
          (item) => !item.is_pinned && new Date(item?.createdAt) >= today
        )
        const yesterdayItems = result.histories.filter(
          (item) =>
            !item.is_pinned &&
            new Date(item?.createdAt) >= yesterday &&
            new Date(item?.createdAt) < today
        )
        const lastWeekItems = result.histories.filter(
          (item) =>
            !item.is_pinned &&
            new Date(item?.createdAt) >= lastWeek &&
            new Date(item?.createdAt) < yesterday
        )
        const olderItems = result.histories.filter(
          (item) => !item.is_pinned && new Date(item?.createdAt) < lastWeek
        )

        const groups = []
        if (pinnedItems.length)
          groups.push({ label: "pinned", items: pinnedItems })
        if (todayItems.length)
          groups.push({ label: "today", items: todayItems })
        if (yesterdayItems.length)
          groups.push({ label: "yesterday", items: yesterdayItems })
        if (lastWeekItems.length)
          groups.push({ label: "last7Days", items: lastWeekItems })
        if (olderItems.length)
          groups.push({ label: "older", items: olderItems })

        return {
          groups,
          hasMore: result.hasMore,
          totalCount: result.totalCount
        }
      } catch (e) {
        setDexiePrivateWindowError(isDatabaseClosedError(e))
        return {
          groups: [],
          hasMore: false,
          totalCount: 0
        }
      }
    },
    getNextPageParam: (lastPage, allPages) => {
      return lastPage.hasMore ? allPages.length + 1 : undefined
    },
    placeholderData: undefined,
    enabled: isOpen,
    initialPageParam: 1
  })

  // Flatten all groups from all pages
  const chatHistories =
    chatHistoriesData?.pages.reduce(
      (acc, page) => {
        // Merge groups with same labels
        page.groups.forEach((group) => {
          const existingGroup = acc.find((g) => g.label === group.label)
          if (existingGroup) {
            existingGroup.items.push(...group.items)
          } else {
            acc.push({ ...group })
          }
        })
        return acc
      },
      [] as Array<{ label: string; items: any[] }>
    ) || []

  // Collect all history IDs for metadata fetching
  const allHistoryIds = chatHistories.flatMap((group) =>
    group.items.map((item) => item.id)
  )

  // Fetch metadata for all visible histories
  const { data: historyMetadata } = useQuery({
    queryKey: ["historyMetadata", allHistoryIds.join(",")],
    queryFn: async () => {
      if (allHistoryIds.length === 0) return new Map()
      return getHistoriesWithMetadata(allHistoryIds)
    },
    enabled: isOpen && allHistoryIds.length > 0,
    staleTime: 30000 // Cache for 30 seconds
  })

  // Helper to format relative time
  const formatRelativeTime = (timestamp: number) => {
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return t("common:justNow", { defaultValue: "Just now" })
    if (minutes < 60)
      return t("common:minutesAgo", {
        count: minutes,
        defaultValue: `${minutes}m ago`
      })
    if (hours < 24)
      return t("common:hoursAgo", {
        count: hours,
        defaultValue: `${hours}h ago`
      })
    if (days < 7)
      return t("common:daysAgo", {
        count: days,
        defaultValue: `${days}d ago`
      })
    return new Date(timestamp).toLocaleDateString()
  }

  // Helper to truncate message preview
  const truncateMessage = (content: string, maxLength: number = 60) => {
    if (!content) return ""
    // Remove markdown formatting for preview
    const cleaned = content.replace(/[#*_`~\[\]]/g, "").trim()
    if (cleaned.length <= maxLength) return cleaned
    return cleaned.substring(0, maxLength).trim() + "..."
  }

  const { mutate: deleteHistory } = useMutation({
    mutationKey: ["deleteHistory"],
    mutationFn: deleteByHistoryId,
    onSuccess: (history_id) => {
      client.invalidateQueries({
        queryKey: ["fetchChatHistory"]
      })
      if (historyId === history_id) {
        clearChat()
        updatePageTitle()
      }
    }
  })

  const { mutate: editHistory } = useMutation({
    mutationKey: ["editHistory"],
    mutationFn: async (data: { id: string; title: string }) => {
      return await updateHistory(data.id, data.title)
    },
    onSuccess: () => {
      client.invalidateQueries({
        queryKey: ["fetchChatHistory"]
      })
    }
  })

  const { mutate: deleteHistoriesByRange, isPending: deleteRangeLoading } =
    useMutation({
      mutationKey: ["deleteHistoriesByRange"],
      mutationFn: async (rangeLabel: string) => {
        setDeleteGroup(rangeLabel)
        return await deleteHistoriesByDateRange(rangeLabel)
      },
      onSuccess: (deletedIds) => {
        client.invalidateQueries({
          queryKey: ["fetchChatHistory"]
        })

        if (deletedIds.includes(historyId)) {
          clearChat()
        }

        message.success(
          t("common:historiesDeleted", { count: deletedIds.length })
        )
      },
      onError: (error) => {
        console.error("Failed to delete histories:", error)
        message.error(t("common:deleteHistoriesError"))
      }
    })

  const handleDeleteHistoriesByRange = async (rangeLabel: string) => {
    const ok = await confirmDanger({
      title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
      content: t(`common:range:deleteConfirm:${rangeLabel}`),
      okText: t("common:delete", { defaultValue: "Delete" }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })
    if (!ok) return
    deleteHistoriesByRange(rangeLabel)
  }

  const { mutate: pinChatHistory, isPending: pinLoading } = useMutation({
    mutationKey: ["pinHistory"],
    mutationFn: async (data: { id: string; is_pinned: boolean }) => {
      return await pinHistory(data.id, data.is_pinned)
    },
    onSuccess: () => {
      client.invalidateQueries({
        queryKey: ["fetchChatHistory"]
      })
    }
  })

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
  }

  const clearSearch = () => {
    setSearchQuery("")
  }

  const toggleServerChatPinned = (chatId: string) => {
    setPinnedServerChatIds((prev) => {
      const current = prev || []
      if (current.includes(chatId)) {
        return current.filter((id) => id !== chatId)
      }
      return [...current, chatId]
    })
  }

  const handleLoadMore = () => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage()
    }
  }

  const loadServerChat = React.useCallback(
    (chat: ServerChatSummary) => {
      selectServerChat(chat)
      onClose()
    },
    [onClose, selectServerChat]
  )

  const loadServerChatById = React.useCallback(
    async (conversationId: string) => {
      const cachedChat = serverChatById.get(conversationId)
      if (cachedChat) {
        loadServerChat(cachedChat)
        return
      }

      try {
        await tldwClient.initialize().catch(() => null)
        const chat = await tldwClient.getChat(conversationId)
        loadServerChat(chat)
      } catch (error) {
        console.error(
          "Failed to load server chat for folder conversation:",
          conversationId,
          error
        )
        message.error(
          t("common:error.friendlyGenericSummary", {
            defaultValue: "Something went wrong while talking to your tldw server."
          })
        )
      }
    },
    [loadServerChat, serverChatById, t]
  )

  const loadLocalConversation = React.useCallback(
    async (conversationId: string) => {
      try {
        const db = new PageAssistDatabase()
        const [history, historyDetails] = await Promise.all([
          db.getChatHistory(conversationId),
          db.getHistoryInfo(conversationId)
        ])

        // Switch to a local Dexie-backed chat; clear any active server-backed session id.
        setServerChatId(null)
        setHistoryId(conversationId)
        setHistory(formatToChatHistory(history))
        setMessages(formatToMessage(history))

        const isLastUsedChatModel = await lastUsedChatModelEnabled()
        if (isLastUsedChatModel && historyDetails?.model_id) {
          setSelectedModel(historyDetails.model_id)
        }

        const lastUsedPrompt = historyDetails?.last_used_prompt
        if (lastUsedPrompt) {
          let promptContent = lastUsedPrompt.prompt_content ?? ""
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id)
            if (prompt) {
              setSelectedSystemPrompt(prompt.id)
              if (!promptContent.trim()) {
                promptContent = prompt.content
              }
            }
          }
          setSystemPrompt(promptContent)
        }

        const session = await getSessionFiles(conversationId)
        setContextFiles(session)

        updatePageTitle(
          historyDetails?.title || t("common:untitled", { defaultValue: "Untitled" })
        )
        navigate("/")
        onClose()

        return { history, historyDetails }
      } catch (error) {
        console.error("Failed to load local chat history from local storage:", error)
        message.error(
          t("common:error.friendlyLocalHistorySummary", {
            defaultValue: "Something went wrong while loading local chat history."
          })
        )
        return null
      }
    },
    [
      navigate,
      onClose,
      setHistory,
      setHistoryId,
      setMessages,
      setSelectedModel,
      setSelectedSystemPrompt,
      setServerChatId,
      setContextFiles,
      setSystemPrompt,
      t
    ]
  )

  // Handle folder selection for a chat
  const handleFolderSelect = async (folderIds: number[]) => {
    const conversationId = folderPickerChatId
    try {
      if (!conversationId) return

      const uniqueFolderIds = Array.from(new Set(folderIds))
      const results = await Promise.allSettled(
        uniqueFolderIds.map((folderId) =>
          addConversationToFolder(conversationId, folderId)
        )
      )

      const succeededFolderIds: number[] = []
      const failedFolderAdds: Array<{ folderId: number; reason: unknown }> = []

      results.forEach((result, index) => {
        const folderId = uniqueFolderIds[index]
        if (result.status === "fulfilled" && result.value) {
          succeededFolderIds.push(folderId)
          return
        }

        failedFolderAdds.push({
          folderId,
          reason:
            result.status === "rejected"
              ? result.reason
              : "addConversationToFolder returned false"
        })
      })

      if (failedFolderAdds.length > 0) {
        console.error("Failed to move chat to folder(s):", {
          conversationId,
          failedFolderAdds,
          succeededFolderIds
        })

        // UX choice: treat multi-folder assignment as atomic (all-or-nothing).
        // If any folder add fails, roll back successful additions so the chat
        // isn't left partially filed. If we want best-effort behavior instead,
        // keep successful additions and surface a partial-success message.
        if (succeededFolderIds.length > 0) {
          const rollbackResults = await Promise.allSettled(
            succeededFolderIds.map((folderId) =>
              removeConversationFromFolder(conversationId, folderId)
            )
          )

          const rollbackFailures: Array<{ folderId: number; reason: unknown }> =
            []

          rollbackResults.forEach((result, index) => {
            const folderId = succeededFolderIds[index]
            if (result.status === "fulfilled" && result.value) {
              return
            }

            rollbackFailures.push({
              folderId,
              reason:
                result.status === "rejected"
                  ? result.reason
                  : "removeConversationFromFolder returned false"
            })
          })

          if (rollbackFailures.length > 0) {
            console.error("Failed to rollback folder move after errors:", {
              conversationId,
              rollbackFailures
            })
          }
        }

        message.error(
          t("common:error.friendlyGenericSummary", {
            defaultValue: "Something went wrong while talking to your tldw server."
          })
        )
        return
      }

      message.success(t("common:success"))
    } catch (error) {
      console.error("Failed to move chat to folder(s):", error)
      message.error(
        t("common:error.friendlyGenericSummary", {
          defaultValue: "Something went wrong while talking to your tldw server."
        })
      )
    } finally {
      setFolderPickerOpen(false)
      setFolderPickerChatId(null)
    }
  }

  const folderConversationIds = React.useMemo(() => {
    if (folderKeywordLinks.length === 0 || conversationKeywordLinks.length === 0) {
      return []
    }

    const folderKeywordIdSet = new Set(
      folderKeywordLinks.map((link) => link.keyword_id)
    )

    const conversationIdSet = new Set<string>()
    conversationKeywordLinks.forEach((link) => {
      if (folderKeywordIdSet.has(link.keyword_id)) {
        conversationIdSet.add(link.conversation_id)
      }
    })

    return Array.from(conversationIdSet)
  }, [conversationKeywordLinks, folderKeywordLinks])

  const loadedConversationTitleById = React.useMemo(() => {
    const titleById = new Map<string, string>()
    serverChats.forEach((chat) => {
      titleById.set(chat.id, chat.title ?? "")
    })
    return titleById
  }, [serverChats])

  const missingFolderConversationIds = React.useMemo(() => {
    return folderConversationIds.filter(
      (conversationId) => !loadedConversationTitleById.has(conversationId)
    )
  }, [folderConversationIds, loadedConversationTitleById])

  const stableMissingFolderConversationIds = React.useMemo(() => {
    return [...missingFolderConversationIds].sort()
  }, [missingFolderConversationIds])

  const { data: missingFolderConversationTitles = [] } = useQuery({
    queryKey: ["folderConversationTitles", stableMissingFolderConversationIds],
    queryFn: async () => {
      await tldwClient.initialize().catch(() => null)
      const results: Array<{ id: string; title: string }> = []
      for (
        let i = 0;
        i < stableMissingFolderConversationIds.length;
        i += FOLDER_CONVERSATION_BATCH_SIZE
      ) {
        const batch = stableMissingFolderConversationIds.slice(
          i,
          i + FOLDER_CONVERSATION_BATCH_SIZE
        )
        const batchResults = await Promise.all(
          batch.map(async (conversationId) => {
            try {
              const chat = await tldwClient.getChat(conversationId)
              return {
                id: conversationId,
                title: chat?.title || ""
              }
            } catch (error) {
              console.error(
                "Failed to load server chat info for folder conversation:",
                conversationId,
                error
              )
              return {
                id: conversationId,
                title: ""
              }
            }
          })
        )
        results.push(...batchResults)
      }
      return results
    },
    enabled:
      isOpen &&
      isConnected &&
      viewMode === "folders" &&
      stableMissingFolderConversationIds.length > 0,
    staleTime: 30_000
  })

  const folderTreeConversations = React.useMemo(() => {
    const titleById = new Map<string, string>(loadedConversationTitleById)
    missingFolderConversationTitles.forEach((entry) => {
      titleById.set(entry.id, entry.title)
    })

    return folderConversationIds.map((conversationId) => ({
      id: conversationId,
      title: titleById.get(conversationId) || ""
    }))
  }, [
    folderConversationIds,
    loadedConversationTitleById,
    missingFolderConversationTitles
  ])

  const renderServerChatRow = (chat: ServerChatHistoryItem) => {
    const isPinned = pinnedServerChatSet.has(chat.id)
    return (
      <div
        key={chat.id}
        className={`flex py-2 px-2 items-center gap-3 relative rounded-md truncate group transition-opacity duration-300 ease-in-out border ${
          serverChatId === chat.id
            ? "bg-surface2 border-border-strong text-text"
            : "bg-surface text-text border-border hover:bg-surface2"
        }`}
      >
        <button
          className="flex flex-col overflow-hidden flex-1 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 rounded"
          onClick={() => {
            loadServerChat(chat)
          }}
        >
          <div className="flex flex-col overflow-hidden flex-1">
            <span className="truncate text-sm">{chat.title}</span>
            <div className="flex flex-wrap items-center gap-2 text-xs text-text-subtle mt-0.5">
              <span className="inline-flex items-center rounded-full bg-surface2 px-2 py-0.5 text-[11px] font-medium lowercase text-text">
                {(chat.state as string) || "in-progress"}
              </span>
              {chat.topic_label && (
                <span
                  className="truncate max-w-[12rem]"
                  title={String(chat.topic_label)}
                >
                  {String(chat.topic_label)}
                </span>
              )}
            </div>
            <span className="flex items-center gap-1 text-xs text-text-subtle">
              {chat.parent_conversation_id ? (
                <Tooltip
                  title={t("common:serverChatForkedTooltip", {
                    chatId: String(chat.parent_conversation_id).slice(0, 8),
                    defaultValue: "Forked from chat {{chatId}}"
                  })}
                >
                  <span className="inline-flex items-center gap-1">
                    <GitBranch className="size-3" />
                    <span>
                      {t("common:serverChatForkedLabel", {
                        defaultValue: "Forked chat"
                      })}
                    </span>
                  </span>
                </Tooltip>
              ) : (
                <span>
                  {t("common:serverChatSourceLabel", {
                    defaultValue: "Server"
                  })}
                </span>
              )}
            </span>
          </div>
        </button>
        <Tooltip title={isPinned ? t("common:unpin") : t("common:pin")}>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              toggleServerChatPinned(chat.id)
            }}
            className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)]"
            aria-label={isPinned ? t("common:unpin") : t("common:pin")}
            aria-pressed={isPinned}
          >
            {isPinned ? (
              <PinOffIcon className="w-4 h-4" />
            ) : (
              <PinIcon className="w-4 h-4" />
            )}
          </button>
        </Tooltip>
      </div>
    )
  }

  return (
    <div
      className={`overflow-y-auto z-99 ${temporaryChat ? "pointer-events-none opacity-50" : ""}`}>
      <div className="sticky top-0 z-10 my-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Input
              placeholder={t("common:search")}
              value={searchQuery}
              onChange={handleSearchChange}
              prefix={<SearchIcon className="w-4 h-4 text-text-subtle" />}
              suffix={
                <button
                  type="button"
                  onClick={clearSearch}
                  className="text-text-subtle hover:text-text"
                  aria-label={t("common:clearSearch", { defaultValue: "Clear search" })}
                  aria-hidden={!searchQuery}
                  tabIndex={searchQuery ? 0 : -1}
                  style={{
                    visibility: searchQuery ? "visible" : "hidden",
                    pointerEvents: searchQuery ? "auto" : "none"
                  }}>
                  ✕
                </button>
              }
              className="w-full rounded-md border border-border bg-surface"
            />
          </div>
          <Tooltip title={t("common:newChat", { defaultValue: "New Chat" })}>
            <Button
              type="text"
              icon={<Plus className="w-4 h-4" />}
              aria-label={t("common:newChat", { defaultValue: "New Chat" })}
              data-testid="sidebar-new-chat"
              onClick={() => {
                clearChat()
                onClose()
              }}
            />
          </Tooltip>
          {isConnected && <FolderToolbar compact />}
        </div>
      </div>

      {serverStatus === "success" &&
        serverChats.length === 0 &&
        (!showLocalChats || !dexiePrivateWindowError) &&
        (!showLocalChats || (status === "success" && chatHistories.length === 0)) && (
          <div className="flex justify-center items-center mt-20 overflow-hidden">
            <Empty description={t("common:noHistory")} />
          </div>
        )}

      {showLocalChats && dexiePrivateWindowError && (
        <div className="flex justify-center items-center mt-20 overflow-hidden">
          <Empty
            description={t("common:privateWindow", {
              defaultValue:
                "Don't worry, this is a known issue on Firefox: IndexedDB does not work in private mode. Please open the extension in a normal window to view your chat history."
            })}
          />
        </div>
      )}

      {showLocalChats && (status === "pending" || isLoading) && (
        <div className="flex justify-center items-center mt-5">
          <Skeleton active paragraph={{ rows: 8 }} />
        </div>
      )}

      {serverStatus === "pending" && (
        <div className="flex justify-center items-center mt-2">
          <Skeleton active paragraph={{ rows: 2 }} />
        </div>
      )}

      {serverStatus === "error" && (
        <div className="flex justify-center items-center mt-2 px-2">
          <span className="text-xs text-text-subtle">
            {t("common:serverChatsUnavailable", {
              defaultValue: isConnected
                ? "Server chats unavailable right now. Check your server logs or try again."
                : "Server chats are available once you connect to your tldw server."
            })}
          </span>
        </div>
      )}

      {showLocalChats && status === "error" && (
        <div className="flex justify-center items-center">
          <span className="text-danger">Error loading history</span>
        </div>
      )}

      {showLocalChats && status === "success" && chatHistories.length > 0 && (
        <div className="flex flex-col gap-2">
          {chatHistories.map((group, groupIndex) => (
            <div key={groupIndex}>
              <div className="flex items-center justify-between mt-2">
                <h3 className="px-2 text-sm font-medium text-text-subtle">
                  {group.label === "searchResults"
                    ? t("common:searchResults")
                    : t(`common:date:${group.label}`)}
                </h3>
                {group.label !== "searchResults" && (
                  <Tooltip
                    title={t(`common:range:tooltip:${group.label}`)}
                    placement="top">
                    <button
                  onClick={() => handleDeleteHistoriesByRange(group.label)}>
                  {deleteRangeLoading && deleteGroup === group.label ? (
                        <Loader2 className="w-4 h-4 text-text-muted hover:text-text animate-spin" />
                      ) : (
                        <Trash2Icon className="w-4 h-4 text-text-muted hover:text-text" />
                      )}
                    </button>
                  </Tooltip>
                )}
              </div>
              <div className="flex flex-col gap-2 mt-2">
                {group.items.map((chat) => {
                  const historyMenuItems: MenuProps["items"] = [
                    {
                      key: "pin",
                      icon: chat.is_pinned ? (
                        <PinOffIcon className="w-4 h-4" />
                      ) : (
                        <PinIcon className="w-4 h-4" />
                      ),
                      label: chat.is_pinned ? t("common:unpin") : t("common:pin"),
                      disabled: pinLoading,
                      onClick: () =>
                        pinChatHistory({
                          id: chat.id,
                          is_pinned: !chat.is_pinned
                        })
                    },
                    ...(isConnected
                      ? [
                          {
                            key: "moveToFolder",
                            icon: <FolderIcon className="w-4 h-4" />,
                            label: t("common:moveToFolder"),
                            onClick: () => {
                              setFolderPickerChatId(chat.id)
                              setFolderPickerOpen(true)
                              setOpenMenuFor(null)
                            }
                          }
                        ]
                      : []),
                    {
                      key: "edit",
                      icon: <PencilIcon className="w-4 h-4" />,
                      label: t("common:edit"),
                      onClick: async () => {
                        const newTitle = await promptInput({
                          title: t("editHistoryTitle", { defaultValue: "Rename chat" }),
                          defaultValue: chat.title,
                          okText: t("common:save", { defaultValue: "Save" }),
                          cancelText: t("common:cancel", { defaultValue: "Cancel" })
                        })
                        if (newTitle && newTitle !== chat.title) {
                          editHistory({ id: chat.id, title: newTitle })
                        }
                      }
                    },
                    {
                      key: "delete",
                      icon: <Trash2 className="w-4 h-4" />,
                      label: t("common:delete"),
                      danger: true,
                      onClick: async () => {
                        const ok = await confirmDanger({
                          title: t("common:confirmTitle", {
                            defaultValue: "Please confirm"
                          }),
                          content: t("deleteHistoryConfirmation"),
                          okText: t("common:delete", {
                            defaultValue: "Delete"
                          }),
                          cancelText: t("common:cancel", {
                            defaultValue: "Cancel"
                          })
                        })
                        if (!ok) return
                        deleteHistory(chat.id)
                      }
                    }
                  ]

                  return (
                    <div
                      key={chat.id}
                      className={`flex py-2 px-2 items-start gap-2 relative rounded-md hover:pr-4 group transition-opacity duration-300 ease-in-out border ${
                        historyId === chat.id
                          ? "bg-surface2 border-border-strong text-text"
                          : "bg-surface text-text border-border hover:bg-surface2"
                      }`}>
                      {chat?.message_source === "copilot" && (
                        <BotIcon className="size-3 text-text-subtle mt-1 flex-shrink-0" />
                      )}
                      {chat?.message_source === "branch" && (
                        <GitBranch className="size-3 text-text-subtle mt-1 flex-shrink-0" />
                      )}
                      <button
                        className="flex-1 overflow-hidden text-start w-full min-w-0"
                        onClick={() => {
                          void loadLocalConversation(chat.id)
                        }}>
                        <div className="flex flex-col gap-0.5">
                          <span className="truncate font-medium">{chat.title}</span>
                          {historyMetadata?.get(chat.id) && (
                            <div className="flex items-center gap-2 text-xs text-text-subtle">
                              <span className="flex items-center gap-1">
                                <MessageSquare className="size-3" />
                                {historyMetadata.get(chat.id)?.messageCount || 0}
                              </span>
                              {historyMetadata.get(chat.id)?.lastMessage && (
                                <span>
                                  {formatRelativeTime(
                                    historyMetadata.get(chat.id)?.lastMessage
                                      ?.createdAt || chat.createdAt
                                  )}
                                </span>
                              )}
                            </div>
                          )}
                          {historyMetadata?.get(chat.id)?.lastMessage && (
                            <span className="text-xs text-text-subtle truncate">
                              {truncateMessage(
                                historyMetadata.get(chat.id)?.lastMessage
                                  ?.content || ""
                              )}
                            </span>
                          )}
                        </div>
                      </button>
                      <div className="flex items-center gap-2">
                        <Dropdown
                          menu={{
                            items: historyMenuItems,
                            id: `history-actions-${chat.id}`
                          }}
                          trigger={["click"]}
                          placement="bottomRight"
                          open={openMenuFor === chat.id}
                          onOpenChange={(o) => setOpenMenuFor(o ? chat.id : null)}>
                          <IconButton
                            className="text-text-subtle opacity-80 hover:opacity-100 h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0"
                            ariaLabel={`${t("option:header.moreActions", "More actions")}: ${chat.title}`}
                            hasPopup="menu"
                            ariaExpanded={openMenuFor === chat.id}
                            ariaControls={`history-actions-${chat.id}`}>
                            <MoreVertical className="w-4 h-4" />
                          </IconButton>
                        </Dropdown>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}

          {/* Load More Button */}
          {hasNextPage && (
            <div className="flex justify-center mt-4 mb-2">
              <Button
                type="default"
                onClick={handleLoadMore}
                loading={isFetchingNextPage}
                icon={
                  !isFetchingNextPage ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : undefined
                }
                className="flex items-center gap-2 text-sm">
                {isFetchingNextPage
                  ? t("common:loading.title", { defaultValue: "Loading..." })
                  : t("common:loadMore")}
              </Button>
            </div>
          )}
        </div>
      )}

      {serverStatus === "success" && serverChats.length > 0 && (
        <div className="mt-4 flex flex-col gap-2 border-t border-border pt-3">
          <div className="flex items-center justify-between mt-1">
            <h3 className="px-2 text-sm font-medium text-text-subtle">
              {t("common:serverChats", { defaultValue: "Server chats" })}
            </h3>
          </div>
          {pinnedServerChats.length > 0 && (
            <div className="flex flex-col gap-2 mt-1">
              <div className="px-2 text-[11px] font-medium text-text-subtle uppercase tracking-wide">
                {t("common:pinned", { defaultValue: "Pinned" })}
              </div>
              {pinnedServerChats.map(renderServerChatRow)}
            </div>
          )}
          {unpinnedServerChats.length > 0 && (
            <div
              className={
                pinnedServerChats.length > 0
                  ? "mt-3 flex flex-col gap-2"
                  : "mt-1 flex flex-col gap-2"
              }
            >
              {unpinnedServerChats.map(renderServerChatRow)}
            </div>
          )}
        </div>
      )}

      {/* Folder Tree View - shown when viewMode is 'folders' */}
      {isConnected && viewMode === 'folders' && (
        <div className="mt-4 border-t border-border pt-3">
          <FolderTree
            onConversationSelect={(conversationId) => {
              void loadServerChatById(conversationId)
            }}
            conversations={folderTreeConversations}
            showConversations
          />
        </div>
      )}

      <div className="mt-4 border-t border-border pt-3">
        <ModeToggle />
      </div>

      {/* Folder Picker Modal */}
      <React.Suspense
        fallback={
          folderPickerOpen ? (
            <Modal
              open={folderPickerOpen}
              onCancel={() => {
                setFolderPickerOpen(false)
                setFolderPickerChatId(null)
              }}
              footer={null}
              title={t("common:moveToFolder")}>
              <div className="flex items-center justify-center py-6">
                <Loader2 className="h-5 w-5 animate-spin text-text-muted" />
              </div>
            </Modal>
          ) : null
        }>
        <FolderPicker
          open={folderPickerOpen}
          onClose={() => {
            setFolderPickerOpen(false)
            setFolderPickerChatId(null)
          }}
          onSelect={handleFolderSelect}
          title={t("common:moveToFolder")}
          allowMultiple
          showCreateNew
        />
      </React.Suspense>
    </div>
  )
}
