import React, { useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient
} from "@tanstack/react-query"
import { Tooltip, Spin, Empty, Skeleton, Dropdown, Button, Input, Modal, message } from "antd"
import {
  MessageSquare,
  Trash2,
  Edit3,
  MoreHorizontal,
  ChevronDown,
  PinIcon,
  PinOffIcon,
  BotIcon,
  GitBranch,
  FolderIcon
} from "lucide-react"

import { pageAssistDatabase } from "@/db/dexie/chat"
import type { HistoryInfo, Message } from "@/db/dexie/types"
import {
  deleteByHistoryId,
  deleteHistoriesByDateRange,
  getHistoriesWithMetadata,
  updateHistory,
  pinHistory,
  getFullChatData,
  restoreChat
} from "@/db/dexie/helpers"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { isDatabaseClosedError } from "@/utils/ff-error"
import { updatePageTitle } from "@/utils/update-page-title"
import { useIsConnected } from "@/hooks/useConnectionState"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { IconButton } from "@/components/Common/IconButton"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useStoreChatModelSettings } from "@/store/model"
import { useLoadLocalConversation } from "@/hooks/useLoadLocalConversation"
import { cn } from "@/libs/utils"

type ChatGroup = {
  label: string
  items: HistoryInfo[]
}

export interface LocalChatListProps {
  searchQuery: string
  selectedChatId: string | null
  onSelectChat?: (chatId: string) => void
  onOpenFolderPicker?: (chatId: string) => void
  className?: string
}

// Relative time helper
const formatRelativeTime = (
  timestamp: number,
  t: (key: string, options?: any) => string
) => {
  const now = Date.now()
  const diff = now - timestamp
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return t("common:justNow", { defaultValue: "Just now" })
  if (minutes < 60) {
    return t("common:minutesAgo", {
      count: minutes,
      defaultValue: `${minutes}m ago`
    })
  }
  if (hours < 24) {
    return t("common:hoursAgo", {
      count: hours,
      defaultValue: `${hours}h ago`
    })
  }
  if (days < 7) {
    return t("common:daysAgo", {
      count: days,
      defaultValue: `${days}d ago`
    })
  }
  return new Date(timestamp).toLocaleDateString()
}

// Message preview helper
const truncateMessage = (content: string, maxLength: number = 60) => {
  if (!content) return ""
  const cleaned = content.replace(/[#*_`~\[\]]/g, "").trim()
  if (cleaned.length <= maxLength) return cleaned
  return cleaned.substring(0, maxLength).trim() + "..."
}

export function LocalChatList({
  searchQuery,
  selectedChatId,
  onSelectChat,
  onOpenFolderPicker,
  className
}: LocalChatListProps) {
  const { t } = useTranslation(["common", "sidepanel", "option"])
  const isConnected = useIsConnected()
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()
  const { showUndoNotification } = useUndoNotification()

  const {
    setMessages,
    setHistory,
    setHistoryId,
    historyId,
    clearChat,
    setSelectedModel,
    setSelectedSystemPrompt,
    setContextFiles,
    setServerChatId
  } = useMessageOption()
  const { setSystemPrompt } = useStoreChatModelSettings()

  const [dexiePrivateWindowError, setDexiePrivateWindowError] = useState(false)
  const [deleteGroup, setDeleteGroup] = useState<string | null>(null)
  const [openMenuFor, setOpenMenuFor] = useState<string | null>(null)
  const [renamingChat, setRenamingChat] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState("")
  const [renameError, setRenameError] = useState<string | null>(null)
  const [loadingChatId, setLoadingChatId] = useState<string | null>(null)
  const loadRequestRef = useRef(0)

  // Local chat history query
  const {
    data: chatHistoriesData,
    status,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading
  } = useInfiniteQuery({
    queryKey: ["fetchChatHistory", searchQuery],
    queryFn: async ({ pageParam = 1 }) => {
      try {
        const db = pageAssistDatabase
        const result = await db.getChatHistoriesPaginated(
          pageParam,
          searchQuery || undefined
        )

        if (searchQuery) {
          return {
            groups:
              result.histories.length > 0
                ? [{ label: "searchResults", items: result.histories }]
                : [],
            hasMore: result.hasMore,
            totalCount: result.totalCount
          }
        }

        const now = new Date()
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
        const yesterday = new Date(today)
        yesterday.setDate(yesterday.getDate() - 1)
        const lastWeek = new Date(today)
        lastWeek.setDate(lastWeek.getDate() - 7)

        const pinnedItems: HistoryInfo[] = []
        const todayItems: HistoryInfo[] = []
        const yesterdayItems: HistoryInfo[] = []
        const lastWeekItems: HistoryInfo[] = []
        const olderItems: HistoryInfo[] = []

        for (const item of result.histories) {
          if (item.is_pinned) {
            pinnedItems.push(item)
            continue
          }

          const date = new Date(item.createdAt)
          const time = Number.isNaN(date.getTime()) ? null : date

          if (!time) {
            olderItems.push(item)
            continue
          }

          if (time >= today) {
            todayItems.push(item)
          } else if (time >= yesterday) {
            yesterdayItems.push(item)
          } else if (time >= lastWeek) {
            lastWeekItems.push(item)
          } else {
            olderItems.push(item)
          }
        }

        const groups: ChatGroup[] = []
        if (pinnedItems.length) groups.push({ label: "pinned", items: pinnedItems })
        if (todayItems.length) groups.push({ label: "today", items: todayItems })
        if (yesterdayItems.length) groups.push({ label: "yesterday", items: yesterdayItems })
        if (lastWeekItems.length) groups.push({ label: "last7Days", items: lastWeekItems })
        if (olderItems.length) groups.push({ label: "older", items: olderItems })

        return {
          groups,
          hasMore: result.hasMore,
          totalCount: result.totalCount
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Failed to fetch chat histories:", error)
        setDexiePrivateWindowError(isDatabaseClosedError(error))
        return {
          groups: [],
          hasMore: false,
          totalCount: 0
        }
      }
    },
    getNextPageParam: (lastPage, allPages) =>
      lastPage.hasMore ? allPages.length + 1 : undefined,
    initialPageParam: 1,
    refetchOnWindowFocus: false,
    staleTime: 30_000
  })

  const chatHistories = useMemo(
    () =>
      chatHistoriesData?.pages.reduce<ChatGroup[]>(
        (acc, page) =>
          page.groups.reduce<ChatGroup[]>((pageAcc, group) => {
            const existingIndex = pageAcc.findIndex((g) => g.label === group.label)
            if (existingIndex === -1) {
              return [...pageAcc, { ...group, items: [...group.items] }]
            }

            const existingGroup = pageAcc[existingIndex]
            const mergedGroup: ChatGroup = {
              ...existingGroup,
              items: [...existingGroup.items, ...group.items]
            }

            return pageAcc.map((entry, index) => (index === existingIndex ? mergedGroup : entry))
          }, acc),
        []
      ) || [],
    [chatHistoriesData]
  )

  const allHistoryIds = useMemo(
    () => chatHistories.flatMap((group) => group.items.map((item) => item.id)),
    [chatHistories]
  )

  const { data: historyMetadata } = useQuery({
    queryKey: ["historyMetadata", allHistoryIds],
    queryFn: async () => {
      if (allHistoryIds.length === 0) return new Map()
      return getHistoriesWithMetadata(allHistoryIds)
    },
    enabled: allHistoryIds.length > 0,
    staleTime: 30_000,
    refetchOnMount: false
  })

  // Mutations
  const { mutate: deleteHistory } = useMutation({
    mutationKey: ["deleteHistory"],
    mutationFn: async (
      history_id: string
    ): Promise<{ deletedId: string; chatData: { historyInfo: HistoryInfo; messages: Message[] } | null }> => {
      // Capture the chat data before deletion for undo
      const chatData = await getFullChatData(history_id)
      const deletedId = await deleteByHistoryId(history_id)
      return { deletedId, chatData }
    },
    onSuccess: ({ deletedId, chatData }) => {
      queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
      const wasActive = historyId === deletedId
      if (wasActive) {
        clearChat()
        updatePageTitle()
      }

      // Show undo notification
      if (chatData) {
        const chatTitle = chatData.historyInfo.title || t("common:untitledChat", "Untitled Chat")
        showUndoNotification({
          title: t("common:undo.chatDeleted", "Chat deleted"),
          description: t("common:undo.chatDeletedDesc", "\"{{title}}\" was removed", { title: chatTitle }),
          onUndo: async () => {
            try {
              await restoreChat(chatData)
              queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
              // If this was the active chat, reload it
              if (wasActive) {
                await loadLocalConversation(chatData.historyInfo.id)
              }
            } catch (error) {
              const errorMessage =
                error && typeof error === "object" && "message" in error
                  ? String((error as { message?: unknown }).message)
                  : t("common:unknownError", { defaultValue: "Unknown error" })
              message.error(
                t("common:undo.restoreChatError", {
                  defaultValue: "Failed to restore chat: {{error}}",
                  error: errorMessage
                })
              )
            }
          }
        })
      }
    },
    onError: (error) => {
      const errorMessage =
        error && typeof error === "object" && "message" in error
          ? String((error as { message?: unknown }).message)
          : t("common:unknownError", { defaultValue: "Unknown error" })
      // eslint-disable-next-line no-console
      console.error("Failed to delete chat history:", error)
      message.error(
        t("common:deleteHistoryError", {
          defaultValue: "Failed to delete chat: {{error}}",
          error: errorMessage
        })
      )
      queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
    }
  })

  const { mutate: editHistory, isPending: renameLoading } = useMutation({
    mutationKey: ["editHistory"],
    mutationFn: async (data: { id: string; title: string }) => {
      return await updateHistory(data.id, data.title)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
      message.success(t("common:updated", { defaultValue: "Updated" }))
    }
  })

  const { mutate: deleteHistoriesByRange, isPending: deleteRangeLoading } =
    useMutation({
      mutationKey: ["deleteHistoriesByRange"],
      onMutate: async (rangeLabel: string) => {
        setDeleteGroup(rangeLabel)
      },
      mutationFn: async (rangeLabel: string) => {
        return await deleteHistoriesByDateRange(rangeLabel)
      },
      onSuccess: (deletedIds: string[]) => {
        queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
        if (historyId && deletedIds.includes(historyId)) {
          clearChat()
        }
        message.success(
          t("common:historiesDeleted", { count: deletedIds.length })
        )
        setDeleteGroup(null)
      },
      onError: () => {
        message.error(t("common:deleteHistoriesError"))
        setDeleteGroup(null)
      }
    })

  const { mutate: pinChatHistory, isPending: pinLoading } = useMutation({
    mutationKey: ["pinHistory"],
    mutationFn: async (data: { id: string; is_pinned: boolean }) => {
      return await pinHistory(data.id, data.is_pinned)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
    },
    onError: (error) => {
      const errorMessage =
        error && typeof error === "object" && "message" in error
          ? String((error as { message?: unknown }).message)
          : t("common:unknownError", { defaultValue: "Unknown error" })
      // eslint-disable-next-line no-console
      console.error("Failed to update pinned chat history:", error)
      message.error(
        t("common:pinHistoryError", {
          defaultValue: "Failed to update pin status: {{error}}",
          error: errorMessage
        })
      )
      queryClient.invalidateQueries({ queryKey: ["fetchChatHistory"] })
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

  const loadLocalConversation = useLoadLocalConversation(
    {
      setServerChatId,
      setHistoryId,
      setHistory,
      setMessages,
      setSelectedModel,
      setSelectedSystemPrompt,
      setSystemPrompt,
      setContextFiles
    },
    {
      t,
      errorLogPrefix: "Failed to load local chat history",
      errorDefaultMessage: "Something went wrong while loading local chat history."
    }
  )

  const effectiveSelectedChatId = selectedChatId ?? historyId ?? null

  const handleRenameSubmit = () => {
    if (!renamingChat) return

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
    editHistory(
      { id: renamingChat, title: newTitle },
      {
        onSuccess: () => {
          setRenamingChat(null)
          setRenameValue("")
        },
        onError: () => {
          message.error(
            t("common:renameChatError", {
              defaultValue: "Failed to rename chat."
            })
          )
        }
      }
    )
  }

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <Skeleton active paragraph={{ rows: 6 }} />
      </div>
    )
  }

  // Error state
  if (status === "error") {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <span className="text-danger">
          {t("common:chatSidebar.loadError", "Failed to load chats")}
        </span>
      </div>
    )
  }

  // Firefox private window error
  if (dexiePrivateWindowError) {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <Empty
          description={t("common:privateWindow", {
            defaultValue:
              "IndexedDB does not work in private mode. Please open the extension in a normal window."
          })}
        />
      </div>
    )
  }

  // Empty state
  if (chatHistories.length === 0) {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <Empty description={t("common:noHistory")} />
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {/* Notification context holder for undo notifications */}
      <Modal
        title={t("common:renameChat", { defaultValue: "Rename chat" })}
        open={!!renamingChat}
        onCancel={() => {
          setRenamingChat(null)
          setRenameValue("")
          setRenameError(null)
        }}
        onOk={handleRenameSubmit}
        confirmLoading={renameLoading}
        okButtonProps={{
          disabled: !renameValue.trim()
        }}
      >
        <Input
          autoFocus
          disabled={renameLoading}
          value={renameValue}
          onChange={(e) => {
            setRenameValue(e.target.value)
            if (renameError) {
              setRenameError(null)
            }
          }}
          onPressEnter={handleRenameSubmit}
          status={renameError ? "error" : undefined}
        />
        {renameError && (
          <div className="mt-1 text-xs text-danger">{renameError}</div>
        )}
      </Modal>
      {chatHistories.map((group) => (
        <div key={group.label}>
          <div className="flex items-center justify-between mt-2 px-1">
            <h3 className="px-2 text-[11px] font-medium text-text-subtle uppercase tracking-wide">
              {group.label === "searchResults"
                ? t("common:searchResults")
                : t(`common:date:${group.label}`)}
            </h3>
            {group.label !== "searchResults" && (
              <Tooltip
                title={t(`common:range:tooltip:${group.label}`)}
                placement="top"
              >
                <button
                  type="button"
                  aria-label={`${t("common:delete", {
                    defaultValue: "Delete"
                  })}: ${t(`common:date:${group.label}`)}`}
                  onClick={() => handleDeleteHistoriesByRange(group.label)}
                  className="p-1 rounded hover:bg-surface2"
                >
                  {deleteRangeLoading && deleteGroup === group.label ? (
                    <Spin size="small" />
                  ) : (
                    <Trash2 className="w-3.5 h-3.5 text-text-muted hover:text-text" />
                  )}
                </button>
              </Tooltip>
            )}
          </div>
          <div className="flex flex-col gap-2 mt-2">
            {group.items.map((chat: HistoryInfo) => {
              const metadata = historyMetadata?.get(chat.id)
              const lastMessage = metadata?.lastMessage
              const isLoadingChat = loadingChatId === chat.id

              return (
                <div
                  key={chat.id}
                  className={cn(
                    "flex py-2 px-2 items-start gap-2 relative rounded-md group transition-opacity duration-300 ease-in-out border",
                    effectiveSelectedChatId === chat.id
                      ? "bg-surface2 border-border-strong text-text"
                      : "bg-surface text-text border-border hover:bg-surface2"
                  )}
                >
                  {chat?.message_source === "copilot" && (
                    <BotIcon className="size-3 text-text-subtle mt-1 flex-shrink-0" />
                  )}
                  {chat?.message_source === "branch" && (
                    <GitBranch className="size-3 text-text-subtle mt-1 flex-shrink-0" />
                  )}
                  <button
                    className="flex-1 overflow-hidden text-start w-full min-w-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 rounded"
                    disabled={isLoadingChat}
                    aria-busy={isLoadingChat}
                    onClick={async () => {
                      const requestId = loadRequestRef.current + 1
                      loadRequestRef.current = requestId
                      setLoadingChatId(chat.id)
                      try {
                        const accepted = await loadLocalConversation(chat.id)
                        if (accepted && loadRequestRef.current === requestId) {
                          onSelectChat?.(chat.id)
                        }
                      } catch (error) {
                        // eslint-disable-next-line no-console
                        console.error("Failed to load conversation:", error)
                        message.error(
                          t("common:chatSidebar.loadError", "Failed to load chat")
                        )
                      } finally {
                        if (loadRequestRef.current === requestId) {
                          setLoadingChatId(null)
                        }
                      }
                    }}
                  >
                    <div className="flex flex-col gap-1 min-w-0">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="truncate font-medium" title={chat.title}>
                          {chat.title}
                        </span>
                        {isLoadingChat && <Spin size="small" />}
                      </div>
                      {metadata && (
                        <div className="flex items-center gap-2 text-xs text-text-subtle">
                          <span className="flex items-center gap-1">
                            <MessageSquare className="size-3" />
                            {metadata.messageCount || 0}
                          </span>
                          {lastMessage && (
                            <span>
                              {formatRelativeTime(
                                lastMessage.createdAt || chat.createdAt,
                                t
                              )}
                            </span>
                          )}
                        </div>
                      )}
                      {lastMessage && (
                        <span className="text-xs text-text-subtle truncate">
                          {truncateMessage(lastMessage.content || "")}
                        </span>
                      )}
                    </div>
                  </button>
                  <Dropdown
                    menu={{
                      items: [
                        {
                          key: "pin",
                          icon: chat.is_pinned ? (
                            <PinOffIcon className="w-4 h-4" />
                          ) : (
                            <PinIcon className="w-4 h-4" />
                          ),
                          label: chat.is_pinned ? t("common:unpin") : t("common:pin"),
                          onClick: () =>
                            pinChatHistory({
                              id: chat.id,
                              is_pinned: !chat.is_pinned
                            }),
                          disabled: pinLoading
                        },
                        ...(isConnected && onOpenFolderPicker
                          ? [
                              {
                                key: "moveToFolder",
                                icon: <FolderIcon className="w-4 h-4" />,
                                label: t("common:moveToFolder"),
                                onClick: () => {
                                  onOpenFolderPicker(chat.id)
                                  setOpenMenuFor(null)
                                }
                              }
                            ]
                          : []),
                        {
                          key: "edit",
                          icon: <Edit3 className="w-4 h-4" />,
                          label: t("common:rename"),
                          onClick: () => {
                            setRenamingChat(chat.id)
                            setRenameValue(chat.title)
                          }
                        },
                        {
                          key: "delete",
                          icon: <Trash2 className="w-4 h-4" />,
                          label: t("common:delete"),
                          onClick: async () => {
                            const ok = await confirmDanger({
                              title: t("common:confirmTitle"),
                              content: t("common:deleteHistoryConfirmation"),
                              okText: t("common:delete"),
                              cancelText: t("common:cancel")
                            })
                            if (!ok) return
                            deleteHistory(chat.id)
                          }
                        }
                      ]
                    }}
                    trigger={["click"]}
                    placement="bottomRight"
                    open={openMenuFor === chat.id}
                    onOpenChange={(open) => setOpenMenuFor(open ? chat.id : null)}
                  >
                    <IconButton
                      className="text-text-subtle opacity-80 hover:opacity-100 h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0"
                      ariaLabel={`${t("option:header.moreActions", "More actions")}: ${chat.title}`}
                      hasPopup="menu"
                      ariaExpanded={openMenuFor === chat.id}
                    >
                      <MoreHorizontal className="w-4 h-4" />
                    </IconButton>
                  </Dropdown>
                </div>
              )
            })}
          </div>
        </div>
      ))}

      {hasNextPage && (
        <div className="flex justify-center mt-4 mb-2">
          <Button
            type="default"
            onClick={() => fetchNextPage()}
            loading={isFetchingNextPage}
            icon={!isFetchingNextPage ? <ChevronDown className="w-4 h-4" /> : undefined}
            className="flex items-center gap-2 text-sm"
          >
            {isFetchingNextPage
              ? t("common:loading.title", { defaultValue: "Loading..." })
              : t("common:loadMore")}
          </Button>
        </div>
      )}
    </div>
  )
}

export default LocalChatList
