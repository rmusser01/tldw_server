import React from "react"
import { message, Tooltip, Modal } from "antd"
import { useQueryClient } from "@tanstack/react-query"
import {
  ChevronLeft,
  Circle,
  CheckCircle2,
  Clock,
  XCircle,
  Pin,
  PinOff,
  Plus,
  Search,
  X,
  Folder,
  FolderPlus,
  Tag,
  Trash2,
  CheckSquare
} from "lucide-react"
import type { TFunction } from "i18next"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"
import { classNames } from "@/libs/class-name"
import type { SidepanelChatTab, ConversationStatus } from "@/store/sidepanel-chat-tabs"
import { useSidepanelChatTabsStore } from "@/store/sidepanel-chat-tabs"
import { useUiModeStore } from "@/store/ui-mode"
import { useDebounce } from "@/hooks/useDebounce"
import { useConnectionState } from "@/hooks/useConnectionState"
import { useServerChatHistory, type ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import { PageAssistDatabase } from "@/db/dexie/chat"
import type { HistoryInfo } from "@/db/dexie/types"
import type { SidepanelChatOwner } from "@/hooks/useSidepanelChatOwner"
import { useFolderStore } from "@/store/folder"
import { useBulkChatOperations } from "@/hooks/useBulkChatOperations"
import { useStorage } from "@plasmohq/storage/hook"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { ModeToggle } from "./ModeToggle"
import { ConversationContextMenu } from "./ConversationContextMenu"
import { FolderPickerModal } from "./FolderPickerModal"
import { exportTabToJSON, exportTabToMarkdown } from "@/utils/conversation-export"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const DEFAULT_SIDEBAR_WIDTH = 288
const SIDEBAR_MIN_WIDTH = 240
const SIDEBAR_MAX_WIDTH = 420

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

const BulkFolderPickerModal = React.lazy(() => import("./BulkFolderPickerModal"))
const BulkTagPickerModal = React.lazy(() => import("./BulkTagPickerModal"))

type SidebarGroup = {
  label: string
  items: SidepanelChatTab[]
}

type SidepanelChatSidebarProps = {
  owner?: Pick<SidepanelChatOwner, "ownerKey" | "isCurrent"> & {
    snapshot: Pick<SidepanelChatOwner["snapshot"], "requestScope">
  }
  open: boolean
  variant: "docked" | "overlay"
  tabs: SidepanelChatTab[]
  activeTabId: string | null
  onSelectTab: (tabId: string) => void
  onCloseTab: (tabId: string) => void
  onNewTab: () => void
  searchQuery: string
  onSearchQueryChange: (value: string) => void
  searchInputRef?: React.RefObject<HTMLInputElement>
  focusSearchTrigger?: number
  onOpenLocalHistory?: (historyId: string) => void
  onOpenServerChat?: (chat: ServerChatHistoryItem) => void
  onClose?: () => void
}

/**
 * Status indicator component
 */
const StatusIndicator: React.FC<{ status: ConversationStatus }> = ({
  status
}) => {
  if (!status) return null

  const icons: Record<string, React.ReactNode> = {
    in_progress: <Circle className="size-2 fill-primary text-primary" />,
    resolved: <CheckCircle2 className="size-2.5 text-success" />,
    backlog: <Clock className="size-2.5 text-text-subtle" />,
    non_viable: <XCircle className="size-2.5 text-danger" />
  }

  return (
    <span className="flex-shrink-0" aria-label={status}>
      {icons[status]}
    </span>
  )
}

const SidebarMetaRow: React.FC<{
  conversationId: string | null
  topic?: string | null
}> = ({ conversationId, topic }) => {
  const { getFoldersForConversation, uiPrefs } = useFolderStore(
    (state) => ({
      getFoldersForConversation: state.getFoldersForConversation,
      uiPrefs: state.uiPrefs
    }),
    shallow
  )
  const normalizedTopic = (topic || "").trim()
  const folders = React.useMemo(
    () => (conversationId ? getFoldersForConversation(conversationId) : []),
    [conversationId, getFoldersForConversation]
  )

  if (!normalizedTopic && folders.length === 0) return null

  const topicLabel = normalizedTopic.startsWith("#")
    ? normalizedTopic
    : `#${normalizedTopic}`

  return (
    <div className="mt-0.5 flex flex-wrap items-center gap-1">
      {normalizedTopic && (
        <span
          className="max-w-[140px] truncate rounded-full border border-border/60 bg-surface2/70 px-1.5 py-0.5 text-[10px] font-medium text-text-subtle"
          title={normalizedTopic}
        >
          {topicLabel}
        </span>
      )}
      {folders.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {folders.slice(0, 3).map((folder) => {
            const color = uiPrefs[folder.id]?.color || "rgb(var(--color-muted))"
            return (
              <span
                key={folder.id}
                className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[10px] font-medium"
                style={{
                  backgroundColor: `${color}20`,
                  color: color
                }}
              >
                <Folder className="size-2" />
                <span className="truncate max-w-[60px]">{folder.name}</span>
              </span>
            )
          })}
          {folders.length > 3 && (
            <span className="text-[10px] text-text-muted">
              +{folders.length - 3}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

const buildGroups = (tabs: SidepanelChatTab[], t: TFunction): SidebarGroup[] => {
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const startOfYesterday = new Date(startOfToday)
  startOfYesterday.setDate(startOfToday.getDate() - 1)

  const today: SidepanelChatTab[] = []
  const yesterday: SidepanelChatTab[] = []
  const older: SidepanelChatTab[] = []

  tabs.forEach((tab) => {
    const updatedAt = new Date(tab.updatedAt)
    if (updatedAt >= startOfToday) {
      today.push(tab)
    } else if (updatedAt >= startOfYesterday) {
      yesterday.push(tab)
    } else {
      older.push(tab)
    }
  })

  const groups: SidebarGroup[] = []
  if (today.length > 0) {
    groups.push({
      label: t("common:today", "Today"),
      items: today
    })
  }
  if (yesterday.length > 0) {
    groups.push({
      label: t("common:yesterday", "Yesterday"),
      items: yesterday
    })
  }
  if (older.length > 0) {
    groups.push({
      label: t("common:older", "Older"),
      items: older
    })
  }
  return groups
}

export const SidepanelChatSidebar = ({
  owner,
  open,
  variant,
  tabs,
  activeTabId,
  onSelectTab,
  onCloseTab,
  onNewTab,
  searchQuery,
  onSearchQueryChange,
  searchInputRef,
  focusSearchTrigger,
  onOpenLocalHistory,
  onOpenServerChat,
  onClose
}: SidepanelChatSidebarProps) => {
  const { t } = useTranslation(["common", "sidepanel"])
  const queryClient = useQueryClient()
  const togglePinned = useSidepanelChatTabsStore(
    (state) => state.togglePinned
  )
  const renameTab = useSidepanelChatTabsStore((state) => state.renameTab)
  const setStatus = useSidepanelChatTabsStore((state) => state.setStatus)
  const uiMode = useUiModeStore((state) => state.mode)
  const [storedSidebarWidth, setStoredSidebarWidth] = useStorage<number>(
    "sidepanelSidebarWidth",
    DEFAULT_SIDEBAR_WIDTH
  )
  const setRouteContext = useChatSurfaceCoordinatorStore(
    (state) => state.setRouteContext
  )
  const setPanelVisible = useChatSurfaceCoordinatorStore(
    (state) => state.setPanelVisible
  )
  const markPanelEngaged = useChatSurfaceCoordinatorStore(
    (state) => state.markPanelEngaged
  )
  const [isResizing, setIsResizing] = React.useState(false)
  const resizeOriginRef = React.useRef<{ startX: number; startWidth: number } | null>(
    null
  )
  const isResizable = variant === "docked" && uiMode === "pro"
  const rawSidebarWidth =
    typeof storedSidebarWidth === "number" && Number.isFinite(storedSidebarWidth)
      ? storedSidebarWidth
      : DEFAULT_SIDEBAR_WIDTH
  const sidebarWidth = clamp(
    rawSidebarWidth,
    SIDEBAR_MIN_WIDTH,
    SIDEBAR_MAX_WIDTH
  )

  const handleResizeStart = React.useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!isResizable) return
      event.preventDefault()
      resizeOriginRef.current = {
        startX: event.clientX,
        startWidth: sidebarWidth
      }
      setIsResizing(true)
    },
    [isResizable, sidebarWidth]
  )

  const handleResizeKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (!isResizable) return
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return
      event.preventDefault()
      const step = event.shiftKey ? 32 : 16
      const delta = event.key === "ArrowRight" ? step : -step
      const nextWidth = clamp(
        sidebarWidth + delta,
        SIDEBAR_MIN_WIDTH,
        SIDEBAR_MAX_WIDTH
      )
      setStoredSidebarWidth(nextWidth)
    },
    [isResizable, sidebarWidth, setStoredSidebarWidth]
  )

  React.useEffect(() => {
    if (!isResizing) return
    const handleMove = (event: MouseEvent) => {
      if (!resizeOriginRef.current) return
      const delta = event.clientX - resizeOriginRef.current.startX
      const nextWidth = clamp(
        resizeOriginRef.current.startWidth + delta,
        SIDEBAR_MIN_WIDTH,
        SIDEBAR_MAX_WIDTH
      )
      setStoredSidebarWidth(nextWidth)
    }
    const handleUp = () => {
      setIsResizing(false)
      resizeOriginRef.current = null
    }
    window.addEventListener("mousemove", handleMove)
    window.addEventListener("mouseup", handleUp)
    return () => {
      window.removeEventListener("mousemove", handleMove)
      window.removeEventListener("mouseup", handleUp)
    }
  }, [isResizing, setStoredSidebarWidth])

  React.useEffect(() => {
    if (!isResizing || typeof document === "undefined") return
    const previousCursor = document.body.style.cursor
    const previousUserSelect = document.body.style.userSelect
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    return () => {
      document.body.style.cursor = previousCursor
      document.body.style.userSelect = previousUserSelect
    }
  }, [isResizing])

  // Context menu handlers
  const handleRename = React.useCallback(
    (tabId: string, newLabel: string) => {
      renameTab(tabId, newLabel)
    },
    [renameTab]
  )

  const handleSetStatus = React.useCallback(
    (tabId: string, status: ConversationStatus) => {
      setStatus(tabId, status)
    },
    [setStatus]
  )

  // Folder picker modal state
  const [folderPickerOpen, setFolderPickerOpen] = React.useState(false)
  const [folderPickerTabId, setFolderPickerTabId] = React.useState<string | null>(null)

  const debouncedSearchQuery = useDebounce(searchQuery, 250)
  const normalizedQuery = searchQuery.trim().toLowerCase()
  const hasSearch = normalizedQuery.length > 0
  const debouncedNormalizedQuery = debouncedSearchQuery.trim().toLowerCase()
  const hasDebouncedSearchQuery = debouncedNormalizedQuery.length > 0
  const isSearchDebouncing = normalizedQuery !== debouncedNormalizedQuery
  const ownerCurrent = Boolean(owner?.isCurrent())
  const { isConnected } = useConnectionState()
  const dbRef = React.useRef<PageAssistDatabase | null>(null)
  const [localSearchRetry, setLocalSearchRetry] = React.useState(0)
  const [localSearch, setLocalSearch] = React.useState<{
    owner: typeof owner
    query: string
    status: "idle" | "pending" | "success" | "error"
    results: HistoryInfo[]
  }>({ owner: undefined, query: "", status: "idle", results: [] })
  const serverSearch = useServerChatHistory(
    debouncedSearchQuery,
    {
      enabled: hasDebouncedSearchQuery && Boolean(owner?.isCurrent()),
      owner: owner ? { key: owner.ownerKey, requestScope: owner.snapshot.requestScope, isCurrent: owner.isCurrent } : undefined,
      mode: "search"
    }
  )
  const localSearchCurrent = localSearch.owner === owner && localSearch.query === normalizedQuery
  const localSearchPending = ownerCurrent && hasSearch && (
    isSearchDebouncing || !localSearchCurrent || localSearch.status === "pending"
  )
  const serverSearchUnavailable = !isConnected || serverSearch.fetchStatus === "paused"
  const searchPending = localSearchPending || (
    ownerCurrent && hasSearch && !serverSearchUnavailable && (
      isSearchDebouncing || serverSearch.isPending || serverSearch.isFetching
    )
  )
  const retryServerSearch = () => {
    if (!owner?.isCurrent() || serverSearchUnavailable || isSearchDebouncing ||
      !hasDebouncedSearchQuery || serverSearch.isFetching) return
    void serverSearch.refetch()
  }
  const retryLocalSearch = () => {
    if (!owner?.isCurrent() || isSearchDebouncing || !hasSearch || localSearchPending) return
    setLocalSearchRetry(value => value + 1)
  }

  React.useEffect(() => {
    setRouteContext({ routeId: "chat", surface: "extension" })
  }, [setRouteContext])

  React.useEffect(() => {
    setPanelVisible("server-history", open)

    return () => {
      setPanelVisible("server-history", false)
    }
  }, [open, setPanelVisible])

  React.useEffect(() => {
    if (hasDebouncedSearchQuery) {
      markPanelEngaged("server-history")
    }
  }, [hasDebouncedSearchQuery, markPanelEngaged])

  React.useEffect(() => {
    if (!focusSearchTrigger) return
    if (searchInputRef?.current) {
      searchInputRef.current.focus()
      searchInputRef.current.select()
    }
  }, [focusSearchTrigger, searchInputRef])

  React.useEffect(() => {
    const normalized = debouncedSearchQuery.trim()
    const query = normalized.toLowerCase()
    if (!normalized || !owner?.isCurrent()) {
      setLocalSearch({ owner, query, status: "idle", results: [] })
      return
    }
    let isActive = true
    setLocalSearch({ owner, query, status: "pending", results: [] })
    if (!dbRef.current) {
      dbRef.current = new PageAssistDatabase()
    }
    const fetchResults = async () => {
      try {
        const results = await dbRef.current!.fullTextSearchChatHistories(
          normalized
        )
        if (isActive && owner.isCurrent()) {
          setLocalSearch({ owner, query, status: "success", results: results.filter(history => history.server_scope_key === owner.ownerKey) })
        }
      } catch {
        if (isActive && owner.isCurrent()) {
          setLocalSearch({ owner, query, status: "error", results: [] })
        }
      }
    }
    void fetchResults()
    return () => {
      isActive = false
    }
  }, [debouncedSearchQuery, owner, localSearchRetry])

  // Bulk selection state
  const [selectionMode, setSelectionMode] = React.useState(false)
  const [selectedTabIds, setSelectedTabIds] = React.useState<string[]>([])
  const [bulkFolderPickerOpen, setBulkFolderPickerOpen] = React.useState(false)
  const [bulkTagPickerOpen, setBulkTagPickerOpen] = React.useState(false)
  const [bulkDeleteConfirmOpen, setBulkDeleteConfirmOpen] = React.useState(false)
  const [isBulkDeleting, setIsBulkDeleting] = React.useState(false)

  const { folderApiAvailable } = useFolderStore(
    (state) => ({
      folderApiAvailable: state.folderApiAvailable
    }),
    shallow
  )

  const handleAddToFolder = React.useCallback((tabId: string) => {
    const tab = tabs.find((t) => t.id === tabId)
    if (tab) {
      setFolderPickerTabId(tab.serverChatId || tab.historyId)
      setFolderPickerOpen(true)
    }
  }, [tabs])

  const handleFolderPickerClose = React.useCallback(() => {
    setFolderPickerOpen(false)
  }, [])

  const handleExportJSON = React.useCallback(
    async (tabId: string) => {
      const tab = tabs.find((t) => t.id === tabId)
      if (!tab?.historyId) {
        message.error(t("sidepanel:contextMenu.exportError", "Cannot export: conversation not saved locally"))
        return
      }
      const success = await exportTabToJSON(tab.historyId)
      if (success) {
        message.success(t("sidepanel:contextMenu.exportSuccess", "Exported successfully"))
      } else {
        message.error(t("sidepanel:contextMenu.exportError", "Export failed"))
      }
    },
    [tabs, t]
  )

  const handleExportMarkdown = React.useCallback(
    async (tabId: string) => {
      const tab = tabs.find((t) => t.id === tabId)
      if (!tab?.historyId) {
        message.error(t("sidepanel:contextMenu.exportError", "Cannot export: conversation not saved locally"))
        return
      }
      const success = await exportTabToMarkdown(tab.historyId)
      if (success) {
        message.success(t("sidepanel:contextMenu.exportSuccess", "Exported successfully"))
      } else {
        message.error(t("sidepanel:contextMenu.exportError", "Export failed"))
      }
    },
    [tabs, t]
  )
  const isPro = uiMode === "pro"
  const itemClass = isPro ? "px-2 py-1 text-caption" : "px-3 py-2 text-body"
  const groupGap = isPro ? "gap-1" : "gap-2"
  const groupSpacing = isPro ? "mb-3" : "mb-4"

  const toggleSelectionMode = React.useCallback(() => {
    setSelectionMode((prev) => !prev)
  }, [])

  const toggleTabSelected = React.useCallback((tabId: string) => {
    setSelectedTabIds((prev) =>
      prev.includes(tabId) ? prev.filter((id) => id !== tabId) : [...prev, tabId]
    )
  }, [])

  const handleRowClick = React.useCallback(
    (tabId: string) => {
      if (selectionMode) {
        toggleTabSelected(tabId)
        return
      }
      onSelectTab(tabId)
    },
    [selectionMode, toggleTabSelected, onSelectTab]
  )

  const renderTabRow = React.useCallback(
    (tab: SidepanelChatTab) => {
      const isSelected = selectedTabIds.includes(tab.id)
      const row = (
        <div
          key={tab.id}
          className={classNames(
            "flex items-center justify-between rounded-md",
            itemClass,
            tab.id === activeTabId
              ? "bg-surface2 text-text"
              : "text-text-muted hover:bg-surface2 hover:text-text",
            selectionMode && isSelected ? "border border-primary/40" : "border border-transparent"
          )}
        >
          {selectionMode && (
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => toggleTabSelected(tab.id)}
              onClick={(e) => e.stopPropagation()}
              className="mr-2 size-3 rounded border-border text-primary accent-primary"
              aria-label={t("sidepanel:multiSelect.toggle", "Toggle selection")}
            />
          )}
          <button
            type="button"
            onClick={() => handleRowClick(tab.id)}
            className="flex-1 min-w-0 text-left"
            title={tab.label}
          >
            <div className="flex items-center gap-1.5">
              <StatusIndicator status={tab.status} />
              <span className="truncate">{tab.label}</span>
            </div>
            <SidebarMetaRow
              conversationId={tab.serverChatId || tab.historyId}
              topic={tab.serverChatTopic}
            />
          </button>
          {!selectionMode && (
            <div className="flex items-center gap-1 flex-shrink-0">
              <Tooltip title={tab.pinned ? t("common:unpin", "Unpin") : t("common:pin", "Pin")}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    togglePinned(tab.id)
                  }}
                  className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text"
                  aria-label={tab.pinned ? t("common:unpin", "Unpin") : t("common:pin", "Pin")}
                  title={tab.pinned ? t("common:unpin", "Unpin") : t("common:pin", "Pin")}
                >
                  {tab.pinned ? <PinOff className="size-3" /> : <Pin className="size-3" />}
                </button>
              </Tooltip>
              <Tooltip title={t("common:close", "Close")}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    onCloseTab(tab.id)
                  }}
                  className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text"
                  aria-label={t("common:close", "Close")}
                  title={t("common:close", "Close")}
                >
                  <X className="size-3" />
                </button>
              </Tooltip>
            </div>
          )}
        </div>
      )

      if (selectionMode) {
        return row
      }

      return (
        <ConversationContextMenu
          key={tab.id}
          tab={tab}
          onRename={handleRename}
          onTogglePin={togglePinned}
          onSetStatus={handleSetStatus}
          onAddToFolder={handleAddToFolder}
          onExportJSON={handleExportJSON}
          onExportMarkdown={handleExportMarkdown}
          onDelete={onCloseTab}
          currentStatus={tab.status}
        >
          {row}
        </ConversationContextMenu>
      )
    },
    [
      activeTabId,
      itemClass,
      selectionMode,
      selectedTabIds,
      t,
      togglePinned,
      toggleTabSelected,
      handleRowClick,
      onCloseTab,
      handleRename,
      handleSetStatus,
      handleAddToFolder,
      handleExportJSON,
      handleExportMarkdown
    ]
  )

  const renderSearchRow = React.useCallback(
    ({
      key,
      label,
      metaLabel,
      onOpen,
      topic
    }: {
      key: string
      label: string
      metaLabel: string
      onOpen: () => void
      topic?: string | null
    }) => (
      <div
        key={key}
        className={classNames(
          "flex items-center justify-between rounded-md border border-transparent",
          itemClass,
          "text-text-muted hover:bg-surface2 hover:text-text"
        )}
      >
        <button
          type="button"
          onClick={onOpen}
          className="flex-1 min-w-0 text-left"
          title={label}
        >
          <div className="flex items-center gap-1.5">
            <span className="truncate">{label}</span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[10px] text-text-subtle">
            <span>{metaLabel}</span>
            {topic && (
              <span className="truncate max-w-[140px]">#{topic}</span>
            )}
          </div>
        </button>
      </div>
    ),
    [itemClass]
  )

  const sortedTabs = React.useMemo(
    () =>
      [...tabs].sort((a, b) => b.updatedAt - a.updatedAt),
    [tabs]
  )

  const filteredTabs = React.useMemo(() => {
    if (!normalizedQuery) return sortedTabs
    return sortedTabs.filter((tab) =>
      tab.label.toLowerCase().includes(normalizedQuery)
    )
  }, [normalizedQuery, sortedTabs])

  const openHistoryIds = React.useMemo(() => {
    const ids = new Set<string>()
    tabs.forEach((tab) => {
      if (tab.historyId) ids.add(tab.historyId)
    })
    return ids
  }, [tabs])

  const openServerChatIds = React.useMemo(() => {
    const ids = new Set<string>()
    tabs.forEach((tab) => {
      if (tab.serverChatId) ids.add(tab.serverChatId)
    })
    return ids
  }, [tabs])

  const filteredLocalResults = React.useMemo(() => {
    if (!hasSearch || !owner?.isCurrent() || !localSearchCurrent || localSearch.status !== "success") return []
    return localSearch.results.filter((history) => !openHistoryIds.has(history.id))
  }, [hasSearch, localSearch, localSearchCurrent, openHistoryIds, owner])

  const localServerIds = React.useMemo(() => {
    const ids = new Set<string>()
    filteredLocalResults.forEach((history) => {
      if (history.server_chat_id) ids.add(history.server_chat_id)
    })
    return ids
  }, [filteredLocalResults])

  const filteredServerResults = React.useMemo(() => {
    if (!hasSearch || !owner?.isCurrent() || isSearchDebouncing || serverSearchUnavailable || serverSearch.isError) return []
    return serverSearch.data.filter((chat) => {
      const chatId = String(chat.id)
      if (openServerChatIds.has(chatId)) return false
      if (localServerIds.has(chatId)) return false
      return true
    })
  }, [hasSearch, isSearchDebouncing, localServerIds, openServerChatIds, serverSearch.data, serverSearch.isError, serverSearchUnavailable, owner])

  const pinnedTabs = filteredTabs.filter((tab) => tab.pinned)
  const unpinnedTabs = filteredTabs.filter((tab) => !tab.pinned)
  const groups = buildGroups(unpinnedTabs, t)

  const visibleTabIds = React.useMemo(() => {
    const ids = new Set<string>()
    pinnedTabs.forEach((tab) => ids.add(tab.id))
    groups.forEach((group) => {
      group.items.forEach((tab) => ids.add(tab.id))
    })
    return Array.from(ids)
  }, [groups, pinnedTabs])

  const tabById = React.useMemo(
    () => new Map(filteredTabs.map((tab) => [tab.id, tab])),
    [filteredTabs]
  )

  const selectedTabs = React.useMemo(
    () =>
      selectedTabIds
        .map((id) => tabById.get(id))
        .filter((tab): tab is SidepanelChatTab => Boolean(tab)),
    [selectedTabIds, tabById]
  )

  const selectedConversationIds = React.useMemo(
    () => {
      const ids = new Set<string>()
      selectedTabs.forEach((tab) => {
        if (tab.serverChatId) ids.add(tab.serverChatId)
      })
      return Array.from(ids)
    },
    [selectedTabs]
  )

  const localOnlySelectedCount = React.useMemo(
    () => selectedTabs.filter((tab) => !tab.serverChatId).length,
    [selectedTabs]
  )
  const { openBulkFolderPicker, openBulkTagPicker, applyBulkDelete } =
    useBulkChatOperations({
      selectedConversationIds,
      folderApiAvailable,
      deleteConversation: (conversationId) => tldwClient.deleteChat(conversationId),
      t,
      setBulkFolderPickerOpen,
      setBulkTagPickerOpen
    })

  React.useEffect(() => {
    setSelectedTabIds((prev) => {
      if (!selectionMode) {
        return prev.length === 0 ? prev : []
      }
      const next = prev.filter((id) => visibleTabIds.includes(id))
      const unchanged =
        next.length === prev.length &&
        next.every((id, index) => id === prev[index])
      return unchanged ? prev : next
    })
    if (!selectionMode) {
      setBulkFolderPickerOpen(false)
      setBulkTagPickerOpen(false)
      setBulkDeleteConfirmOpen(false)
    }
  }, [selectionMode, visibleTabIds])

  const handleSelectAllVisible = React.useCallback(() => {
    setSelectedTabIds(visibleTabIds)
  }, [visibleTabIds])

  const clearSelection = React.useCallback(() => {
    setSelectedTabIds([])
  }, [])

  const openBulkDeleteConfirm = React.useCallback(() => {
    setBulkDeleteConfirmOpen(true)
  }, [])

  const handleBulkFolderPickerClose = React.useCallback(() => {
    setBulkFolderPickerOpen(false)
  }, [])

  const handleBulkTagPickerClose = React.useCallback(() => {
    setBulkTagPickerOpen(false)
  }, [])

  const handleBulkDeleteConfirmClose = React.useCallback(() => {
    setBulkDeleteConfirmOpen(false)
  }, [])

  const handleBulkDelete = async () => {
    if (selectedTabs.length === 0) return

    setIsBulkDeleting(true)
    try {
      const failedConversationIds = new Set<string>()
      if (selectedConversationIds.length > 0) {
        const result = await applyBulkDelete()
        if (!result) return
        result.failedConversationIds.forEach((id) => failedConversationIds.add(id))
        queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
      }

      const remainingSelection: string[] = []
      selectedTabs.forEach((tab) => {
        if (tab.serverChatId && failedConversationIds.has(tab.serverChatId)) {
          remainingSelection.push(tab.id)
          return
        }
        onCloseTab(tab.id)
      })

      if (localOnlySelectedCount > 0) {
        message.info(
          t("sidepanel:multiSelect.localOnlyClosed", {
            defaultValue: "{{count}} local chat(s) were closed.",
            count: localOnlySelectedCount
          })
        )
      }

      setSelectedTabIds(remainingSelection)
      setBulkDeleteConfirmOpen(false)
    } finally {
      setIsBulkDeleting(false)
    }
  }

  const panelClass = classNames(
    "relative flex h-full min-h-0 flex-col border-r border-border bg-surface",
    variant === "overlay"
      ? `fixed left-0 top-0 z-40 w-72 shadow-card transition-transform ${
          open ? "translate-x-0" : "-translate-x-full"
        }`
      : "shadow-none"
  )

  return (
    <aside
      className={panelClass}
      data-testid="sidepanel-chat-sidebar"
      style={variant === "docked" ? { width: sidebarWidth } : undefined}
    >
      <div
        className={classNames(
          "flex items-center justify-between border-b border-border",
          isPro ? "px-4 py-2" : "px-4 py-3"
        )}
      >
        <div className="text-body font-semibold text-text">
          {t("common:chatSidebar.title", "Chats")}
        </div>
        <div className="flex items-center gap-1">
          <Tooltip title={t("common:chatSidebar.newChat", "New Chat")}>
            <button
              type="button"
              data-testid="sidepanel-sidebar-new-chat"
              onClick={onNewTab}
              className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
              aria-label={t("common:chatSidebar.newChat", "New Chat")}
              title={t("common:chatSidebar.newChat", "New Chat")}
            >
              <Plus className="size-4" />
            </button>
          </Tooltip>
          <Tooltip
            title={
              selectionMode
                ? t("sidepanel:multiSelect.exit", "Exit selection")
                : t("sidepanel:multiSelect.enter", "Select chats")
            }
          >
            <button
              type="button"
              onClick={toggleSelectionMode}
              className={classNames(
                "rounded-md p-2",
                selectionMode
                  ? "bg-surface2 text-text"
                  : "text-text-muted hover:bg-surface2 hover:text-text"
              )}
              aria-pressed={selectionMode}
              aria-label={
                selectionMode
                  ? t("sidepanel:multiSelect.exit", "Exit selection")
                  : t("sidepanel:multiSelect.enter", "Select chats")
              }
            >
              <CheckSquare className="size-4" />
            </button>
          </Tooltip>
          {variant === "overlay" ? (
            <Tooltip title={t("common:close", "Close")}>
              <button
                type="button"
                aria-label={t("common:close", "Close")}
                onClick={() => onClose?.()}
                className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
                title={t("common:close", "Close")}
              >
                <X className="size-4" />
              </button>
            </Tooltip>
          ) : (
            <Tooltip title={t("common:chatSidebar.collapse", "Collapse sidebar")}>
              <button
                type="button"
                aria-label={t("common:chatSidebar.collapse", "Collapse sidebar")}
                onClick={() => onClose?.()}
                className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
                title={t("common:chatSidebar.collapse", "Collapse sidebar")}
              >
                <ChevronLeft className="size-4" />
              </button>
            </Tooltip>
          )}
        </div>
      </div>

      <div className="border-b border-border px-4 py-3">
        <label className="panel-input flex items-center gap-2">
          <Search className="size-4 text-text-subtle" aria-hidden="true" />
          <input
            type="text"
            value={searchQuery}
            onChange={(event) => onSearchQueryChange(event.target.value)}
            placeholder={t("common:chatSidebar.search", "Search chats...")}
            data-testid="sidepanel-sidebar-search"
            className="w-full bg-transparent text-body text-text outline-none placeholder:text-text-subtle"
            ref={searchInputRef}
          />
        </label>
      </div>

      {selectionMode && (
        <div className="border-b border-border px-4 py-2">
          <div className="flex items-center justify-between text-xs text-text-subtle">
            <span>
              {t("sidepanel:multiSelect.count", {
                defaultValue: "{{count}} selected",
                count: selectedTabIds.length
              })}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSelectAllVisible}
                className="text-text-subtle hover:text-text"
                disabled={visibleTabIds.length === 0}
              >
                {t("sidepanel:multiSelect.selectAll", "Select all")}
              </button>
              <button
                type="button"
                onClick={clearSelection}
                className="text-text-subtle hover:text-text"
                disabled={selectedTabIds.length === 0}
              >
                {t("sidepanel:multiSelect.clear", "Clear")}
              </button>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={openBulkFolderPicker}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
              disabled={selectedTabIds.length === 0}
            >
              <FolderPlus className="size-3.5" />
              {t("sidepanel:multiSelect.addToFolder", "Add to folders")}
            </button>
            <button
              type="button"
              onClick={openBulkTagPicker}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
              disabled={selectedTabIds.length === 0}
            >
              <Tag className="size-3.5" />
              {t("sidepanel:multiSelect.addTags", "Add tags")}
            </button>
            <button
              type="button"
              onClick={openBulkDeleteConfirm}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-danger hover:bg-surface2"
              disabled={selectedTabIds.length === 0}
              aria-label={t("common:delete", "Delete")}
              title={t("common:delete", "Delete")}
            >
              <Trash2 className="size-3.5" />
              {t("common:delete", "Delete")}
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto px-3 py-3">
        {hasSearch ? (
          <>
            {!ownerCurrent && (
              <div role="status" className="px-2 py-3 text-xs text-text-subtle">
                {t("common:chatSidebar.searchUnverified", "History search is unavailable until your account is verified.")}
              </div>
            )}
            {searchPending && (
              <div role="status" className="px-2 py-3 text-xs text-text-subtle">
                {t("common:chatSidebar.searching", "Searching chat history…")}
              </div>
            )}
            {ownerCurrent && !isSearchDebouncing && serverSearchUnavailable && (
              <div role="status" className="px-2 py-3 text-xs text-text-subtle">
                {t("common:chatSidebar.searchOffline", "Server history search is unavailable while disconnected.")}
              </div>
            )}
            {ownerCurrent && !isSearchDebouncing && !serverSearchUnavailable && serverSearch.isError && !serverSearch.isFetching && (
              <div role="alert" className="px-2 py-3 text-xs text-text-subtle">
                {t("common:chatSidebar.serverSearchFailed", "Server history search failed. Try again.")}
                <button
                  type="button"
                  className="ml-2 underline hover:text-text"
                  aria-label={t("common:chatSidebar.retryServerSearch", "Retry server history search")}
                  onClick={retryServerSearch}
                >
                  {t("common:retry", "Retry")}
                </button>
              </div>
            )}
            {ownerCurrent && !isSearchDebouncing && localSearchCurrent && localSearch.status === "error" && (
              <div role="alert" className="px-2 py-3 text-xs text-text-subtle">
                {t("common:chatSidebar.localSearchFailed", "Local history search failed. Try again.")}
                <button
                  type="button"
                  className="ml-2 underline hover:text-text"
                  aria-label={t("common:chatSidebar.retryLocalSearch", "Retry local history search")}
                  onClick={retryLocalSearch}
                >
                  {t("common:retry", "Retry")}
                </button>
              </div>
            )}
            {pinnedTabs.length > 0 && (
              <div className={groupSpacing}>
                <div className="panel-section-label">
                  {t("common:pinned", "Pinned")}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {pinnedTabs.map((tab) => renderTabRow(tab))}
                </div>
              </div>
            )}

            {groups.map((group) => (
              <div key={group.label} className={groupSpacing}>
                <div className="panel-section-label">
                  {group.label}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {group.items.map((tab) => renderTabRow(tab))}
                </div>
              </div>
            ))}

            {filteredLocalResults.length > 0 && (
              <div className={groupSpacing}>
                <div className="panel-section-label">
                  {t("common:chatSidebar.historyResults", "History")}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {filteredLocalResults.map((history) =>
                    renderSearchRow({
                      key: `local-${history.id}`,
                      label:
                        history.title ||
                        t("common:untitled", { defaultValue: "Untitled" }),
                      metaLabel: t("common:chatSidebar.localLabel", "Local"),
                      onOpen: () => onOpenLocalHistory?.(history.id)
                    })
                  )}
                </div>
              </div>
            )}

            {filteredServerResults.length > 0 && (
              <div className={groupSpacing}>
                <div className="panel-section-label">
                  {t("common:chatSidebar.serverResults", "Server")}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {filteredServerResults.map((chat) =>
                    renderSearchRow({
                      key: `server-${chat.id}`,
                      label:
                        chat.title ||
                        t("common:untitled", { defaultValue: "Untitled" }),
                      metaLabel: t("common:chatSidebar.serverLabel", "Server"),
                      topic: chat.topic_label ?? null,
                      onOpen: () => onOpenServerChat?.(chat)
                    })
                  )}
                </div>
              </div>
            )}

            {ownerCurrent && !searchPending && !serverSearchUnavailable && serverSearch.isSuccess &&
              localSearchCurrent && localSearch.status === "success" &&
              pinnedTabs.length === 0 &&
              groups.length === 0 &&
              filteredLocalResults.length === 0 &&
              filteredServerResults.length === 0 && (
                <div className="px-2 py-3 text-xs text-text-subtle">
                  {t("common:chatSidebar.noResults", "No matches found")}
                </div>
              )}
          </>
        ) : (
          <>
            {pinnedTabs.length > 0 && (
              <div className={groupSpacing}>
                <div className="panel-section-label">
                  {t("common:pinned", "Pinned")}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {pinnedTabs.map((tab) => renderTabRow(tab))}
                </div>
              </div>
            )}

            {groups.map((group) => (
              <div key={group.label} className={groupSpacing}>
                <div className="panel-section-label">
                  {group.label}
                </div>
                <div className={classNames("flex flex-col", groupGap)}>
                  {group.items.map((tab) => renderTabRow(tab))}
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      <div className="border-t border-border px-4 py-3">
        <ModeToggle />
      </div>

      <FolderPickerModal
        open={folderPickerOpen}
        onClose={handleFolderPickerClose}
        conversationId={folderPickerTabId}
      />

      <React.Suspense fallback={null}>
        {bulkFolderPickerOpen && (
          <BulkFolderPickerModal
            open={bulkFolderPickerOpen}
            conversationIds={selectedConversationIds}
            onClose={handleBulkFolderPickerClose}
            onSuccess={clearSelection}
          />
        )}
        {bulkTagPickerOpen && (
          <BulkTagPickerModal
            open={bulkTagPickerOpen}
            conversationIds={selectedConversationIds}
            onClose={handleBulkTagPickerClose}
            onSuccess={clearSelection}
          />
        )}
      </React.Suspense>

      <Modal
        open={bulkDeleteConfirmOpen}
        onCancel={handleBulkDeleteConfirmClose}
        onOk={handleBulkDelete}
        okText={t("common:delete", "Delete")}
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true, loading: isBulkDeleting }}
        title={t(
          "sidepanel:multiSelect.deleteConfirmTitle",
          "Delete selected chats?"
        )}
        destroyOnHidden
      >
        <p>
          {t(
            "sidepanel:multiSelect.deleteConfirmBody",
            "This will delete the selected chats from the server."
          )}
        </p>
      </Modal>
      {isResizable && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label={t("common:chatSidebar.resize", "Resize sidebar")}
          aria-valuemin={SIDEBAR_MIN_WIDTH}
          aria-valuemax={SIDEBAR_MAX_WIDTH}
          aria-valuenow={Math.round(sidebarWidth)}
          tabIndex={0}
          onMouseDown={handleResizeStart}
          onKeyDown={handleResizeKeyDown}
          className={classNames(
            "absolute right-0 top-0 h-full w-1.5 cursor-col-resize transition-colors",
            isResizing ? "bg-border" : "hover:bg-border/60"
          )}
        />
      )}
    </aside>
  )
}
