import React, { useState, useMemo } from "react"
import { useNavigate, useLocation } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Input, Tooltip, Segmented } from "antd"
import {
  Plus,
  Search,
  Settings,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CheckSquare
} from "lucide-react"
import {
  SIDEBAR_ACTIVE_TAB_SETTING,
  SIDEBAR_SHORTCUTS_COLLAPSED_SETTING,
  SIDEBAR_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"
import { useSetting } from "@/hooks/useSetting"

import { useDebounce } from "@/hooks/useDebounce"
import {
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
  useServerChatHistory
} from "@/hooks/useServerChatHistory"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useStoreMessageOption } from "@/store/option"
import { useFolderStore } from "@/store/folder"
import { useRouteTransitionStore } from "@/store/route-transition"
import {
  shouldEnableOptionalResource,
  useChatSurfaceCoordinatorStore
} from "@/store/chat-surface-coordinator"
import { cn } from "@/libs/utils"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import { ServerChatList } from "./ChatSidebar/ServerChatList"
import { FolderChatList } from "./ChatSidebar/FolderChatList"
import {
  normalizeSidebarShortcutSelection,
  type SidebarShortcutAction
} from "./ChatSidebar/shortcut-actions"
import { isSidebarShortcutRouteActive } from "./ChatSidebar/shortcut-active"
import { QuickChatHelperButton } from "@/components/Common/QuickChatHelper"
import { NotesDockButton } from "@/components/Common/NotesDock"
import { ModeToggle } from "@/components/Sidepanel/Chat/ModeToggle"

interface ChatSidebarProps {
  /** Whether sidebar is collapsed */
  collapsed?: boolean
  /** Toggle collapsed state */
  onToggleCollapse?: () => void
  /** Additional class names */
  className?: string
  /** Monotonic signal from parent layouts when the sidebar should reset open state */
  openResetKey?: number
}

type SidebarTab = "server" | "folders"

export function ChatSidebar({
  collapsed = false,
  onToggleCollapse,
  className,
  openResetKey
}: ChatSidebarProps) {
  const { t } = useTranslation(["common", "sidepanel", "option", "settings"])
  const navigate = useNavigate()
  const location = useLocation()
  const [searchQuery, setSearchQuery] = useState("")
  const debouncedSearchQuery = useDebounce(searchQuery, 300)
  const [selectionMode, setSelectionMode] = useState(false)
  const [recentCollapsed, setRecentCollapsed] = useState(true)
  const hasSearchQuery = searchQuery.trim().length > 0
  const recentHistoryVisible = !recentCollapsed || hasSearchQuery

  // Tab state persisted in UI settings
  const [currentTab, setCurrentTab] = useSetting(SIDEBAR_ACTIVE_TAB_SETTING)
  const [shortcutsCollapsed, setShortcutsCollapsed] = useSetting(
    SIDEBAR_SHORTCUTS_COLLAPSED_SETTING
  )
  const showShortcuts = shortcutsCollapsed !== true
  const [shortcutSelection] = useSetting(SIDEBAR_SHORTCUT_SELECTION_SETTING)
  const setPanelVisible = useChatSurfaceCoordinatorStore(
    (state) => state.setPanelVisible
  )
  const markPanelEngaged = useChatSurfaceCoordinatorStore(
    (state) => state.markPanelEngaged
  )
  const serverHistoryOverviewEnabled = useChatSurfaceCoordinatorStore(
    (state) => shouldEnableOptionalResource(state, "server-history")
  )
  const serverHistoryPanelVisible =
    !collapsed && recentHistoryVisible && currentTab === "server"

  const clearChat = useClearChat()
  const temporaryChat = useStoreMessageOption((state) => state.temporaryChat)
  const startRouteTransition = useRouteTransitionStore((state) => state.start)

  // Folder conversation count for tab badge
  const conversationKeywordLinks = useFolderStore((s) => s.conversationKeywordLinks)
  const folderConversationCount = useMemo(
    () => new Set(conversationKeywordLinks.map((link) => link.conversation_id)).size,
    [conversationKeywordLinks]
  )

  // Server chat count for tab badge
  const { total: serverChatCount = 0 } = useServerChatHistory("", {
    enabled: serverHistoryPanelVisible && serverHistoryOverviewEnabled,
    mode: "overview",
    page: 1,
    limit: SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
    filterMode: "all"
  })

  const sidebarShortcuts = useMemo(
    () => normalizeSidebarShortcutSelection(shortcutSelection),
    [shortcutSelection]
  )

  const handleShortcutAction = (item: SidebarShortcutAction) => {
    if (item.kind === "event") {
      if (item.eventName === "tldw:open-quick-ingest") {
        handleIngest()
        return
      }
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent(item.eventName))
      }
      return
    }
    navigateWithLoading(item.path)
  }

  const isShortcutRouteActive = React.useCallback(
    (item: SidebarShortcutAction) =>
      item.kind === "route" &&
      isSidebarShortcutRouteActive(item.path, location.pathname),
    [location.pathname]
  )

  const renderShortcutIcon = (item: SidebarShortcutAction) => {
    const isActive = isShortcutRouteActive(item)
    return (
      <Tooltip title={t(item.labelKey, item.labelDefault)} placement="right">
        <button
          aria-label={t(item.labelKey, item.labelDefault)}
          aria-current={isActive ? "page" : undefined}
          onClick={() => handleShortcutAction(item)}
          data-testid={`chat-sidebar-shortcut-${item.id}`}
          className={cn(
            "p-2 rounded-lg",
            focusRingClasses,
            isActive
              ? "border border-border bg-surface text-text"
              : "text-text-muted hover:bg-surface hover:text-text"
          )}
        >
          <item.icon className="size-4" />
        </button>
      </Tooltip>
    )
  }

  const renderSidebarShortcut = (item: SidebarShortcutAction) => {
    const isActive = isShortcutRouteActive(item)
    return (
      <button
        key={item.id}
        onClick={() => handleShortcutAction(item)}
        aria-current={isActive ? "page" : undefined}
        data-testid={`chat-sidebar-shortcut-${item.id}`}
        className={cn(
          "flex items-center gap-2 w-full px-2 py-2 rounded text-sm",
          focusRingClasses,
          isActive
            ? "border border-border bg-surface text-text"
            : "text-text-muted hover:bg-surface hover:text-text"
        )}
      >
        <item.icon className="size-4" />
        <span>{t(item.labelKey, item.labelDefault)}</span>
      </button>
    )
  }

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextValue = e.target.value
    if (currentTab === "server" && nextValue.trim().length > 0) {
      markPanelEngaged("server-history")
    }
    setSearchQuery(nextValue)
  }

  const handleNewChat = () => {
    clearChat()
  }

  const handleIngest = () => {
    requestQuickIngestOpen()
  }

  const navigateWithLoading = React.useCallback(
    (path: string) => {
      if (path === location.pathname) {
        return
      }
      void setShortcutsCollapsed(true)
      startRouteTransition(path)
      navigate(path)
    },
    [location.pathname, navigate, setShortcutsCollapsed, startRouteTransition]
  )

  const resetToolsFirst = React.useCallback(() => {
    if (shortcutsCollapsed === true) {
      void setShortcutsCollapsed(false)
    }
    setRecentCollapsed(true)
    setSelectionMode(false)
  }, [setShortcutsCollapsed, shortcutsCollapsed])

  const previousCollapsedRef = React.useRef<boolean | null>(null)
  React.useEffect(() => {
    const wasCollapsed = previousCollapsedRef.current
    if (!collapsed && (wasCollapsed === null || wasCollapsed === true)) {
      resetToolsFirst()
    }
    previousCollapsedRef.current = collapsed
  }, [collapsed, resetToolsFirst])

  const previousOpenResetKeyRef = React.useRef(openResetKey)
  React.useEffect(() => {
    if (previousOpenResetKeyRef.current === openResetKey) {
      return
    }
    previousOpenResetKeyRef.current = openResetKey
    if (!collapsed) {
      resetToolsFirst()
    }
  }, [collapsed, openResetKey, resetToolsFirst])

  React.useEffect(() => {
    if (currentTab !== "server" && selectionMode) {
      setSelectionMode(false)
    }
  }, [currentTab, selectionMode])

  React.useEffect(() => {
    setPanelVisible("server-history", serverHistoryPanelVisible)

    return () => {
      setPanelVisible("server-history", false)
    }
  }, [serverHistoryPanelVisible, setPanelVisible])

  React.useEffect(() => {
    if (serverHistoryPanelVisible) {
      markPanelEngaged("server-history")
    }
  }, [markPanelEngaged, serverHistoryPanelVisible])

  const previousPathRef = React.useRef(location.pathname)
  React.useEffect(() => {
    if (previousPathRef.current !== location.pathname) {
      previousPathRef.current = location.pathname
      void setShortcutsCollapsed(true)
    }
  }, [location.pathname, setShortcutsCollapsed])

  // Build tab options with counts
  const tabOptions: Array<{ value: SidebarTab; label: string }> = [
    {
      value: "server",
      label: `${t("common:chatSidebar.tabs.server", "Server")}${serverChatCount > 0 ? ` (${serverChatCount})` : ""}`
    },
    {
      value: "folders",
      label: `${t("common:chatSidebar.tabs.folders", "Folders")}${folderConversationCount > 0 ? ` (${folderConversationCount})` : ""}`
    }
  ]
  const settingsShortcutActive = isSidebarShortcutRouteActive(
    "/settings",
    location.pathname
  )
  const focusRingClasses =
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  const recentConversationsExpanded = recentHistoryVisible
  const toggleRecentConversations = React.useCallback(() => {
    setRecentCollapsed((prev) => {
      const next = !prev
      if (next) {
        setSelectionMode(false)
      } else if (currentTab === "server") {
        markPanelEngaged("server-history")
      }
      return next
    })
  }, [currentTab, markPanelEngaged])

  // Collapsed view - just icons
  if (collapsed) {
    return (
      <div
        data-testid="chat-sidebar"
        className={cn(
          "flex flex-col h-screen items-center py-4 gap-2 w-12 overflow-hidden border-r border-border bg-surface2",
          className
        )}
      >
        <Tooltip
          title={t("common:chatSidebar.expand", "Expand sidebar")}
          placement="right"
        >
          <button
            aria-label={t("common:chatSidebar.expand", "Expand sidebar")}
            data-testid="chat-sidebar-toggle"
            onClick={onToggleCollapse}
            className={cn(
              "p-2 rounded-lg text-text-muted hover:bg-surface hover:text-text",
              focusRingClasses
            )}
          >
            <ChevronRight className="size-4" />
          </button>
        </Tooltip>

        <div className="h-px w-6 bg-border my-2" />

        <Tooltip
          title={t("common:chatSidebar.newChat", "New Chat")}
          placement="right"
        >
          <button
            aria-label={t("common:chatSidebar.newChat", "New Chat")}
            data-testid="chat-sidebar-new-chat"
            onClick={handleNewChat}
            className={cn(
              "p-2 rounded-lg text-text-muted hover:bg-surface hover:text-primary",
              focusRingClasses
            )}
          >
            <Plus className="size-4" />
          </button>
        </Tooltip>

        <div className="h-px w-6 bg-border my-2" />

        <div className="flex min-h-0 flex-1 w-full flex-col items-center gap-2 overflow-y-auto overflow-x-hidden px-1">
          {sidebarShortcuts.map((item) => (
            <React.Fragment key={item.id}>
              {renderShortcutIcon(item)}
            </React.Fragment>
          ))}
        </div>

        <div className="flex shrink-0 flex-col items-center gap-2">
          <NotesDockButton
            appearance="ghost"
            tooltipPlacement="right"
            ariaLabel={t("option:notesDock.tooltipSidebar", "Open Notes Dock")}
          />

          <QuickChatHelperButton
            variant="inline"
            showToggle={false}
            appearance="ghost"
            tooltipPlacement="right"
            ariaLabel={t(
              "option:quickChatHelper.tooltipSidebar",
              "Open Quick Chat Helper (sidebar)"
            )}
          />

          <Tooltip
            title={t("common:chatSidebar.settings", "Settings")}
            placement="right"
          >
            <button
              aria-label={t("common:chatSidebar.settings", "Settings")}
              aria-current={settingsShortcutActive ? "page" : undefined}
              data-testid="chat-sidebar-settings"
              onClick={() => navigate("/settings")}
              className={cn(
                "p-2 rounded-lg",
                focusRingClasses,
                settingsShortcutActive
                  ? "border border-border bg-surface text-text"
                  : "text-text-muted hover:bg-surface hover:text-text"
              )}
            >
              <Settings className="size-4" />
            </button>
          </Tooltip>
        </div>
      </div>
    )
  }

  // Expanded view
  return (
    <div
      data-testid="chat-sidebar"
      className={cn(
        "flex flex-col h-screen border-r border-border bg-surface2",
        className
      )}
      style={{ width: "var(--sidebar-width)" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <h2 className="font-semibold text-text">
          {t("common:chatSidebar.title", "Chats")}
        </h2>
        <div className="flex items-center gap-1">
          <Tooltip title={t("common:chatSidebar.newChat", "New Chat")}>
            <button
              aria-label={t("common:chatSidebar.newChat", "New Chat")}
              data-testid="chat-sidebar-new-chat"
              onClick={handleNewChat}
              className={cn(
                "p-2 rounded text-text-muted hover:bg-surface hover:text-primary",
                focusRingClasses
              )}
            >
              <Plus className="size-4" />
            </button>
          </Tooltip>
          {recentHistoryVisible && currentTab === "server" && (
            <Tooltip
              title={
                selectionMode
                  ? t("sidepanel:multiSelect.exit", "Exit selection")
                  : t("sidepanel:multiSelect.enter", "Select chats")
              }
            >
              <button
                type="button"
                onClick={() => setSelectionMode((prev) => !prev)}
                className={cn(
                  "rounded p-2",
                  focusRingClasses,
                  selectionMode
                    ? "bg-surface text-text"
                    : "text-text-muted hover:bg-surface hover:text-text"
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
          )}
          <Tooltip
            title={t("common:chatSidebar.collapse", "Collapse sidebar")}
          >
            <button
              aria-label={t("common:chatSidebar.collapse", "Collapse sidebar")}
              data-testid="chat-sidebar-toggle"
              onClick={onToggleCollapse}
              className={cn(
                "p-2 rounded text-text-muted hover:bg-surface hover:text-text",
                focusRingClasses
              )}
            >
              <ChevronLeft className="size-4" />
            </button>
          </Tooltip>
        </div>
      </div>

      {/* Quick Actions */}
      <button
        type="button"
        aria-expanded={showShortcuts}
        aria-controls="chat-sidebar-shortcuts"
        onClick={() => {
          void setShortcutsCollapsed(showShortcuts)
        }}
        className={cn(
          "group flex w-full items-center justify-between px-3 py-2 text-left hover:bg-surface",
          focusRingClasses
        )}
        title={t("common:chatSidebar.shortcuts", "Shortcuts")}
      >
        <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          {t("common:chatSidebar.shortcuts", "Shortcuts")}
        </span>
        <ChevronDown
          className={cn(
            "size-4 text-text-muted transition-transform group-hover:text-text",
            showShortcuts ? "rotate-0" : "-rotate-90"
          )}
        />
      </button>
      {showShortcuts && (
        <div id="chat-sidebar-shortcuts" className="px-3 pb-2 space-y-1">
          {sidebarShortcuts.length > 0 ? (
            sidebarShortcuts.map((item) => renderSidebarShortcut(item))
          ) : (
            <div className="px-2 py-2 text-xs text-text-subtle">
              {t(
                "settings:uiCustomization.shortcuts.empty",
                "No shortcuts selected"
              )}
            </div>
          )}
        </div>
      )}

      <div className="h-px bg-border mx-3" />

      {/* Recent Conversations */}
      <button
        type="button"
        aria-expanded={recentConversationsExpanded}
        aria-controls="chat-sidebar-recent-conversations"
        onClick={toggleRecentConversations}
        className={cn(
          "group flex w-full items-center justify-between px-3 py-2 text-left hover:bg-surface",
          focusRingClasses
        )}
        title={t(
          "common:chatSidebar.recentConversations",
          "Recent conversations"
        )}
      >
        <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          {t("common:chatSidebar.recentConversations", "Recent conversations")}
        </span>
        <ChevronDown
          className={cn(
            "size-4 text-text-muted transition-transform group-hover:text-text",
            recentConversationsExpanded ? "rotate-0" : "-rotate-90"
          )}
        />
      </button>

      {recentHistoryVisible && (
        <div
          id="chat-sidebar-recent-conversations"
          className="flex min-h-0 flex-1 flex-col"
        >
          {/* Search */}
          <div className="px-3 py-2 border-b border-border">
            <Input
              data-testid="chat-sidebar-search"
              prefix={<Search className="size-3.5 text-text-subtle" />}
              placeholder={t("common:chatSidebar.search", "Search chats...")}
              value={searchQuery}
              onChange={handleSearchChange}
              size="small"
              className="bg-surface"
              allowClear
            />
          </div>

          {/* Tabs */}
          <div className="px-3 py-2 border-b border-border">
            <label id="chat-sidebar-tab-label" className="sr-only">
              {t("common:chatSidebar.tabsLabel", "Chat view")}
            </label>
            <Segmented<SidebarTab>
              value={currentTab}
              onChange={(value) => {
                void setCurrentTab(value)
              }}
              options={tabOptions}
              block
              size="small"
              className="w-full"
              aria-labelledby="chat-sidebar-tab-label"
            />
          </div>

          {/* Tab Content */}
          <div
            className={cn(
              "flex-1 overflow-y-auto",
              temporaryChat ? "pointer-events-none opacity-50" : ""
            )}
          >
            {currentTab === "server" && (
              <ServerChatList
                searchQuery={debouncedSearchQuery}
                selectionMode={selectionMode}
              />
            )}

            {currentTab === "folders" && (
              <FolderChatList />
            )}
          </div>
        </div>
      )}

      {!recentHistoryVisible && (
        <div
          id="chat-sidebar-recent-conversations"
          className="flex-1"
          hidden
        />
      )}

      {/* Footer */}
      <div className="border-t border-border px-3 py-2">
        <button
          onClick={() => navigate("/settings")}
          aria-current={settingsShortcutActive ? "page" : undefined}
          data-testid="chat-sidebar-settings"
          className={cn(
            "flex items-center gap-2 w-full px-2 py-2 rounded text-sm",
            focusRingClasses,
            settingsShortcutActive
              ? "border border-border bg-surface text-text"
              : "text-text-muted hover:bg-surface hover:text-text"
          )}
        >
          <Settings className="size-4" />
          <span>{t("common:chatSidebar.settings", "Settings")}</span>
        </button>
        <div className="mt-2 border-t border-border pt-2">
          <div className="flex items-center gap-2">
            <div className="flex-1">
              <ModeToggle />
            </div>
            <div className="flex flex-col gap-1">
              <NotesDockButton
                appearance="ghost"
                className="shrink-0"
                tooltipPlacement="top"
                ariaLabel={t("option:notesDock.tooltipSidebar", "Open Notes Dock")}
              />
              <QuickChatHelperButton
                variant="inline"
                showToggle={false}
                appearance="ghost"
                className="shrink-0"
                ariaLabel={t(
                  "option:quickChatHelper.tooltipSidebar",
                  "Open Quick Chat Helper (sidebar)"
                )}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatSidebar
