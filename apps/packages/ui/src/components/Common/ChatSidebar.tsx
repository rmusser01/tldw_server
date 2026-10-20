import React, { useState, useMemo } from "react"
import { useNavigate, useLocation } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Input, Tooltip, Segmented, type InputRef } from "antd"
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
  SIDEBAR_RECENT_COLLAPSED_SETTING,
  SIDEBAR_SHORTCUTS_COLLAPSED_SETTING,
  SIDEBAR_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"
import { useSetting } from "@/hooks/useSetting"

import { useDebounce } from "@/hooks/useDebounce"
import {
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
  useServerChatHistory
} from "@/hooks/useServerChatHistory"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import type { VisibleServerChats } from "@/db/dexie/chat"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useShallow } from "zustand/react/shallow"
import { useFolderStore } from "@/store/folder"
import { useStoreMessageOption } from "@/store/option"
import { useRouteTransitionStore } from "@/store/route-transition"
import {
  shouldEnableOptionalResource,
  useChatSurfaceCoordinatorStore
} from "@/store/chat-surface-coordinator"
import { cn } from "@/libs/utils"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import { ServerChatList } from "./ChatSidebar/ServerChatList"
import { LocalChatList } from "./ChatSidebar/LocalChatList"
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
  /** Called after an explicit conversation target is accepted. */
  onConversationSelected?: () => void
}

type SidebarTab = "all" | "server" | "local" | "folders"

export function ChatSidebar({
  collapsed = false,
  onToggleCollapse,
  className,
  openResetKey,
  onConversationSelected
}: ChatSidebarProps) {
  const { t } = useTranslation(["common", "sidepanel", "option", "settings"])
  const navigate = useNavigate()
  const location = useLocation()
  const [searchQuery, setSearchQuery] = useState("")
  const hasSearchQuery = searchQuery.trim().length > 0
  const debouncedSearchQuery = useDebounce(searchQuery, 300)
  const [selectionMode, setSelectionMode] = useState(false)
  const [recentCollapsed, setRecentCollapsed, recentPreference] = useSetting(
    SIDEBAR_RECENT_COLLAPSED_SETTING
  )
  // Read the saved source before starting any source-specific list.
  const [savedTab, setCurrentTab, tabPreference] = useSetting(SIDEBAR_ACTIVE_TAB_SETTING)
  const tabReady = tabPreference?.isLoading !== true
  const currentTab: SidebarTab = savedTab === "folders" && hasSearchQuery ? "all" : savedTab
  const [visibleServerChats, setVisibleServerChats] = useState<VisibleServerChats | null>(null)
  const readOwner = useChatDraftOwner(() => setVisibleServerChats(null))
  const ownerReady = readOwner.verificationState === "ready" && !!readOwner.ownerKey && !!readOwner.requestScope && readOwner.isCurrent()
  const historyOwner = React.useMemo(() => ownerReady ? {
    key: readOwner.ownerKey!, requestScope: readOwner.requestScope!, isCurrent: readOwner.isCurrent
  } : undefined, [ownerReady, readOwner.ownerKey, readOwner.requestScope, readOwner.isCurrent])

  const searchInput = React.useRef<InputRef>(null)
  const focusSearchOnOpen = React.useRef<{
    location: typeof location
    initiator: HTMLButtonElement
  } | null>(null)
  React.useLayoutEffect(() => {
    const requested = focusSearchOnOpen.current
    if (!requested) return
    if (collapsed || requested.location !== location) {
      focusSearchOnOpen.current = null
      return
    }
    if (!tabReady) return
    focusSearchOnOpen.current = null
    const active = document.activeElement
    // A newer connected control owns focus; consume the old request quietly.
    if (active !== document.body && active?.isConnected &&
        active !== requested.initiator && active !== searchInput.current?.input) return
    searchInput.current?.focus()
  }, [collapsed, location, tabReady])
  const recentHistoryVisible = tabReady && (hasSearchQuery ||
    (recentPreference?.isLoading !== true && !recentCollapsed))

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
    !collapsed && recentHistoryVisible && (currentTab === "server" || currentTab === "all")

  const clearChat = useClearChat()
  const temporaryChat = useStoreMessageOption((state) => state.temporaryChat)
  const startRouteTransition = useRouteTransitionStore((state) => state.start)

  // Count only the verified view's successful server folder snapshot.
  const folderData = useFolderStore(useShallow(state => ({
    folders: state.folders, keywords: state.keywords, folderKeywordLinks: state.folderKeywordLinks,
    conversationKeywordLinks: state.conversationKeywordLinks, serverDataScope: state.serverDataScope,
    isLoading: state.isLoading, error: state.error, folderApiAvailable: state.folderApiAvailable,
    getConversationsForFolder: state.getConversationsForFolder
  })))
  const folderConversationCount = useMemo(() => {
    if (!ownerReady || folderData.serverDataScope?.requestScope !== historyOwner?.requestScope ||
      !folderData.serverDataScope?.isCurrent() || folderData.isLoading || folderData.error ||
      folderData.folderApiAvailable !== true) return 0
    return new Set(folderData.folders.filter(folder => !folder.deleted)
      .flatMap(folder => folderData.getConversationsForFolder(folder.id))).size
  }, [folderData, ownerReady, historyOwner])

  // Server chat count for tab badge
  const { total: serverChatCount = 0 } = useServerChatHistory("", {
    enabled: ownerReady && serverHistoryPanelVisible && serverHistoryOverviewEnabled,
    owner: historyOwner,
    scope: {type:"global"},
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
    if (currentTab !== "local" && nextValue.trim().length > 0) {
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
      startRouteTransition(path)
      navigate(path)
    },
    [location.pathname, navigate, startRouteTransition]
  )

  // Reopening resets transient management, not explicit expansion preferences.
  React.useEffect(() => {
    setSelectionMode(false)
  }, [collapsed, openResetKey])

  React.useEffect(() => {
    if ((currentTab !== "server" && currentTab !== "all" || temporaryChat) && selectionMode) {
      setSelectionMode(false)
    }
  }, [currentTab, selectionMode, temporaryChat])

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

  // Build tab options with counts
  const tabOptions: Array<{ value: SidebarTab; label: string }> = [
    { value: "all", label: t("common:chatSidebar.tabs.all", "All") },
    {
      value: 'server',
      label:
        ownerReady && serverChatCount > 0
          ? t('common:chatSidebar.tabs.serverTotal', 'Server ({{count}} total)', {
              count: serverChatCount,
            })
          : t('common:chatSidebar.tabs.server', 'Server'),
    },
    {
      value: "local",
      label: t("common:chatSidebar.tabs.local", "On this device")
    },
    {
      value: 'folders',
      label:
        folderConversationCount > 0
          ? t(
              'common:chatSidebar.tabs.foldersChats',
              'Folders ({count, plural, one {# chat} other {# chats}})',
              { count: folderConversationCount }
            )
          : t('common:chatSidebar.tabs.folders', 'Folders'),
    },
  ];
  const settingsShortcutActive = isSidebarShortcutRouteActive('/settings', location.pathname);
  const focusRingClasses =
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  const recentConversationsExpanded = recentHistoryVisible
  const toggleRecentConversations = React.useCallback(() => {
    if (recentConversationsExpanded) {
      setSearchQuery("")
      setSelectionMode(false)
      void setRecentCollapsed(true)
    } else {
      void setRecentCollapsed(false)
      if (currentTab === "server") markPanelEngaged("server-history")
    }
  }, [currentTab, markPanelEngaged, recentConversationsExpanded, setRecentCollapsed])

  // Collapsed view - just icons
  if (collapsed) {
    return (
      <div
        data-testid="chat-sidebar"
      data-chat-sidebar
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
            aria-expanded={false}
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

        <Tooltip title={t("common:chatSidebar.findConversations", "Find conversations")} placement="right">
          <button
            type="button"
            aria-label={t("common:chatSidebar.findConversations", "Find conversations")}
            onClick={(event) => {
              focusSearchOnOpen.current = { location, initiator: event.currentTarget }
              event.currentTarget.focus()
              onToggleCollapse?.()
            }}
            className={cn("p-2 rounded-lg text-text-muted hover:bg-surface hover:text-text", focusRingClasses)}
          >
            <Search aria-hidden="true" className="size-4" />
          </button>
        </Tooltip>

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
        data-chat-sidebar
      className={cn(
        "flex h-screen min-h-0 flex-col overflow-hidden border-r border-border bg-surface2",
        className
      )}
      style={{ width: "var(--sidebar-width)" }}
    >
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between px-3 py-3 border-b border-border">
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
          {recentHistoryVisible && (currentTab === "server" || currentTab === "all") && !temporaryChat && (
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
              aria-expanded={true}
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

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* Search stays reachable even when recent history is collapsed. */}
        <div className="px-3 py-2 border-b border-border">
          <label htmlFor="chat-sidebar-search-input" className="mb-1 block text-xs font-semibold text-text-muted">
            {t("common:chatSidebar.findConversations", "Find conversations")}
          </label>
          <Input
            ref={searchInput}
            disabled={!tabReady}
            id="chat-sidebar-search-input"
            data-testid="chat-sidebar-search"
            prefix={<Search aria-hidden="true" className="size-3.5 text-text-subtle" />}
            placeholder={currentTab === "local" ? t("common:chatSidebar.local.search", "Search on-device chats…") : t("common:chatSidebar.search", "Search chats...")}
            aria-label={t("common:chatSidebar.findConversations", "Find conversations")}
            aria-describedby="chat-sidebar-search-scope"
            value={searchQuery}
            onChange={handleSearchChange}
            size="small"
            className="bg-surface"
            allowClear
          />
          <p id="chat-sidebar-search-scope" className="mt-1 text-xs text-text-muted">
            {!tabReady
              ? t("common:loading.title", "Loading…")
              : currentTab === "folders"
              ? t("common:chatSidebar.searchFromFolders", "Typing searches Server and On this device history.")
              : currentTab === "all"
              ? t("common:chatSidebar.discovery.allScope", "Search Server and On this device history.")
              : t("common:chatSidebar.searchScope", "Search {{source}} history.", {
                  source: currentTab === "local"
                    ? t("common:chatSidebar.tabs.local", "On this device")
                    : t("common:chatSidebar.tabs.server", "Server")
                })}
          </p>
        </div>

        {/* Recent Conversations */}
        <button
          type="button"
          aria-expanded={recentConversationsExpanded}
          disabled={!tabReady || recentPreference?.isLoading === true}
          aria-busy={!tabReady || recentPreference?.isLoading === true}
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
          <div id="chat-sidebar-recent-conversations">
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
                orientation="vertical"
                className="w-full [&_.ant-segmented-item-label]:px-2"
                styles={{
                  label: { whiteSpace: 'normal', overflowWrap: 'anywhere', textAlign: 'left' },
                }}
                aria-labelledby="chat-sidebar-tab-label"
              />
            </div>

            {/* Tab Content. Temporary mode opens saved chats read-only (H1): rows stay
                selectable without management actions, and folder management stays off. */}
            <div>
              {(currentTab === "server" || currentTab === "all") && (
                <section aria-label={t("common:chatSidebar.tabs.server", "Server")}>
                {currentTab === "all" && <h3 className="px-3 pt-3 text-sm font-semibold">{t("common:chatSidebar.discovery.serverGroup", "Server results")}</h3>}
                <ServerChatList
                  searchQuery={debouncedSearchQuery}
                  selectionMode={selectionMode}
                  readOnly={temporaryChat}
                  onConversationSelected={onConversationSelected}
                  onVisibleChatsChange={setVisibleServerChats}
                />
                </section>
              )}

              {(currentTab === "local" || currentTab === "all") && (
                <section aria-label={t("common:chatSidebar.tabs.local", "On this device")}>
                {currentTab === "all" && <h3 className="px-3 pt-3 text-sm font-semibold">{t("common:chatSidebar.discovery.deviceGroup", "On this device results")}</h3>}
                <LocalChatList searchQuery={debouncedSearchQuery} readOnly={temporaryChat}
                  includeCachedServerChats
                  visibleServerChats={currentTab === "all" ? visibleServerChats : null}
                  onSelectChat={() => onConversationSelected?.()} />
                </section>
              )}

              {currentTab === "folders" && (
                <FolderChatList owner={historyOwner} verificationState={readOwner.verificationState}
                  onRetryVerification={readOwner.retryVerification} readOnly={temporaryChat}
                  onConversationSelected={onConversationSelected} />
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
        <div className="h-px bg-border mx-3" />

        {/* Platform shortcuts */}
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
          title={t("common:chatSidebar.navigation", "Navigation")}
        >
          <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
            {t("common:chatSidebar.navigation", "Navigation")}
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

      </div>

      {/* Footer */}
      <div className="shrink-0 border-t border-border px-3 py-2">
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
