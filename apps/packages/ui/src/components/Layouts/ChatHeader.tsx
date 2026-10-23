import React from "react"
import type { TFunction } from "i18next"
import { Tooltip, Input, Dropdown, type MenuProps } from "antd"
import {
  Bell,
  BellOff,
  CogIcon,
  House,
  LoaderCircle,
  Menu,
  MoreHorizontal,
  Keyboard,
  MessageSquareDashed,
  Moon,
  Search,
  Share2,
  Signpost,
  SquarePen,
  Sun,
  TriangleAlert,
  UserCircle2
} from "lucide-react"
import { HeaderShortcuts } from "./HeaderShortcuts"
import { focusFirstVisibleElement } from "@/utils/focus-return"
import logoImage from "~/assets/icon.png"
import type { NotificationLifecycleState } from "@/services/notification-lifecycle"
import {
  getChatPersistenceCopy,
  type ChatPersistenceKind
} from "@/utils/chat-persistence-status"

import {
  getPersistenceModeLabel,
  getPersistenceModePillLabel
} from "@/hooks/playground/usePersistenceMode"

type HeaderNotificationState = NotificationLifecycleState

type ChatHeaderProps = {
  t: TFunction
  temporaryChat: boolean
  historyId?: string | null
  chatTitle: string
  isEditingTitle: boolean
  onTitleChange: (value: string) => void
  onTitleEditStart: () => void
  onTitleCommit: (value: string) => void | Promise<void>
  onToggleSidebar?: () => void
  sidebarCollapsed?: boolean
  onOpenCompanionHome: () => void
  onOpenCommandPalette: () => void
  onOpenShortcutsModal: () => void
  onOpenSettings: () => void
  onOpenShareModal?: () => void
  shareStatusLabel?: string | null
  shareButtonDisabled?: boolean
  onToggleTheme?: () => void
  themeMode?: "system" | "dark" | "light"
  onStartSavedChat?: () => void
  onStartTemporaryChat?: () => void
  onStartCharacterChat?: () => void
  activeCharacterName?: string | null
  showChatTitle?: boolean
  showSessionModeBadge?: boolean
  /**
   * Where the chat is saved, derived from acknowledged state. Defaults to
   * "local" so the badge never claims the server without evidence.
   */
  persistenceKind?: ChatPersistenceKind
  shortcutsExpanded: boolean
  onToggleShortcuts: (next?: boolean) => void
  commandKeyLabel: string
  /** Unread notification count (0 or undefined hides badge) */
  notificationCount?: number
  /** Callback when notification bell is clicked */
  onOpenNotifications?: () => void
  notificationState?: HeaderNotificationState
  onRetryNotifications?: () => void | Promise<void>
}

const toText = (value: unknown): string =>
  typeof value === "string" ? value : String(value)

export function ChatHeader({
  t,
  temporaryChat,
  historyId,
  chatTitle,
  isEditingTitle,
  onTitleChange,
  onTitleEditStart,
  onTitleCommit,
  onToggleSidebar,
  sidebarCollapsed = false,
  onOpenCompanionHome,
  onOpenCommandPalette,
  onOpenShortcutsModal,
  onOpenSettings,
  onOpenShareModal,
  shareStatusLabel = null,
  shareButtonDisabled = false,
  onToggleTheme,
  themeMode = "dark",
  onStartSavedChat,
  onStartTemporaryChat,
  onStartCharacterChat,
  activeCharacterName,
  showChatTitle = true,
  showSessionModeBadge = true,
  persistenceKind,
  shortcutsExpanded,
  onToggleShortcuts,
  commandKeyLabel,
  notificationCount,
  onOpenNotifications,
  notificationState = "active",
  onRetryNotifications
}: ChatHeaderProps) {
  const [headerMenuOpen, setHeaderMenuOpen] = React.useState(false)
  const headerMenuTriggerRef = React.useRef<HTMLButtonElement>(null)
  const [notificationPopoverOpen, setNotificationPopoverOpen] =
    React.useState(false)
  const notificationTriggerRef = React.useRef<HTMLButtonElement>(null)
  const notificationPopoverRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (!headerMenuOpen) return
    // RC Dropdown's delayed autofocus can refocus its closed menu; cancel ours on close.
    const frame = requestAnimationFrame(() => {
      const menu = document.getElementById("chat-header-actions-menu")
      if (menu)
        focusFirstVisibleElement(
          '[role="menuitem"]:not([aria-disabled="true"])',
          menu
        )
    })
    return () => cancelAnimationFrame(frame)
  }, [headerMenuOpen])
  const logoSrc =
    typeof logoImage === "string"
      ? logoImage
      : ((logoImage as { src?: string })?.src ?? "")
  const showSidebarToggle = Boolean(onToggleSidebar)
  const sidebarLabel = sidebarCollapsed
    ? toText(t("common:chatSidebar.expand", "Expand sidebar"))
    : toText(t("common:chatSidebar.collapse", "Collapse sidebar"))
  const shortcutsToggleLabel = shortcutsExpanded
    ? toText(t("option:header.hidePages", "Hide pages"))
    : toText(t("option:header.showPages", "Show pages"))
  const companionHomeLabel = toText(
    t("option:header.companionHome", "Companion Home")
  )
  const resolvedPersistenceKind: ChatPersistenceKind = temporaryChat
    ? "temporary"
    : persistenceKind && persistenceKind !== "temporary"
      ? persistenceKind
      : "local"
  const persistenceCopy = getChatPersistenceCopy(t, resolvedPersistenceKind)
  const showHistoryMode =
    temporaryChat || (resolvedPersistenceKind === "local" && !historyId)
  const persistenceBadgeLabel = showHistoryMode
    ? getPersistenceModePillLabel(t, temporaryChat)
    : persistenceCopy.pill
  const persistenceBadgeHelp = showHistoryMode
    ? getPersistenceModeLabel(t, temporaryChat)
    : persistenceCopy.description
  const persistenceBadgeClass =
    resolvedPersistenceKind === "server"
      ? "border-success/40 bg-success/10 text-text"
      : resolvedPersistenceKind === "serverUnconfirmed"
        ? "border-warn/50 bg-warn/10 text-text"
        : resolvedPersistenceKind === "temporary" ||
            resolvedPersistenceKind === "serverFailed"
          ? "border-warn/50 bg-warn/10 text-text"
          : "border-border bg-surface2 text-text-muted"
  const canEditTitle =
    showChatTitle && !temporaryChat && historyId && historyId !== "temp"
  const isDarkTheme = themeMode !== "light"
  const themeToggleLabel = isDarkTheme
    ? toText(t("common:theme.switchToLight", "Switch to light theme"))
    : toText(t("common:theme.switchToDark", "Switch to dark theme"))
  const commandPaletteLabel = toText(
    t("common:shortcuts.openCommandPalette", "Open command palette")
  )
  const commandPaletteTitle = toText(
    t("common:commandPalette.searchCommands", "Search commands")
  )
  const commandPaletteAccessibleLabel = `${commandPaletteTitle} - ${commandPaletteLabel}`
  const showSavedChatAction = Boolean(onStartSavedChat)
  const showTemporaryChatAction = Boolean(onStartTemporaryChat)
  const showCharacterChatAction = Boolean(onStartCharacterChat)
  const shareButtonLabel = shareStatusLabel
    ? toText(
        t(
          "playground:header.shareStatusAria",
          "Share conversation ({{status}})",
          {
            status: shareStatusLabel
          } as any
        )
      )
    : toText(t("playground:header.shareConversation", "Share conversation"))
  const focusRingClasses =
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  const notificationCopy =
    notificationState === "idle"
      ? null
      : {
          active: {
            name: toText(
              t("option:header.notificationsActiveName", {
                count: notificationCount ?? 0,
                defaultValue: `Notifications, ${notificationCount ?? 0} unread`
              })
            ),
            title: toText(t("option:header.notifications", "Notifications")),
            description:
              (notificationCount ?? 0) > 0
                ? toText(
                    t("option:header.notificationsUnreadDescription", {
                      count: notificationCount,
                      defaultValue: `${notificationCount} unread notification${notificationCount === 1 ? "" : "s"}.`
                    })
                  )
                : toText(
                    t(
                      "option:header.notificationsCaughtUp",
                      "You are caught up."
                    )
                  ),
            announcement: toText(
              t(
                "option:header.notificationsActiveAnnouncement",
                "Notifications are active"
              )
            )
          },
          connecting: {
            name: toText(
              t(
                "option:header.notificationsConnectingName",
                "Notifications, connecting"
              )
            ),
            title: toText(
              t(
                "option:header.notificationsConnectingTitle",
                "Notifications are connecting"
              )
            ),
            description: toText(
              t(
                "option:header.notificationsConnectingDescription",
                "Checking for new notifications."
              )
            ),
            announcement: toText(
              t(
                "option:header.notificationsConnectingAnnouncement",
                "Notifications are connecting"
              )
            )
          },
          degraded: {
            name: toText(
              t(
                "option:header.notificationsReconnectingName",
                "Notifications, reconnecting"
              )
            ),
            title: toText(
              t(
                "option:header.notificationsReconnectingTitle",
                "Notifications are reconnecting"
              )
            ),
            description: toText(
              t(
                "option:header.notificationsReconnectingDescription",
                "Recent notifications may be delayed while the connection recovers."
              )
            ),
            announcement: toText(
              t(
                "option:header.notificationsReconnectingAnnouncement",
                "Notifications are reconnecting"
              )
            )
          },
          "auth-required": {
            name: toText(
              t(
                "option:header.notificationsSignInName",
                "Notifications, sign-in required"
              )
            ),
            title: toText(
              t("option:header.notificationsSignInTitle", "Sign in required")
            ),
            description: toText(
              t(
                "option:header.notificationsSignInDescription",
                "Sign in again to resume personal notifications."
              )
            ),
            announcement: toText(
              t(
                "option:header.notificationsSignInAnnouncement",
                "Notifications require sign in"
              )
            )
          },
          unavailable: {
            name: toText(
              t(
                "option:header.notificationsUnavailableName",
                "Notifications unavailable for this account"
              )
            ),
            title: toText(
              t(
                "option:header.notificationsUnavailableTitle",
                "Notifications unavailable for this account"
              )
            ),
            description: toText(
              t(
                "option:header.notificationsUnavailableDescription",
                "Notifications are not available for this account."
              )
            ),
            announcement: toText(
              t(
                "option:header.notificationsUnavailableAnnouncement",
                "Notifications are unavailable"
              )
            )
          }
        }[notificationState]

  React.useEffect(() => {
    if (!notificationPopoverOpen) return
    notificationPopoverRef.current
      ?.querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return
      setNotificationPopoverOpen(false)
      notificationTriggerRef.current?.focus()
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null
      if (!target) return
      if (notificationPopoverRef.current?.contains(target)) return
      if (notificationTriggerRef.current?.contains(target)) return
      setNotificationPopoverOpen(false)
    }
    document.addEventListener("keydown", onKeyDown)
    document.addEventListener("pointerdown", onPointerDown)
    return () => {
      document.removeEventListener("keydown", onKeyDown)
      document.removeEventListener("pointerdown", onPointerDown)
    }
  }, [notificationPopoverOpen])

  React.useEffect(() => {
    if (notificationState === "idle") setNotificationPopoverOpen(false)
  }, [notificationState])

  const openNotifications = () => {
    setNotificationPopoverOpen(false)
    onOpenNotifications?.()
  }

  const NotificationIcon =
    notificationState === "active"
      ? Bell
      : notificationState === "connecting"
        ? LoaderCircle
        : notificationState === "unavailable"
          ? BellOff
          : TriangleAlert

  const notificationControl = onOpenNotifications && notificationCopy && (
    <div className="relative">
      <Tooltip title={notificationCopy.title}>
        <button
          ref={notificationTriggerRef}
          type="button"
          onClick={() => setNotificationPopoverOpen((open) => !open)}
          aria-label={notificationCopy.name}
          aria-haspopup="dialog"
          aria-expanded={notificationPopoverOpen}
          aria-controls="chat-header-notification-status"
          className={`relative inline-flex items-center justify-center min-h-11 min-w-11 rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
          data-testid="chat-header-notifications-bell"
        >
          <NotificationIcon
            className={`size-4 ${notificationState === "connecting" ? "animate-spin" : ""}`}
            aria-hidden="true"
          />
          {notificationState === "active" && (notificationCount ?? 0) > 0 && (
            <span
              aria-hidden="true"
              className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-bold text-white"
            >
              {notificationCount! > 99 ? "99+" : notificationCount}
            </span>
          )}
        </button>
      </Tooltip>
      {notificationPopoverOpen ? (
        <div
          ref={notificationPopoverRef}
          id="chat-header-notification-status"
          role="dialog"
          aria-label={notificationCopy.title}
          className="absolute right-0 top-full z-50 mt-2 w-72 rounded-lg border border-border bg-surface p-3 text-left shadow-lg"
        >
          <p className="text-sm font-semibold text-text">
            {notificationCopy.title}
          </p>
          <p className="mt-1 text-xs leading-5 text-text-muted">
            {notificationCopy.description}
          </p>
          <div className="mt-3 flex items-center gap-2">
            {(notificationState === "degraded" ||
              notificationState === "unavailable") &&
            onRetryNotifications ? (
              <button
                type="button"
                onClick={() => void onRetryNotifications()}
                className={`rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primaryStrong ${focusRingClasses}`}
              >
                {toText(t("option:header.notificationsTryAgain", "Try again"))}
              </button>
            ) : null}
            <button
              type="button"
              onClick={openNotifications}
              className={`rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 ${focusRingClasses}`}
            >
              {toText(
                t("option:header.notificationsOpen", "Open notifications")
              )}
            </button>
          </div>
        </div>
      ) : null}
      <span role="status" aria-live="polite" className="sr-only">
        {notificationCopy.announcement}
      </span>
    </div>
  )
  const runMenuAction = (action: (() => void) | undefined) => {
    setHeaderMenuOpen(false)
    headerMenuTriggerRef.current?.focus()
    action?.()
  }
  const menuItems: MenuProps["items"] = [
    ...(onOpenShareModal
      ? [
          {
            key: "share",
            label: shareButtonLabel,
            icon: <Share2 className="size-4" aria-hidden="true" />,
            disabled: shareButtonDisabled,
            className: "sm:!hidden",
            onClick: () => runMenuAction(onOpenShareModal)
          }
        ]
      : []),
    ...(onStartTemporaryChat
      ? [
          {
            key: "temporary",
            label: toText(
              t(
                "playground:header.newTemporaryChat",
                "Temporary chat (on this device)"
              )
            ),
            icon: <MessageSquareDashed className="size-4" aria-hidden="true" />,
            onClick: () => runMenuAction(onStartTemporaryChat)
          }
        ]
      : []),
    ...(onStartCharacterChat
      ? [
          {
            key: "character",
            label: toText(
              t("playground:header.newCharacterChat", "Character chat")
            ),
            icon: <UserCircle2 className="size-4" aria-hidden="true" />,
            onClick: () => runMenuAction(onStartCharacterChat)
          }
        ]
      : []),
    { type: "divider" },
    {
      key: "home",
      label: companionHomeLabel,
      icon: <House className="size-4" aria-hidden="true" />,
      onClick: () => runMenuAction(onOpenCompanionHome)
    },
    {
      key: "pages",
      label: shortcutsExpanded
        ? toText(t("option:header.closePageLauncher", "Close page launcher"))
        : toText(t("option:header.browsePages", "Browse pages")),
      icon: <Signpost className="size-4" aria-hidden="true" />,
      onClick: () => runMenuAction(() => onToggleShortcuts(!shortcutsExpanded))
    },
    { type: "divider" },
    {
      key: "settings",
      label: toText(t("sidepanel:header.openSettingsAria", "Open settings")),
      icon: <CogIcon className="size-4" aria-hidden="true" />,
      onClick: () => runMenuAction(onOpenSettings)
    },
    ...(onToggleTheme
      ? [
          {
            key: "theme",
            label: themeToggleLabel,
            icon: isDarkTheme ? (
              <Sun className="size-4" aria-hidden="true" />
            ) : (
              <Moon className="size-4" aria-hidden="true" />
            ),
            onClick: () => runMenuAction(onToggleTheme)
          }
        ]
      : []),
    {
      key: "shortcuts",
      label: toText(
        t("option:header.keyboardShortcutsAria", "Show keyboard shortcuts")
      ),
      icon: <Keyboard className="size-4" aria-hidden="true" />,
      onClick: () => runMenuAction(onOpenShortcutsModal)
    }
  ]
  const headerActionClass = `inline-flex min-h-11 min-w-11 shrink-0 items-center justify-center gap-2 rounded-md px-2 text-sm text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`

  const sessionBadges = showSessionModeBadge ? (
    <div className="flex min-w-0 max-w-full flex-wrap items-center gap-1">
      <span
        className={`inline-flex shrink-0 items-center rounded border px-1.5 py-0.5 text-[11px] ${persistenceBadgeClass}`}
        title={persistenceBadgeHelp}
        data-testid="chat-header-persistence-badge"
        data-persistence-kind={resolvedPersistenceKind}
      >
        {persistenceBadgeLabel}
      </span>
      {activeCharacterName ? (
        <span
          className={`${showChatTitle ? "hidden sm:inline-flex" : "inline-flex"} max-w-[180px] items-center gap-1 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-text`}
          title={toText(
            t(
              "playground:header.modeCharacterHelp",
              "Character mode is active."
            )
          )}
        >
          <UserCircle2 className="size-3" aria-hidden="true" />
          <span className="truncate">
            {toText(t("playground:header.modeCharacter", "Character"))}:{" "}
            {activeCharacterName}
          </span>
        </span>
      ) : null}
      {shareStatusLabel ? (
        <span
          className="inline-flex max-w-full items-center gap-1 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-text sm:max-w-[220px]"
          title={shareButtonLabel}
          data-testid="chat-header-share-status"
        >
          <Share2 className="size-3" aria-hidden="true" />
          <span className="truncate">{shareStatusLabel}</span>
        </span>
      ) : null}
    </div>
  ) : null

  return showChatTitle ? (
    <header
      data-istemporary-chat={temporaryChat}
      data-ischat-route="true"
      data-testid="chat-conversation-header"
      className="z-20 w-full border-b border-border bg-surface"
    >
      <div className="flex min-h-14 w-full items-center gap-1 px-2 py-1.5 sm:gap-2 sm:px-4">
        {showSidebarToggle && (
          <button
            type="button"
            onClick={onToggleSidebar}
            aria-label={sidebarLabel}
            title={sidebarLabel}
            data-testid="chat-header-sidebar-toggle"
            className={headerActionClass}
          >
            <Menu className="size-4" aria-hidden="true" />
          </button>
        )}
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-semibold leading-5 text-text">
            {canEditTitle ? (
              isEditingTitle ? (
                <Input
                  size="small"
                  autoFocus
                  aria-label={toText(
                    t("option:header.renameConversation", "Rename conversation")
                  )}
                  placeholder={toText(
                    t("option:header.untitledChat", "Untitled")
                  )}
                  value={chatTitle}
                  onChange={(event) => onTitleChange(event.target.value)}
                  onPressEnter={() => void onTitleCommit(chatTitle)}
                  onBlur={() => void onTitleCommit(chatTitle)}
                />
              ) : (
                <button
                  type="button"
                  onClick={onTitleEditStart}
                  className={`block w-full truncate rounded text-left ${focusRingClasses}`}
                  title={toText(
                    t("option:header.renameConversation", "Rename conversation")
                  )}
                >
                  {chatTitle || t("option:header.untitledChat", "Untitled")}
                </button>
              )
            ) : temporaryChat ? (
              t(
                "playground:header.newTemporaryChat",
                "Temporary chat (on this device)"
              )
            ) : (
              chatTitle || t("option:newChat", "New Chat")
            )}
          </h1>
          {sessionBadges}
        </div>
        <div className="flex shrink-0 items-center gap-0.5 sm:gap-1">
          <button
            type="button"
            onClick={onOpenCommandPalette}
            aria-label={commandPaletteAccessibleLabel}
            title={commandPaletteTitle}
            data-testid="chat-header-command-palette-trigger"
            className={headerActionClass}
          >
            <Search className="size-4" aria-hidden="true" />
            <span className="hidden md:inline">{commandPaletteTitle}</span>
            <kbd className="hidden rounded border border-border px-1 text-xs text-text-muted lg:inline">
              {commandKeyLabel}K
            </kbd>
          </button>
          {onOpenShareModal && (
            <button
              type="button"
              onClick={onOpenShareModal}
              disabled={shareButtonDisabled}
              aria-label={shareButtonLabel}
              title={shareButtonLabel}
              data-testid="chat-header-share-button"
              className={`${headerActionClass} hidden disabled:cursor-not-allowed disabled:opacity-55 sm:inline-flex`}
            >
              <Share2 className="size-4" aria-hidden="true" />
              <span className="hidden md:inline">
                {t("playground:header.shareShort", "Share")}
              </span>
            </button>
          )}
          {onStartSavedChat && (
            <button
              type="button"
              onClick={onStartSavedChat}
              aria-label={toText(
                t("playground:header.newSavedChat", "New saved chat")
              )}
              title={toText(
                t("playground:header.newSavedChat", "New saved chat")
              )}
              data-testid="new-chat-button"
              className={`${headerActionClass} text-text`}
            >
              <SquarePen className="size-4" aria-hidden="true" />
              <span className="hidden sm:inline">
                {t("option:newChat", "New Chat")}
              </span>
            </button>
          )}
          {notificationControl}
          <Dropdown
            destroyOnHidden
            trigger={["click"]}
            placement="bottomRight"
            open={headerMenuOpen}
            onOpenChange={setHeaderMenuOpen}
            menu={{
              id: "chat-header-actions-menu",
              className:
                "min-w-60 max-w-[calc(100vw-1rem)] [&_.ant-dropdown-menu-item]:min-h-11 [&_.ant-dropdown-menu-title-content]:whitespace-normal",
              items: menuItems,
              onKeyDown: (event) => {
                if (event.key === "Escape") {
                  event.preventDefault()
                  event.stopPropagation()
                  setHeaderMenuOpen(false)
                  headerMenuTriggerRef.current?.focus()
                }
              }
            }}
          >
            <button
              ref={headerMenuTriggerRef}
              type="button"
              aria-label={toText(
                t("option:header.moreActions", "More actions")
              )}
              title={toText(t("option:header.moreActions", "More actions"))}
              aria-haspopup="menu"
              aria-expanded={headerMenuOpen}
              data-testid="chat-header-more-actions"
              className={headerActionClass}
            >
              <MoreHorizontal className="size-4" aria-hidden="true" />
            </button>
          </Dropdown>
        </div>
      </div>
      <HeaderShortcuts
        expanded={shortcutsExpanded}
        onExpandedChange={onToggleShortcuts}
      />
    </header>
  ) : (
    <header
      data-istemporary-chat={temporaryChat}
      data-ischat-route="true"
      className="z-20 flex w-full flex-col border-b border-border bg-surface/95 backdrop-blur"
    >
      <div className="flex w-full flex-wrap items-center justify-between gap-2 px-4 py-2">
        <div className="flex min-w-0 max-w-full flex-wrap items-center gap-2">
          {showSidebarToggle && (
            <Tooltip title={sidebarLabel} placement="bottom">
              <button
                type="button"
                onClick={onToggleSidebar}
                aria-label={sidebarLabel}
                data-testid="chat-header-sidebar-toggle"
                className={`shrink-0 rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={sidebarLabel}
              >
                <Menu className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          )}
          <div className="flex shrink-0 items-center gap-2 text-text">
            <img
              src={logoSrc}
              alt={toText(t("common:pageAssist", "tldw Assistant"))}
              className="h-5 w-auto"
            />
            <span className="text-sm font-medium">
              {toText(t("common:pageAssist", "tldw Assistant"))}
            </span>
            <Tooltip title={companionHomeLabel}>
              <button
                type="button"
                onClick={onOpenCompanionHome}
                aria-label={companionHomeLabel}
                className={`inline-flex items-center justify-center rounded-md p-1.5 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={companionHomeLabel}
                data-testid="chat-header-companion-home"
              >
                <House className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
            <Tooltip title={shortcutsToggleLabel}>
              <button
                type="button"
                onClick={() => onToggleShortcuts(!shortcutsExpanded)}
                aria-label={shortcutsToggleLabel}
                aria-expanded={shortcutsExpanded}
                className={`inline-flex items-center justify-center rounded-md p-1.5 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={shortcutsToggleLabel}
                data-testid="chat-toggle-shortcuts"
              >
                <Signpost className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
          {sessionBadges}
        </div>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-1 sm:gap-2">
          <button
            type="button"
            onClick={onOpenCommandPalette}
            aria-label={commandPaletteAccessibleLabel}
            className={`hidden items-center gap-2 rounded-md px-3 py-1.5 text-xs text-text-muted transition hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
            title={commandPaletteTitle}
            data-testid="chat-header-command-palette-trigger"
          >
            <Search className="size-4" aria-hidden="true" />
            <span>{commandPaletteTitle}</span>
            <span className="rounded border border-border px-1.5 py-0.5 text-xs text-text-muted">
              {commandKeyLabel}K
            </span>
          </button>
          {showSavedChatAction ? (
            <Tooltip
              title={t("playground:header.newSavedChat", "New saved chat")}
            >
              <button
                type="button"
                onClick={onStartSavedChat}
                aria-label={
                  t(
                    "playground:header.newSavedChat",
                    "New saved chat"
                  ) as string
                }
                className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={t("playground:header.newSavedChat", "New saved chat")}
                data-testid="new-chat-button"
              >
                <SquarePen className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          ) : null}
          {showTemporaryChatAction ? (
            <Tooltip
              title={t(
                "playground:header.newTemporaryChat",
                "Temporary chat (on this device)"
              )}
            >
              <button
                type="button"
                onClick={onStartTemporaryChat}
                aria-label={
                  t(
                    "playground:header.newTemporaryChat",
                    "Temporary chat (on this device)"
                  ) as string
                }
                className={`hidden items-center justify-center rounded-md px-2 py-1.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
                title={t(
                  "playground:header.newTemporaryChat",
                  "Temporary chat (on this device)"
                )}
              >
                {t("playground:header.temporaryShort", "Temp")}
              </button>
            </Tooltip>
          ) : null}
          {showCharacterChatAction ? (
            <Tooltip
              title={t("playground:header.newCharacterChat", "Character chat")}
            >
              <button
                type="button"
                onClick={onStartCharacterChat}
                aria-label={
                  t(
                    "playground:header.newCharacterChat",
                    "Character chat"
                  ) as string
                }
                className={`hidden items-center justify-center rounded-md px-2 py-1.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
                title={t(
                  "playground:header.newCharacterChat",
                  "Character chat"
                )}
              >
                {t("playground:header.characterShort", "Character")}
              </button>
            </Tooltip>
          ) : null}
          {onOpenShareModal && (
            <Tooltip title={shareButtonLabel}>
              <button
                type="button"
                onClick={onOpenShareModal}
                disabled={shareButtonDisabled}
                aria-label={shareButtonLabel}
                className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-60 ${focusRingClasses}`}
                title={shareButtonLabel}
                data-testid="chat-header-share-button"
              >
                <Share2 className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          )}
          <Tooltip title={t("sidepanel:header.settingsShortLabel", "Settings")}>
            <button
              type="button"
              onClick={onOpenSettings}
              aria-label={
                t(
                  "sidepanel:header.openSettingsAria",
                  "Open settings"
                ) as string
              }
              className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
              title={t("sidepanel:header.settingsShortLabel", "Settings")}
            >
              <CogIcon className="size-4" aria-hidden="true" />
            </button>
          </Tooltip>
          {notificationControl}
          {onToggleTheme && (
            <Tooltip title={themeToggleLabel}>
              <button
                type="button"
                onClick={onToggleTheme}
                aria-label={themeToggleLabel}
                className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={themeToggleLabel}
                data-testid="chat-header-theme-toggle"
              >
                {isDarkTheme ? (
                  <Sun className="size-4" aria-hidden="true" />
                ) : (
                  <Moon className="size-4" aria-hidden="true" />
                )}
              </button>
            </Tooltip>
          )}
          <Tooltip
            title={t(
              "option:header.keyboardShortcuts",
              "Keyboard shortcuts (?)"
            )}
          >
            <button
              type="button"
              onClick={onOpenShortcutsModal}
              aria-label={
                t(
                  "option:header.keyboardShortcutsAria",
                  "Show keyboard shortcuts"
                ) as string
              }
              className={`inline-flex items-center justify-center rounded-md p-1.5 text-text-subtle hover:bg-surface2 hover:text-text ${focusRingClasses}`}
              title={t("option:header.keyboardShortcuts", "Keyboard shortcuts")}
            >
              <kbd className="rounded border border-border px-1.5 py-0.5 text-xs font-medium text-text-subtle">
                ?
              </kbd>
            </button>
          </Tooltip>
        </div>
      </div>
      <HeaderShortcuts
        expanded={shortcutsExpanded}
        onExpandedChange={onToggleShortcuts}
      />
    </header>
  )
}
