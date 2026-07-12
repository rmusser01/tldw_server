import React from "react"
import type { TFunction } from "i18next"
import { Tooltip, Input } from "antd"
import {
  Bell,
  BellOff,
  CogIcon,
  House,
  LoaderCircle,
  Menu,
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
import logoImage from "~/assets/icon.png"
import type { NotificationLifecycleState } from "@/services/notification-lifecycle"

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
  shortcutsExpanded,
  onToggleShortcuts,
  commandKeyLabel,
  notificationCount,
  onOpenNotifications,
  notificationState = "active",
  onRetryNotifications
}: ChatHeaderProps) {
  const [notificationPopoverOpen, setNotificationPopoverOpen] = React.useState(false)
  const notificationTriggerRef = React.useRef<HTMLButtonElement>(null)
  const notificationPopoverRef = React.useRef<HTMLDivElement>(null)
  const logoSrc =
    typeof logoImage === "string"
      ? logoImage
      : (logoImage as { src?: string })?.src ?? ""
  const showSidebarToggle = Boolean(onToggleSidebar)
  const sidebarLabel = sidebarCollapsed
    ? toText(t("common:chatSidebar.expand", "Expand sidebar"))
    : toText(t("common:chatSidebar.collapse", "Collapse sidebar"))
  const shortcutsToggleLabel = shortcutsExpanded
    ? toText(t("option:header.hideShortcuts", "Hide shortcuts"))
    : toText(t("option:header.showShortcuts", "Show shortcuts"))
  const companionHomeLabel = toText(
    t("option:header.companionHome", "Companion Home")
  )
  const canEditTitle =
    showChatTitle && !temporaryChat && historyId && historyId !== "temp"
  const isDarkTheme = themeMode !== "light"
  const themeToggleLabel = isDarkTheme
    ? toText(t("common:theme.switchToLight", "Switch to light theme"))
    : toText(t("common:theme.switchToDark", "Switch to dark theme"))
  const commandPaletteLabel = toText(
    t("common:shortcuts.openCommandPalette", "Open command palette")
  )
  const commandPaletteTitle = toText(t("common:search", "Search"))
  const commandPaletteAccessibleLabel =
    `${commandPaletteTitle} - ${commandPaletteLabel}`
  const showSavedChatAction = Boolean(onStartSavedChat)
  const showTemporaryChatAction = Boolean(onStartTemporaryChat)
  const showCharacterChatAction = Boolean(onStartCharacterChat)
  const shareButtonLabel = shareStatusLabel
    ? toText(
        t("playground:header.shareStatusAria", "Share conversation ({{status}})", {
          status: shareStatusLabel
        } as any)
      )
    : toText(t("playground:header.shareConversation", "Share conversation"))
  const focusRingClasses =
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  const notificationCopy = notificationState === "idle" ? null : {
    active: {
      name: toText(t(
        "option:header.notificationsActiveName",
        `Notifications, ${notificationCount ?? 0} unread`,
        { count: notificationCount ?? 0 }
      )),
      title: toText(t("option:header.notifications", "Notifications")),
      description: (notificationCount ?? 0) > 0
        ? toText(t(
            "option:header.notificationsUnreadDescription",
            `${notificationCount} unread notification${notificationCount === 1 ? "" : "s"}.`,
            { count: notificationCount }
          ))
        : toText(t("option:header.notificationsCaughtUp", "You are caught up.")),
      announcement: toText(t("option:header.notificationsActiveAnnouncement", "Notifications are active"))
    },
    connecting: {
      name: toText(t("option:header.notificationsConnectingName", "Notifications, connecting")),
      title: toText(t("option:header.notificationsConnectingTitle", "Notifications are connecting")),
      description: toText(t("option:header.notificationsConnectingDescription", "Checking for new notifications.")),
      announcement: toText(t("option:header.notificationsConnectingAnnouncement", "Notifications are connecting"))
    },
    degraded: {
      name: toText(t("option:header.notificationsReconnectingName", "Notifications, reconnecting")),
      title: toText(t("option:header.notificationsReconnectingTitle", "Notifications are reconnecting")),
      description: toText(t("option:header.notificationsReconnectingDescription", "Recent notifications may be delayed while the connection recovers.")),
      announcement: toText(t("option:header.notificationsReconnectingAnnouncement", "Notifications are reconnecting"))
    },
    "auth-required": {
      name: toText(t("option:header.notificationsSignInName", "Notifications, sign-in required")),
      title: toText(t("option:header.notificationsSignInTitle", "Sign in required")),
      description: toText(t("option:header.notificationsSignInDescription", "Sign in again to resume personal notifications.")),
      announcement: toText(t("option:header.notificationsSignInAnnouncement", "Notifications require sign in"))
    },
    unavailable: {
      name: toText(t("option:header.notificationsUnavailableName", "Notifications unavailable for this account")),
      title: toText(t("option:header.notificationsUnavailableTitle", "Notifications unavailable for this account")),
      description: toText(t("option:header.notificationsUnavailableDescription", "Notifications are not available for this account.")),
      announcement: toText(t("option:header.notificationsUnavailableAnnouncement", "Notifications are unavailable"))
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

  const NotificationIcon = notificationState === "active"
    ? Bell
    : notificationState === "connecting"
      ? LoaderCircle
      : notificationState === "unavailable"
        ? BellOff
        : TriangleAlert

  return (
    <header
      data-istemporary-chat={temporaryChat}
      data-ischat-route="true"
      className="z-20 flex w-full flex-col border-b border-border bg-surface/95 backdrop-blur data-[istemporary-chat='true']:bg-purple-900 data-[ischat-route='true']:bg-surface/95"
    >
      <div className="flex w-full flex-wrap items-center justify-between gap-2 px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          {showSidebarToggle && (
            <Tooltip title={sidebarLabel} placement="bottom">
              <button
                type="button"
                onClick={onToggleSidebar}
                aria-label={sidebarLabel}
                data-testid="chat-header-sidebar-toggle"
                className={`rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={sidebarLabel}
              >
                <Menu className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          )}
          <div className="flex items-center gap-2 text-text">
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
          {canEditTitle && (
            <div className="min-w-[140px] max-w-[220px] truncate">
              {isEditingTitle ? (
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
                  onChange={(e) => onTitleChange(e.target.value)}
                  onPressEnter={() => {
                    void onTitleCommit(chatTitle)
                  }}
                  onBlur={() => {
                    void onTitleCommit(chatTitle)
                  }}
                />
              ) : (
                <button
                  type="button"
                  onClick={onTitleEditStart}
                  className={`truncate text-left text-xs text-text-muted hover:text-text ${focusRingClasses}`}
                  title={chatTitle || "Untitled"}
                >
                  {chatTitle || t("option:header.untitledChat", "Untitled")}
                </button>
              )}
            </div>
          )}
          {showSessionModeBadge ? (
            <div className="flex items-center gap-1">
              <span
                className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                  temporaryChat
                    ? "border-warn/50 bg-warn/10 text-warn"
                    : "border-success/40 bg-success/10 text-success"
                }`}
                title={
                  temporaryChat
                    ? toText(
                        t(
                          "playground:header.modeTemporaryHelp",
                          "Temporary chat. Messages are not saved."
                        )
                      )
                    : toText(
                        t(
                          "playground:header.modeSavedHelp",
                          "Saved chat. History is persisted."
                        )
                      )
                }
              >
                {temporaryChat
                  ? toText(t("playground:header.modeTemporary", "Temporary"))
                  : toText(t("playground:header.modeSaved", "Saved"))}
              </span>
              {activeCharacterName ? (
                <span
                  className="inline-flex max-w-[180px] items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primaryStrong"
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
                  className="inline-flex max-w-[220px] items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primaryStrong"
                  title={shareButtonLabel}
                  data-testid="chat-header-share-status"
                >
                  <Share2 className="size-3" aria-hidden="true" />
                  <span className="truncate">{shareStatusLabel}</span>
                </span>
              ) : null}
            </div>
          ) : null}
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
            <span className="rounded border border-border px-1.5 py-0.5 text-xs text-text-subtle">
              {commandKeyLabel}K
            </span>
          </button>
          {showSavedChatAction ? (
            <Tooltip title={t("playground:header.newSavedChat", "New saved chat")}>
              <button
                type="button"
                onClick={onStartSavedChat}
                aria-label={t("playground:header.newSavedChat", "New saved chat") as string}
                className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                title={t("playground:header.newSavedChat", "New saved chat")}
                data-testid="new-chat-button"
              >
                <SquarePen className="size-4" aria-hidden="true" />
              </button>
            </Tooltip>
          ) : null}
          {showTemporaryChatAction ? (
            <Tooltip title={t("playground:header.newTemporaryChat", "Temporary chat (not saved)")}>
              <button
                type="button"
                onClick={onStartTemporaryChat}
                aria-label={t("playground:header.newTemporaryChat", "Temporary chat (not saved)") as string}
                className={`hidden items-center justify-center rounded-md px-2 py-1.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
                title={t("playground:header.newTemporaryChat", "Temporary chat (not saved)")}
              >
                {t("playground:header.temporaryShort", "Temp")}
              </button>
            </Tooltip>
          ) : null}
          {showCharacterChatAction ? (
            <Tooltip title={t("playground:header.newCharacterChat", "Character chat")}>
              <button
                type="button"
                onClick={onStartCharacterChat}
                aria-label={t("playground:header.newCharacterChat", "Character chat") as string}
                className={`hidden items-center justify-center rounded-md px-2 py-1.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
                title={t("playground:header.newCharacterChat", "Character chat")}
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
              aria-label={t("sidepanel:header.openSettingsAria", "Open settings") as string}
              className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
              title={t("sidepanel:header.settingsShortLabel", "Settings")}
            >
              <CogIcon className="size-4" aria-hidden="true" />
            </button>
          </Tooltip>
          {onOpenNotifications && notificationCopy && (
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
                  className={`relative inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                  data-testid="chat-header-notifications-bell"
                >
                  <NotificationIcon
                    className={`size-4 ${notificationState === "connecting" ? "animate-spin" : ""}`}
                    aria-hidden="true"
                  />
                  {notificationState === "active" && (notificationCount ?? 0) > 0 && (
                    <span aria-hidden="true" className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-bold text-white">
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
                  <p className="text-sm font-semibold text-text">{notificationCopy.title}</p>
                  <p className="mt-1 text-xs leading-5 text-text-muted">{notificationCopy.description}</p>
                  <div className="mt-3 flex items-center gap-2">
                    {(notificationState === "degraded" || notificationState === "unavailable") && onRetryNotifications ? (
                      <button type="button" onClick={() => void onRetryNotifications()} className={`rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primaryStrong ${focusRingClasses}`}>
                        {toText(t("option:header.notificationsTryAgain", "Try again"))}
                      </button>
                    ) : null}
                    <button type="button" onClick={openNotifications} className={`rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 ${focusRingClasses}`}>
                      {toText(t("option:header.notificationsOpen", "Open notifications"))}
                    </button>
                  </div>
                </div>
              ) : null}
              <span role="status" aria-live="polite" className="sr-only">
                {notificationCopy.announcement}
              </span>
            </div>
          )}
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
          <Tooltip title={t("option:header.keyboardShortcuts", "Keyboard shortcuts (?)")}>
            <button
              type="button"
              onClick={onOpenShortcutsModal}
              aria-label={t("option:header.keyboardShortcutsAria", "Show keyboard shortcuts") as string}
              className={`inline-flex items-center justify-center rounded-md p-1.5 text-text-subtle hover:bg-surface2 hover:text-text ${focusRingClasses}`}
              title={t("option:header.keyboardShortcuts", "Keyboard shortcuts")}
            >
              <kbd className="rounded border border-border px-1.5 py-0.5 text-xs font-medium text-text-subtle">?</kbd>
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
