import React from "react"
import type { TFunction } from "i18next"
import { Tooltip, Input } from "antd"
import {
  Bell,
  CogIcon,
  Menu,
  Moon,
  Search,
  Share2,
  Signpost,
  SquarePen,
  Sun,
  UserCircle2
} from "lucide-react"
import { HeaderShortcuts } from "./HeaderShortcuts"
import logoImage from "~/assets/icon.png"

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
  onOpenNotifications
}: ChatHeaderProps) {
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
  const canEditTitle =
    showChatTitle && !temporaryChat && historyId && historyId !== "temp"
  const isDarkTheme = themeMode !== "light"
  const themeToggleLabel = isDarkTheme
    ? toText(t("common:theme.switchToLight", "Switch to light theme"))
    : toText(t("common:theme.switchToDark", "Switch to dark theme"))
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
            className={`hidden items-center gap-2 rounded-md px-3 py-1.5 text-xs text-text-muted transition hover:bg-surface2 hover:text-text sm:inline-flex ${focusRingClasses}`}
            title={toText(t("common:search", "Search"))}
          >
            <Search className="size-4" aria-hidden="true" />
            <span>{toText(t("common:search", "Search"))}</span>
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
          {onOpenNotifications && (
            <Tooltip title={t("option:header.notifications", "Notifications")}>
              <button
                type="button"
                onClick={onOpenNotifications}
                aria-label={t("option:header.notificationsAria", "Open notifications") as string}
                className={`relative inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
                data-testid="chat-header-notifications-bell"
              >
                <Bell className="size-4" aria-hidden="true" />
                {(notificationCount ?? 0) > 0 && (
                  <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-bold text-white">
                    {notificationCount! > 99 ? "99+" : notificationCount}
                  </span>
                )}
              </button>
            </Tooltip>
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
