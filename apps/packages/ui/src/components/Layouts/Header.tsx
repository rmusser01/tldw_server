import React from "react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router-dom"
import { Button, Input, InputNumber, Modal, Tooltip } from "antd"
import { isMac } from "@/hooks/keyboard/useKeyboardShortcuts"
import { getTitleById, updateHistory } from "@/db"
import { useMessageOption } from "~/hooks/useMessageOption"
import { useSetting } from "@/hooks/useSetting"
import { HEADER_SHORTCUTS_EXPANDED_SETTING } from "@/services/settings/ui-settings"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { ChatHeader } from "./ChatHeader"
import { TtsClipsDrawer } from "@/components/Sidepanel/Chat/TtsClipsDrawer"
import { useDarkMode } from "@/hooks/useDarkmode"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import type { Character } from "@/types/character"
import { dispatchOpenAssistantSelect } from "@/utils/assistant-select-events"
import { dispatchCharacterChatModeIntent } from "@/utils/character-chat-mode-intent"
import { buildCharacterChatPath } from "@/routes/route-paths"
import {
  tldwClient,
  type ConversationShareLinkSummary,
} from "@/services/tldw/TldwApiClient"
import {
  buildConversationShareUrl,
  getActiveShareLinkCount,
  isShareLinkActive,
  isShareLinkRevoked,
  sortShareLinksByCreatedDesc,
} from "./chat-share-links"
import { getHeaderActionPolicy } from "./header-action-policy"

type Props = {
  onToggleSidebar?: () => void
  sidebarCollapsed?: boolean
  /** Unread notification count (passed through to ChatHeader bell) */
  notificationCount?: number
  /** Callback when notification bell is clicked */
  onOpenNotifications?: () => void
}

export const Header: React.FC<Props> = ({
  onToggleSidebar,
  sidebarCollapsed = false,
  notificationCount,
  onOpenNotifications
}) => {
  const { t } = useTranslation([
    "option",
    "common",
    "settings",
    "playground"
  ])
  const cmdKey = isMac ? "⌘" : "Ctrl+"
  const [headerShortcutsExpanded, setHeaderShortcutsExpanded] = useSetting(
    HEADER_SHORTCUTS_EXPANDED_SETTING
  )
  const hostedMode = isHostedTldwDeployment()
  const { mode: themeMode, toggleDarkMode } = useDarkMode()
  const { clearChat, historyId, temporaryChat, setTemporaryChat, serverChatId } =
    useMessageOption()
  const [selectedCharacter, setSelectedCharacter] = useSelectedCharacter<Character | null>(
    null
  )
  const navigate = useNavigate()
  const location = useLocation()
  const [chatTitle, setChatTitle] = React.useState("")
  const [isEditingTitle, setIsEditingTitle] = React.useState(false)
  const [ttsClipsOpen, setTtsClipsOpen] = React.useState(false)
  const [shareModalOpen, setShareModalOpen] = React.useState(false)
  const [shareLinks, setShareLinks] = React.useState<ConversationShareLinkSummary[]>([])
  const [shareLinksLoading, setShareLinksLoading] = React.useState(false)
  const [shareCreateLoading, setShareCreateLoading] = React.useState(false)
  const [shareRevokeLoadingId, setShareRevokeLoadingId] = React.useState<string | null>(null)
  const [shareTtlHours, setShareTtlHours] = React.useState(24)
  const [shareLabel, setShareLabel] = React.useState("")
  const [shareError, setShareError] = React.useState<string | null>(null)
  const [copiedShareId, setCopiedShareId] = React.useState<string | null>(null)
  const headerActionPolicy = React.useMemo(
    () => getHeaderActionPolicy(location.pathname),
    [location.pathname]
  )
  const isChatRoute = headerActionPolicy.showChatSessionActions

  React.useEffect(() => {
    ;(async () => {
      try {
        if (historyId && historyId !== "temp" && !temporaryChat) {
          const title = await getTitleById(historyId)
          setChatTitle(title || "")
        } else {
          setChatTitle("")
        }
      } catch {}
    })()
  }, [historyId, temporaryChat])

  const saveTitle = async (value: string) => {
    try {
      if (historyId && historyId !== "temp" && !temporaryChat) {
        await updateHistory(historyId, value.trim() || "Untitled")
      }
    } catch (e) {
      console.error("Failed to update chat title", e)
    }
  }

  const openCommandPalette = React.useCallback(() => {
    if (typeof window === "undefined") return
    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
  }, [])

  const openShortcutsModal = React.useCallback(() => {
    if (typeof window === "undefined") return
    window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
  }, [])

  const toggleHeaderShortcuts = React.useCallback((next?: boolean) => {
    void setHeaderShortcutsExpanded((prev) =>
      typeof next === "boolean" ? next : !prev
    ).catch(() => {
      // ignore storage write failures
    })
  }, [setHeaderShortcutsExpanded])

  const handleTitleEditStart = React.useCallback(() => {
    setIsEditingTitle(true)
  }, [])

  const handleTitleCommit = React.useCallback(
    async (value: string) => {
      setIsEditingTitle(false)
      await saveTitle(value)
    },
    [saveTitle]
  )

  const startSavedChat = React.useCallback(() => {
    setTemporaryChat(false)
    void setSelectedCharacter(null)
    clearChat()
  }, [clearChat, setSelectedCharacter, setTemporaryChat])

  const startTemporaryChat = React.useCallback(() => {
    setTemporaryChat(true)
    void setSelectedCharacter(null)
    clearChat()
  }, [clearChat, setSelectedCharacter, setTemporaryChat])

  const startCharacterChat = React.useCallback(() => {
    setTemporaryChat(false)
    const characterId = selectedCharacter?.id ?? null
    if (!isChatRoute) {
      navigate(buildCharacterChatPath({ characterId }))
      return
    }
    dispatchCharacterChatModeIntent({
      source: "chat-header",
      characterId
    })
    if (!selectedCharacter?.id) {
      dispatchOpenAssistantSelect({
        tab: "character",
        source: "chat-header"
      })
      return
    }
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
    }
  }, [isChatRoute, navigate, selectedCharacter, setTemporaryChat])

  const refreshShareLinks = React.useCallback(async () => {
    if (!serverChatId) {
      setShareLinks([])
      return
    }
    setShareLinksLoading(true)
    setShareError(null)
    try {
      const response = await tldwClient.listConversationShareLinks(serverChatId)
      const links = Array.isArray(response?.links) ? response.links : []
      setShareLinks(sortShareLinksByCreatedDesc(links))
    } catch {
      setShareError(
        t(
          "playground:header.shareLoadError",
          "Unable to load share links for this chat."
        ) as string
      )
    } finally {
      setShareLinksLoading(false)
    }
  }, [serverChatId, t])

  React.useEffect(() => {
    if (!shareModalOpen) return
    void refreshShareLinks()
  }, [refreshShareLinks, shareModalOpen])

  React.useEffect(() => {
    if (!isChatRoute || temporaryChat || !serverChatId) {
      setShareLinks([])
      return
    }
    void refreshShareLinks()
  }, [isChatRoute, refreshShareLinks, serverChatId, temporaryChat])

  const activeShareCount = React.useMemo(
    () => getActiveShareLinkCount(shareLinks),
    [shareLinks]
  )
  const shareStatusLabel = React.useMemo(() => {
    if (activeShareCount > 0) {
      return t("playground:header.shareStatusActive", "{{count}} active link(s)", {
        count: activeShareCount
      }) as string
    }
    if (shareLinks.length > 0) {
      return t(
        "playground:header.shareStatusInactive",
        "Shared links inactive"
      ) as string
    }
    return null
  }, [activeShareCount, shareLinks.length, t])

  const openShareModal = React.useCallback(() => {
    setShareError(null)
    setShareModalOpen(true)
  }, [])

  const openWorkflowAutomation = React.useCallback(() => {
    const query = serverChatId
      ? `?source=chat-share&conversationId=${encodeURIComponent(serverChatId)}`
      : "?source=chat-share"
    setShareModalOpen(false)
    navigate(`/workflow-editor${query}`)
  }, [navigate, serverChatId])

  const handleCreateShareLink = React.useCallback(async () => {
    if (!serverChatId || shareCreateLoading) return
    const ttlSeconds = Math.max(300, Math.round((shareTtlHours || 1) * 3600))
    setShareCreateLoading(true)
    setShareError(null)
    try {
      const created = await tldwClient.createConversationShareLink(serverChatId, {
        permission: "view",
        ttl_seconds: ttlSeconds,
        label: shareLabel.trim() || undefined
      })
      const shareUrl = buildConversationShareUrl(
        typeof window !== "undefined" ? window.location.origin : "",
        {
          share_path: created?.share_path,
          token: created?.token
        }
      )
      if (shareUrl && typeof navigator !== "undefined" && navigator.clipboard) {
        await navigator.clipboard.writeText(shareUrl)
      }
      await refreshShareLinks()
      setCopiedShareId(String(created.share_id))
      if (typeof window !== "undefined") {
        window.setTimeout(() => setCopiedShareId((prev) => (prev === String(created.share_id) ? null : prev)), 1800)
      }
    } catch {
      setShareError(
        t(
          "playground:header.shareCreateError",
          "Unable to create a share link."
        ) as string
      )
    } finally {
      setShareCreateLoading(false)
    }
  }, [
    refreshShareLinks,
    serverChatId,
    shareCreateLoading,
    shareLabel,
    shareTtlHours,
    t
  ])

  const handleCopyShareLink = React.useCallback(
    async (link: ConversationShareLinkSummary) => {
      const shareUrl = buildConversationShareUrl(
        typeof window !== "undefined" ? window.location.origin : "",
        link
      )
      if (!shareUrl) {
        setShareError(
          t(
            "playground:header.shareCopyUnavailable",
            "This share link is unavailable because it is revoked or expired."
          ) as string
        )
        return
      }
      try {
        await navigator.clipboard.writeText(shareUrl)
        setCopiedShareId(link.id)
        if (typeof window !== "undefined") {
          window.setTimeout(() => {
            setCopiedShareId((prev) => (prev === link.id ? null : prev))
          }, 1800)
        }
      } catch {
        setShareError(
          t(
            "playground:header.shareCopyError",
            "Unable to copy share link."
          ) as string
        )
      }
    },
    [t]
  )

  const handleRevokeShareLink = React.useCallback(
    async (shareId: string) => {
      if (!serverChatId || !shareId) return
      setShareRevokeLoadingId(shareId)
      setShareError(null)
      try {
        await tldwClient.revokeConversationShareLink(serverChatId, shareId)
        await refreshShareLinks()
      } catch {
        setShareError(
          t(
            "playground:header.shareRevokeError",
            "Unable to revoke share link."
          ) as string
        )
      } finally {
        setShareRevokeLoadingId(null)
      }
    },
    [refreshShareLinks, serverChatId, t]
  )

  const shareButtonDisabled =
    temporaryChat || !serverChatId || (isChatRoute && historyId === "temp")

  return (
    <>
      <ChatHeader
        t={t}
        temporaryChat={temporaryChat}
        historyId={historyId}
        chatTitle={chatTitle}
        isEditingTitle={isEditingTitle}
        onTitleChange={setChatTitle}
        onTitleEditStart={handleTitleEditStart}
        onTitleCommit={handleTitleCommit}
        onToggleSidebar={onToggleSidebar}
        sidebarCollapsed={sidebarCollapsed}
        onOpenCommandPalette={openCommandPalette}
        onOpenShortcutsModal={openShortcutsModal}
        onOpenShareModal={
          headerActionPolicy.showShareConversation ? openShareModal : undefined
        }
        shareStatusLabel={
          headerActionPolicy.showShareConversation ? shareStatusLabel : null
        }
        shareButtonDisabled={shareButtonDisabled}
        onOpenSettings={() => navigate(hostedMode ? "/account" : "/settings/tldw")}
        onToggleTheme={toggleDarkMode}
        themeMode={themeMode}
        onStartSavedChat={
          headerActionPolicy.showChatSessionActions ? startSavedChat : undefined
        }
        onStartTemporaryChat={
          headerActionPolicy.showChatSessionActions
            ? startTemporaryChat
            : undefined
        }
        onStartCharacterChat={
          headerActionPolicy.showChatSessionActions
            ? startCharacterChat
            : undefined
        }
        activeCharacterName={selectedCharacter?.name || null}
        showChatTitle={headerActionPolicy.showChatTitle}
        showSessionModeBadge={headerActionPolicy.showSessionModeBadge}
        shortcutsExpanded={headerShortcutsExpanded}
        onToggleShortcuts={toggleHeaderShortcuts}
        commandKeyLabel={cmdKey}
        notificationCount={notificationCount}
        onOpenNotifications={onOpenNotifications}
      />
      {ttsClipsOpen ? (
        <TtsClipsDrawer
          open={ttsClipsOpen}
          onClose={() => setTtsClipsOpen(false)}
        />
      ) : null}
      <Modal
        open={shareModalOpen}
        onCancel={() => setShareModalOpen(false)}
        title={t("playground:header.shareModalTitle", "Share conversation")}
        destroyOnHidden
        footer={null}
      >
        <div className="space-y-3" data-testid="chat-share-modal">
          <p className="text-xs text-text-muted">
            {t(
              "playground:header.shareModalDescription",
              "Create read-only share links. Shared views hide authoring controls and preserve message citations/artifacts."
            )}
          </p>
          <div
            className="rounded-md border border-border bg-surface2/50 px-2 py-2"
            data-testid="chat-share-role-scope"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-semibold text-text">
                {t("playground:header.shareRoleLabel", "Access role")}
              </span>
              <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted">
                {t("playground:header.shareRoleView", "Read-only viewer")}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-text-muted">
              {t(
                "playground:header.shareRoleDescription",
                "Recipients can view messages, citations, and artifacts, but cannot send, edit, or delete."
              )}
            </p>
            <div className="mt-2 flex justify-end">
              <Button
                size="small"
                onClick={openWorkflowAutomation}
                data-testid="chat-share-open-workflows"
              >
                {t(
                  "playground:header.shareAutomationShortcut",
                  "Open automation workflows"
                )}
              </Button>
            </div>
          </div>
          <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
            <div className="space-y-2">
              <label className="flex flex-col gap-1 text-xs text-text-muted">
                <span>{t("playground:header.shareLabel", "Label")}</span>
                <Input
                  value={shareLabel}
                  placeholder={t(
                    "playground:header.shareLabelPlaceholder",
                    "Optional purpose label"
                  )}
                  onChange={(event) => setShareLabel(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-text-muted">
                <span>{t("playground:header.shareTtl", "Expires in (hours)")}</span>
                <InputNumber
                  min={1}
                  max={24 * 30}
                  value={shareTtlHours}
                  onChange={(value) =>
                    setShareTtlHours(
                      typeof value === "number" && Number.isFinite(value)
                        ? value
                        : 24
                    )
                  }
                />
              </label>
            </div>
            <div className="flex items-end justify-end">
              <Button
                type="primary"
                loading={shareCreateLoading}
                onClick={handleCreateShareLink}
                disabled={!serverChatId || temporaryChat}
                data-testid="chat-share-create-button"
              >
                {t("playground:header.shareCreate", "Create link")}
              </Button>
            </div>
          </div>
          {shareError ? (
            <div
              role="alert"
              className="rounded-md border border-danger/30 bg-danger/10 px-2 py-1 text-xs text-danger"
            >
              {shareError}
            </div>
          ) : null}
          <div className="space-y-2 rounded-md border border-border bg-surface2/60 px-2 py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-text">
                {t("playground:header.shareExisting", "Share links")}
              </span>
              <span className="text-[11px] text-text-muted">
                {t("playground:header.shareActiveCount", "{{count}} active", {
                  count: activeShareCount
                })}
              </span>
            </div>
            {shareLinksLoading ? (
              <p className="text-xs text-text-muted">
                {t("common:loading", "Loading...")}
              </p>
            ) : shareLinks.length === 0 ? (
              <p className="text-xs text-text-muted">
                {t(
                  "playground:header.shareEmpty",
                  "No share links yet for this conversation."
                )}
              </p>
            ) : (
              <div className="space-y-2">
                {shareLinks.map((link) => {
                  const active = isShareLinkActive(link)
                  const revoked = isShareLinkRevoked(link)
                  const statusText = revoked
                    ? t("playground:header.shareStatusRevoked", "Revoked")
                    : active
                      ? t("playground:header.shareStatusReady", "Active")
                      : t("playground:header.shareStatusExpired", "Expired")
                  return (
                    <div
                      key={link.id}
                      className="rounded border border-border bg-surface px-2 py-2"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <span className="font-medium text-text">
                          {link.id.slice(0, 8)}
                        </span>
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] ${
                            active
                              ? "border-primary/30 bg-primary/10 text-primaryStrong"
                              : "border-border bg-surface2 text-text-muted"
                          }`}
                        >
                          {statusText}
                        </span>
                      </div>
                      <div className="mt-1 text-[11px] text-text-muted">
                        {t("playground:header.shareExpiresAt", "Expires: {{value}}", {
                          value: new Date(link.expires_at).toLocaleString()
                        })}
                      </div>
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <Tooltip
                          title={
                            !active
                              ? t(
                                  "playground:header.shareCopyUnavailable",
                                  "This share link is unavailable because it is revoked or expired."
                                )
                              : undefined
                          }
                        >
                          <Button
                            size="small"
                            onClick={() => void handleCopyShareLink(link)}
                            disabled={!active}
                            data-testid={`chat-share-copy-${link.id}`}
                          >
                            {copiedShareId === link.id
                              ? t("common:copied", "Copied")
                              : t("common:copy", "Copy")}
                          </Button>
                        </Tooltip>
                        <Button
                          size="small"
                          danger
                          onClick={() => void handleRevokeShareLink(link.id)}
                          disabled={!active}
                          loading={shareRevokeLoadingId === link.id}
                          data-testid={`chat-share-revoke-${link.id}`}
                        >
                          {t("playground:header.shareRevoke", "Revoke")}
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
          <div className="flex justify-end">
            <Button onClick={() => setShareModalOpen(false)}>
              {t("common:close", "Close")}
            </Button>
          </div>
        </div>
      </Modal>
    </>
  )
}
