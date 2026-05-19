import logoImage from "@/assets/icon.png"
import { useMessage } from "@/hooks/useMessage"
import { Link } from "react-router-dom"
import { Tooltip } from "antd"
import { Bot, CogIcon, ExternalLink, LayoutDashboard, Menu, Pencil, Volume2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import React from "react"
import { StatusDot } from "./StatusDot"
import { browser } from "wxt/browser"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { TtsClipsDrawer } from "./TtsClipsDrawer"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"

type SidepanelHeaderSimpleProps = {
  sidebarOpen?: boolean
  setSidebarOpen?: (open: boolean) => void
  activeTitle?: string
  onRenameTitle?: (nextTitle: string) => void
}

/**
 * Simplified sidepanel header with minimal controls:
 * - Status dot (connection indicator)
 * - Logo + title
 * - New chat button
 * - Settings link
 *
 * All other controls moved to ControlRow in the composer area.
 */
export const SidepanelHeaderSimple = ({
  sidebarOpen: propSidebarOpen,
  setSidebarOpen: propSetSidebarOpen,
  activeTitle,
  onRenameTitle
}: SidepanelHeaderSimpleProps = {}) => {
  const { temporaryChat } = useMessage()
  const { t } = useTranslation(["sidepanel", "common", "option"])
  const notification = useAntdNotification()
  const { capabilities } = useServerCapabilities()
  const [localSidebarOpen, setLocalSidebarOpen] = React.useState(false)
  const [ttsClipsOpen, setTtsClipsOpen] = React.useState(false)
  const [isEditingTitle, setIsEditingTitle] = React.useState(false)
  const [draftTitle, setDraftTitle] = React.useState(activeTitle || "")
  const titleBlurActionRef = React.useRef<"submit" | "cancel" | null>(null)
  const titleInputRef = React.useRef<HTMLInputElement>(null)
  const isControlled = typeof propSidebarOpen === "boolean"
  const sidebarOpen = isControlled
    ? (propSidebarOpen as boolean)
    : localSidebarOpen
  const logoSrc =
    typeof logoImage === "string"
      ? logoImage
      : (logoImage as { src?: string })?.src ?? ""
  const sidebarToggleLabel = sidebarOpen
    ? t("common:chatSidebar.collapse", "Collapse sidebar")
    : t("common:chatSidebar.expand", "Expand sidebar")

  const handleSidebarToggle = React.useCallback(() => {
    const next = !sidebarOpen
    if (!isControlled) {
      setLocalSidebarOpen(next)
      return
    }
    propSetSidebarOpen?.(next)
  }, [isControlled, propSetSidebarOpen, sidebarOpen])

  React.useEffect(() => {
    setDraftTitle(activeTitle || "")
    setIsEditingTitle(false)
  }, [activeTitle])

  React.useEffect(() => {
    if (!isEditingTitle) return
    titleInputRef.current?.focus()
    titleInputRef.current?.select()
  }, [isEditingTitle])

  const cancelTitleEdit = React.useCallback(() => {
    setDraftTitle(activeTitle || "")
    setIsEditingTitle(false)
  }, [activeTitle])

  const submitTitleEdit = React.useCallback(() => {
    const trimmed = draftTitle.trim()
    if (!trimmed) {
      setDraftTitle(activeTitle || "")
      setIsEditingTitle(false)
      return
    }
    if (trimmed !== (activeTitle || "").trim()) {
      onRenameTitle?.(trimmed)
    }
    setIsEditingTitle(false)
  }, [activeTitle, draftTitle, onRenameTitle])

  const openFullScreen = React.useCallback(() => {
    const url = browser.runtime.getURL("/options.html#/")
    const showFailure = () => {
      notification.error({
        message: t(
          "sidepanel:header.openFullScreenFailed",
          "Unable to open full-screen view."
        )
      })
    }
    const openFallback = () => {
      const opened = window.open(url, "_blank")
      if (!opened) {
        showFailure()
      }
    }
    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch((error) => {
        console.error("Failed to open full-screen tab:", error)
        openFallback()
      })
      return
    }
    openFallback()
  }, [notification, t])

  const openDashboard = React.useCallback(() => {
    const url = browser.runtime.getURL("/options.html#/flashcards")
    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch(() => {
        window.open(url, "_blank")
      })
      return
    }
    window.open(url, "_blank")
  }, [])

  return (
    <div
      data-istemporary-chat={temporaryChat}
      data-testid="chat-header"
      className="absolute top-0 z-10 flex h-11 w-full items-center justify-between border-b border-border bg-surface/95 px-3 py-2 text-body text-text backdrop-blur data-[istemporary-chat='true']:bg-purple-50 data-[istemporary-chat='true']:dark:bg-purple-900/30"
    >
      {/* Left: Sidebar toggle + Status dot + Logo + Title */}
      <div className="flex items-center gap-2">
        <Tooltip
          title={sidebarToggleLabel}>
          <button
            type="button"
            aria-label={sidebarToggleLabel}
            aria-expanded={sidebarOpen}
            onClick={handleSidebarToggle}
            className="rounded-md p-1.5 text-text-muted hover:bg-surface2 hover:text-text"
            title={sidebarToggleLabel}
          >
            <Menu className="size-4" />
          </button>
        </Tooltip>
        <StatusDot />
        <div className="flex items-center text-text">
          <img
            className="h-5 w-auto"
            src={logoSrc}
            alt={t("common:pageAssist")}
          />
          <span className="ml-2 text-sm font-medium">
            {t("common:pageAssist")}
          </span>
        </div>
        {activeTitle && (
          <div className="ml-2 hidden max-w-[12rem] sm:inline">
            {isEditingTitle && onRenameTitle ? (
              <input
                ref={titleInputRef}
                value={draftTitle}
                onChange={(event) => setDraftTitle(event.target.value)}
                onBlur={() => {
                  if (!onRenameTitle) return
                  const action = titleBlurActionRef.current
                  titleBlurActionRef.current = null
                  if (action === "cancel") {
                    cancelTitleEdit()
                    return
                  }
                  submitTitleEdit()
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault()
                    titleBlurActionRef.current = "submit"
                    event.currentTarget.blur()
                  }
                  if (event.key === "Escape") {
                    event.preventDefault()
                    titleBlurActionRef.current = "cancel"
                    event.currentTarget.blur()
                  }
                }}
                aria-label={t(
                  "sidepanel:header.renameTitle",
                  "Rename conversation"
                )}
                className="w-full rounded-md border border-border bg-surface px-2 py-0.5 text-caption text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              />
            ) : (
              onRenameTitle ? (
                <button
                  type="button"
                  onClick={() => {
                    setIsEditingTitle(true)
                  }}
                  className="group flex max-w-full items-center gap-1 text-caption text-text-muted hover:text-text"
                  title={activeTitle}
                  aria-label={t(
                    "sidepanel:header.renameTitle",
                    "Rename conversation"
                  )}
                >
                  <span className="truncate">{activeTitle}</span>
                  <Pencil className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-80" />
                </button>
              ) : (
                <span
                  className="flex max-w-full items-center gap-1 text-caption text-text-muted"
                  title={activeTitle}
                >
                  <span className="truncate">{activeTitle}</span>
                </span>
              )
            )}
          </div>
        )}
      </div>

      {/* Right: Full-screen + Settings */}
      <div className="flex items-center gap-1">
        <Tooltip title={t("playground:ttsClips.title", "TTS clips")}>
          <button
            type="button"
            onClick={() => setTtsClipsOpen(true)}
            aria-label={t("playground:ttsClips.title", "TTS clips")}
            className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
            title={t("playground:ttsClips.title", "TTS clips")}
            data-testid="chat-open-tts-clips"
          >
            <Volume2 className="size-4" aria-hidden="true" />
          </button>
        </Tooltip>
        {capabilities?.hasPersona ? (
          <Tooltip title={t("sidepanel:persona.openPersona", "Open Persona Garden")}>
            <Link
              to="/persona"
              aria-label={t("sidepanel:persona.openPersona", "Open Persona Garden") as string}
              className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
              title={t("sidepanel:persona.openPersona", "Open Persona Garden")}
              data-testid="chat-open-persona"
            >
              <Bot className="size-4" aria-hidden="true" />
            </Link>
          </Tooltip>
        ) : null}
        <Tooltip title={t("sidepanel:header.openDashboard", "Open dashboard")}>
          <button
            type="button"
            onClick={openDashboard}
            aria-label={t("sidepanel:header.openDashboard", "Open dashboard")}
            className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
            title={t("sidepanel:header.openDashboard", "Open dashboard")}
            data-testid="chat-open-dashboard"
          >
            <LayoutDashboard className="size-4" aria-hidden="true" />
          </button>
        </Tooltip>
        <Tooltip title={t("sidepanel:header.openFullScreen", "Open Full-Screen")}>
          <button
            type="button"
            onClick={openFullScreen}
            aria-label={t("sidepanel:header.openFullScreen", "Open Full-Screen")}
            className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
            title={t("sidepanel:header.openFullScreen", "Open Full-Screen")}
            data-testid="chat-open-full-screen"
          >
            <ExternalLink className="size-4" aria-hidden="true" />
          </button>
        </Tooltip>
        <Tooltip title={t("sidepanel:header.settingsShortLabel", "Settings")}>
          <Link
            to="/settings"
            aria-label={t("sidepanel:header.openSettingsAria") as string}
            className="rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text"
            title={t("sidepanel:header.settingsShortLabel", "Settings")}
          >
            <CogIcon className="size-4" aria-hidden="true" />
          </Link>
        </Tooltip>
      </div>
      {ttsClipsOpen ? (
        <TtsClipsDrawer
          open={ttsClipsOpen}
          onClose={() => setTtsClipsOpen(false)}
        />
      ) : null}
    </div>
  )
}
