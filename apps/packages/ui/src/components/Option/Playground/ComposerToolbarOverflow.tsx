import React from "react"
import { useTranslation } from "react-i18next"
import { Popover } from "antd"
import {
  BookText,
  Globe,
  MicIcon,
  Search,
  Sparkles,
  Gauge,
  SlidersHorizontal,
  Users
} from "lucide-react"

export type ComposerToolbarRolePlayActions = {
  onOpenSystemPrompts?: () => void
  onOpenGenerationStyle?: () => void
  onOpenRolePlaySetup?: () => void
}

export type ComposerToolbarOverflowProps = {
  isProMode: boolean
  isConnectionReady: boolean
  contextToolsOpen: boolean
  onToggleKnowledgePanel: (tab: string) => void
  webSearch: boolean
  onToggleWebSearch: () => void
  hasWebSearch: boolean
  onOpenModelSettings: () => void
  hasDictation: boolean
  speechAvailable: boolean
  speechUsesServer: boolean
  isListening: boolean
  isServerDictating: boolean
  voiceChatEnabled: boolean
  onDictationToggle: () => void
  temporaryChat: boolean
  onFocusConnectionCard: () => void
  rolePlayActions?: ComposerToolbarRolePlayActions
}

/** Overflow menu item for toolbar actions */
const OverflowItem: React.FC<{
  icon: React.ReactNode
  label: string
  onClick?: () => void
  active?: boolean
  disabled?: boolean
}> = ({ icon, label, onClick, active, disabled }) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50 ${
      active ? "bg-primary/10 text-primaryStrong" : "text-text hover:text-text"
    }`}
  >
    <span className="flex h-4 w-4 items-center justify-center text-text-subtle">{icon}</span>
    <span>{label}</span>
  </button>
)

/**
 * Mobile overflow popover for the composer toolbar.
 * Collapses secondary actions (search, web search, dictation, settings)
 * into a single "more options" button on small screens.
 */
export const ComposerToolbarOverflow = React.memo(function ComposerToolbarOverflow(
  props: ComposerToolbarOverflowProps
) {
  const { t } = useTranslation(["playground", "common", "option"])
  const {
    isProMode,
    isConnectionReady,
    contextToolsOpen,
    onToggleKnowledgePanel,
    webSearch,
    onToggleWebSearch,
    hasWebSearch,
    onOpenModelSettings,
    hasDictation,
    speechAvailable,
    speechUsesServer,
    isListening,
    isServerDictating,
    voiceChatEnabled,
    onDictationToggle,
    temporaryChat,
    onFocusConnectionCard,
    rolePlayActions
  } = props

  const [overflowOpen, setOverflowOpen] = React.useState(false)

  const overflowItems = React.useMemo(() => {
    const items: React.ReactNode[] = []
    items.push(
      <OverflowItem
        key="search"
        icon={<Search className="w-3.5 h-3.5" />}
        label={
          contextToolsOpen
            ? (t("playground:composer.contextKnowledgeClose", "Close Search & Context") as string)
            : (t("playground:composer.contextKnowledge", "Search & Context") as string)
        }
        onClick={() => onToggleKnowledgePanel("search")}
        active={contextToolsOpen}
      />
    )
    if (rolePlayActions?.onOpenSystemPrompts) {
      items.push(
        <OverflowItem
          key="system-prompts"
          icon={<BookText className="w-3.5 h-3.5" />}
          label={t("playground:templates.button", "System prompts") as string}
          onClick={rolePlayActions.onOpenSystemPrompts}
        />
      )
    }
    if (rolePlayActions?.onOpenGenerationStyle) {
      items.push(
        <OverflowItem
          key="generation-style"
          icon={<Sparkles className="w-3.5 h-3.5" />}
          label={
            t("playground:presets.generationStyle", "Generation style") as string
          }
          onClick={rolePlayActions.onOpenGenerationStyle}
        />
      )
    }
    if (rolePlayActions?.onOpenRolePlaySetup) {
      items.push(
        <OverflowItem
          key="role-play-setup"
          icon={<Users className="w-3.5 h-3.5" />}
          label={
            t("playground:composer.rolePlaySetup", "Role-play setup") as string
          }
          onClick={rolePlayActions.onOpenRolePlaySetup}
        />
      )
    }
    if (hasWebSearch) {
      items.push(
        <OverflowItem
          key="web"
          icon={<Globe className="w-3.5 h-3.5" />}
          label={t("playground:tools.webSearch", "Web search") as string}
          onClick={onToggleWebSearch}
          active={webSearch}
        />
      )
    }
    if (hasDictation) {
      const isDictating = speechAvailable &&
        ((speechUsesServer && isServerDictating) ||
          (!speechUsesServer && isListening))
      items.push(
        <OverflowItem
          key="dictation"
          icon={<MicIcon className="w-3.5 h-3.5" />}
          label={
            isDictating
              ? (t("playground:actions.speechStop", "Stop dictation") as string)
              : (t("playground:actions.speechStart", "Start dictation") as string)
          }
          onClick={onDictationToggle}
          active={isDictating}
          disabled={!speechAvailable || voiceChatEnabled}
        />
      )
    }
    items.push(
      <OverflowItem
        key="settings"
        icon={<Gauge className="w-3.5 h-3.5" />}
        label={t("playground:composer.chatSettings", "Chat Settings") as string}
        onClick={onOpenModelSettings}
      />
    )
    if (isProMode && !temporaryChat && !isConnectionReady) {
      items.push(
        <OverflowItem
          key="connect"
          icon={<span className="w-3.5 h-3.5 text-primary">●</span>}
          label={t("playground:composer.persistence.connectToSave", "Connect your server to sync chats.") as string}
          onClick={onFocusConnectionCard}
        />
      )
    }
    return items
  }, [
    contextToolsOpen, onToggleKnowledgePanel, rolePlayActions, hasWebSearch, webSearch,
    onToggleWebSearch, hasDictation, speechAvailable,
    speechUsesServer, isServerDictating, isListening, voiceChatEnabled,
    onDictationToggle, onOpenModelSettings, isProMode, temporaryChat,
    isConnectionReady, onFocusConnectionCard, t
  ])

  if (overflowItems.length === 0) return null

  return (
    <Popover
      open={overflowOpen}
      onOpenChange={setOverflowOpen}
      trigger="click"
      placement="topRight"
      content={
        <div className="flex min-w-[200px] flex-col py-1" onClick={() => setOverflowOpen(false)}>
          {overflowItems}
        </div>
      }
    >
      <button
        type="button"
        onMouseDown={(event) => event.preventDefault()}
        aria-label={t("common:moreActions", "More options") as string}
        aria-expanded={overflowOpen}
        className="inline-flex h-9 w-9 items-center justify-center rounded-md text-text-muted transition hover:bg-surface2 hover:text-text"
      >
        <SlidersHorizontal className="h-4 w-4" />
      </button>
    </Popover>
  )
})
