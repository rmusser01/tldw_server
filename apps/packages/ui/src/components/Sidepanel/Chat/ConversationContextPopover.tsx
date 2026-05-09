import React from "react"
import { Popover, Tooltip } from "antd"
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  CircleDashed,
  Languages,
  Layers,
  Loader2,
  User2
} from "lucide-react"
import { useTranslation } from "react-i18next"

import { CharacterSelect } from "./CharacterSelect"
import {
  formatContextSourceLabel,
  resolveContextReadiness,
  summarizeConversationContextPieces
} from "./conversation-context-utils"
import type { ConversationContextComposition } from "@/types/conversation-context"
import type { ConversationContextCompositionStatus } from "@/hooks/chat/useConversationContextComposition"

type ConversationContextPopoverProps = {
  chatId?: string | null
  selectedCharacterId: string | null
  setSelectedCharacterId: (id: string | null) => void
  composition?: ConversationContextComposition | null
  compositionStatus: ConversationContextCompositionStatus
  disabled?: boolean
  className?: string
  iconClassName?: string
}

const triggerBaseClass =
  "flex min-w-[44px] sm:min-w-[104px] min-h-[44px] sm:min-h-0 h-9 items-center justify-center gap-2 rounded px-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"

const toneClass = {
  ready: "text-text-muted hover:bg-surface2 hover:text-text",
  partial: "bg-warning/10 text-warning hover:bg-warning/20",
  blocked: "bg-error/10 text-error hover:bg-error/20",
  loading: "text-text-muted hover:bg-surface2 hover:text-text"
} as const

const readinessIcon = (tone: keyof typeof toneClass) => {
  if (tone === "loading") return <Loader2 className="size-3.5 animate-spin" />
  if (tone === "blocked") return <AlertTriangle className="size-3.5" />
  if (tone === "partial") return <CircleDashed className="size-3.5" />
  return <CheckCircle2 className="size-3.5" />
}

const Section = ({
  icon,
  title,
  value,
  detail
}: {
  icon: React.ReactNode
  title: string
  value: string
  detail?: React.ReactNode
}) => (
  <div className="border-t border-border py-2 first:border-t-0">
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-2">
        <span className="text-text-subtle">{icon}</span>
        <span className="truncate text-sm font-medium text-text">{title}</span>
      </div>
      <span className="shrink-0 text-xs text-text-subtle">{value}</span>
    </div>
    {detail ? <div className="mt-2 text-xs text-text-subtle">{detail}</div> : null}
  </div>
)

export const ConversationContextPopover: React.FC<
  ConversationContextPopoverProps
> = ({
  chatId,
  selectedCharacterId,
  setSelectedCharacterId,
  composition,
  compositionStatus,
  disabled = false,
  className = "",
  iconClassName = "size-4"
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [open, setOpen] = React.useState(false)
  const summary = React.useMemo(
    () => summarizeConversationContextPieces(composition),
    [composition]
  )
  const readiness = React.useMemo(
    () => resolveContextReadiness({ composition, status: compositionStatus }),
    [composition, compositionStatus]
  )
  const hasOptionalContext =
    summary.worldbooks.total > 0 || summary.dictionaries.total > 0
  const worldbookValue =
    summary.worldbooks.total > 0
      ? `${summary.worldbooks.matched} ${t("sidepanel:conversationContext.matched", "matched")} / ${summary.worldbooks.total} ${t("sidepanel:conversationContext.configured", "configured")}`
      : t("sidepanel:conversationContext.none", "None")
  const dictionaryValue =
    summary.dictionaries.total > 0
      ? `${summary.dictionaries.active} ${t("sidepanel:conversationContext.active", "active")} / ${summary.dictionaries.total} ${t("sidepanel:conversationContext.configured", "configured")}`
      : t("sidepanel:conversationContext.none", "None")

  const previewSections = composition?.previewSections ?? []
  const sources = React.useMemo(() => {
    const labels = new Set<string>()
    for (const piece of composition?.pieces ?? []) {
      labels.add(formatContextSourceLabel(piece.source))
    }
    return Array.from(labels)
  }, [composition])

  const content = (
    <div
      className="w-[320px] max-w-[calc(100vw-32px)] p-3"
      onKeyDown={(event) => {
        if (event.key === "Escape") setOpen(false)
      }}
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Layers className="size-4 text-text-subtle" />
          <span className="truncate text-sm font-semibold text-text">
            {t("sidepanel:conversationContext.title", "Conversation context")}
          </span>
        </div>
        <span
          className={`shrink-0 rounded px-2 py-0.5 text-xs ${
            readiness.tone === "ready"
              ? "bg-success/10 text-success"
              : readiness.tone === "partial" || readiness.tone === "loading"
                ? "bg-warning/10 text-warning"
                : "bg-error/10 text-error"
          }`}
        >
          {readiness.label}
        </span>
      </div>

      <Section
        icon={<User2 className="size-3.5" />}
        title={t("sidepanel:conversationContext.character", "Character")}
        value={
          selectedCharacterId
            ? t("sidepanel:conversationContext.selected", "Selected")
            : t("sidepanel:conversationContext.none", "None")
        }
        detail={
          <CharacterSelect
            selectedCharacterId={selectedCharacterId}
            setSelectedCharacterId={setSelectedCharacterId}
            iconClassName="size-4"
            className="px-2 text-text-muted hover:text-text"
          />
        }
      />
      <Section
        icon={<BookOpen className="size-3.5" />}
        title={t("sidepanel:conversationContext.worldbooks", "Worldbooks")}
        value={worldbookValue}
        detail={
          summary.worldbooks.total > 0 ? (
            <span>
              {summary.worldbooks.explicit}{" "}
              {t("sidepanel:conversationContext.chatScoped", "chat-scoped")}
              {summary.worldbooks.inherited > 0
                ? `, ${summary.worldbooks.inherited} ${t(
                    "sidepanel:conversationContext.inherited",
                    "inherited"
                  )}`
                : ""}
            </span>
          ) : null
        }
      />
      <Section
        icon={<Languages className="size-3.5" />}
        title={t("sidepanel:conversationContext.dictionaries", "Dictionaries")}
        value={dictionaryValue}
        detail={
          summary.dictionaries.total > 0 ? (
            <span>
              {summary.dictionaries.explicit}{" "}
              {t("sidepanel:conversationContext.chatScoped", "chat-scoped")}
            </span>
          ) : null
        }
      />

      {!hasOptionalContext && !selectedCharacterId ? (
        <div className="border-t border-border py-2 text-xs text-text-subtle">
          {t(
            "sidepanel:conversationContext.noOptionalContext",
            "No optional context"
          )}
        </div>
      ) : null}

      {previewSections.length > 0 ? (
        <div className="border-t border-border pt-2">
          <div className="mb-1 text-xs font-medium text-text-subtle">
            {t("sidepanel:conversationContext.preview", "Preview")}
          </div>
          <div className="max-h-32 space-y-2 overflow-auto pr-1">
            {previewSections.map((section, index) => (
              <div key={`${section.name}-${index}`}>
                <div className="text-xs font-medium text-text">
                  {section.name}
                </div>
                <div className="line-clamp-3 whitespace-pre-wrap text-xs text-text-subtle">
                  {section.content}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {sources.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1 border-t border-border pt-2">
          {sources.map((source) => (
            <span
              key={source}
              className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] text-text-subtle"
            >
              {source}
            </span>
          ))}
        </div>
      ) : null}

      {!chatId ? (
        <div className="mt-2 border-t border-border pt-2 text-xs text-text-subtle">
          {t(
            "sidepanel:conversationContext.unsavedChat",
            "Edits are available after the chat is created."
          )}
        </div>
      ) : null}
    </div>
  )

  return (
    <Popover
      trigger="click"
      open={open}
      onOpenChange={setOpen}
      placement="topLeft"
      content={content}
    >
      <Tooltip
        title={t(
          "sidepanel:conversationContext.tooltip",
          "Conversation context"
        )}
      >
        <button
          type="button"
          data-testid="conversation-context-trigger"
          className={`${triggerBaseClass} ${toneClass[readiness.tone]} ${className}`}
          disabled={disabled}
          aria-disabled={disabled}
          aria-label={t(
            "sidepanel:conversationContext.tooltip",
            "Conversation context"
          )}
          aria-haspopup="dialog"
          aria-expanded={open}
        >
          <Layers className={iconClassName} />
          <span className="hidden truncate sm:inline">
            {t("sidepanel:conversationContext.shortLabel", "Context")}
          </span>
          <span className="flex h-4 w-4 shrink-0 items-center justify-center">
            {readinessIcon(readiness.tone)}
          </span>
        </button>
      </Tooltip>
    </Popover>
  )
}

export default ConversationContextPopover
