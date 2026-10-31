import React from "react"
import { Tooltip } from "antd"
import { ThumbsDown, ThumbsUp } from "lucide-react"
import { useTranslation } from "react-i18next"
import { MessageSource } from "@/components/Common/Playground/MessageSource"
import type { FeedbackThumb } from "@/store/feedback"

type Props = {
  source: any
  sourceKey: string
  sourceIndex?: number
  pinnedState?: "active" | "inactive" | null
  selected?: FeedbackThumb
  disabled?: boolean
  onRate?: (sourceKey: string, source: any, thumb: FeedbackThumb) => void
  onSourceClick?: (source: any) => void
  onTrackClick?: (source: any, index?: number) => void
  onTrackCitation?: (source: any, index?: number) => void
  onTrackDwell?: (source: any, dwellMs: number, index?: number) => void
  onAskWithSource?: (source: any) => void
  onOpenKnowledgePanel?: () => void
}

const buttonBase =
  "inline-flex h-5 w-5 items-center justify-center rounded-full border text-[10px] transition-colors"

export const SourceFeedback = ({
  source,
  sourceKey,
  sourceIndex,
  pinnedState = null,
  selected = null,
  disabled = false,
  onRate,
  onSourceClick,
  onTrackClick,
  onTrackCitation,
  onTrackDwell,
  onAskWithSource,
  onOpenKnowledgePanel
}: Props) => {
  const { t } = useTranslation("playground")

  const handleSourceClick = React.useCallback(
    (payload: any) => {
      onTrackClick?.(payload, sourceIndex)
      onSourceClick?.(payload)
    },
    [onSourceClick, onTrackClick, sourceIndex]
  )

  const handleSourceNavigate = React.useCallback(
    (payload: any) => {
      onTrackClick?.(payload, sourceIndex)
      onTrackCitation?.(payload, sourceIndex)
    },
    [onTrackCitation, onTrackClick, sourceIndex]
  )

  const handleSourceDwell = React.useCallback(
    (payload: any, dwellMs: number) => {
      onTrackDwell?.(payload, dwellMs, sourceIndex)
    },
    [onTrackDwell, sourceIndex]
  )

  const isDisabled = disabled

  return (
    <div data-source-feedback className="flex items-center gap-2">
      <MessageSource
        source={source}
        onSourceClick={handleSourceClick}
        onSourceNavigate={handleSourceNavigate}
        onSourceDwell={handleSourceDwell}
        onOpenKnowledgePanel={onOpenKnowledgePanel}
      />
      {pinnedState && (
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
            pinnedState === "active"
              ? "border-success/40 bg-success/10 text-success"
              : "border-border bg-surface2 text-text-muted"
          }`}
          title={
            pinnedState === "active"
              ? t("sources.pinnedUsed", "Pinned and used in this answer")
              : t("sources.pinnedNotUsed", "Pinned but not used in this answer")
          }
        >
          {pinnedState === "active"
            ? t("sources.pinnedUsedBadge", "Pinned: used")
            : t("sources.pinnedIdleBadge", "Pinned: not used")}
        </span>
      )}
      {onAskWithSource && (
        <button
          type="button"
          onClick={() => onAskWithSource(source)}
          className="rounded-md border border-border bg-surface2 px-2 py-0.5 text-[10px] font-medium text-text-subtle transition hover:bg-surface hover:text-text"
        >
          {t("sources.askWithSource", "Ask with this source")}
        </button>
      )}
      <div className="flex items-center gap-1">
        <Tooltip title={t("feedback.sourceHelpful", "Helpful source")}>
          <button
            type="button"
            aria-label={t("feedback.sourceHelpful", "Helpful source")}
            aria-pressed={selected === "up"}
            disabled={isDisabled}
            onClick={() => onRate?.(sourceKey, source, "up")}
            title={t("feedback.sourceHelpful", "Helpful source")}
            className={`${buttonBase} ${
              selected === "up"
                ? "border-success/30 bg-success/10 text-success"
                : "border-border text-text-subtle hover:bg-surface2"
            } ${isDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <ThumbsUp className="h-3 w-3" />
          </button>
        </Tooltip>
        <Tooltip title={t("feedback.sourceUnhelpful", "Unhelpful source")}>
          <button
            type="button"
            aria-label={t("feedback.sourceUnhelpful", "Unhelpful source")}
            aria-pressed={selected === "down"}
            disabled={isDisabled}
            onClick={() => onRate?.(sourceKey, source, "down")}
            title={t("feedback.sourceUnhelpful", "Unhelpful source")}
            className={`${buttonBase} ${
              selected === "down"
                ? "border-danger/30 bg-danger/10 text-danger"
                : "border-border text-text-subtle hover:bg-surface2"
            } ${isDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <ThumbsDown className="h-3 w-3" />
          </button>
        </Tooltip>
      </div>
    </div>
  )
}
