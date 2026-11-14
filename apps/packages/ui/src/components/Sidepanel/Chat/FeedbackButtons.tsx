import React from "react"
import { Tooltip } from "antd"
import { MessageSquareText, ThumbsDown, ThumbsUp } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { FeedbackThumb } from "@/store/feedback"

type Props = {
  selected?: FeedbackThumb
  disabled?: boolean
  isSubmitting?: boolean
  onThumbUp?: () => void
  onThumbDown?: () => void
  onOpenDetails?: () => void
  showThanks?: boolean
  disabledReason?: string
  compact?: boolean
  className?: string
}

const baseButton =
  "inline-flex items-center justify-center rounded-full border transition-colors"

const getThumbClass = (active: boolean, kind: "up" | "down") => {
  if (active) {
    return kind === "up"
      ? "border-success/30 bg-success/10 text-success"
      : "border-danger/30 bg-danger/10 text-danger"
  }
  return "border-border text-text-muted hover:bg-surface2"
}

export const FeedbackButtons = ({
  selected = null,
  disabled = false,
  isSubmitting = false,
  onThumbUp,
  onThumbDown,
  onOpenDetails,
  showThanks = false,
  disabledReason,
  compact = false,
  className
}: Props) => {
  const { t } = useTranslation("playground")
  const label = t("feedback.prompt", "Was this helpful?")
  const disabledHint = disabledReason
    ? disabledReason
    : t(
        "feedback.disabled",
        "Feedback is available after the response finishes."
      )

  const isDisabled = disabled || isSubmitting
  const buttonSizeClass = "h-11 w-11 min-h-[44px] min-w-[44px]"
  const containerClass = compact
    ? "flex flex-wrap items-center gap-2 text-[11px] text-text-muted"
    : "mt-2 flex flex-wrap items-center gap-2 text-caption text-text-muted"

  return (
    <div className={[containerClass, className].filter(Boolean).join(" ")}>
      <span>{label}</span>
      <div className="flex items-center gap-1">
        <Tooltip
          title={
            isDisabled
              ? disabledHint
              : t("feedback.thumbsUp", "Helpful")
          }>
          <button
            type="button"
            aria-label={t("feedback.thumbsUp", "Helpful")}
            aria-pressed={selected === "up"}
            disabled={isDisabled}
            onClick={onThumbUp}
            title={
              isDisabled
                ? disabledHint
                : t("feedback.thumbsUp", "Helpful")
            }
            className={`${baseButton} ${buttonSizeClass} ${getThumbClass(
              selected === "up",
              "up"
            )} ${isDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <ThumbsUp className="h-3 w-3" />
          </button>
        </Tooltip>
        <Tooltip
          title={
            isDisabled
              ? disabledHint
              : t("feedback.thumbsDown", "Not helpful")
          }>
          <button
            type="button"
            aria-label={t("feedback.thumbsDown", "Not helpful")}
            aria-pressed={selected === "down"}
            disabled={isDisabled}
            onClick={onThumbDown}
            title={
              isDisabled
                ? disabledHint
                : t("feedback.thumbsDown", "Not helpful")
            }
            className={`${baseButton} ${buttonSizeClass} ${getThumbClass(
              selected === "down",
              "down"
            )} ${isDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <ThumbsDown className="h-3 w-3" />
          </button>
        </Tooltip>
        <Tooltip
          title={
            isDisabled ? disabledHint : t("feedback.more", "More feedback")
          }>
          <button
            type="button"
            aria-label={t("feedback.more", "More feedback")}
            disabled={isDisabled}
            onClick={onOpenDetails}
            title={
              isDisabled ? disabledHint : t("feedback.more", "More feedback")
            }
            className={`${baseButton} ${buttonSizeClass} border-border text-text-muted hover:bg-surface2 ${
              isDisabled ? "cursor-not-allowed opacity-50" : ""
            }`}>
            {/* Not an ellipsis: the reply's own "More actions" is the only one. */}
            <MessageSquareText className="h-3 w-3" />
          </button>
        </Tooltip>
      </div>
      {showThanks && (
        <span className="text-[11px] font-medium text-success">
          {t("feedback.thanks", "Thanks for the feedback")}
        </span>
      )}
    </div>
  )
}
