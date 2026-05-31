import { Tag } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"

interface ReviewProgressProps {
  dueCount: number
  reviewedCount: number
  deckName?: string
  availableNowCount?: number
  scheduledDueCount?: number
}

export const ReviewProgress: React.FC<ReviewProgressProps> = ({
  dueCount,
  reviewedCount,
  deckName,
  availableNowCount,
  scheduledDueCount
}) => {
  const { t } = useTranslation(["option"])
  const remaining = Math.max(0, dueCount - reviewedCount)
  // Average time per card in seconds (based on typical flashcard review time of 10-20s)
  const avgTimePerCard = 15
  const estimatedMinutes = Math.ceil((remaining * avgTimePerCard) / 60)

  const statusMessageParts = [t("option:flashcards.progressStatus", {
    defaultValue: "{{remaining}} cards remaining, {{reviewed}} reviewed",
    remaining,
    reviewed: reviewedCount
  })]
  if (typeof availableNowCount === "number") {
    statusMessageParts.push(
      t("option:flashcards.availableNowCount", {
        defaultValue: "Available now: {{count}}",
        count: availableNowCount
      })
    )
  }
  if (typeof scheduledDueCount === "number") {
    statusMessageParts.push(
      t("option:flashcards.scheduledDueCount", {
        defaultValue: "Scheduled due: {{count}}",
        count: scheduledDueCount
      })
    )
  }
  const statusMessage = statusMessageParts.join(", ")

  if (dueCount === 0) return null

  return (
    <div
      className="flex items-center gap-4 p-3 rounded-lg bg-surface2 mb-4"
      data-testid="flashcards-review-progress"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {/* Screen reader only status */}
      <span className="sr-only">{statusMessage}</span>

      <div className="flex items-center gap-2">
        <span className="text-2xl font-bold text-primary" aria-hidden="true">{remaining}</span>
        <span className="text-sm text-text-muted" aria-hidden="true">
          {t("option:flashcards.cardsRemaining", { defaultValue: "cards remaining" })}
        </span>
      </div>
      <div className="h-8 w-px bg-border" aria-hidden="true" />
      <div className="text-sm text-text-muted" aria-hidden="true">
        <span className="font-medium text-text">{reviewedCount}</span>{" "}
        {t("option:flashcards.reviewed", { defaultValue: "reviewed" })}
      </div>
      {typeof availableNowCount === "number" && (
        <>
          <div className="h-8 w-px bg-border" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.availableNowCount", {
              defaultValue: "Available now: {{count}}",
              count: availableNowCount
            })}
          </div>
        </>
      )}
      {typeof scheduledDueCount === "number" && (
        <>
          <div className="h-8 w-px bg-border" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.scheduledDueCount", {
              defaultValue: "Scheduled due: {{count}}",
              count: scheduledDueCount
            })}
          </div>
        </>
      )}
      {remaining > 0 && (
        <>
          <div className="h-8 w-px bg-border" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            ~{estimatedMinutes}{" "}
            {t("option:flashcards.minutesLeft", { defaultValue: "min left" })}
          </div>
        </>
      )}
      {deckName && <Tag className="ml-auto">{deckName}</Tag>}
    </div>
  )
}
