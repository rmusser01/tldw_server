import { Tag } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"

interface ReviewProgressProps {
  dueCount: number
  reviewedCount: number
  deckName?: string
  availableNowCount?: number
  scheduledDueCount?: number
  newCount?: number
  learningCount?: number
}

export const ReviewProgress: React.FC<ReviewProgressProps> = ({
  dueCount,
  reviewedCount,
  deckName,
  availableNowCount,
  scheduledDueCount,
  newCount,
  learningCount
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
  if (typeof newCount === "number") {
    statusMessageParts.push(
      t("option:flashcards.newQueueCount", {
        defaultValue: "new: {{count}}",
        count: newCount
      })
    )
  }
  if (typeof learningCount === "number") {
    statusMessageParts.push(
      t("option:flashcards.learningQueueCount", {
        defaultValue: "learning: {{count}}",
        count: learningCount
      })
    )
  }
  if (typeof scheduledDueCount === "number") {
    statusMessageParts.push(
      t("option:flashcards.dueQueueCount", {
        defaultValue: "due: {{count}}",
        count: scheduledDueCount
      })
    )
  }
  const statusMessage = statusMessageParts.join(", ")

  if (dueCount === 0) return null

  return (
    <div
      className="mb-4 flex max-w-full flex-wrap items-center gap-2 rounded-lg bg-surface2 p-3 sm:gap-4"
      data-testid="flashcards-review-progress"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {/* Screen reader only status */}
      <span className="sr-only">{statusMessage}</span>

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-text" aria-hidden="true">
          {t("option:flashcards.studyQueue", { defaultValue: "Study queue" })}
        </span>
        <span className="text-2xl font-bold text-primary" aria-hidden="true">{remaining}</span>
        <span className="text-sm text-text-muted" aria-hidden="true">
          {t("option:flashcards.cardsRemaining", { defaultValue: "cards remaining" })}
        </span>
      </div>
      <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />
      <div className="text-sm text-text-muted" aria-hidden="true">
        <span className="font-medium text-text">{reviewedCount}</span>{" "}
        {t("option:flashcards.reviewed", { defaultValue: "reviewed" })}
      </div>
      {typeof availableNowCount === "number" && (
        <>
          <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.availableNowCount", {
              defaultValue: "Available now: {{count}}",
              count: availableNowCount
            })}
          </div>
        </>
      )}
      {typeof newCount === "number" && (
        <>
          <div className="h-8 w-px bg-border" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.newQueueCount", {
              defaultValue: "new: {{count}}",
              count: newCount
            })}
          </div>
        </>
      )}
      {typeof learningCount === "number" && (
        <>
          <div className="h-8 w-px bg-border" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.learningQueueCount", {
              defaultValue: "learning: {{count}}",
              count: learningCount
            })}
          </div>
        </>
      )}
      {typeof scheduledDueCount === "number" && (
        <>
          <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            {t("option:flashcards.dueQueueCount", {
              defaultValue: "due: {{count}}",
              count: scheduledDueCount
            })}
          </div>
        </>
      )}
      {remaining > 0 && (
        <>
          <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />
          <div className="text-sm text-text-muted" aria-hidden="true">
            ~{estimatedMinutes}{" "}
            {t("option:flashcards.minutesLeft", { defaultValue: "min left" })}
          </div>
        </>
      )}
      {deckName && (
        <Tag
          className="!m-0 min-w-0 max-w-full truncate sm:ml-auto"
          data-testid="flashcards-review-progress-deck-name"
        >
          {deckName}
        </Tag>
      )}
    </div>
  )
}
