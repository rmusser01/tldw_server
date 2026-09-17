import React from "react"
import { Card, Progress, Skeleton, Tag, Typography } from "antd"
import { useTranslation } from "react-i18next"
import type { FlashcardAnalyticsSummary } from "@/services/flashcards"

const { Text } = Typography

interface ReviewAnalyticsSummaryProps {
  summary: FlashcardAnalyticsSummary | null | undefined
  isLoading?: boolean
  selectedDeckId?: number | null
}

const formatPercent = (value?: number | null): string => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—"
  return `${value.toFixed(1)}%`
}

const formatAnswerTime = (value?: number | null): string => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—"
  return `${(value / 1000).toFixed(1)}s`
}

export const ReviewAnalyticsSummary: React.FC<ReviewAnalyticsSummaryProps> = ({
  summary,
  isLoading = false,
  selectedDeckId
}) => {
  const { t } = useTranslation(["option"])

  if (isLoading && !summary) {
    return (
      <Card className="mb-4">
        <Skeleton active paragraph={{ rows: 4 }} />
      </Card>
    )
  }

  if (!summary) return null

  const deckRows = [...(summary.decks || [])]
    .filter((deck) => deck.total > 0)
    .sort((a, b) => {
      if (a.deck_id === selectedDeckId) return -1
      if (b.deck_id === selectedDeckId) return 1
      if (b.due !== a.due) return b.due - a.due
      return a.deck_name.localeCompare(b.deck_name)
    })

  return (
    <Card className="mb-4" data-testid="flashcards-review-analytics-summary">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <div className="rounded border border-border p-2">
          <Text type="secondary" className="block text-xs">
            {t("option:flashcards.reviewedToday", { defaultValue: "Reviewed today" })}
          </Text>
          <Text strong className="text-lg">
            {summary.reviewed_today}
          </Text>
        </div>
        <div className="rounded border border-border p-2">
          <Text type="secondary" className="block text-xs">
            {t("option:flashcards.retentionRate", { defaultValue: "Retention rate" })}
          </Text>
          <Text strong className="text-lg">
            {formatPercent(summary.retention_rate_today)}
          </Text>
        </div>
        <div className="rounded border border-border p-2">
          <Text type="secondary" className="block text-xs">
            {t("option:flashcards.lapseRate", { defaultValue: "Lapse rate" })}
          </Text>
          <Text strong className="text-lg">
            {formatPercent(summary.lapse_rate_today)}
          </Text>
        </div>
        <div className="rounded border border-border p-2">
          <Text type="secondary" className="block text-xs">
            {t("option:flashcards.averageAnswerTime", {
              defaultValue: "Avg answer time"
            })}
          </Text>
          <Text strong className="text-lg">
            {formatAnswerTime(summary.avg_answer_time_ms_today)}
          </Text>
        </div>
        <div className="rounded border border-border p-2">
          <Text type="secondary" className="block text-xs">
            {t("option:flashcards.studyStreak", { defaultValue: "Study streak" })}
          </Text>
          <Text strong className="text-lg">
            {t("option:flashcards.studyStreakDays", {
              defaultValue: "{count, plural, one {# day} other {# days}}",
              count: summary.study_streak_days
            })}
          </Text>
        </div>
      </div>

      {deckRows.length > 0 && (
        <div className="mt-4 space-y-2">
          <Text strong>
            {t("option:flashcards.deckProgress", { defaultValue: "Deck progress" })}
          </Text>
          {deckRows.slice(0, 8).map((deck) => {
            const reviewedShare =
              deck.total > 0 ? ((deck.total - deck.new) / deck.total) * 100 : 0
            return (
              <div
                key={deck.deck_id}
                className={`rounded border p-2 ${deck.deck_id === selectedDeckId ? "border-primary" : "border-border"}`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Text strong>{deck.deck_name}</Text>
                  <div className="flex flex-wrap gap-1">
                    <Tag>{t("option:flashcards.deckTotal", { defaultValue: "Total: {{count}}", count: deck.total })}</Tag>
                    <Tag color="blue">{t("option:flashcards.deckNew", { defaultValue: "New: {{count}}", count: deck.new })}</Tag>
                    <Tag color="gold">{t("option:flashcards.deckLearning", { defaultValue: "Learning: {{count}}", count: deck.learning })}</Tag>
                    <Tag color="red">{t("option:flashcards.deckDue", { defaultValue: "Due: {{count}}", count: deck.due })}</Tag>
                    <Tag color="green">{t("option:flashcards.deckMature", { defaultValue: "Mature: {{count}}", count: deck.mature })}</Tag>
                  </div>
                </div>
                <Progress
                  className="mt-1"
                  percent={Math.max(0, Math.min(100, reviewedShare))}
                  showInfo={false}
                  size="small"
                />
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}

export default ReviewAnalyticsSummary
