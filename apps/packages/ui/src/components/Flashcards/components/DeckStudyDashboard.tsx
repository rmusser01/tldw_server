import React from "react"
import { Button, Typography } from "antd"
import { CalendarClock, Download, Flame, Pencil, Play } from "lucide-react"
import { useTranslation } from "react-i18next"

import type { Deck, FlashcardDeckProgress } from "@/services/flashcards"

const { Text } = Typography

export type DeckStudyDashboardProps = {
  decks?: Deck[] | null
  deckProgress?: FlashcardDeckProgress[] | null
  selectedDeckId?: number | null
  isLoading?: boolean
  onReviewDeck: (deckId: number) => void
  onCramDeck: (deckId: number) => void
  onManageDeck: (deckId: number) => void
  onOpenScheduler?: (deckId: number) => void
  onExportDeck?: (deckId: number) => void
}

type DeckStudyRow = FlashcardDeckProgress & {
  deckName: string
  readyCount: number
}

const asCount = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0

const buildRows = (
  decks: Deck[] | null | undefined,
  deckProgress: FlashcardDeckProgress[] | null | undefined
): DeckStudyRow[] => {
  const deckNamesById = new Map((decks ?? []).map((deck) => [deck.id, deck.name]))
  return [...(deckProgress ?? [])]
    .filter((deck) => asCount(deck.total) > 0)
    .map((deck) => {
      const deckName =
        deckNamesById.get(deck.deck_id) ??
        (typeof deck.deck_name === "string" && deck.deck_name.trim()
          ? deck.deck_name
          : `Deck ${deck.deck_id}`)
      return {
        ...deck,
        new: asCount(deck.new),
        learning: asCount(deck.learning),
        due: asCount(deck.due),
        mature: asCount(deck.mature),
        total: asCount(deck.total),
        deckName,
        readyCount: asCount(deck.due) + asCount(deck.learning) + asCount(deck.new)
      }
    })
    .sort((left, right) => {
      if (right.readyCount !== left.readyCount) return right.readyCount - left.readyCount
      if (right.due !== left.due) return right.due - left.due
      if (right.total !== left.total) return right.total - left.total
      return left.deckName.localeCompare(right.deckName)
    })
}

const CountPill: React.FC<{ label: string }> = ({ label }) => (
  <span className="inline-flex h-6 items-center rounded border border-border/70 px-2 text-xs text-text-secondary">
    {label}
  </span>
)

/**
 * Compact deck-level launch surface for experienced flashcard study workflows.
 */
export const DeckStudyDashboard: React.FC<DeckStudyDashboardProps> = ({
  decks,
  deckProgress,
  selectedDeckId,
  isLoading = false,
  onReviewDeck,
  onCramDeck,
  onManageDeck,
  onOpenScheduler,
  onExportDeck
}) => {
  const { t } = useTranslation(["option"])
  const rows = React.useMemo(() => buildRows(decks, deckProgress), [decks, deckProgress])

  if (isLoading && rows.length === 0) {
    return (
      <section
        className="mb-4 rounded border border-border/70 bg-surface2/40 p-3"
        data-testid="flashcards-deck-study-dashboard"
      >
        <Text type="secondary">
          {t("option:flashcards.deckStudyDashboardLoading", {
            defaultValue: "Loading deck study options..."
          })}
        </Text>
      </section>
    )
  }

  if (rows.length === 0) return null

  return (
    <section
      className="mb-4 rounded border border-border/70 bg-surface2/40 p-3"
      data-testid="flashcards-deck-study-dashboard"
    >
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <Text strong className="block">
            {t("option:flashcards.deckStudyDashboardTitle", {
              defaultValue: "Deck study dashboard"
            })}
          </Text>
          <Text type="secondary" className="text-sm">
            {t("option:flashcards.deckStudyDashboardDescription", {
              defaultValue: "Choose the right deck and study mode without opening each deck first."
            })}
          </Text>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        {rows.map((row) => {
          const reviewedShare =
            row.total > 0 ? Math.max(0, Math.min(100, ((row.total - row.new) / row.total) * 100)) : 0
          const isSelected = selectedDeckId === row.deck_id
          return (
            <div
              key={row.deck_id}
              className={`rounded border bg-surface p-3 ${
                isSelected ? "border-primary" : "border-border/70"
              }`}
              data-testid={`flashcards-deck-study-dashboard-row-${row.deck_id}`}
              aria-current={isSelected ? "true" : undefined}
            >
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Text strong className="truncate">
                      {row.deckName}
                    </Text>
                    {isSelected && (
                      <span className="rounded bg-primary/10 px-2 py-0.5 text-xs text-primary">
                        {t("option:flashcards.deckStudyDashboardSelected", {
                          defaultValue: "Selected"
                        })}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    <CountPill
                      label={t("option:flashcards.deckDue", {
                        defaultValue: "Due: {{count}}",
                        count: row.due
                      })}
                    />
                    <CountPill
                      label={t("option:flashcards.deckNew", {
                        defaultValue: "New: {{count}}",
                        count: row.new
                      })}
                    />
                    <CountPill
                      label={t("option:flashcards.deckLearning", {
                        defaultValue: "Learning: {{count}}",
                        count: row.learning
                      })}
                    />
                    <CountPill
                      label={t("option:flashcards.deckMature", {
                        defaultValue: "Mature: {{count}}",
                        count: row.mature
                      })}
                    />
                  </div>
                  <div
                    className="mt-2 h-1.5 overflow-hidden rounded bg-surface3"
                    aria-label={t("option:flashcards.deckStudyDashboardProgress", {
                      defaultValue: "{{deckName}} reviewed share",
                      deckName: row.deckName
                    })}
                  >
                    <div
                      className="h-full rounded bg-primary"
                      style={{ width: `${reviewedShare}%` }}
                    />
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5 lg:justify-end">
                  <Button
                    type="primary"
                    size="small"
                    icon={<Play className="size-4" aria-hidden="true" />}
                    disabled={row.readyCount === 0}
                    onClick={() => onReviewDeck(row.deck_id)}
                  >
                    {row.readyCount > 0
                      ? t("option:flashcards.deckStudyDashboardReviewReady", {
                          defaultValue: "Review {{count}} ready",
                          count: row.readyCount
                        })
                      : t("option:flashcards.deckStudyDashboardCaughtUp", {
                          defaultValue: "Caught up"
                        })}
                  </Button>
                  <Button
                    size="small"
                    icon={<Flame className="size-4" aria-hidden="true" />}
                    onClick={() => onCramDeck(row.deck_id)}
                  >
                    {t("option:flashcards.reviewModeCram", { defaultValue: "Cram" })}
                  </Button>
                  <Button
                    size="small"
                    icon={<Pencil className="size-4" aria-hidden="true" />}
                    onClick={() => onManageDeck(row.deck_id)}
                  >
                    {t("option:flashcards.edit", { defaultValue: "Edit" })}
                  </Button>
                  {onOpenScheduler && (
                    <Button
                      size="small"
                      icon={<CalendarClock className="size-4" aria-hidden="true" />}
                      onClick={() => onOpenScheduler(row.deck_id)}
                    >
                      {t("option:flashcards.tabScheduler", { defaultValue: "Scheduler" })}
                    </Button>
                  )}
                  {onExportDeck && (
                    <Button
                      size="small"
                      icon={<Download className="size-4" aria-hidden="true" />}
                      onClick={() => onExportDeck(row.deck_id)}
                    >
                      {t("option:flashcards.export", { defaultValue: "Export" })}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default DeckStudyDashboard
