import React from "react"
import { Button, Card, List, Tag, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { EmptyState } from "@/components/ui/feedback/EmptyState"
import { LoadingState } from "@/components/ui/feedback/LoadingState"
import type { Deck, FlashcardReviewSessionSummary } from "@/services/flashcards"
import { useRecentFlashcardReviewSessionsQuery } from "../hooks"
import { formatFlashcardAbsoluteDateTime } from "../utils/date-display"

const { Text } = Typography

export interface RecentStudySessionsProps {
  deckId?: number | null
  decks?: Deck[]
  selectedSessionId?: number | null
  onOpenSession: (sessionId: number) => void
  isActive: boolean
}

const getSessionModeLabel = (
  session: FlashcardReviewSessionSummary,
  t: ReturnType<typeof useTranslation>["t"]
) => {
  const mode = String(session.review_mode || "").toLowerCase()
  const scope = String(session.scope_key || "").toLowerCase()

  if (mode === "cram" || scope.startsWith("cram:")) {
    return t("option:flashcards.recentStudySessionsModeCram", {
      defaultValue: "Cram review"
    })
  }
  if (mode === "due" || scope.startsWith("due:")) {
    return t("option:flashcards.recentStudySessionsModeDue", {
      defaultValue: "Due review"
    })
  }
  if (mode === "deck" || scope.startsWith("deck:")) {
    return t("option:flashcards.recentStudySessionsModeDeck", {
      defaultValue: "Deck review"
    })
  }
  if (mode === "tag" || scope.startsWith("tag:")) {
    return t("option:flashcards.recentStudySessionsModeTag", {
      defaultValue: "Tag review"
    })
  }
  if (mode === "study_pack" || scope.startsWith("study_pack:")) {
    return t("option:flashcards.recentStudySessionsModeStudyPack", {
      defaultValue: "Study pack review"
    })
  }

  return t("option:flashcards.recentStudySessionsModeGeneric", {
    defaultValue: "Review session"
  })
}

const getReviewedCountLabel = (
  session: FlashcardReviewSessionSummary,
  t: ReturnType<typeof useTranslation>["t"]
) => {
  const count = session.cards_reviewed
  if (typeof count !== "number" || !Number.isFinite(count) || count < 0) {
    return null
  }

  return t("option:flashcards.recentStudySessionsReviewedCount", {
    defaultValue: "{{count}} cards reviewed",
    count
  })
}

/**
 * Shows recently completed flashcard review sessions and lets the user reopen one.
 */
export const RecentStudySessions: React.FC<RecentStudySessionsProps> = ({
  deckId,
  decks = [],
  selectedSessionId,
  onOpenSession,
  isActive
}) => {
  const { t } = useTranslation(["option"])
  const recentSessionsQuery = useRecentFlashcardReviewSessionsQuery(
    {
      deckId,
      status: "completed",
      limit: 8
    },
    {
      enabled: isActive
    }
  )

  const sessions = recentSessionsQuery.data ?? []
  const deckNamesById = React.useMemo(() => {
    const names = new Map<number, string>()
    for (const deck of decks) {
      if (deck.name.trim()) names.set(deck.id, deck.name)
    }
    return names
  }, [decks])
  const errorMessage =
    recentSessionsQuery.error instanceof Error
      ? recentSessionsQuery.error.message
      : t("option:flashcards.recentStudySessionsLoadFailedFallback", {
          defaultValue: "Failed to load recent study sessions."
        })

  return (
    <Card
      size="small"
      title={t("option:flashcards.recentStudySessionsTitle", {
        defaultValue: "Recent study sessions"
      })}
    >
      {recentSessionsQuery.isLoading ? (
        <LoadingState
          mode="spinner"
          size="sm"
          label={t("option:flashcards.recentStudySessionsLoading", {
            defaultValue: "Loading recent study sessions..."
          })}
        />
      ) : recentSessionsQuery.isError ? (
        <EmptyState
          variant="inline"
          size="sm"
          title={t("option:flashcards.recentStudySessionsLoadFailed", {
            defaultValue: "Failed to load recent study sessions"
          })}
          description={errorMessage}
          primaryAction={{
            label: t("option:flashcards.recentStudySessionsRetry", {
              defaultValue: "Retry"
            }),
            onClick: () => void recentSessionsQuery.refetch()
          }}
        />
      ) : sessions.length === 0 ? (
        <EmptyState
          variant="inline"
          size="sm"
          title={t("option:flashcards.recentStudySessionsEmpty", {
            defaultValue: "No completed study sessions yet."
          })}
        />
      ) : (
        <List
          dataSource={sessions}
          renderItem={(session) => {
            const isSelected = selectedSessionId === session.id
            const deckName =
              session.deck_id == null ? null : deckNamesById.get(session.deck_id) ?? null
            const deckLabel =
              deckName ??
              (session.deck_id != null
                ? t("option:flashcards.recentStudySessionsDeckFallback", {
                    defaultValue: "Deck {{id}}",
                    id: session.deck_id
                  })
                : t("option:flashcards.recentStudySessionsAllDecks", {
                    defaultValue: "All decks"
                  }))
            const modeLabel = getSessionModeLabel(session, t)
            const reviewedCountLabel = getReviewedCountLabel(session, t)
            const completedAtLabel = formatFlashcardAbsoluteDateTime(
              session.completed_at ?? session.last_activity_at
            )

            return (
              <List.Item key={session.id}>
                <div className="flex w-full flex-col gap-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Text strong>{deckLabel}</Text>
                    <Button
                      type={isSelected ? "primary" : "default"}
                      onClick={() => onOpenSession(session.id)}
                    >
                      {isSelected
                        ? t("option:flashcards.recentStudySessionsViewingCompleted", {
                            defaultValue: "Viewing completed session"
                          })
                        : t("option:flashcards.recentStudySessionsViewCompleted", {
                            defaultValue: "View completed session"
                          })}
                    </Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag color="green">
                      {t("option:flashcards.recentStudySessionsCompletedTag", {
                        defaultValue: "Completed"
                      })}
                    </Tag>
                    <Tag>{modeLabel}</Tag>
                    {reviewedCountLabel ? <Tag>{reviewedCountLabel}</Tag> : null}
                  </div>
                  {completedAtLabel ? (
                    <Text type="secondary" className="text-xs">
                      {t("option:flashcards.recentStudySessionsCompletedAt", {
                        defaultValue: "Completed {{time}}",
                        time: completedAtLabel
                      })}
                    </Text>
                  ) : null}
                </div>
              </List.Item>
            )
          }}
        />
      )}
    </Card>
  )
}

export default RecentStudySessions
