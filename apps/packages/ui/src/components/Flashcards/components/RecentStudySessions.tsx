import React from "react"
import { Button, Card, List, Space, Tag, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { EmptyState } from "@/components/ui/feedback/EmptyState"
import { LoadingState } from "@/components/ui/feedback/LoadingState"
import { useRecentFlashcardReviewSessionsQuery } from "../hooks"

const { Text } = Typography

export interface RecentStudySessionsProps {
  deckId?: number | null
  selectedSessionId?: number | null
  onOpenSession: (sessionId: number) => void
  isActive: boolean
}

/**
 * Shows recently completed flashcard review sessions and lets the user reopen one.
 */
export const RecentStudySessions: React.FC<RecentStudySessionsProps> = ({
  deckId,
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
            return (
              <List.Item key={session.id}>
                <Space direction="vertical" size={6} className="w-full">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag color="green">
                      {t("option:flashcards.recentStudySessionsCompletedTag", {
                        defaultValue: "Completed"
                      })}
                    </Tag>
                    <Tag>
                      {t("option:flashcards.recentStudySessionsSessionNumber", {
                        defaultValue: "Session #{{id}}",
                        id: session.id
                      })}
                    </Tag>
                    {session.deck_id != null ? (
                      <Tag>
                        {t("option:flashcards.recentStudySessionsDeckNumber", {
                          defaultValue: "Deck {{id}}",
                          id: session.deck_id
                        })}
                      </Tag>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Text type="secondary" className="text-xs">
                      {session.scope_key}
                    </Text>
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
                </Space>
              </List.Item>
            )
          }}
        />
      )}
    </Card>
  )
}

export default RecentStudySessions
