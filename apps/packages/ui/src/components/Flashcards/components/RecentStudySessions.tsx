import React from "react"
import { Button, Card, Empty, List, Space, Tag, Typography } from "antd"

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
  const recentSessionsQuery =
    useRecentFlashcardReviewSessionsQuery(
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

  return (
    <Card size="small" title="Recent study sessions">
      {recentSessionsQuery.isLoading ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Loading recent study sessions..." />
      ) : sessions.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No completed study sessions yet." />
      ) : (
        <List
          dataSource={sessions}
          renderItem={(session) => {
            const isSelected = selectedSessionId === session.id
            return (
              <List.Item key={session.id}>
                <Space direction="vertical" size={6} className="w-full">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag color="green">Completed</Tag>
                    <Tag>Session #{session.id}</Tag>
                    {session.deck_id != null ? <Tag>Deck {session.deck_id}</Tag> : null}
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
                        ? "Viewing snapshot"
                        : `Reopen snapshot for session ${session.id}`}
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
