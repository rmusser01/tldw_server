import React from "react"
import { Button, Card, Input, Space, Tag, Typography } from "antd"
import { MinusCircleOutlined, PlusOutlined, ReloadOutlined } from "@ant-design/icons"

const { Text } = Typography

export type TopicBuilderRankReason = "weakness" | "adjacent" | "exploratory" | "candidate"

export type TopicBuilderTopic = {
  id: string
  label: string
  rankReason: TopicBuilderRankReason
  selected: boolean
  evidenceClass?: string | null
  isManual?: boolean
}

export interface TopicBuilderProps {
  topics: TopicBuilderTopic[]
  onAddTopic: () => void
  onRemoveTopic: (topicId: string) => void
  onRenameTopic: (topicId: string, nextLabel: string) => void
  onToggleTopic: (topicId: string) => void
  onResetTopics: () => void
}

const TITLE_CASE_LABELS: Record<TopicBuilderRankReason, string> = {
  weakness: "Weakness",
  adjacent: "Adjacent",
  exploratory: "Exploratory",
  candidate: "Candidate"
}

const titleizeEvidenceClass = (value: string | null | undefined): string => {
  const normalized = String(value || "").trim()
  if (!normalized) {
    return "Unknown"
  }
  return normalized
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

export const TopicBuilder: React.FC<TopicBuilderProps> = ({
  topics,
  onAddTopic,
  onRemoveTopic,
  onRenameTopic,
  onToggleTopic,
  onResetTopics
}) => {
  return (
    <Card size="small" title="Topic Builder">
      <Space direction="vertical" size={12} className="w-full">
        <div className="flex flex-wrap gap-2">
          <Button icon={<PlusOutlined />} onClick={onAddTopic}>
            Add topic
          </Button>
          <Button icon={<ReloadOutlined />} onClick={onResetTopics}>
            Reset topics
          </Button>
        </div>

        <Space direction="vertical" size={10} className="w-full">
          {topics.length === 0 ? (
            <Text type="secondary">No topics yet.</Text>
          ) : (
            topics.map((topic, index) => (
              <div key={topic.id} className="rounded border border-border bg-surface p-3">
                <Space direction="vertical" size={8} className="w-full">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag color={topic.rankReason === "exploratory" ? "blue" : "gold"}>
                      {TITLE_CASE_LABELS[topic.rankReason]}
                    </Tag>
                    <Tag color={topic.selected ? "green" : "default"}>
                      {topic.selected ? "Selected" : "Excluded"}
                    </Tag>
                    <Tag color="geekblue">
                      {`Evidence: ${titleizeEvidenceClass(topic.evidenceClass)}`}
                    </Tag>
                    {topic.isManual ? <Tag>Manual</Tag> : null}
                    <Text type="secondary">Topic {index + 1}</Text>
                  </div>
                  <Input
                    aria-label={`Topic ${index + 1}`}
                    value={topic.label}
                    onChange={(event) => {
                      onRenameTopic(topic.id, event.target.value)
                    }}
                    onBlur={(event) => {
                      const normalized = event.target.value.trim().replace(/\s+/g, " ")
                      if (normalized !== topic.label) {
                        onRenameTopic(topic.id, normalized)
                      }
                    }}
                  />
                  <div className="flex justify-between gap-2">
                    <Button
                      size="small"
                      type={topic.selected ? "primary" : "default"}
                      aria-label={`Toggle selection for topic ${index + 1}`}
                      aria-pressed={topic.selected}
                      onClick={() => onToggleTopic(topic.id)}
                    >
                      {topic.selected ? "Selected" : "Excluded"}
                    </Button>
                    <Button
                      size="small"
                      icon={<MinusCircleOutlined />}
                      aria-label={`Remove topic ${index + 1}`}
                      onClick={() => onRemoveTopic(topic.id)}
                    >
                      Remove
                    </Button>
                  </div>
                </Space>
              </div>
            ))
          )}
        </Space>
      </Space>
    </Card>
  )
}

export default TopicBuilder
