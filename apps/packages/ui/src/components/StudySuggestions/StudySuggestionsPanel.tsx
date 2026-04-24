import React from "react"
import { Alert, Button, Card, Descriptions, Empty, Space, Tag, Typography } from "antd"
import { ReloadOutlined } from "@ant-design/icons"

import { TopicBuilder, type TopicBuilderRankReason, type TopicBuilderTopic } from "./TopicBuilder"
import { useStudySuggestions } from "./hooks/useStudySuggestions"
import type {
  SuggestionAnchorType,
  StudySuggestionActionRequest,
  StudySuggestionActionResponse,
  StudySuggestionSnapshotResponse
} from "@/services/studySuggestions"

const { Text } = Typography

const normalizeLabel = (value: string): string => value.trim().replace(/\s+/g, " ")

const normalizeSelectedLabels = (topics: TopicBuilderTopic[]): string[] => {
  const seen = new Set<string>()
  const labels: string[] = []

  topics.forEach((topic) => {
    if (!topic.selected) {
      return
    }
    const normalized = normalizeLabel(topic.label)
    if (!normalized) {
      return
    }
    const dedupeKey = normalized.toLowerCase()
    if (seen.has(dedupeKey)) {
      return
    }
    seen.add(dedupeKey)
    labels.push(normalized)
  })

  return labels
}

const buildSelectedTopicEdits = (
  topics: TopicBuilderTopic[]
): Array<{ id: string; label: string }> => {
  return topics
    .filter((topic) => topic.selected && !topic.isManual)
    .map((topic) => ({
      id: topic.id,
      label: normalizeLabel(topic.label)
    }))
    .filter((topic) => topic.label.length > 0)
}

const mapRankReason = (value: unknown): TopicBuilderRankReason => {
  const normalized = normalizeLabel(String(value || "")).toLowerCase()
  if (normalized === "weakness" || normalized === "adjacent" || normalized === "exploratory") {
    return normalized
  }
  return "candidate"
}

const readTopicLabel = (raw: Record<string, unknown>): string => {
  const displayLabel = normalizeLabel(String(raw.display_label ?? ""))
  if (displayLabel) {
    return displayLabel
  }
  return normalizeLabel(String(raw.canonical_label ?? ""))
}

const readTopicId = (raw: Record<string, unknown>, index: number): string => {
  const id = raw.id
  const normalizedId = normalizeLabel(String(id || ""))
  return normalizedId || `topic-${index + 1}`
}

const readEvidenceClass = (raw: Record<string, unknown>): string | null => {
  const evidenceReasons = raw.evidence_reasons
  if (Array.isArray(evidenceReasons)) {
    const normalizedReasons = evidenceReasons
      .map((reason) => normalizeLabel(String(reason || "")))
      .filter(Boolean)
    if (normalizedReasons.length > 0) {
      return normalizedReasons.join(", ")
    }
  }

  return normalizeLabel(String(raw.type || raw.source_type || "")) || null
}

const hasSourceAwareEvidence = (
  snapshot: StudySuggestionSnapshotResponse | null
): boolean => {
  if (!snapshot) {
    return false
  }
  return Object.values(snapshot.live_evidence || {}).some((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return false
    }
    const raw = entry as Record<string, unknown>
    if (raw.source_available === false) {
      return false
    }
    return Boolean(raw.source_type || raw.source_id || raw.excerpt_text)
  })
}

const buildTopicsFromSnapshot = (
  snapshot: StudySuggestionSnapshotResponse | null,
  options?: { suppressSourceAwareAdjacency?: boolean }
): TopicBuilderTopic[] => {
  const rawTopics = snapshot?.snapshot.payload &&
    typeof snapshot.snapshot.payload === "object" &&
    !Array.isArray(snapshot.snapshot.payload)
      ? (snapshot.snapshot.payload as { topics?: unknown }).topics
      : []

  if (!Array.isArray(rawTopics)) {
    return []
  }

  return rawTopics
    .map((topic, index) => {
      if (!topic || typeof topic !== "object") {
        return null
      }
      const raw = topic as Record<string, unknown>
      const label = readTopicLabel(raw)
      if (!label) {
        return null
      }
      const normalizedRankReason = mapRankReason(raw.status || raw.type)
      const rankReason =
        options?.suppressSourceAwareAdjacency && normalizedRankReason === "adjacent"
          ? "exploratory"
          : normalizedRankReason
      return {
        id: readTopicId(raw, index),
        label,
        rankReason,
        selected: raw.selected !== false,
        evidenceClass: options?.suppressSourceAwareAdjacency
          ? "exploratory"
          : readEvidenceClass(raw),
        isManual: false
      }
    })
    .filter((topic): topic is TopicBuilderTopic => topic != null)
}

const titleizeSummaryKey = (value: string): string => {
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

const buildSummaryEntries = (
  snapshot: StudySuggestionSnapshotResponse | null
): Array<{ key: string; label: string; value: string }> => {
  const rawSummary = snapshot?.snapshot.payload &&
    typeof snapshot.snapshot.payload === "object" &&
    !Array.isArray(snapshot.snapshot.payload)
      ? (snapshot.snapshot.payload as { summary?: unknown }).summary
      : null

  if (!rawSummary || typeof rawSummary !== "object" || Array.isArray(rawSummary)) {
    return []
  }

  return Object.entries(rawSummary)
    .filter(([, value]) =>
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    )
    .map(([key, value]) => ({
      key,
      label: titleizeSummaryKey(key),
      value: String(value)
    }))
}

const addManualTopic = (topics: TopicBuilderTopic[]): TopicBuilderTopic[] => {
  const nextIndex = topics.length + 1
  return [
    ...topics,
    {
      id: `manual-${Date.now()}-${nextIndex}`,
      label: "New topic",
      rankReason: "exploratory",
      selected: true,
      evidenceClass: "manual",
      isManual: true
    }
  ]
}

export interface StudySuggestionsPanelProps {
  anchorType: SuggestionAnchorType
  anchorId: number
  onActionResult?: (
    response: StudySuggestionActionResponse,
    request: StudySuggestionActionRequest
  ) => void | Promise<void>
}

export const StudySuggestionsPanel: React.FC<StudySuggestionsPanelProps> = ({
  anchorType,
  anchorId,
  onActionResult
}) => {
  const { status, snapshot, isLoading, isRefreshing, refresh, performAction } =
    useStudySuggestions(anchorType, anchorId)
  const [topics, setTopics] = React.useState<TopicBuilderTopic[]>([])
  const [lastAction, setLastAction] = React.useState<StudySuggestionActionResponse | null>(null)
  const suppressSourceAwareAdjacency =
    snapshot?.snapshot.activity_type === "flashcard_review_session" &&
    !hasSourceAwareEvidence(snapshot)

  React.useEffect(() => {
    setTopics(buildTopicsFromSnapshot(snapshot, { suppressSourceAwareAdjacency }))
  }, [snapshot?.snapshot.id, suppressSourceAwareAdjacency])

  const selectedTopicIds = topics
    .filter((topic) => topic.selected && !topic.isManual)
    .map((topic) => topic.id)
  const selectedTopicEdits = buildSelectedTopicEdits(topics)
  const manualTopicLabels = normalizeSelectedLabels(topics.filter((topic) => topic.isManual))
  const summaryEntries = React.useMemo(() => buildSummaryEntries(snapshot), [snapshot?.snapshot.id])
  const followUpConfigs = React.useMemo(() => {
    if (snapshot?.snapshot.activity_type === "flashcard_review_session") {
      return [
        {
          key: "follow_up_flashcards",
          label: "Create flashcards",
          targetService: "flashcards" as const,
          targetType: "deck",
          actionKind: "follow_up_flashcards",
          buttonType: "primary" as const
        },
        {
          key: "follow_up_quiz",
          label: "Create quiz",
          targetService: "quiz" as const,
          targetType: "quiz",
          actionKind: "follow_up_quiz",
          buttonType: "default" as const
        }
      ]
    }

    if (snapshot?.snapshot.service === "quiz") {
      return [
        {
          key: "follow_up_quiz",
          label: "Create quiz",
          targetService: "quiz" as const,
          targetType: "quiz",
          actionKind: "follow_up_quiz",
          buttonType: "primary" as const
        },
        {
          key: "follow_up_flashcards",
          label: "Create flashcards",
          targetService: "flashcards" as const,
          targetType: "deck",
          actionKind: "follow_up_flashcards",
          buttonType: "default" as const
        }
      ]
    }

    return [
      {
        key: "follow_up_quiz",
        label: "Create quiz",
        targetService: "quiz" as const,
        targetType: "quiz",
        actionKind: "follow_up_quiz",
        buttonType: "primary" as const
      }
    ]
  }, [snapshot?.snapshot.activity_type, snapshot?.snapshot.service])

  const sessionCopy = React.useMemo(() => {
    if (snapshot?.snapshot.activity_type === "flashcard_review_session") {
      return suppressSourceAwareAdjacency
        ? "Exploratory follow-up"
        : "Flashcard follow-up"
    }
    return "Quiz follow-up"
  }, [snapshot?.snapshot.activity_type, suppressSourceAwareAdjacency])

  const handleAddTopic = () => {
    setTopics((current) => addManualTopic(current))
  }

  const handleRemoveTopic = (topicId: string) => {
    setTopics((current) => current.filter((topic) => topic.id !== topicId))
  }

  const handleRenameTopic = (topicId: string, nextLabel: string) => {
    setTopics((current) =>
      current.map((topic) =>
        topic.id === topicId
          ? { ...topic, label: normalizeLabel(nextLabel) || "New topic" }
          : topic
      )
    )
  }

  const handleResetTopics = () => {
    setTopics(buildTopicsFromSnapshot(snapshot, { suppressSourceAwareAdjacency }))
  }

  const handleToggleTopic = (topicId: string) => {
    setTopics((current) =>
      current.map((topic) =>
        topic.id === topicId ? { ...topic, selected: !topic.selected } : topic
      )
    )
  }

  const handleRefresh = async () => {
    await refresh()
  }

  const handleFollowUp = async (config: (typeof followUpConfigs)[number]) => {
    if (snapshot == null) return
    const request: StudySuggestionActionRequest = {
      targetService: config.targetService,
      targetType: config.targetType,
      actionKind: config.actionKind,
      selectedTopicIds,
      selectedTopicEdits,
      manualTopicLabels,
      hasExplicitSelection: true,
      generatorVersion: "v1",
      forceRegenerate: false
    }
    const response = await performAction(request)
    setLastAction(response)
    await onActionResult?.(response, request)
  }

  if (!snapshot && isLoading) {
    return (
      <Card size="small" title="Study suggestions">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="Loading study suggestions..."
        />
      </Card>
    )
  }

  if (!snapshot && status === "failed") {
    return (
      <Card
        size="small"
        title="Study suggestions"
        extra={<Tag color="red">failed</Tag>}
      >
        <Space orientation="vertical" size={16} className="w-full">
          <Alert
            type="error"
            showIcon
            title="Study suggestions are unavailable right now."
          />
          <Button
            icon={<ReloadOutlined />}
            loading={isRefreshing}
            onClick={() => {
              void handleRefresh()
            }}
          >
            Retry
          </Button>
        </Space>
      </Card>
    )
  }

  if (!snapshot) {
    return (
      <Card size="small" title="Study suggestions">
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No study suggestions yet." />
      </Card>
    )
  }

  return (
    <Card
      size="small"
      title="Study suggestions"
      extra={<Tag color={status === "failed" ? "red" : status === "pending" ? "gold" : "green"}>{status}</Tag>}
    >
      <Space orientation="vertical" size={16} className="w-full">
        <div className="space-y-1">
          <Text strong>{sessionCopy}</Text>
          <div className="flex flex-wrap gap-2">
            <Tag>{snapshot.snapshot.activity_type}</Tag>
            {snapshot.snapshot.refreshed_from_snapshot_id != null ? (
              <Tag color="blue">Refreshed</Tag>
            ) : null}
          </div>
        </div>

        {lastAction ? (
          <Alert
            type={lastAction.disposition === "opened_existing" ? "info" : "success"}
            showIcon
            title={lastAction.disposition === "opened_existing" ? "Open existing" : "Generated"}
          />
        ) : null}

        <div className="flex flex-wrap gap-2">
          <Button
            icon={<ReloadOutlined />}
            loading={isRefreshing}
            onClick={() => {
              void handleRefresh()
            }}
          >
            {status === "failed" ? "Retry" : "Refresh"}
          </Button>
          {followUpConfigs.map((config) => (
            <Button
              key={config.key}
              type={config.buttonType}
              onClick={() => {
                void handleFollowUp(config)
              }}
            >
              {config.label}
            </Button>
          ))}
        </div>

        {summaryEntries.length > 0 ? (
          <Card size="small" title="Summary">
            <Descriptions size="small" column={1}>
              {summaryEntries.map((entry) => (
                <Descriptions.Item key={entry.key} label={entry.label}>
                  {entry.value}
                </Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
        ) : null}

        <TopicBuilder
          topics={topics}
          onAddTopic={handleAddTopic}
          onRemoveTopic={handleRemoveTopic}
          onRenameTopic={handleRenameTopic}
          onToggleTopic={handleToggleTopic}
          onResetTopics={handleResetTopics}
        />
      </Space>
    </Card>
  )
}

export default StudySuggestionsPanel
