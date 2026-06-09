import React, { useMemo, useState } from "react"
import { Alert, Button, Empty, Input, Select, Space, Table, Tag, Typography } from "antd"
import type { ColumnsType } from "antd/es/table"

import { formatScheduledTaskTimestamp } from "./scheduled-task-status"
import {
  filterScheduledTaskResults,
  type ScheduledTaskResultItem,
  type ScheduledTaskResultOwner,
  type ScheduledTaskResultSignalKind,
  type ScheduledTaskResultState,
  type ScheduledTaskResultsCapabilityMode
} from "./scheduled-task-results"

export interface ScheduledTaskResultsPanelProps {
  results: ScheduledTaskResultItem[]
  taskCount: number
  capabilityMode: ScheduledTaskResultsCapabilityMode
  onCreateTask: () => void
  onOpenResult: (result: ScheduledTaskResultItem) => void
}

type ResultStateFilter =
  | "all"
  | "result"
  | "failure"
  | "running"
  | "completed_no_results"

type ReviewStateFilter = "all" | "unreviewed" | "reviewed"

const resultStateOptions: Array<{ value: ResultStateFilter; label: string }> = [
  { value: "all", label: "All states" },
  { value: "result", label: "Found results" },
  { value: "failure", label: "Needs attention" },
  { value: "running", label: "Running" },
  { value: "completed_no_results", label: "Completed/no results" }
]

const reviewStateOptions: Array<{ value: ReviewStateFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "unreviewed", label: "Unreviewed" },
  { value: "reviewed", label: "Reviewed" }
]

const severityToTagColor = (severity: ScheduledTaskResultItem["severity"]): string => {
  switch (severity) {
    case "success":
      return "green"
    case "warning":
      return "gold"
    case "error":
      return "red"
    default:
      return "processing"
  }
}

const resultStatusLabel = (result: ScheduledTaskResultItem): string => {
  if (result.signalKind === "result") return "Found results"
  if (result.state === "blocked") return "Blocked"
  if (result.signalKind === "failure") return "Needs attention"
  if (result.state === "paused") return "Paused"
  if (result.signalKind === "running") return "Running now"
  return "Completed/no results"
}

const stateFilterToSignalKinds = (
  stateFilter: ResultStateFilter
): ScheduledTaskResultSignalKind[] | null => {
  switch (stateFilter) {
    case "result":
      return ["result"]
    case "failure":
      return ["failure"]
    case "running":
      return ["running"]
    case "completed_no_results":
      return ["completed_no_results"]
    default:
      return null
  }
}

const stateFilterToStates = (
  stateFilter: ResultStateFilter
): ScheduledTaskResultState[] | null => {
  if (stateFilter === "completed_no_results") {
    return ["completed_no_results"]
  }
  return null
}

const hasActiveFilters = ({
  searchText,
  stateFilter,
  taskTypeFilter,
  ownerFilter,
  reviewStateFilter
}: {
  searchText: string
  stateFilter: ResultStateFilter
  taskTypeFilter: string
  ownerFilter: string
  reviewStateFilter: ReviewStateFilter
}): boolean =>
  Boolean(searchText.trim()) ||
  stateFilter !== "all" ||
  taskTypeFilter !== "all" ||
  ownerFilter !== "all" ||
  reviewStateFilter !== "all"

export const ScheduledTaskResultsPanel: React.FC<ScheduledTaskResultsPanelProps> = ({
  results,
  taskCount,
  capabilityMode,
  onCreateTask,
  onOpenResult
}) => {
  const [searchText, setSearchText] = useState("")
  const [stateFilter, setStateFilter] = useState<ResultStateFilter>("all")
  const [taskTypeFilter, setTaskTypeFilter] = useState("all")
  const [ownerFilter, setOwnerFilter] = useState<ScheduledTaskResultOwner | "all">("all")
  const [reviewStateFilter, setReviewStateFilter] = useState<ReviewStateFilter>("all")
  const canShowReviewState = capabilityMode !== "projected_signals"

  const taskTypeOptions = useMemo(() => {
    const uniqueTypes = Array.from(new Set(results.map((result) => result.taskTypeLabel)))
    return [
      { value: "all", label: "All types" },
      ...uniqueTypes.map((type) => ({ value: type, label: type }))
    ]
  }, [results])

  const ownerOptions = useMemo(() => {
    const ownerMap = new Map<ScheduledTaskResultOwner, string>()
    results.forEach((result) => {
      ownerMap.set(result.owner, result.ownerLabel)
    })
    return [
      { value: "all", label: "All owners" },
      ...Array.from(ownerMap.entries()).map(([value, label]) => ({ value, label }))
    ]
  }, [results])

  const filteredResults = useMemo(() => {
    const signalKinds = stateFilterToSignalKinds(stateFilter)
    const states = stateFilterToStates(stateFilter)
    const reviewState =
      canShowReviewState && reviewStateFilter !== "all" ? reviewStateFilter : "all"
    const byStructuredFilters = filterScheduledTaskResults(results, {
      ...(signalKinds ? { signalKinds } : {}),
      ...(states ? { states } : {}),
      ...(ownerFilter !== "all" ? { owners: [ownerFilter] } : {}),
      reviewState
    })
    const normalizedSearch = searchText.trim().toLowerCase()

    return byStructuredFilters.filter((result) => {
      const taskTypeMatches =
        taskTypeFilter === "all" || result.taskTypeLabel === taskTypeFilter
      const searchMatches =
        !normalizedSearch ||
        [
          result.title,
          result.summary,
          result.ownerLabel,
          result.taskTypeLabel,
          result.sourceLabel,
          result.matchedRuleLabel,
          result.outputLabel,
          resultStatusLabel(result)
        ].some((value) => String(value || "").toLowerCase().includes(normalizedSearch))

      return taskTypeMatches && searchMatches
    })
  }, [
    canShowReviewState,
    ownerFilter,
    results,
    reviewStateFilter,
    searchText,
    stateFilter,
    taskTypeFilter
  ])

  const resetFilters = () => {
    setSearchText("")
    setStateFilter("all")
    setTaskTypeFilter("all")
    setOwnerFilter("all")
    setReviewStateFilter("all")
  }

  const activeFilters = hasActiveFilters({
    searchText,
    stateFilter,
    taskTypeFilter,
    ownerFilter,
    reviewStateFilter
  })

  const columns: ColumnsType<ScheduledTaskResultItem> = [
    {
      title: "Result",
      key: "result",
      render: (_, result) => (
        <Space orientation="vertical" size={4}>
          <Typography.Text strong>{result.title}</Typography.Text>
          <Typography.Text type="secondary">{result.summary}</Typography.Text>
          <Space wrap size={4}>
            <Tag>{result.taskTypeLabel}</Tag>
            <Tag>{result.ownerLabel}</Tag>
          </Space>
        </Space>
      )
    },
    {
      title: "State",
      key: "state",
      render: (_, result) => (
        <Tag color={severityToTagColor(result.severity)}>
          {resultStatusLabel(result)}
        </Tag>
      )
    },
    {
      title: "Last signal",
      key: "occurredAt",
      render: (_, result) => (
        <Typography.Text>
          {formatScheduledTaskTimestamp(result.occurredAt, "No run time")}
        </Typography.Text>
      )
    },
    {
      title: "Actions",
      key: "actions",
      render: (_, result) => (
        <Button
          size="small"
          aria-label={`Open signal for ${result.title}`}
          onClick={() => onOpenResult(result)}
        >
          Open signal
        </Button>
      )
    }
  ]

  if (taskCount === 0) {
    return (
      <Space orientation="vertical" size={16} style={{ width: "100%" }}>
        <PanelHeader capabilityMode={capabilityMode} />
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space orientation="vertical" size={4}>
              <Typography.Text strong>No scheduled tasks yet</Typography.Text>
              <Typography.Text type="secondary">
                Results and failures appear here after an automation runs.
              </Typography.Text>
            </Space>
          }
        >
          <Button type="primary" onClick={onCreateTask}>
            Create scheduled task
          </Button>
        </Empty>
      </Space>
    )
  }

  if (results.length === 0) {
    return (
      <Space orientation="vertical" size={16} style={{ width: "100%" }}>
        <PanelHeader capabilityMode={capabilityMode} />
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space orientation="vertical" size={4}>
              <Typography.Text strong>No results to review</Typography.Text>
              <Typography.Text type="secondary">
                The latest scheduled runs have not produced new results or failures.
              </Typography.Text>
            </Space>
          }
        />
      </Space>
    )
  }

  return (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
      <PanelHeader capabilityMode={capabilityMode} />
      <Space wrap>
        <Input
          allowClear
          aria-label="Search scheduled task results"
          placeholder="Search results"
          value={searchText}
          onChange={(event) => setSearchText(event.target.value)}
          style={{ width: 240 }}
        />
        <Space align="center" size={8}>
          <Typography.Text type="secondary">Result state</Typography.Text>
          <Select
            aria-label="Result state filter"
            value={stateFilter}
            onChange={(value) => setStateFilter(value)}
            options={resultStateOptions}
            style={{ width: 190 }}
          />
        </Space>
        <Space align="center" size={8}>
          <Typography.Text type="secondary">Task type</Typography.Text>
          <Select
            aria-label="Task type filter"
            value={taskTypeFilter}
            onChange={(value) => setTaskTypeFilter(value)}
            options={taskTypeOptions}
            style={{ width: 190 }}
          />
        </Space>
        <Space align="center" size={8}>
          <Typography.Text type="secondary">Owner</Typography.Text>
          <Select
            aria-label="Owner filter"
            value={ownerFilter}
            onChange={(value) => setOwnerFilter(value)}
            options={ownerOptions}
            style={{ width: 170 }}
          />
        </Space>
        {canShowReviewState ? (
          <Space align="center" size={8}>
            <Typography.Text type="secondary">Review state</Typography.Text>
            <Select
              aria-label="Review state filter"
              value={reviewStateFilter}
              onChange={(value) => setReviewStateFilter(value)}
              options={reviewStateOptions}
              style={{ width: 160 }}
            />
          </Space>
        ) : null}
      </Space>

      {filteredResults.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space orientation="vertical" size={4}>
              <Typography.Text strong>No results match these filters</Typography.Text>
              <Typography.Text type="secondary">
                Adjust the result state, task type, owner, or search filters.
              </Typography.Text>
            </Space>
          }
        >
          {activeFilters ? <Button onClick={resetFilters}>Clear filters</Button> : null}
        </Empty>
      ) : (
        <Table<ScheduledTaskResultItem>
          rowKey="id"
          columns={columns}
          dataSource={filteredResults}
          pagination={false}
          scroll={{ x: "max-content" }}
        />
      )}
    </Space>
  )
}

const PanelHeader: React.FC<{ capabilityMode: ScheduledTaskResultsCapabilityMode }> = ({
  capabilityMode
}) => (
  <Space orientation="vertical" size={12} style={{ width: "100%" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <Typography.Title level={3} style={{ marginBottom: 0 }}>
        Scheduled task results
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
        Review outputs, failures, and run state from recurring automations.
        Source-specific setup stays in the owning workspace.
      </Typography.Paragraph>
    </div>
    <Alert
      type="info"
      showIcon
      title={capabilityMode === "projected_signals" ? "Latest automation signals" : "Results"}
      description={
        capabilityMode === "projected_signals"
          ? "Latest signals inferred from task status. Durable review state appears when the results API is available."
          : "Review state comes from the scheduled-task results API."
      }
    />
  </Space>
)

export default ScheduledTaskResultsPanel
