import React from "react"
import { Button, Card, Col, Row, Space, Tag, Typography } from "antd"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import {
  SCHEDULED_TASK_ATTENTION_STATUS_KEYS,
  getScheduledTaskProductStatus
} from "./scheduled-task-status"
import type { ScheduledTaskResultItem } from "./scheduled-task-results"

export interface ScheduledTaskOverviewProps {
  tasks: ScheduledTask[]
  partial: boolean
  results?: ScheduledTaskResultItem[]
  onOpenResult?: (result: ScheduledTaskResultItem) => void
}

const countLabel = (count: number, singular: string, plural = `${singular}s`): string =>
  `${count} ${count === 1 ? singular : plural}`

const getTimestamp = (task: ScheduledTask): number | null => {
  if (!task.enabled) return null
  if (!task.next_run_at) return null

  const timestamp = new Date(task.next_run_at).getTime()
  return Number.isFinite(timestamp) ? timestamp : null
}

const formatRunDate = (timestamp: number | null): string => {
  if (timestamp === null) return "No upcoming runs"

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(timestamp))
}

const findNextRunTimestamp = (tasks: ScheduledTask[]): number | null => {
  let nextRun: number | null = null

  for (const task of tasks) {
    const timestamp = getTimestamp(task)
    if (timestamp === null) continue
    if (nextRun === null || timestamp < nextRun) {
      nextRun = timestamp
    }
  }

  return nextRun
}

interface OverviewPanelProps {
  title: string
  value: string
  children?: React.ReactNode
}

const OverviewPanel: React.FC<OverviewPanelProps> = ({
  title,
  value,
  children
}) => (
  <Card size="small" style={{ height: "100%" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <Typography.Text type="secondary">{title}</Typography.Text>
      <Typography.Text strong style={{ fontSize: 20, lineHeight: 1.2 }}>
        {value}
      </Typography.Text>
      {children}
    </div>
  </Card>
)

export const ScheduledTaskOverview: React.FC<ScheduledTaskOverviewProps> = ({
  tasks,
  partial,
  results = [],
  onOpenResult
}) => {
  const statuses = tasks.map(getScheduledTaskProductStatus)
  const needsAttentionCount = statuses.filter(
    (status) => SCHEDULED_TASK_ATTENTION_STATUS_KEYS.includes(status.key)
  ).length
  const runningNowCount = statuses.filter(
    (status) => status.key === "running"
  ).length
  const nextRunTimestamp = findNextRunTimestamp(tasks)
  const latestResult = results[0] ?? null

  return (
    <section aria-label="Scheduled task overview">
      <Row gutter={[12, 12]}>
        <Col xs={24} sm={12} lg={6}>
          <OverviewPanel
            title="Total scheduled tasks"
            value={countLabel(tasks.length, "scheduled task")}
          >
            {partial ? <Tag color="gold">Partial data</Tag> : <Tag color="blue">Loaded</Tag>}
          </OverviewPanel>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <OverviewPanel
            title="Needs attention"
            value={`${needsAttentionCount} needs attention`}
          >
            {needsAttentionCount > 0 ? <Tag color="red">Review required</Tag> : null}
          </OverviewPanel>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <OverviewPanel
            title="Running now"
            value={`${runningNowCount} running now`}
          >
            {runningNowCount > 0 ? <Tag color="processing">Active</Tag> : null}
          </OverviewPanel>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <OverviewPanel
            title="Next upcoming run"
            value={formatRunDate(nextRunTimestamp)}
          >
            {nextRunTimestamp === null ? <Tag color="gold">Not scheduled</Tag> : null}
          </OverviewPanel>
        </Col>
      </Row>
      {latestResult ? (
        <Card size="small" style={{ marginTop: 12 }}>
          <Space orientation="vertical" size={6} style={{ width: "100%" }}>
            <Typography.Text type="secondary">Latest result signal</Typography.Text>
            <Typography.Text strong>{latestResult.title}</Typography.Text>
            <Typography.Text type="secondary">{latestResult.summary}</Typography.Text>
            {onOpenResult ? (
              <Button
                size="small"
                onClick={() => onOpenResult(latestResult)}
              >
                Open latest result signal
              </Button>
            ) : null}
          </Space>
        </Card>
      ) : null}
    </section>
  )
}

export default ScheduledTaskOverview
