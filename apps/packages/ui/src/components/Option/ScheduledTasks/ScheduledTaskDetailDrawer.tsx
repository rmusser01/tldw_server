import React from "react"
import { Button, Descriptions, Drawer, Space, Tag, Typography } from "antd"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import {
  buildWatchlistTaskLinks,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel,
  type ScheduledTaskProductStatus
} from "./scheduled-task-status"

export interface ScheduledTaskDetailDrawerProps {
  open: boolean
  task: ScheduledTask | null
  onClose: () => void
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}

const WATCHLISTS_WORKSPACE_COPY =
  "Watchlists remains the full workspace for monitor setup, source tuning, run activity, and reports."

const isNativeReminder = (task: ScheduledTask): boolean =>
  task.primitive === "reminder_task" && task.edit_mode === "native"

const statusToneToTagColor = (status: ScheduledTaskProductStatus): string => {
  switch (status.tone) {
    case "success":
      return "green"
    case "processing":
      return "processing"
    case "warning":
      return "gold"
    case "error":
      return "red"
    default:
      return "default"
  }
}

const formatTimestamp = (
  timestamp: string | null | undefined,
  fallback: string
): string => {
  if (!timestamp) return fallback

  const parsed = new Date(timestamp)
  if (Number.isNaN(parsed.getTime())) {
    return timestamp
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(parsed)
}

const sourceValueToText = (value: unknown): string | null => {
  if (value === null || value === undefined) return null

  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed ? trimmed : null
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }

  return null
}

const renderOptionalDescriptionItem = (
  label: string,
  value: unknown
): React.ReactNode => {
  const text = sourceValueToText(value)
  if (!text) return null

  return (
    <Descriptions.Item key={label} label={label}>
      {text}
    </Descriptions.Item>
  )
}

const renderSourceReferenceItems = (task: ScheduledTask): React.ReactNode => {
  const sourceRef = task.source_ref || {}

  if (task.primitive === "watchlist_job") {
    return (
      <>
        {renderOptionalDescriptionItem("Watchlists job id", sourceRef.job_id)}
        {renderOptionalDescriptionItem("Watchlists scope", sourceRef.scope)}
      </>
    )
  }

  return (
    <>
      {renderOptionalDescriptionItem("Reminder task id", sourceRef.task_id)}
      {renderOptionalDescriptionItem("Link type", sourceRef.link_type)}
      {renderOptionalDescriptionItem("Link id", sourceRef.link_id)}
      {renderOptionalDescriptionItem("Link URL", sourceRef.link_url)}
    </>
  )
}

const renderTaskActions = ({
  task,
  onEditReminder,
  onDeleteReminder
}: {
  task: ScheduledTask
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}): React.ReactNode => {
  if (isNativeReminder(task)) {
    return (
      <Space wrap>
        <Button type="primary" onClick={() => onEditReminder(task)}>
          Edit reminder
        </Button>
        <Button danger onClick={() => onDeleteReminder(task)}>
          Delete reminder
        </Button>
      </Space>
    )
  }

  const links = buildWatchlistTaskLinks(task)

  return (
    <Space wrap>
      {links.settingsUrl ? (
        <Button type="link" href={links.settingsUrl} target="_self">
          Open monitor settings
        </Button>
      ) : null}
      {links.activityUrl ? (
        <Button type="link" href={links.activityUrl} target="_self">
          Open activity
        </Button>
      ) : null}
      {links.reportsUrl ? (
        <Button type="link" href={links.reportsUrl} target="_self">
          Open reports
        </Button>
      ) : null}
      {links.latestRunUrl ? (
        <Button type="link" href={links.latestRunUrl} target="_self">
          Open latest run
        </Button>
      ) : null}
      {links.latestOutputUrl ? (
        <Button type="link" href={links.latestOutputUrl} target="_self">
          Open latest report
        </Button>
      ) : null}
    </Space>
  )
}

export const ScheduledTaskDetailDrawer: React.FC<ScheduledTaskDetailDrawerProps> = ({
  open,
  task,
  onClose,
  onEditReminder,
  onDeleteReminder
}) => {
  const productStatus = task ? getScheduledTaskProductStatus(task) : null

  return (
    <Drawer
      title={task?.title ?? "Scheduled task details"}
      open={open}
      onClose={onClose}
      size={520}
    >
      {task && productStatus ? (
        <Space orientation="vertical" size="large" style={{ width: "100%" }}>
          {task.description ? (
            <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
              {task.description}
            </Typography.Paragraph>
          ) : null}

          <Descriptions bordered size="small" column={1}>
            <Descriptions.Item label="Product status">
              <Space orientation="vertical" size={2}>
                <Tag color={statusToneToTagColor(productStatus)}>
                  {productStatus.label}
                </Tag>
                <Typography.Text type="secondary">
                  {productStatus.description}
                </Typography.Text>
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="Task type">
              {getScheduledTaskTypeLabel(task)}
            </Descriptions.Item>
            <Descriptions.Item label="Management owner">
              {isNativeReminder(task) ? "Managed here" : "Managed in Watchlists"}
            </Descriptions.Item>
            <Descriptions.Item label="Schedule summary">
              {task.schedule_summary || "Manual"}
            </Descriptions.Item>
            <Descriptions.Item label="Timezone">
              {task.timezone || "No timezone"}
            </Descriptions.Item>
            <Descriptions.Item label="Last run">
              {formatTimestamp(task.last_run_at, "No completed runs yet")}
            </Descriptions.Item>
            <Descriptions.Item label="Next run">
              {formatTimestamp(task.next_run_at, "No upcoming run")}
            </Descriptions.Item>
            {renderSourceReferenceItems(task)}
          </Descriptions>

          <div>
            <Typography.Title level={5}>Actions</Typography.Title>
            {renderTaskActions({ task, onEditReminder, onDeleteReminder })}
          </div>

          {task.primitive === "watchlist_job" ? (
            <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
              {WATCHLISTS_WORKSPACE_COPY}
            </Typography.Paragraph>
          ) : null}
        </Space>
      ) : (
        <Typography.Text type="secondary">Select a task to inspect.</Typography.Text>
      )}
    </Drawer>
  )
}

export default ScheduledTaskDetailDrawer
