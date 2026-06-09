import React from "react"
import { Button, Descriptions, Drawer, Space, Typography } from "antd"
import { Badge as DesignSystemBadge } from "@/components/ui/primitives"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import {
  formatScheduledTaskTimestamp,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel,
  isNativeReminderTask,
  scheduledTaskStatusToneToBadgeVariant
} from "./scheduled-task-status"
import { WatchlistTaskActionLinks } from "./WatchlistTaskActionLinks"
import type { ScheduledTaskResultItem } from "./scheduled-task-results"

export interface ScheduledTaskDetailDrawerProps {
  open: boolean
  task: ScheduledTask | null
  latestResult?: ScheduledTaskResultItem | null
  onClose: () => void
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}

const WATCHLISTS_WORKSPACE_COPY =
  "Watchlists remains the full workspace for monitor setup, source tuning, run activity, and reports."

const SOURCE_VALUE_MAX_LENGTH = 96

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

const truncateSourceValue = (value: string): string =>
  value.length > SOURCE_VALUE_MAX_LENGTH
    ? `${value.slice(0, SOURCE_VALUE_MAX_LENGTH - 3)}...`
    : value

const renderOptionalDescriptionItem = (
  label: string,
  value: unknown
): React.ReactNode => {
  const text = sourceValueToText(value)
  if (!text) return null
  const displayText = truncateSourceValue(text)

  return (
    <Descriptions.Item key={label} label={label}>
      <Typography.Text title={text}>{displayText}</Typography.Text>
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
  if (isNativeReminderTask(task)) {
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

  return <WatchlistTaskActionLinks task={task} />
}

export const ScheduledTaskDetailDrawer: React.FC<ScheduledTaskDetailDrawerProps> = ({
  open,
  task,
  latestResult = null,
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
                <DesignSystemBadge
                  variant={scheduledTaskStatusToneToBadgeVariant(productStatus)}
                >
                  {productStatus.label}
                </DesignSystemBadge>
                <Typography.Text type="secondary">
                  {productStatus.description}
                </Typography.Text>
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="Task type">
              {getScheduledTaskTypeLabel(task)}
            </Descriptions.Item>
            <Descriptions.Item label="Management owner">
              {isNativeReminderTask(task) ? "Managed here" : "Managed in Watchlists"}
            </Descriptions.Item>
            <Descriptions.Item label="Schedule summary">
              {task.schedule_summary || "Manual"}
            </Descriptions.Item>
            <Descriptions.Item label="Timezone">
              {task.timezone || "No timezone"}
            </Descriptions.Item>
            <Descriptions.Item label="Last run">
              {formatScheduledTaskTimestamp(task.last_run_at, "No completed runs yet")}
            </Descriptions.Item>
            <Descriptions.Item label="Next run">
              {formatScheduledTaskTimestamp(task.next_run_at, "No upcoming run")}
            </Descriptions.Item>
            {renderSourceReferenceItems(task)}
          </Descriptions>

          <div>
            <Typography.Title level={5}>Actions</Typography.Title>
            <Space orientation="vertical" size={8}>
              {latestResult ? (
                <div>
                  <Button href={latestResult.primaryHref}>
                    Open latest result signal
                  </Button>
                </div>
              ) : null}
              {renderTaskActions({ task, onEditReminder, onDeleteReminder })}
            </Space>
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
