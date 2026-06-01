import React, { useMemo, useState } from "react"
import { Button, Input, Space, Table, Tag, Typography } from "antd"
import type { ColumnsType } from "antd/es/table"
import type {
  ScheduledTask,
  ScheduledTaskPrimitive
} from "@/services/scheduled-tasks-control-plane"
import {
  SCHEDULED_TASK_ATTENTION_STATUS_KEYS,
  buildWatchlistTaskLinks,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel,
  type ScheduledTaskProductStatus,
  type ScheduledTaskStatusKey
} from "./scheduled-task-status"

export interface ScheduledTaskTableRowActionContext {
  task: ScheduledTask
}

export interface ScheduledTaskTableProps {
  tasks: ScheduledTask[]
  onCreateReminder: () => void
  onInspectTask: (task: ScheduledTask) => void
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}

type ScheduledTaskStatusFilter = "all" | ScheduledTaskStatusKey
type ScheduledTaskTypeFilter = "all" | ScheduledTaskPrimitive

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

const typeTagColor = (task: ScheduledTask): string =>
  task.primitive === "watchlist_job" ? "gold" : "blue"

const rowActionLabel = (action: string, task: ScheduledTask): string =>
  `${action} ${task.title}`

const watchlistLinkLabel = (action: string, task: ScheduledTask): string =>
  `${action} for ${task.title}`

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

const matchesStatusFilter = (
  productStatus: ScheduledTaskProductStatus,
  statusFilter: ScheduledTaskStatusFilter
): boolean => {
  if (statusFilter === "all") return true
  if (statusFilter === "needs_attention") {
    return SCHEDULED_TASK_ATTENTION_STATUS_KEYS.includes(productStatus.key)
  }
  return productStatus.key === statusFilter
}

const matchesSearchText = (
  task: ScheduledTask,
  productStatus: ScheduledTaskProductStatus,
  searchText: string
): boolean => {
  const normalizedSearch = searchText.trim().toLowerCase()
  if (!normalizedSearch) return true

  return [
    task.title,
    task.description,
    task.schedule_summary,
    getScheduledTaskTypeLabel(task),
    productStatus.label
  ].some((value) => String(value || "").toLowerCase().includes(normalizedSearch))
}

const statusFilterOptions: Array<{
  value: ScheduledTaskStatusFilter
  label: string
}> = [
  { value: "all", label: "All statuses" },
  { value: "needs_attention", label: "Needs attention" },
  { value: "running", label: "Running now" },
  { value: "waiting", label: "Waiting" },
  { value: "found_results", label: "Found results" },
  { value: "blocked", label: "Blocked" },
  { value: "paused", label: "Paused" },
  { value: "disabled", label: "Disabled" },
  { value: "draft", label: "Draft" },
  { value: "completed", label: "Completed" }
]

const typeFilterOptions: Array<{
  value: ScheduledTaskTypeFilter
  label: string
}> = [
  { value: "all", label: "All types" },
  { value: "reminder_task", label: "Reminder" },
  { value: "watchlist_job", label: "Watchlist monitor" }
]

export const ScheduledTaskTable: React.FC<ScheduledTaskTableProps> = ({
  tasks,
  onCreateReminder,
  onInspectTask,
  onEditReminder,
  onDeleteReminder
}) => {
  const [searchText, setSearchText] = useState("")
  const [statusFilter, setStatusFilter] =
    useState<ScheduledTaskStatusFilter>("all")
  const [typeFilter, setTypeFilter] = useState<ScheduledTaskTypeFilter>("all")

  const filteredTasks = useMemo(
    () =>
      tasks.filter((task) => {
        const productStatus = getScheduledTaskProductStatus(task)
        const statusMatches = matchesStatusFilter(productStatus, statusFilter)
        const typeMatches =
          typeFilter === "all" || task.primitive === typeFilter
        const searchMatches = matchesSearchText(task, productStatus, searchText)

        return statusMatches && typeMatches && searchMatches
      }),
    [tasks, searchText, statusFilter, typeFilter]
  )

  const columns: ColumnsType<ScheduledTask> = [
    {
      title: "Task",
      dataIndex: "title",
      key: "title",
      render: (_, task) => (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <Typography.Text strong>{task.title}</Typography.Text>
          <Typography.Text type="secondary">
            {task.description || task.schedule_summary || "—"}
          </Typography.Text>
          <div>
            <Tag color={typeTagColor(task)}>{getScheduledTaskTypeLabel(task)}</Tag>
          </div>
        </div>
      )
    },
    {
      title: "Status",
      key: "status",
      render: (_, task) => {
        const productStatus = getScheduledTaskProductStatus(task)

        return (
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div>
              <Tag color={statusToneToTagColor(productStatus)}>
                {productStatus.label}
              </Tag>
            </div>
            <Typography.Text type="secondary">
              {productStatus.description}
            </Typography.Text>
          </div>
        )
      }
    },
    {
      title: "Schedule",
      key: "schedule",
      render: (_, task) => (
        <div style={{ display: "flex", flexDirection: "column" }}>
          <Typography.Text>{task.schedule_summary || "Manual"}</Typography.Text>
          <Typography.Text type="secondary">
            {task.timezone ? `Timezone: ${task.timezone}` : "No timezone"}
          </Typography.Text>
        </div>
      )
    },
    {
      title: "Last run",
      key: "last_run_at",
      render: (_, task) => (
        <Typography.Text>
          {formatTimestamp(task.last_run_at, "No completed runs yet")}
        </Typography.Text>
      )
    },
    {
      title: "Next run",
      key: "next_run_at",
      render: (_, task) => (
        <Typography.Text>
          {formatTimestamp(task.next_run_at, "No upcoming run")}
        </Typography.Text>
      )
    },
    {
      title: "Management",
      key: "management",
      render: (_, task) => (
        <Tag color={isNativeReminder(task) ? "blue" : "gold"}>
          {isNativeReminder(task) ? "Managed here" : "Managed in Watchlists"}
        </Tag>
      )
    },
    {
      title: "Actions",
      key: "actions",
      render: (_, task) => {
        if (isNativeReminder(task)) {
          return (
            <Space wrap>
              <Button
                size="small"
                aria-label={rowActionLabel("Inspect", task)}
                onClick={() => onInspectTask(task)}
              >
                Inspect
              </Button>
              <Button
                size="small"
                aria-label={rowActionLabel("Edit", task)}
                onClick={() => onEditReminder(task)}
              >
                Edit
              </Button>
              <Button
                size="small"
                danger
                aria-label={rowActionLabel("Delete", task)}
                onClick={() => onDeleteReminder(task)}
              >
                Delete
              </Button>
            </Space>
          )
        }

        const links = buildWatchlistTaskLinks(task)

        return (
          <Space wrap>
            <Button
              size="small"
              aria-label={rowActionLabel("Inspect", task)}
              onClick={() => onInspectTask(task)}
            >
              Inspect
            </Button>
            {links.settingsUrl ? (
              <Button
                size="small"
                type="link"
                href={links.settingsUrl}
                target="_self"
                aria-label={watchlistLinkLabel("Open monitor settings", task)}
              >
                Open monitor settings
              </Button>
            ) : null}
            {links.activityUrl ? (
              <Button
                size="small"
                type="link"
                href={links.activityUrl}
                target="_self"
                aria-label={watchlistLinkLabel("Open activity", task)}
              >
                Open activity
              </Button>
            ) : null}
            {links.reportsUrl ? (
              <Button
                size="small"
                type="link"
                href={links.reportsUrl}
                target="_self"
                aria-label={watchlistLinkLabel("Open reports", task)}
              >
                Open reports
              </Button>
            ) : null}
            {links.latestRunUrl ? (
              <Button
                size="small"
                type="link"
                href={links.latestRunUrl}
                target="_self"
                aria-label={watchlistLinkLabel("Open latest run", task)}
              >
                Open latest run
              </Button>
            ) : null}
            {links.latestOutputUrl ? (
              <Button
                size="small"
                type="link"
                href={links.latestOutputUrl}
                target="_self"
                aria-label={watchlistLinkLabel("Open latest report", task)}
              >
                Open latest report
              </Button>
            ) : null}
          </Space>
        )
      }
    }
  ]

  return (
    <Table<ScheduledTask>
      rowKey="id"
      columns={columns}
      dataSource={filteredTasks}
      pagination={false}
      scroll={{ x: "max-content" }}
      locale={{ emptyText: "No scheduled tasks match these filters." }}
      title={() => (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Space style={{ width: "100%", justifyContent: "space-between" }}>
            <Typography.Title level={4} style={{ margin: 0 }}>
              Scheduled tasks
            </Typography.Title>
            <Button type="primary" onClick={onCreateReminder}>
              Create scheduled task
            </Button>
          </Space>
          <Space wrap>
            <Input
              allowClear
              aria-label="Search scheduled tasks"
              placeholder="Search tasks"
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              style={{ width: 240 }}
            />
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Typography.Text type="secondary">Status</Typography.Text>
              <select
                aria-label="Status filter"
                value={statusFilter}
                onChange={(event) =>
                  setStatusFilter(event.target.value as ScheduledTaskStatusFilter)
                }
                style={{ height: 32 }}
              >
                {statusFilterOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Typography.Text type="secondary">Type</Typography.Text>
              <select
                aria-label="Type filter"
                value={typeFilter}
                onChange={(event) =>
                  setTypeFilter(event.target.value as ScheduledTaskTypeFilter)
                }
                style={{ height: 32 }}
              >
                {typeFilterOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </Space>
        </div>
      )}
    />
  )
}

export default ScheduledTaskTable
