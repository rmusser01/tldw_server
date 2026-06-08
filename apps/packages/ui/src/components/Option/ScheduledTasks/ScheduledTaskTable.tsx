import React, { useMemo, useState } from "react"
import { Button, Input, Select, Space, Table, Tag, Typography } from "antd"
import type { ColumnsType } from "antd/es/table"
import type {
  ScheduledTask,
  ScheduledTaskPrimitive
} from "@/services/scheduled-tasks-control-plane"
import {
  SCHEDULED_TASK_ATTENTION_STATUS_KEYS,
  formatScheduledTaskTimestamp,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel,
  isNativeReminderTask,
  scheduledTaskStatusToneToTagColor,
  type ScheduledTaskProductStatus,
  type ScheduledTaskStatusKey
} from "./scheduled-task-status"
import { WatchlistTaskActionLinks } from "./WatchlistTaskActionLinks"

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

const typeTagColor = (task: ScheduledTask): string =>
  task.primitive === "watchlist_job" ? "gold" : "blue"

const rowActionLabel = (action: string, task: ScheduledTask): string =>
  `${action} ${task.title}`

const watchlistLinkLabel = (action: string, task: ScheduledTask): string =>
  `${action} for ${task.title}`

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
              <Tag color={scheduledTaskStatusToneToTagColor(productStatus)}>
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
          {formatScheduledTaskTimestamp(task.last_run_at, "No completed runs yet")}
        </Typography.Text>
      )
    },
    {
      title: "Next run",
      key: "next_run_at",
      render: (_, task) => (
        <Typography.Text>
          {formatScheduledTaskTimestamp(task.next_run_at, "No upcoming run")}
        </Typography.Text>
      )
    },
    {
      title: "Management",
      key: "management",
      render: (_, task) => (
        <Tag color={isNativeReminderTask(task) ? "blue" : "gold"}>
          {isNativeReminderTask(task) ? "Managed here" : "Managed in Watchlists"}
        </Tag>
      )
    },
    {
      title: "Actions",
      key: "actions",
      render: (_, task) => {
        if (isNativeReminderTask(task)) {
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

        return (
          <Space wrap>
            <Button
              size="small"
              aria-label={rowActionLabel("Inspect", task)}
              onClick={() => onInspectTask(task)}
            >
              Inspect
            </Button>
            <WatchlistTaskActionLinks
              task={task}
              size="small"
              getAriaLabel={watchlistLinkLabel}
            />
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
            <Space align="center" size={8}>
              <Typography.Text type="secondary">Status</Typography.Text>
              <Select
                id="scheduled-task-status-filter"
                aria-label="Status filter"
                value={statusFilter}
                onChange={(value) => setStatusFilter(value)}
                options={statusFilterOptions}
                style={{ width: 160 }}
              />
            </Space>
            <Space align="center" size={8}>
              <Typography.Text type="secondary">Type</Typography.Text>
              <Select
                id="scheduled-task-type-filter"
                aria-label="Type filter"
                value={typeFilter}
                onChange={(value) => setTypeFilter(value)}
                options={typeFilterOptions}
                style={{ width: 180 }}
              />
            </Space>
          </Space>
        </div>
      )}
    />
  )
}

export default ScheduledTaskTable
