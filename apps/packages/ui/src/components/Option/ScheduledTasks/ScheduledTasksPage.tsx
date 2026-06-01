import React, { useState } from "react"
import { Alert, Button, Empty, Space, Spin, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  createScheduledTaskReminder,
  deleteScheduledTaskReminder,
  listScheduledTasks,
  updateScheduledTaskReminder,
  type ScheduledTask,
  type CreateScheduledTaskReminderPayload,
  type UpdateScheduledTaskReminderPayload
} from "@/services/scheduled-tasks-control-plane"
import { ScheduledTaskTable } from "./ScheduledTaskTable"
import { ReminderTaskEditor } from "./ReminderTaskEditor"
import { ScheduledTaskOverview } from "./ScheduledTaskOverview"

const SCHEDULED_TASKS_PATH = "/api/v1/scheduled-tasks"

const LoadingState: React.FC = () => (
  <div role="status" aria-live="polite">
    <Space>
      <Spin size="small" />
      <Typography.Text type="secondary">Loading tasks and latest run state</Typography.Text>
    </Space>
  </div>
)

export const ScheduledTasksPage: React.FC = () => {
  const navigate = useNavigate()
  const { t } = useTranslation(["scheduledTasks", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [editorOpen, setEditorOpen] = useState(false)
  const [editingTask, setEditingTask] = useState<ScheduledTask | null>(null)
  const [selectedTask, setSelectedTask] = useState<ScheduledTask | null>(null)
  const [saving, setSaving] = useState(false)
  const [scheduledTasksSupported, setScheduledTasksSupported] = useState<
    boolean | null
  >(null)

  React.useEffect(() => {
    if (connectionConfigLoading) return

    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) {
      setScheduledTasksSupported(true)
      return
    }

    let cancelled = false

    const probeScheduledTasksSupport = async () => {
      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (!response.ok) {
          if (!cancelled) {
            setScheduledTasksSupported(true)
          }
          return
        }

        const spec = await response.json()
        const paths =
          spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
            ? (spec.paths as Record<string, unknown>)
            : null

        if (!cancelled) {
          setScheduledTasksSupported(Boolean(paths && SCHEDULED_TASKS_PATH in paths))
        }
      } catch {
        if (!cancelled) {
          setScheduledTasksSupported(true)
        }
      }
    }

    void probeScheduledTasksSupport()

    return () => {
      cancelled = true
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading])

  const tasksQuery = useQuery({
    queryKey: ["scheduled-tasks"],
    queryFn: listScheduledTasks,
    enabled: scheduledTasksSupported === true
  })

  const tasks = tasksQuery.data?.items ?? []
  const hasLoadedTasks = Boolean(tasksQuery.data)
  const hasWatchlistJob = tasks.some((task) => task.primitive === "watchlist_job")
  const isLoadingTasks =
    connectionConfigLoading ||
    scheduledTasksSupported === null ||
    (scheduledTasksSupported === true && tasksQuery.isLoading)
  const canShowScheduledTasksWorkbench =
    scheduledTasksSupported !== false && !isLoadingTasks && !tasksQuery.isError

  const closeTaskDetail = () => {
    if (selectedTask === null) return
    setSelectedTask(null)
  }

  const openCreateReminder = () => {
    closeTaskDetail()
    setEditingTask(null)
    setEditorOpen(true)
  }

  const openEditReminder = (task: ScheduledTask) => {
    closeTaskDetail()
    setEditingTask(task)
    setEditorOpen(true)
  }

  const openTaskDetail = (task: ScheduledTask) => setSelectedTask(task)

  const closeEditor = () => {
    setEditorOpen(false)
    setEditingTask(null)
  }

  const refreshTasks = async () => {
    await tasksQuery.refetch()
  }

  const handleSubmit = async (
    payload: CreateScheduledTaskReminderPayload | UpdateScheduledTaskReminderPayload
  ) => {
    setSaving(true)
    try {
      if (editingTask) {
        await updateScheduledTaskReminder(editingTask.id, payload as UpdateScheduledTaskReminderPayload)
        message.success("Reminder task updated")
      } else {
        await createScheduledTaskReminder(payload as CreateScheduledTaskReminderPayload)
        message.success("Reminder task created")
      }
      closeEditor()
      await refreshTasks()
    } catch (error: any) {
      message.error(error?.message || "Unable to save reminder task")
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteReminder = async (task: ScheduledTask) => {
    try {
      await deleteScheduledTaskReminder(task.id)
      message.success("Reminder task deleted")
      await refreshTasks()
    } catch (error: any) {
      message.error(error?.message || "Unable to delete reminder task")
    }
  }

  const partialErrors = tasksQuery.data?.errors ?? []
  const unsupportedState = scheduledTasksSupported === false
    ? buildCapabilityState({
        featureName: "Scheduled tasks",
        capabilityName: "scheduled task management",
        endpoint: SCHEDULED_TASKS_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        reason: "unsupported"
      })
    : null
  const loadErrorState = tasksQuery.isError
    ? buildCapabilityState({
        featureName: "Scheduled tasks",
        capabilityName: "scheduled task management",
        endpoint: SCHEDULED_TASKS_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: tasksQuery.error
      })
    : null
  const partialState = tasksQuery.data?.partial
    ? buildCapabilityState({
        featureName: "Scheduled tasks",
        capabilityName: "scheduled task management",
        endpoint: SCHEDULED_TASKS_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        reason: "partial",
        partialErrors,
        message: "Some scheduled-task data loaded while one dependency could not be reached."
      })
    : null
  const scheduledTasksFeatureName = t("scheduledTasks:title", "Scheduled tasks")

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <Typography.Title level={2} style={{ marginBottom: 0 }}>
          {scheduledTasksFeatureName}
        </Typography.Title>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
          {t(
            "scheduledTasks:description",
            "Track reminders, Watchlist monitors, and recurring automation from one place. Use domain workspaces like Watchlists for deep source and output configuration."
          )}
        </Typography.Paragraph>
      </div>

      {isLoadingTasks ? <LoadingState /> : null}
      {scheduledTasksSupported === false ? (
        <RecoveryCallout
          state={unsupportedState?.state ?? "unavailable"}
          title={unsupportedState?.title ?? "Scheduled tasks are unavailable on this server"}
          message={
            unsupportedState?.message ??
            "The connected server does not advertise scheduled task management."
          }
          diagnostics={unsupportedState?.diagnostics}
          primaryAction={{
            label: "Health & diagnostics",
            onClick: () => navigate("/settings/health")
          }}
        />
      ) : null}
      {tasksQuery.isError ? (
        <RecoveryCallout
          state={loadErrorState?.state ?? "error"}
          title={loadErrorState?.title ?? "Unable to load scheduled tasks"}
          message={
            loadErrorState?.message ??
            "The scheduled tasks overview could not be loaded."
          }
          diagnostics={loadErrorState?.diagnostics}
          primaryAction={{
            label: "Try again",
            onClick: () => {
              void tasksQuery.refetch()
            }
          }}
          secondaryActions={[
            {
              label: "Health & diagnostics",
              onClick: () => navigate("/settings/health")
            }
          ]}
        />
      ) : null}
      {tasksQuery.data?.partial ? (
        <RecoveryCallout
          state={partialState?.state ?? "degraded"}
          title={partialState?.title ?? "Scheduled tasks are partially available"}
          message={
            partialState?.message ??
            "Some scheduled task data loaded, but one dependency could not be reached."
          }
          diagnostics={partialState?.diagnostics}
          primaryAction={{
            label: "Try again",
            onClick: () => {
              void tasksQuery.refetch()
            }
          }}
        />
      ) : null}

      {canShowScheduledTasksWorkbench ? (
        <>
          {hasLoadedTasks ? (
            <ScheduledTaskOverview
              tasks={tasks}
              partial={Boolean(tasksQuery.data?.partial)}
            />
          ) : null}

          {hasLoadedTasks && hasWatchlistJob ? (
            <Alert
              type="info"
              showIcon
              title="Watchlists remains the full workspace for monitor setup, source tuning, run activity, and reports."
            />
          ) : null}

          {hasLoadedTasks && tasks.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <Typography.Text strong>No scheduled tasks yet.</Typography.Text>
                  <Typography.Text type="secondary">
                    Create a reminder now. Automation templates for GitHub, YouTube, RAG, and
                    agents are planned follow-up phases.
                  </Typography.Text>
                </div>
              }
            >
              <Button type="primary" onClick={openCreateReminder}>
                Create scheduled task
              </Button>
            </Empty>
          ) : null}

          {hasLoadedTasks && tasks.length > 0 ? (
            <ScheduledTaskTable
              tasks={tasks}
              onCreateReminder={openCreateReminder}
              onInspectTask={openTaskDetail}
              onEditReminder={openEditReminder}
              onDeleteReminder={handleDeleteReminder}
            />
          ) : null}

          <ReminderTaskEditor
            open={editorOpen}
            task={editingTask}
            saving={saving}
            onClose={closeEditor}
            onSubmit={handleSubmit}
          />
        </>
      ) : null}
    </div>
  )
}

export default ScheduledTasksPage
