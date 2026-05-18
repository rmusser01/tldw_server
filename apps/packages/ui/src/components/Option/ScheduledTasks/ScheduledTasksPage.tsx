import React, { useState } from "react"
import { Spin, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  StatePanel,
  buildCapabilityState,
  classifyCapabilityError,
  messageFromError,
  statusFromError
} from "@/components/ui/state"
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

const SCHEDULED_TASKS_PATH = "/api/v1/scheduled-tasks"

export const ScheduledTasksPage: React.FC = () => {
  const navigate = useNavigate()
  const { t } = useTranslation(["scheduledTasks", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [editorOpen, setEditorOpen] = useState(false)
  const [editingTask, setEditingTask] = useState<ScheduledTask | null>(null)
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

  const openCreateReminder = () => {
    setEditingTask(null)
    setEditorOpen(true)
  }

  const openEditReminder = (task: ScheduledTask) => {
    setEditingTask(task)
    setEditorOpen(true)
  }

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
  const serverUrl = connectionConfig?.serverUrl?.trim()
  const scheduledTasksFeatureName = t("scheduledTasks:title", "Scheduled tasks")
  const scheduledTasksCapabilityName = t(
    "scheduledTasks:capability.scheduledTasks",
    "scheduled tasks"
  )
  const unsupportedState = buildCapabilityState({
    kind: "unavailable",
    featureName: scheduledTasksFeatureName,
    capabilityName: scheduledTasksCapabilityName,
    method: "GET",
    endpoint: SCHEDULED_TASKS_PATH,
    serverUrl,
    primaryAction: {
      label: t("scheduledTasks:actions.checkServerSetup", "Check server setup"),
      onClick: () => {
        navigate("/settings/health")
      }
    }
  })
  const loadErrorState = tasksQuery.isError
    ? buildCapabilityState({
        kind: classifyCapabilityError(tasksQuery.error),
        featureName: scheduledTasksFeatureName,
        capabilityName: scheduledTasksCapabilityName,
        method: "GET",
        endpoint: SCHEDULED_TASKS_PATH,
        status: statusFromError(tasksQuery.error),
        rawMessage: messageFromError(tasksQuery.error) || t(
          "scheduledTasks:errors.overviewLoadFailed",
          "The scheduled tasks overview could not be loaded."
        ),
        primaryAction: {
          label: t("common:actions.retry", "Try again"),
          onClick: () => {
            void tasksQuery.refetch()
          }
        }
      })
    : null
  const degradedState = tasksQuery.data?.partial
    ? buildCapabilityState({
        kind: "degraded",
        featureName: scheduledTasksFeatureName,
        rawMessage: partialErrors.length
          ? partialErrors.join(", ")
          : t(
              "scheduledTasks:errors.partialOverview",
              "The overview is partially available."
            ),
        primaryAction: {
          label: t("common:actions.refresh", "Refresh"),
          onClick: () => {
            void tasksQuery.refetch()
          }
        }
      })
    : null

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <Typography.Title level={2} style={{ marginBottom: 0 }}>
          {scheduledTasksFeatureName}
        </Typography.Title>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
          {t(
            "scheduledTasks:description",
            "Review reminder tasks here. Watchlist jobs remain managed from Watchlists."
          )}
        </Typography.Paragraph>
      </div>

      {connectionConfigLoading || scheduledTasksSupported === null ? <Spin /> : null}
      {scheduledTasksSupported === false ? (
        <StatePanel
          state={unsupportedState.state}
          title={unsupportedState.title}
          message={unsupportedState.message}
          diagnostics={unsupportedState.diagnostics}
          primaryAction={unsupportedState.primaryAction}
        />
      ) : null}
      {tasksQuery.isLoading ? <Spin /> : null}
      {loadErrorState ? (
        <StatePanel
          state={loadErrorState.state}
          title={loadErrorState.title}
          message={loadErrorState.message}
          diagnostics={loadErrorState.diagnostics}
          primaryAction={loadErrorState.primaryAction}
          role="alert"
        />
      ) : null}
      {degradedState ? (
        <StatePanel
          state={degradedState.state}
          title={degradedState.title}
          message={degradedState.message}
          diagnostics={degradedState.diagnostics}
          primaryAction={degradedState.primaryAction}
        />
      ) : null}

      {scheduledTasksSupported === false ? null : (
        <>
          <ScheduledTaskTable
            tasks={tasks}
            onCreateReminder={openCreateReminder}
            onEditReminder={openEditReminder}
            onDeleteReminder={handleDeleteReminder}
          />

          <ReminderTaskEditor
            open={editorOpen}
            task={editingTask}
            saving={saving}
            onClose={closeEditor}
            onSubmit={handleSubmit}
          />
        </>
      )}
    </div>
  )
}

export default ScheduledTasksPage
