import React, { useState } from "react"
import { Alert, Button, Empty, Space, Spin, Tabs, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate, useSearchParams } from "react-router-dom"
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
import { ScheduledTaskDetailDrawer } from "./ScheduledTaskDetailDrawer"
import { ScheduledTaskCreatePanel } from "./ScheduledTaskCreatePanel"
import { DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES } from "./scheduled-task-template-capabilities"
import {
  SCHEDULED_TASK_TABS,
  buildScheduledTaskSearch,
  parseScheduledTaskRouteState,
  type ScheduledTaskTabId
} from "./scheduled-task-route-state"
import {
  findScheduledTaskResultByRouteState,
  projectScheduledTaskResults
} from "./scheduled-task-results"
import {
  getScheduledTaskTemplate,
  type ScheduledTaskTemplateId
} from "./scheduled-task-templates"

const SCHEDULED_TASKS_PATH = "/api/v1/scheduled-tasks"
const SCHEDULED_TASKS_SUPPORT_PROBE_TIMEOUT_MS = 8000

const LoadingState: React.FC = () => (
  <div role="status" aria-live="polite">
    <Space>
      <Spin size="small" />
      <Typography.Text type="secondary">Loading tasks and latest run state</Typography.Text>
    </Space>
  </div>
)

export const ScheduledTasksPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { t } = useTranslation(["scheduledTasks", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [editorOpen, setEditorOpen] = useState(false)
  const [editingTask, setEditingTask] = useState<ScheduledTask | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [createdTaskFallback, setCreatedTaskFallback] = useState<ScheduledTask | null>(null)
  const [saving, setSaving] = useState(false)
  const [scheduledTasksSupported, setScheduledTasksSupported] = useState<
    boolean | null
  >(null)
  const routeState = React.useMemo(
    () =>
      parseScheduledTaskRouteState(searchParams, {
        defaultTab: location.pathname.replace(/\/+$/, "").endsWith("/scheduled-tasks/results")
          ? "results"
          : "overview"
      }),
    [location.pathname, searchParams]
  )

  const updateRoute = React.useCallback(
    ({
      tab,
      templateId,
      taskId,
      runId,
      resultId
    }: {
      tab: ScheduledTaskTabId
      templateId?: string | null
      taskId?: string | null
      runId?: string | null
      resultId?: string | null
    }) => {
      setSearchParams(
        new URLSearchParams(
          buildScheduledTaskSearch({ tab, templateId, taskId, runId, resultId })
        )
      )
    },
    [setSearchParams]
  )

  React.useEffect(() => {
    if (connectionConfigLoading) return

    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) {
      setScheduledTasksSupported(true)
      return
    }

    let cancelled = false
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => {
      controller.abort()
    }, SCHEDULED_TASKS_SUPPORT_PROBE_TIMEOUT_MS)

    const probeScheduledTasksSupport = async () => {
      try {
        const response = await fetch(`${serverUrl}/openapi.json`, {
          signal: controller.signal
        })
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
      } finally {
        window.clearTimeout(timeoutId)
      }
    }

    void probeScheduledTasksSupport()

    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
      controller.abort()
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading])

  const tasksQuery = useQuery({
    queryKey: ["scheduled-tasks"],
    queryFn: listScheduledTasks,
    enabled: scheduledTasksSupported === true
  })

  const tasks = tasksQuery.data?.items ?? []
  const projectedResults = React.useMemo(
    () => projectScheduledTaskResults(tasks, { includeCompletedNoResults: true }),
    [tasks]
  )
  const selectedResult = React.useMemo(
    () =>
      routeState.tab === "results"
        ? findScheduledTaskResultByRouteState(projectedResults, routeState)
        : null,
    [projectedResults, routeState]
  )
  const selectedTask = React.useMemo(
    () => {
      const refreshedTask = tasks.find((task) => task.id === selectedTaskId)
      if (refreshedTask) return refreshedTask
      return createdTaskFallback?.id === selectedTaskId ? createdTaskFallback : null
    },
    [createdTaskFallback, selectedTaskId, tasks]
  )
  const hasLoadedTasks = Boolean(tasksQuery.data)
  const hasWatchlistJob = tasks.some((task) => task.primitive === "watchlist_job")
  const selectedTemplate = React.useMemo(
    () =>
      routeState.tab === "create"
        ? getScheduledTaskTemplate(routeState.templateId)
        : null,
    [routeState.tab, routeState.templateId]
  )
  const selectedTemplateId = selectedTemplate?.id ?? null
  const isLoadingTasks =
    connectionConfigLoading ||
    scheduledTasksSupported === null ||
    (scheduledTasksSupported === true && tasksQuery.isLoading)
  const canShowScheduledTasksWorkbench =
    scheduledTasksSupported !== false && !isLoadingTasks && !tasksQuery.isError

  React.useEffect(() => {
    if (routeState.tab !== "tasks") {
      if (selectedTaskId !== null) {
        setSelectedTaskId(null)
      }
      return
    }

    if (!hasLoadedTasks) return

    if (!routeState.taskId) {
      if (selectedTaskId !== null) {
        setSelectedTaskId(null)
      }
      return
    }

    const routeTaskExists = tasks.some((task) => task.id === routeState.taskId)
    if (routeTaskExists) {
      if (selectedTaskId !== routeState.taskId) {
        setSelectedTaskId(routeState.taskId)
      }
      return
    }

    if (createdTaskFallback?.id === routeState.taskId) {
      if (selectedTaskId !== routeState.taskId) {
        setSelectedTaskId(routeState.taskId)
      }
      return
    }

    if (selectedTaskId === routeState.taskId) {
      setSelectedTaskId(null)
      updateRoute({ tab: "tasks" })
      return
    }

    if (selectedTaskId !== null) {
      setSelectedTaskId(null)
    }
  }, [
    createdTaskFallback?.id,
    hasLoadedTasks,
    routeState.tab,
    routeState.taskId,
    selectedTaskId,
    tasks,
    updateRoute
  ])

  React.useEffect(() => {
    if (!createdTaskFallback) return

    if (tasks.some((task) => task.id === createdTaskFallback.id)) {
      setCreatedTaskFallback(null)
      return
    }

    if (selectedTaskId !== createdTaskFallback.id) {
      setCreatedTaskFallback(null)
    }
  }, [createdTaskFallback, selectedTaskId, tasks])

  const closeTaskDetail = () => {
    if (selectedTaskId === null) return
    setCreatedTaskFallback(null)
    setSelectedTaskId(null)
    updateRoute({ tab: "tasks" })
  }

  const openCreateReminder = () => {
    closeTaskDetail()
    setEditingTask(null)
    updateRoute({ tab: "create" })
  }

  const openEditReminder = (task: ScheduledTask) => {
    closeTaskDetail()
    setEditingTask(task)
    setEditorOpen(true)
  }

  const openTaskDetail = (task: ScheduledTask) => {
    if (createdTaskFallback?.id !== task.id) {
      setCreatedTaskFallback(null)
    }
    setSelectedTaskId(task.id)
    updateRoute({ tab: "tasks", taskId: task.id })
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

  const handleCreateReminderFromPanel = async (
    payload: CreateScheduledTaskReminderPayload
  ) => {
    setSaving(true)
    try {
      const createdTask = await createScheduledTaskReminder(payload)
      setCreatedTaskFallback(createdTask)
      setSelectedTaskId(createdTask.id)
      updateRoute({ tab: "tasks", taskId: createdTask.id })
      message.success("Reminder scheduled. Status appears in Tasks.")
      await refreshTasks()
    } catch (error: any) {
      message.error(error?.message || "Unable to save reminder task")
    } finally {
      setSaving(false)
    }
  }

  const handleTabChange = (tab: string) => {
    updateRoute({ tab: tab as ScheduledTaskTabId })
  }

  const handleSelectTemplate = (templateId: ScheduledTaskTemplateId | null) => {
    updateRoute({ tab: "create", templateId })
  }

  const handleDeleteReminder = async (task: ScheduledTask) => {
    try {
      await deleteScheduledTaskReminder(task.id)
      message.success("Reminder task deleted")
      closeTaskDetail()
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
  const missingRouteTask =
    routeState.tab === "tasks" &&
    hasLoadedTasks &&
    Boolean(routeState.taskId) &&
    !tasks.some((task) => task.id === routeState.taskId) &&
    selectedTaskId !== routeState.taskId
  const hasResultRouteTarget =
    routeState.tab === "results" &&
    Boolean(routeState.resultId || routeState.runId || routeState.taskId)
  const missingRouteResult =
    hasLoadedTasks &&
    hasResultRouteTarget &&
    selectedResult === null

  const renderOverviewTab = () => (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
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
    </Space>
  )

  const openResultSignal = (
    resultId: string | null,
    runId: string | null,
    taskId: string
  ) => {
    updateRoute({ tab: "results", resultId, runId, taskId })
  }

  const renderResultsTab = () => (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
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
        title="Latest automation signals"
        description="Latest signals inferred from task status. Durable review state appears when the results API is available."
      />

      {missingRouteResult ? (
        <Alert type="warning" showIcon title="Result signal not found." />
      ) : null}

      {selectedResult ? (
        <Alert
          type={selectedResult.severity === "error" ? "error" : "success"}
          showIcon
          title={`Selected signal: ${selectedResult.title}`}
          description={selectedResult.summary}
        />
      ) : null}

      {hasLoadedTasks && tasks.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <Typography.Text strong>No scheduled tasks yet</Typography.Text>
              <Typography.Text type="secondary">
                Results and failures appear here after an automation runs.
              </Typography.Text>
            </div>
          }
        >
          <Button type="primary" onClick={openCreateReminder}>
            Create scheduled task
          </Button>
        </Empty>
      ) : null}

      {hasLoadedTasks && tasks.length > 0 && projectedResults.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <Typography.Text strong>No results to review</Typography.Text>
              <Typography.Text type="secondary">
                The latest scheduled runs have not produced new results or failures.
              </Typography.Text>
            </div>
          }
        />
      ) : null}

      {projectedResults.length > 0 ? (
        <section aria-label="Scheduled task results">
          <Space orientation="vertical" size={12} style={{ width: "100%" }}>
            {projectedResults.map((result) => (
              <div
                key={result.id}
                style={{
                  border: "1px solid var(--ant-color-border, #d9d9d9)",
                  borderRadius: 8,
                  padding: 12
                }}
              >
                <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                  <Typography.Text strong>{result.title}</Typography.Text>
                  <Typography.Text type="secondary">{result.summary}</Typography.Text>
                  <Typography.Text type="secondary">
                    {result.ownerLabel} - {result.state.replace(/_/g, " ")}
                  </Typography.Text>
                  <Button
                    type="link"
                    style={{ alignSelf: "flex-start", paddingInline: 0 }}
                    onClick={() => {
                      openResultSignal(result.resultId, result.runId, result.taskId)
                    }}
                  >
                    Open signal for {result.title}
                  </Button>
                </Space>
              </div>
            ))}
          </Space>
        </section>
      ) : null}
    </Space>
  )

  const renderTasksTab = () => (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
      {missingRouteTask ? <Alert type="warning" showIcon title="Task not found." /> : null}

      {hasLoadedTasks && tasks.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <Typography.Text strong>No scheduled tasks yet.</Typography.Text>
              <Typography.Text type="secondary">
                Create a reminder now. Watch and Ingest setup continue in their owner
                workspaces until capability, preview, duplicate, creation, and result
                contracts are available.
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
    </Space>
  )

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
      {routeState.invalidTab ? (
        <Alert
          type="warning"
          showIcon
          title="That tab is not available. Showing Overview."
        />
      ) : null}
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
          <Tabs
            activeKey={routeState.tab}
            onChange={handleTabChange}
            items={SCHEDULED_TASK_TABS.map((tab) => ({
              key: tab.id,
              label: tab.label,
              children:
                tab.id === "overview"
                  ? renderOverviewTab()
                  : tab.id === "results"
                    ? renderResultsTab()
                  : tab.id === "tasks"
                    ? renderTasksTab()
                    : (
                        <ScheduledTaskCreatePanel
                          selectedTemplateId={selectedTemplateId}
                          onSelectTemplate={handleSelectTemplate}
                          onCreateReminder={handleCreateReminderFromPanel}
                          savingReminder={saving}
                          templateCapabilities={DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES}
                        />
                      )
            }))}
          />

          <ReminderTaskEditor
            open={editorOpen}
            task={editingTask}
            saving={saving}
            onClose={closeEditor}
            onSubmit={handleSubmit}
          />
          <ScheduledTaskDetailDrawer
            open={Boolean(selectedTask)}
            task={selectedTask}
            onClose={closeTaskDetail}
            onEditReminder={openEditReminder}
            onDeleteReminder={handleDeleteReminder}
          />
        </>
      ) : null}
    </div>
  )
}

export default ScheduledTasksPage
