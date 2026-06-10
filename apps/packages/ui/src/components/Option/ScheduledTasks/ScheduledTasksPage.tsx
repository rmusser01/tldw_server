import React, { useState } from "react"
import { Drawer, Space, Tabs, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate, useSearchParams } from "react-router-dom"
import { EmptyState, LoadingState as DesignSystemLoadingState } from "@/components/ui/feedback"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  archiveScheduledTaskDefinition,
  createScheduledTaskDefinition,
  createScheduledTaskReminder,
  createScheduledTaskPreview,
  deleteScheduledTaskReminder,
  duplicateScheduledTaskDefinition,
  getScheduledTaskCapabilities,
  getScheduledTaskDefinition,
  listScheduledTaskDefinitionAudit,
  listScheduledTaskPreviews,
  listScheduledTasks,
  pauseScheduledTaskDefinition,
  resumeScheduledTaskDefinition,
  updateScheduledTaskDefinition,
  updateScheduledTaskReminder,
  type ScheduledTask,
  type ScheduledTaskDefinitionResponse,
  type ScheduledTaskPreviewCreateRequest,
  type CreateScheduledTaskReminderPayload,
  type UpdateScheduledTaskReminderPayload
} from "@/services/scheduled-tasks-control-plane"
import { ScheduledTaskTable } from "./ScheduledTaskTable"
import { ReminderTaskEditor } from "./ReminderTaskEditor"
import { ScheduledTaskOverview } from "./ScheduledTaskOverview"
import { ScheduledTaskDetailDrawer } from "./ScheduledTaskDetailDrawer"
import { ScheduledTaskCreatePanel } from "./ScheduledTaskCreatePanel"
import {
  ScheduledTaskAutomationDefinitionEditor,
  type ScheduledTaskAutomationEditorScheduleKind,
  type ScheduledTaskAutomationDefinitionEditorValues
} from "./ScheduledTaskAutomationDefinitionEditor"
import { ScheduledTaskResultsPanel } from "./ScheduledTaskResultsPanel"
import { ScheduledTaskResultDetailDrawer } from "./ScheduledTaskResultDetailDrawer"
import { DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES } from "./scheduled-task-template-capabilities"
import {
  SCHEDULED_TASK_TABS,
  buildScheduledTaskSearch,
  parseScheduledTaskRouteState,
  type ScheduledTaskTabId
} from "./scheduled-task-route-state"
import {
  findScheduledTaskResultByRouteState,
  projectScheduledTaskResults,
  type ScheduledTaskResultItem
} from "./scheduled-task-results"
import {
  getScheduledTaskTemplate,
  type ScheduledTaskTemplateId
} from "./scheduled-task-templates"

const SCHEDULED_TASKS_PATH = "/api/v1/scheduled-tasks"
const SCHEDULED_TASKS_SUPPORT_PROBE_TIMEOUT_MS = 8000
const SUPPORTED_AUTOMATION_SCHEDULE_KINDS = new Set([
  "one_time",
  "interval",
  "daily",
  "weekly",
  "cron"
])

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const readStringField = (
  source: Record<string, unknown> | null | undefined,
  key: string
): string => {
  const value = source?.[key]
  return typeof value === "string" ? value : ""
}

const readStringListField = (
  source: Record<string, unknown> | null | undefined,
  key: string
): string => {
  const value = source?.[key]
  if (Array.isArray(value)) {
    return value
      .filter((entry): entry is string => typeof entry === "string" && Boolean(entry.trim()))
      .join(", ")
  }
  return typeof value === "string" ? value : ""
}

const stringifyEditorJson = (value: unknown): string => {
  if (!isRecord(value)) return "{}"

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return "{}"
  }
}

const stringifyEditorText = (value: unknown): string => {
  if (typeof value === "string") return value
  if (!isRecord(value)) return ""

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return ""
  }
}

const normalizeAutomationScheduleKind = (
  value: unknown
): ScheduledTaskAutomationEditorScheduleKind =>
  typeof value === "string" && SUPPORTED_AUTOMATION_SCHEDULE_KINDS.has(value)
    ? (value as ScheduledTaskAutomationEditorScheduleKind)
    : "daily"

const getAutomationDefinitionIdForTask = (task: ScheduledTask): string => {
  const sourceDefinitionId = task.source_ref?.definition_id
  if (typeof sourceDefinitionId === "string" && sourceDefinitionId.trim()) {
    return sourceDefinitionId
  }
  return task.id
}

const buildAutomationEditorInitialValues = (
  definition: ScheduledTaskDefinitionResponse
): ScheduledTaskAutomationDefinitionEditorValues => {
  const schedule = isRecord(definition.schedule) ? definition.schedule : {}
  const input = isRecord(definition.input) ? definition.input : {}
  const config = isRecord(definition.config) ? definition.config : {}
  const visibilityPolicy = isRecord(definition.visibility_policy)
    ? definition.visibility_policy
    : {}
  const approvalPolicy = isRecord(definition.approval_policy)
    ? definition.approval_policy
    : {}
  const visibility = visibilityPolicy.visibility === "shared" ? "shared" : "private"
  const approvalMode = approvalPolicy.mode === "manual" ? "manual" : "none"

  return {
    name: definition.name,
    description: definition.description ?? "",
    schedule,
    scheduleKind: normalizeAutomationScheduleKind(schedule.kind),
    cron: readStringField(schedule, "cron"),
    timezone: readStringField(schedule, "timezone") || "UTC",
    visibility,
    question: readStringField(input, "question"),
    successCriteria: readStringField(input, "success_criteria"),
    scopeJson: stringifyEditorJson(input.scope),
    agentRef: stringifyEditorText(input.agent_ref),
    message: readStringField(input, "message"),
    allowedToolClasses: readStringListField(config, "allowed_tool_classes"),
    deniedToolClasses: readStringListField(config, "denied_tool_classes"),
    approvalMode
  }
}

export const ScheduledTasksPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { t } = useTranslation(["scheduledTasks", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [editorOpen, setEditorOpen] = useState(false)
  const [editingTask, setEditingTask] = useState<ScheduledTask | null>(null)
  const [editingAutomationTask, setEditingAutomationTask] =
    useState<ScheduledTask | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [createdTaskFallback, setCreatedTaskFallback] = useState<ScheduledTask | null>(null)
  const [saving, setSaving] = useState(false)
  const [automationErrorMessage, setAutomationErrorMessage] = useState<string | null>(null)
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
      const normalizedPath = location.pathname.replace(/\/+$/, "")
      const onResultsAlias = normalizedPath.endsWith("/scheduled-tasks/results")
      if (tab === "overview" && onResultsAlias) {
        navigate("/scheduled-tasks")
        return
      }

      setSearchParams(
        new URLSearchParams(
          buildScheduledTaskSearch({ tab, templateId, taskId, runId, resultId })
        )
      )
    },
    [location.pathname, navigate, setSearchParams]
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
  const automationCapabilitiesQuery = useQuery({
    queryKey: ["scheduled-task-automation-capabilities"],
    queryFn: getScheduledTaskCapabilities,
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
  const selectedTaskLatestResult = React.useMemo(
    () =>
      selectedTask
        ? projectedResults.find((result) => result.taskId === selectedTask.id) ?? null
        : null,
    [projectedResults, selectedTask]
  )
  const selectedAutomationDefinitionId = React.useMemo(() => {
    if (selectedTask?.primitive !== "automation_definition") return null
    const sourceDefinitionId = selectedTask.source_ref?.definition_id
    return typeof sourceDefinitionId === "string" && sourceDefinitionId.trim()
      ? sourceDefinitionId
      : selectedTask.id
  }, [selectedTask])
  const selectedAutomationDefinitionQuery = useQuery({
    queryKey: ["scheduled-task-definition", selectedAutomationDefinitionId],
    queryFn: () => getScheduledTaskDefinition(selectedAutomationDefinitionId as string),
    enabled: Boolean(selectedAutomationDefinitionId)
  })
  const selectedAutomationPreviewsQuery = useQuery({
    queryKey: ["scheduled-task-definition-previews", selectedAutomationDefinitionId],
    queryFn: () =>
      listScheduledTaskPreviews({
        definition_id: selectedAutomationDefinitionId,
        limit: 10
      }),
    enabled: Boolean(selectedAutomationDefinitionId)
  })
  const selectedAutomationAuditQuery = useQuery({
    queryKey: ["scheduled-task-definition-audit", selectedAutomationDefinitionId],
    queryFn: () =>
      listScheduledTaskDefinitionAudit(selectedAutomationDefinitionId as string, {
        limit: 10
      }),
    enabled: Boolean(selectedAutomationDefinitionId)
  })
  const editingAutomationDefinitionId = React.useMemo(
    () =>
      editingAutomationTask
        ? getAutomationDefinitionIdForTask(editingAutomationTask)
        : null,
    [editingAutomationTask]
  )
  const editingAutomationDefinitionQuery = useQuery({
    queryKey: ["scheduled-task-definition-edit", editingAutomationDefinitionId],
    queryFn: () => getScheduledTaskDefinition(editingAutomationDefinitionId as string),
    enabled: Boolean(editingAutomationDefinitionId)
  })
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

  const closeResultDetail = () => {
    updateRoute({ tab: "results" })
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

  const openEditAutomationDefinition = (task: ScheduledTask) => {
    setAutomationErrorMessage(null)
    setEditorOpen(false)
    setEditingTask(null)
    setEditingAutomationTask(task)
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

  const closeAutomationDefinitionEditor = () => {
    setEditingAutomationTask(null)
  }

  const refreshTasks = async () => {
    await tasksQuery.refetch()
  }

  const readScheduledTaskControlPlaneError = (
    error: unknown,
    fallback: string
  ): string => {
    if (error && typeof error === "object") {
      const details = "details" in error ? (error as { details?: unknown }).details : null
      const detail =
        details && typeof details === "object" && "detail" in details
          ? (details as { detail?: unknown }).detail
          : "detail" in error
            ? (error as { detail?: unknown }).detail
            : null
      if (detail && typeof detail === "object") {
        const code =
          "code" in detail ? String((detail as { code?: unknown }).code || "") : ""
        const detailMessage =
          "message" in detail
            ? String((detail as { message?: unknown }).message || "")
            : ""
        if (code && detailMessage) return `${code}: ${detailMessage}`
        if (detailMessage) return detailMessage
        if (code) return code
      }
      if ("message" in error && typeof (error as { message?: unknown }).message === "string") {
        return (error as { message: string }).message
      }
    }
    return fallback
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

  const handlePreviewAutomationDefinition = async (
    payload: ScheduledTaskPreviewCreateRequest
  ) => {
    setAutomationErrorMessage(null)
    try {
      return await createScheduledTaskPreview(payload)
    } catch (error) {
      const messageText = readScheduledTaskControlPlaneError(
        error,
        "Unable to preview definition"
      )
      setAutomationErrorMessage(messageText)
      throw error
    }
  }

  const handleCreateAutomationDefinition = async (payload: {
    preview_id: string
    initial_lifecycle?: "configured" | "paused"
  }) => {
    setAutomationErrorMessage(null)
    try {
      return await createScheduledTaskDefinition(payload)
    } catch (error) {
      const messageText = readScheduledTaskControlPlaneError(
        error,
        "Unable to save definition"
      )
      setAutomationErrorMessage(messageText)
      throw error
    }
  }

  const handleAutomationDefinitionCreated = async (
    definition: ScheduledTaskDefinitionResponse | unknown
  ) => {
    await refreshTasks()
    const definitionId =
      definition && typeof definition === "object" && "id" in definition
        ? String((definition as { id?: unknown }).id || "")
        : ""
    if (definitionId) {
      setSelectedTaskId(`automation_definition:${definitionId}`)
    }
    updateRoute({ tab: "tasks", taskId: definitionId ? `automation_definition:${definitionId}` : null })
  }

  const handleUpdateAutomationDefinition = async (definitionId: string, payload: { preview_id: string }) => {
    setAutomationErrorMessage(null)
    try {
      const updated = await updateScheduledTaskDefinition(definitionId, payload)
      await refreshTasks()
      return updated
    } catch (error) {
      const messageText = readScheduledTaskControlPlaneError(
        error,
        "Unable to update definition"
      )
      setAutomationErrorMessage(messageText)
      throw error
    }
  }

  const handleAutomationLifecycleAction = async (
    task: ScheduledTask,
    action: "pause" | "resume" | "archive" | "duplicate"
  ) => {
    setAutomationErrorMessage(null)
    try {
      if (action === "pause") {
        await pauseScheduledTaskDefinition(task.id)
      } else if (action === "resume") {
        await resumeScheduledTaskDefinition(task.id)
      } else if (action === "archive") {
        await archiveScheduledTaskDefinition(task.id)
      } else {
        await duplicateScheduledTaskDefinition(task.id, {
          name: `${task.title} copy`
        })
      }
      await refreshTasks()
    } catch (error) {
      const messageText = readScheduledTaskControlPlaneError(
        error,
        "Unable to update automation definition"
      )
      setAutomationErrorMessage(messageText)
      message.error(messageText)
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
          results={projectedResults}
          onOpenResult={openResultSignal}
        />
      ) : null}

      {hasLoadedTasks && hasWatchlistJob ? (
        <DesignSystemAlert
          variant="info"
          title="Watchlists remains the full workspace for monitor setup, source tuning, run activity, and reports."
        />
      ) : null}
    </Space>
  )

  const openResultSignal = (result: ScheduledTaskResultItem) => {
    updateRoute({
      tab: "results",
      resultId: result.resultId,
      runId: result.runId,
      taskId: result.taskId
    })
  }

  const openTaskResults = (task: ScheduledTask) => {
    updateRoute({ tab: "results", taskId: task.id })
  }

  const renderResultsTab = () => (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
      {missingRouteResult ? (
        <DesignSystemAlert variant="warning" title="Result signal not found." />
      ) : null}
      <ScheduledTaskResultsPanel
        results={projectedResults}
        taskCount={tasks.length}
        capabilityMode="projected_signals"
        onCreateTask={openCreateReminder}
        onOpenResult={openResultSignal}
      />
    </Space>
  )

  const renderTasksTab = () => (
    <Space orientation="vertical" size={16} style={{ width: "100%" }}>
      {missingRouteTask ? (
        <DesignSystemAlert variant="warning" title="Task not found." />
      ) : null}

      {hasLoadedTasks && tasks.length === 0 ? (
        <EmptyState
          variant="inline"
          title="No scheduled tasks yet."
          description="Create a reminder now. Watch and Ingest setup continue in their owner workspaces until capability, preview, duplicate, creation, and result contracts are available."
          primaryAction={{
            label: "Create scheduled task",
            onClick: openCreateReminder
          }}
        />
      ) : null}

      {hasLoadedTasks && tasks.length > 0 ? (
        <ScheduledTaskTable
          tasks={tasks}
          results={projectedResults}
          onCreateReminder={openCreateReminder}
          onInspectTask={openTaskDetail}
          onOpenTaskResults={openTaskResults}
          onEditReminder={openEditReminder}
          onDeleteReminder={handleDeleteReminder}
          onEditAutomationDefinition={openEditAutomationDefinition}
          onPauseAutomationDefinition={(task) => {
            void handleAutomationLifecycleAction(task, "pause")
          }}
          onResumeAutomationDefinition={(task) => {
            void handleAutomationLifecycleAction(task, "resume")
          }}
          onArchiveAutomationDefinition={(task) => {
            void handleAutomationLifecycleAction(task, "archive")
          }}
          onDuplicateAutomationDefinition={(task) => {
            void handleAutomationLifecycleAction(task, "duplicate")
          }}
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

      {isLoadingTasks ? (
        <div role="status" aria-live="polite">
          <DesignSystemLoadingState
            mode="inline"
            size="sm"
            label="Loading tasks and latest run state"
            className="w-full"
          />
        </div>
      ) : null}
      {routeState.invalidTab ? (
        <DesignSystemAlert
          variant="warning"
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
      {automationErrorMessage ? (
        <DesignSystemAlert variant="error" title={automationErrorMessage} />
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
                          automationCapabilities={automationCapabilitiesQuery.data ?? null}
                          onPreviewAutomationDefinition={handlePreviewAutomationDefinition}
                          onCreateAutomationDefinition={handleCreateAutomationDefinition}
                          onAutomationDefinitionCreated={(definition) => {
                            void handleAutomationDefinitionCreated(definition)
                          }}
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
          {selectedTask ? (
            <ScheduledTaskDetailDrawer
              open
              task={selectedTask}
              latestResult={selectedTaskLatestResult}
              automationDefinition={selectedAutomationDefinitionQuery.data ?? null}
              automationPreviewHistory={selectedAutomationPreviewsQuery.data?.items ?? []}
              automationAuditEvents={selectedAutomationAuditQuery.data?.items ?? []}
              onClose={closeTaskDetail}
              onEditReminder={openEditReminder}
              onDeleteReminder={handleDeleteReminder}
              onEditAutomationDefinition={openEditAutomationDefinition}
              onPauseAutomationDefinition={(task) => {
                void handleAutomationLifecycleAction(task, "pause")
              }}
              onResumeAutomationDefinition={(task) => {
                void handleAutomationLifecycleAction(task, "resume")
              }}
              onArchiveAutomationDefinition={(task) => {
                void handleAutomationLifecycleAction(task, "archive")
              }}
              onDuplicateAutomationDefinition={(task) => {
                void handleAutomationLifecycleAction(task, "duplicate")
              }}
            />
          ) : null}
          {editingAutomationTask ? (
            <Drawer
              title="Edit automation definition"
              open
              onClose={closeAutomationDefinitionEditor}
              size={680}
            >
              {editingAutomationDefinitionQuery.isLoading ? (
                <div role="status" aria-live="polite">
                  <DesignSystemLoadingState
                    mode="inline"
                    size="sm"
                    label="Loading automation definition"
                    className="w-full"
                  />
                </div>
              ) : editingAutomationDefinitionQuery.data ? (
                <ScheduledTaskAutomationDefinitionEditor
                  key={`${editingAutomationDefinitionQuery.data.id}:${editingAutomationDefinitionQuery.data.version}`}
                  family={editingAutomationDefinitionQuery.data.family}
                  mode="update"
                  definitionId={editingAutomationDefinitionQuery.data.id}
                  definitionVersion={editingAutomationDefinitionQuery.data.version}
                  initialValues={buildAutomationEditorInitialValues(
                    editingAutomationDefinitionQuery.data
                  )}
                  onPreview={handlePreviewAutomationDefinition}
                  onUpdate={(payload) =>
                    handleUpdateAutomationDefinition(
                      editingAutomationDefinitionQuery.data.id,
                      payload
                    )
                  }
                  onCancel={closeAutomationDefinitionEditor}
                  onSaved={() => {
                    closeAutomationDefinitionEditor()
                  }}
                />
              ) : (
                <Typography.Text type="secondary">
                  Automation definition could not be loaded.
                </Typography.Text>
              )}
            </Drawer>
          ) : null}
          {routeState.tab === "results" && selectedResult ? (
            <ScheduledTaskResultDetailDrawer
              open
              result={selectedResult}
              onClose={closeResultDetail}
              onReviewResult={() => undefined}
              onRetryRun={() => undefined}
            />
          ) : null}
        </>
      ) : null}
    </div>
  )
}

export default ScheduledTasksPage
