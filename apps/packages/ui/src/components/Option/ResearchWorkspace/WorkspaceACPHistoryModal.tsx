import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { Button, Collapse, Empty, Modal, Spin, Tag } from "antd"
import { ExternalLink } from "lucide-react"

import { Alert as DSAlert } from "@/components/ui/primitives"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  getWorkspaceTaskDetail,
  listCanonicalWorkspaceProjects,
  listCanonicalWorkspaceSessions,
  listWorkspaceProjectTasks,
  type WorkspaceHistoryRequestOptions
} from "@/services/workspace-acp-history"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { formatErrorMessage } from "@/utils/format-error-message"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import type {
  ACPSessionListItem
} from "@/services/acp/types"

const AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE =
  "Agent orchestration is not available on this server."
const AGENT_ORCHESTRATION_UNSUPPORTED_CODE =
  "AGENT_ORCHESTRATION_UNSUPPORTED"
const MAX_TASK_ROWS = 12
const MAX_RECENT_RUNS = 6

type CanonicalWorkspaceLink = {
  canonical_workspace_id?: string | null
  link_status?: string | null
}

type ProjectSummary = {
  id: number
  name: string
  metadata?: Record<string, unknown> | null
  canonical_workspace?: CanonicalWorkspaceLink | null
}

type TaskSummary = {
  id: number
  project_id: number
  title: string
  status: string
  created_at?: string | null
  updated_at?: string | null
  metadata?: Record<string, unknown> | null
  canonical_workspace?: CanonicalWorkspaceLink | null
}

type RunItem = {
  id: number
  task_id: number
  session_id?: string | null
  agent_type?: string | null
  status: string
  result_summary?: string | null
  error?: string | null
  started_at?: string | null
  completed_at?: string | null
  session?: {
    session_id?: string | null
    available?: boolean
    links?: Record<string, string>
  } | null
  history?: {
    audit_event_count?: number
    artifact_count?: number
    diagnostic_count?: number
    event_count?: number
    result?: {
      preview?: string | null
    } | null
  } | null
  failure_context?: {
    reason_code?: string | null
    message?: string | null
  } | null
}

type TaskDetail = TaskSummary & {
  runs?: RunItem[]
}

type WorkspaceTaskRow = {
  project: ProjectSummary
  task: TaskSummary
}

type TaskDetailState = {
  runs?: RunItem[]
  task?: TaskDetail
  loading?: boolean
  error?: string | null
}

type WorkspaceACPHistoryError =
  | { kind: "message"; message: string }
  | { kind: "unsupported" }

export type WorkspaceACPRunArtifactSaveInput = {
  workspaceId: string
  scope: ServicePromptSnapshot
  isCurrent: () => boolean
  projectId: number
  projectName: string
  taskId: number
  taskTitle: string
  runId: number
  status: string
  resultPreview: string
  sessionId: string | null
  agentType?: string | null
  startedAt?: string | null
  completedAt?: string | null
  links?: Record<string, string>
  history?: {
    audit_event_count?: number
    artifact_count?: number
    diagnostic_count?: number
    event_count?: number
    result?: {
      preview?: string | null
    } | null
  } | null
}

export interface WorkspaceACPHistoryModalProps {
  open: boolean
  workspaceId?: string | null
  workspaceName?: string | null
  onCancel: () => void
  onOpenAgentTasks: () => void
  onSaveRunArtifact?: (input: WorkspaceACPRunArtifactSaveInput) => Promise<void>
}

const normalizeListPayload = <T,>(payload: unknown, key: string): T[] => {
  if (Array.isArray(payload)) return payload as T[]
  if (payload && typeof payload === "object") {
    const value = (payload as Record<string, unknown>)[key]
    if (Array.isArray(value)) return value as T[]
  }
  return []
}

const normalizeWorkspaceId = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

const getCanonicalWorkspaceId = (
  item: {
    canonical_workspace?: CanonicalWorkspaceLink | null
    metadata?: Record<string, unknown> | null
  }
): string | null =>
  normalizeWorkspaceId(item.canonical_workspace?.canonical_workspace_id) ||
  normalizeWorkspaceId(item.metadata?.canonical_workspace_id)

const getRunTimestamp = (run: RunItem): number => {
  const candidate = run.completed_at || run.started_at
  if (!candidate) return 0
  const timestamp = Date.parse(candidate)
  return Number.isNaN(timestamp) ? 0 : timestamp
}

const getTaskTimestamp = (task: TaskSummary): number => {
  const candidate = task.updated_at || task.created_at
  if (!candidate) return task.id
  const timestamp = Date.parse(candidate)
  return Number.isNaN(timestamp) ? task.id : timestamp
}

const formatTaskTimestamp = (task: TaskSummary): string => {
  const candidate = task.updated_at || task.created_at
  if (!candidate) return ""
  const timestamp = Date.parse(candidate)
  if (Number.isNaN(timestamp)) return ""
  return new Date(timestamp).toLocaleString()
}

const getDirectSessionTimestamp = (session: ACPSessionListItem): number => {
  const candidate = session.last_activity_at || session.created_at
  if (!candidate) return 0
  const timestamp = Date.parse(candidate)
  return Number.isNaN(timestamp) ? 0 : timestamp
}

const getSessionId = (run: RunItem): string | null =>
  normalizeWorkspaceId(run.session?.session_id) ||
  normalizeWorkspaceId(run.session_id)

const getDirectSessionWorkspaceId = (session: ACPSessionListItem): string | null =>
  normalizeWorkspaceId(session.workspace_context?.workspace_id) ||
  normalizeWorkspaceId(session.workspace_id)

const statusColor = (status: string): string => {
  if (status === "completed" || status === "complete") return "success"
  if (status === "failed" || status === "triage") return "error"
  if (status === "running" || status === "inprogress") return "processing"
  return "default"
}

const isCompletedRunStatus = (status: string): boolean => {
  const normalizedStatus = status.toLowerCase()
  return normalizedStatus === "completed" || normalizedStatus === "complete"
}

const formatCountLabel = (count: number, singular: string, plural: string): string =>
  `${count} ${count === 1 ? singular : plural}`

const createUnsupportedError = (): Error & { code: string } =>
  Object.assign(new Error(AGENT_ORCHESTRATION_UNSUPPORTED_CODE), {
    code: AGENT_ORCHESTRATION_UNSUPPORTED_CODE
  })

const isUnsupportedError = (
  error: unknown
): error is Error & { code: string } =>
  Boolean(
    error &&
      typeof error === "object" &&
      "code" in error &&
      (error as { code?: string }).code === AGENT_ORCHESTRATION_UNSUPPORTED_CODE
  )

type WorkspaceRunCardProps = {
  projectId: number
  projectName: string
  taskId: number
  taskTitle: string
  run: RunItem
  onOpenSessionRoute: (run: RunItem, view?: string) => void
  saveEnabled: boolean
  savingRunId: number | null
  onSaveRun?: (
    input: Omit<
      WorkspaceACPRunArtifactSaveInput,
      "workspaceId" | "scope" | "isCurrent"
    >
  ) => void
}

const WorkspaceACPRunCard: React.FC<WorkspaceRunCardProps> = ({
  projectId,
  projectName,
  taskId,
  taskTitle,
  run,
  onOpenSessionRoute,
  saveEnabled,
  savingRunId,
  onSaveRun
}) => {
  const { t } = useTranslation(["playground", "common"])
  const sessionId = getSessionId(run)
  const failureMessage = run.failure_context?.message || run.error || null
  const resultPreview = run.result_summary?.trim()
    ? run.result_summary
    : run.history?.result?.preview || null
  const artifactCount = run.history?.artifact_count ?? 0
  const diagnosticCount = run.history?.diagnostic_count ?? 0
  const auditEventCount = run.history?.audit_event_count ?? 0
  const eventCount = run.history?.event_count ?? 0
  const observableActivityLabel = t(
    "playground:workspace.acpRunObservableActivity",
    "Observable activity: {{artifactCount}} artifacts/files, {{diagnosticCount}} diagnostics/warnings, {{auditEventCount}} audit/approvals, {{eventCount}} events/tool activity",
    {
      artifactCount,
      diagnosticCount,
      auditEventCount,
      eventCount
    }
  )
  const canSaveRunArtifact = Boolean(
    saveEnabled &&
      resultPreview?.trim() &&
      isCompletedRunStatus(run.status)
  )

  return (
    <div
      key={`${taskId}-${run.id}`}
      className="rounded-lg border border-border p-4"
    >
      <div className="flex flex-wrap items-start justify-end gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Tag color={statusColor(run.status)}>{run.status}</Tag>
          {run.agent_type && <Tag>{run.agent_type}</Tag>}
        </div>
      </div>

      {sessionId && (
        <div className="mt-2 text-xs text-muted-foreground">{sessionId}</div>
      )}

      <div
        className="mt-3 grid grid-cols-4 gap-2 text-xs text-muted-foreground"
        aria-label={observableActivityLabel}
      >
        <span>
          {t(
            "playground:workspace.acpArtifactsFilesCount",
            "{{count}} artifacts/files",
            { count: artifactCount }
          )}
        </span>
        <span>
          {t(
            "playground:workspace.acpDiagnosticsWarningsCount",
            "{{count}} diagnostics/warnings",
            { count: diagnosticCount }
          )}
        </span>
        <span>
          {t(
            "playground:workspace.acpAuditApprovalsCount",
            "{{count}} audit/approvals",
            { count: auditEventCount }
          )}
        </span>
        <span>
          {t(
            "playground:workspace.acpEventsActivityCount",
            "{{count}} events/tool activity",
            { count: eventCount }
          )}
        </span>
      </div>

      {failureMessage && (
        <div className="mt-3 rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-300">
          {failureMessage}
        </div>
      )}
      {!failureMessage && resultPreview && (
        <div className="mt-3 text-sm">{resultPreview}</div>
      )}

      {(sessionId || canSaveRunArtifact) && (
        <div className="mt-3 flex flex-wrap gap-2">
          {sessionId && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => onOpenSessionRoute(run)}
            >
              {t("playground:workspace.openSession", "Open session")}
            </Button>
          )}
          {sessionId && run.session?.links?.diagnostics && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => onOpenSessionRoute(run, "diagnostics")}
            >
              {t("playground:workspace.openDiagnostics", "Open diagnostics")}
            </Button>
          )}
          {sessionId && run.session?.links?.artifacts && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => onOpenSessionRoute(run, "artifacts")}
            >
              {t("playground:workspace.openArtifacts", "Open artifacts")}
            </Button>
          )}
          {canSaveRunArtifact && (
            <Button
              size="small"
              aria-label="Save to Studio"
              loading={savingRunId === run.id}
              disabled={savingRunId !== null}
              onClick={() =>
                onSaveRun?.({
                  projectId,
                  projectName,
                  taskId,
                  taskTitle,
                  runId: run.id,
                  status: run.status,
                  resultPreview: resultPreview?.trim() || "",
                  sessionId,
                  agentType: run.agent_type,
                  startedAt: run.started_at,
                  completedAt: run.completed_at,
                  links: run.session?.links,
                  history: run.history
                })
              }
            >
              {t("playground:workspace.saveRunToStudio", "Save to Studio")}
            </Button>
          )}
          {sessionId && run.session?.links?.audit && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => onOpenSessionRoute(run, "audit")}
            >
              {t("playground:workspace.openAudit", "Open audit")}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

export const WorkspaceACPHistoryModal: React.FC<
  WorkspaceACPHistoryModalProps
> = ({
  open,
  workspaceId,
  workspaceName,
  onCancel,
  onOpenAgentTasks,
  onSaveRunArtifact
}) => {
  const { t } = useTranslation(["playground", "common"])
  const navigate = useNavigate()
  const {
    config: connectionConfig,
    loading: connectionConfigLoading
  } = useCanonicalConnectionConfig()
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<WorkspaceACPHistoryError | null>(null)
  const [taskRows, setTaskRows] = React.useState<WorkspaceTaskRow[]>([])
  const [directSessions, setDirectSessions] = React.useState<ACPSessionListItem[]>([])
  const [expandedTaskIds, setExpandedTaskIds] = React.useState<string[]>([])
  const [taskDetails, setTaskDetails] = React.useState<
    Record<number, TaskDetailState>
  >({})

  // WP-44: task detail layers are fetched lazily on row expand (first page
  // only) instead of eagerly fanning out 2+N+M requests on open.
  const detailFetchAbortRef = React.useRef<AbortController | null>(null)
  const inFlightDetailTasksRef = React.useRef<Set<number>>(new Set())
  const detailFetchRef = React.useRef<
    ((taskId: number) => Promise<TaskDetail>) | null
  >(null)

  const [historyAuthority, setHistoryAuthority] = React.useState<{
    workspaceId: string; scope: ServicePromptSnapshot; isCurrent: () => boolean
  } | null>(null)
  const [savingRun, setSavingRun] = React.useState<number | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const saving = React.useRef(false)

  React.useLayoutEffect(() => {
    if (!open) return

    let cancelled = false
    let retired = false
    const abortController = new AbortController()
    detailFetchAbortRef.current = abortController
    inFlightDetailTasksRef.current = new Set()
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; abortController.abort() }, 15000)
    const deadline = performance.now() + 15000
    let scope: ServicePromptSnapshot | null = null
    const isCurrent = () => !cancelled && !retired && !!scope &&
      !scope.scopeSignal.aborted && !scope.scopeInvalidatedSignal.aborted
    const retire = () => {
      if (cancelled) return
      setTaskRows([])
      setDirectSessions([])
      setExpandedTaskIds([])
      setTaskDetails({})
      setHistoryAuthority(null)
      setError(null)
      setLoading(false)
      setSaveError("The account or server changed. Reopen history before saving.")
    }
    const stopAccountWatch = watchChatAccountChanges((invalidated) => {
      if (invalidated) { retired = true; abortController.abort(); retire() }
    })
    const fetchJson = async <T,>(
      request: (options: WorkspaceHistoryRequestOptions) => Promise<T>,
      projectsRequest = false
    ): Promise<T> => {
      if (!isCurrent()) throw createServicePromptScopeChangedError()
      try {
        const payload = await request({
          ...requestScopeFields(scope!.requestScope),
          abortSignal: abortController.signal,
          timeoutMs: Math.max(1, Math.ceil(deadline - performance.now()))
        })
        if (!isCurrent()) throw createServicePromptScopeChangedError()
        return payload
      } catch (err) {
        if ((err as { status?: number })?.status === 404 &&
          projectsRequest &&
          ((err as { details?: { detail?: unknown } })?.details?.detail === "Not Found" ||
            (err instanceof Error && err.message === "Not Found"))) throw createUnsupportedError()
        throw err
      }
    }
    // WP-44: detail layers stay outside the open-flow deadline so a lazy
    // expand well after the list loaded still gets a full request budget.
    // The fetch keeps the verified-principal scope headers (dev) so lazy
    // reads run under the same authority as the list reads.
    detailFetchRef.current = async (taskId: number): Promise<TaskDetail> => {
      if (!isCurrent()) throw createServicePromptScopeChangedError()
      const detail = await getWorkspaceTaskDetail<TaskDetail>(taskId, {
        ...requestScopeFields(scope!.requestScope),
        abortSignal: abortController.signal
      })
      if (!isCurrent()) throw createServicePromptScopeChangedError()
      return detail
    }

    const loadHistory = async () => {
      const canonicalWorkspaceId = workspaceId?.trim()
      setLoading(true)
      setError(null)
      setTaskRows([])
      setDirectSessions([])
      setExpandedTaskIds([])
      setTaskDetails({})
      setHistoryAuthority(null)
      setSaveError(null)

      try {
        if (!canonicalWorkspaceId) {
          throw new Error(
            "Select or save a workspace before loading ACP run history."
          )
        }

        if (!connectionConfig) throw new Error(connectionConfigLoading
          ? "Backend connection is still loading." : "Backend connection is not configured.")
        scope = await loadServicePromptSnapshot([], { signal: abortController.signal })
        if (!isCurrent()) throw createServicePromptScopeChangedError()
        scope.scopeInvalidatedSignal.addEventListener("abort", retire, { once: true })

        const agentTasksPromise = (async (): Promise<WorkspaceTaskRow[]> => {
          const projectsPayload = await fetchJson<unknown>(
            (options) => listCanonicalWorkspaceProjects(canonicalWorkspaceId, options), true
          )
          const projects = normalizeListPayload<ProjectSummary>(
            projectsPayload,
            "projects"
          )
          const matchingProjects = projects.filter(
            (project) => getCanonicalWorkspaceId(project) === canonicalWorkspaceId
          )

          return (
            await Promise.all(
              matchingProjects.map(async (project) => {
                const tasksPayload = await fetchJson<unknown>(
                  (options) => listWorkspaceProjectTasks(project.id, options)
                )
                const tasks = normalizeListPayload<TaskSummary>(
                  tasksPayload,
                  "tasks"
                )
                return tasks.map((task) => ({ project, task }))
              })
            )
          )
            .flat()
            .sort(
              (left, right) =>
                getTaskTimestamp(right.task) - getTaskTimestamp(left.task)
            )
            .slice(0, MAX_TASK_ROWS)
        })()

        const directSessionsPromise = (async (): Promise<ACPSessionListItem[]> => {
          const sessionsPayload = await fetchJson<unknown>(
            (options) => listCanonicalWorkspaceSessions(canonicalWorkspaceId, MAX_RECENT_RUNS, options)
          )
          return normalizeListPayload<ACPSessionListItem>(
            sessionsPayload,
            "sessions"
          )
            .filter(
              (session) =>
                getDirectSessionWorkspaceId(session) === canonicalWorkspaceId
            )
            .sort(
              (left, right) =>
                getDirectSessionTimestamp(right) - getDirectSessionTimestamp(left)
            )
            .slice(0, MAX_RECENT_RUNS)
        })()

        const [agentTasksResult, directSessionsResult] =
          await Promise.allSettled([agentTasksPromise, directSessionsPromise])
        if (timedOut) throw new Error("Loading history exceeded 15 seconds. Reopen history to retry.")
        if (!isCurrent()) throw createServicePromptScopeChangedError()

        if (
          agentTasksResult.status === "rejected" &&
          directSessionsResult.status === "rejected"
        ) {
          const errors = [
            agentTasksResult.reason,
            directSessionsResult.reason
          ]
          throw errors.find(isUnsupportedError) || errors[0]
        }
        if (
          agentTasksResult.status === "rejected" &&
          !isUnsupportedError(agentTasksResult.reason) &&
          directSessionsResult.status === "fulfilled" &&
          directSessionsResult.value.length === 0
        ) {
          throw agentTasksResult.reason
        }
        if (
          directSessionsResult.status === "rejected" &&
          !isUnsupportedError(directSessionsResult.reason) &&
          agentTasksResult.status === "fulfilled" &&
          agentTasksResult.value.length === 0
        ) {
          throw directSessionsResult.reason
        }

        if (isCurrent()) {
          setHistoryAuthority({ workspaceId: canonicalWorkspaceId, scope: scope!, isCurrent })
          setTaskRows(
            agentTasksResult.status === "fulfilled"
              ? agentTasksResult.value
              : []
          )
          setDirectSessions(
            directSessionsResult.status === "fulfilled"
              ? directSessionsResult.value
              : []
          )
        }
      } catch (err) {
        if (cancelled || retired || scope?.scopeInvalidatedSignal.aborted) return
        if (!cancelled) {
          const detail = (err as { details?: { detail?: unknown } })?.details?.detail
          const value = detail && typeof detail === "object"
            ? (detail as Record<string, unknown>).message ?? (detail as Record<string, unknown>).code
            : detail
          setError(
            isUnsupportedError(err)
              ? { kind: "unsupported" }
              : {
                  kind: "message",
                  message:
                    timedOut ? "Loading history exceeded 15 seconds. Reopen history to retry." :
                      formatErrorMessage(value, err instanceof Error ? err.message : "Failed to load ACP run history")
                }
          )
        }
      } finally {
        clearTimeout(timer)
        if (!cancelled && !retired && !scope?.scopeInvalidatedSignal.aborted) {
          setLoading(false)
        }
      }
    }

    void loadHistory()

    return () => {
      cancelled = true
      clearTimeout(timer)
      stopAccountWatch()
      abortController.abort()
      scope?.scopeInvalidatedSignal.removeEventListener("abort", retire)
      scope?.release()
      detailFetchRef.current = null
      detailFetchAbortRef.current = null
      inFlightDetailTasksRef.current = new Set()
    }
  }, [connectionConfig, connectionConfigLoading, open, workspaceId])

  const saveRun = async (input: Omit<WorkspaceACPRunArtifactSaveInput, "workspaceId" | "scope" | "isCurrent">) => {
    if (saving.current || !historyAuthority?.isCurrent() || !onSaveRunArtifact) return
    saving.current = true
    setSavingRun(input.runId)
    setSaveError(null)
    try {
      await onSaveRunArtifact({ ...input, ...historyAuthority })
    } catch (err) {
      if (historyAuthority.isCurrent())
        setSaveError(err instanceof Error ? err.message : "Could not save. Retry Save to Studio.")
    } finally {
      saving.current = false
      setSavingRun(null)
    }
  }

  const loadTaskDetail = React.useCallback(
    (taskId: number) => {
      if (inFlightDetailTasksRef.current.has(taskId)) return
      inFlightDetailTasksRef.current.add(taskId)

      setTaskDetails((current) => ({
        ...current,
        [taskId]: { ...current[taskId], loading: true, error: null }
      }))

      void (async () => {
        try {
          const fetchTaskDetail = detailFetchRef.current
          if (!fetchTaskDetail) throw new Error("ACP run history is still loading.")
          const detail = await fetchTaskDetail(taskId)
          setTaskDetails((current) => ({
            ...current,
            [taskId]: {
              ...current[taskId],
              // Merge the detail layer over the list layer where possible.
              task: detail,
              runs: [...(detail.runs || [])].sort(
                (left, right) => getRunTimestamp(right) - getRunTimestamp(left)
              ),
              loading: false,
              error: null
            }
          }))
        } catch (err) {
          // Allow a later expand to retry after a failed detail fetch.
          inFlightDetailTasksRef.current.delete(taskId)
          const detail = (err as { details?: { detail?: unknown } })?.details?.detail
          const value = detail && typeof detail === "object"
            ? (detail as Record<string, unknown>).message ?? (detail as Record<string, unknown>).code
            : detail
          setTaskDetails((current) => ({
            ...current,
            [taskId]: {
              ...current[taskId],
              loading: false,
              error: formatErrorMessage(
                value,
                err instanceof Error ? err.message : "Failed to load task runs"
              )
            }
          }))
        }
      })()
    },
    []
  )

  const handleExpandChange = React.useCallback(
    (keys: string | string[]) => {
      const nextKeys = Array.isArray(keys) ? keys : [keys]
      setExpandedTaskIds(nextKeys)
      for (const key of nextKeys) {
        const taskId = Number(key)
        if (!Number.isFinite(taskId)) continue
        const existing = taskDetails[taskId]
        if (existing?.runs || existing?.loading) continue
        loadTaskDetail(taskId)
      }
    },
    [loadTaskDetail, taskDetails]
  )

  const openSessionRoute = React.useCallback(
    (run: RunItem, view?: string) => {
      const sessionId = getSessionId(run)
      if (!sessionId) return
      const params = new URLSearchParams({ session: sessionId })
      if (view) params.set("view", view)
      navigate(`/acp-playground?${params.toString()}`)
    },
    [navigate]
  )

  const openDirectSessionRoute = React.useCallback(
    (sessionId: string, view?: string) => {
      const params = new URLSearchParams({ session: sessionId })
      if (view) params.set("view", view)
      navigate(`/acp-playground?${params.toString()}`)
    },
    [navigate]
  )

  const errorDescription =
    error?.kind === "unsupported"
      ? t(
          "playground:workspace.orchestrationUnsupported",
          AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE
        )
      : error?.message
  const hasHistory = taskRows.length > 0 || directSessions.length > 0

  return (
    <Modal
      title={t("playground:workspace.acpRunHistory", "ACP run history")}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="agent-tasks" onClick={onOpenAgentTasks}>
          {t("playground:workspace.openAgentTasks", "Open Agent Tasks")}
        </Button>,
        <Button key="close" type="primary" onClick={onCancel}>
          {t("common:close", "Close")}
        </Button>
      ]}
      width={760}
    >
      <div className="space-y-4">
        <div className="text-sm text-muted-foreground">
          {t("playground:workspace.acpRunHistoryWorkspace", {
            defaultValue: "Recent agent runs linked to {{workspace}}.",
            workspace: workspaceName?.trim() || workspaceId || "this workspace"
          })}
        </div>

        {loading && (
          <div className="flex justify-center py-8">
            <Spin />
          </div>
        )}

        {!loading && error && (
          <DSAlert
            variant="error"
            title={t(
              "playground:workspace.acpRunHistoryLoadFailed",
              "Could not load ACP run history"
            )}
          >
            {errorDescription}
          </DSAlert>
        )}

        {saveError && <DSAlert variant="error">{saveError}</DSAlert>}
        {!loading && !error && !hasHistory && (
          <Empty
            description={t(
              "playground:workspace.acpRunHistoryEmpty",
              "No ACP runs linked to this workspace yet"
            )}
          />
        )}

        {!loading && !error && hasHistory && (
          <div className="space-y-3">
            {taskRows.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium">
                  {t("playground:workspace.acpTaskHistory", "Agent tasks")}
                </div>
                <Collapse
                  activeKey={expandedTaskIds}
                  onChange={handleExpandChange}
                  items={taskRows.map(({ project, task }) => {
                    const detailState = taskDetails[task.id]
                    const displayTask = detailState?.task || task
                    return {
                      key: String(task.id),
                      label: (
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <div className="text-sm font-medium">
                              {displayTask.title}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {project.name}
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            <Tag color={statusColor(displayTask.status)}>
                              {displayTask.status}
                            </Tag>
                            {formatTaskTimestamp(displayTask) && (
                              <span className="text-xs text-muted-foreground">
                                {formatTaskTimestamp(displayTask)}
                              </span>
                            )}
                          </div>
                        </div>
                      ),
                      children: (
                        <div className="space-y-3">
                          {detailState?.loading && (
                            <div className="flex justify-center py-4">
                              <Spin />
                            </div>
                          )}
                          {!detailState?.loading && detailState?.error && (
                            <DSAlert
                              variant="error"
                              title={t(
                                "playground:workspace.acpTaskRunsLoadFailed",
                                "Could not load runs for this task"
                              )}
                            >
                              {detailState.error}
                            </DSAlert>
                          )}
                          {!detailState?.loading &&
                            !detailState?.error &&
                            (detailState?.runs || []).length === 0 && (
                              <Empty
                                description={t(
                                  "playground:workspace.acpTaskNoRuns",
                                  "No runs recorded for this task"
                                )}
                              />
                            )}
                          {!detailState?.loading &&
                            !detailState?.error &&
                            (detailState?.runs || []).map((run) => (
                              <WorkspaceACPRunCard
                                key={run.id}
                                projectId={project.id}
                                projectName={project.name}
                                taskId={task.id}
                                taskTitle={displayTask.title}
                                run={run}
                                onOpenSessionRoute={openSessionRoute}
                                saveEnabled={Boolean(
                                  onSaveRunArtifact && historyAuthority?.isCurrent()
                                )}
                                savingRunId={savingRun}
                                onSaveRun={(input) => void saveRun(input)}
                              />
                            ))}
                        </div>
                      )
                    }
                  })}
                />
              </div>
            )}
            {directSessions.length > 0 && (
              <div className="space-y-3">
                <div className="text-sm font-medium">
                  {t(
                    "playground:workspace.directAcpSessions",
                    "Direct ACP sessions"
                  )}
                </div>
                {directSessions.map((session) => {
                  const mcpCount =
                    session.workspace_context?.mcp_server_count ?? 0
                  return (
                    <div
                      key={session.session_id}
                      className="rounded-lg border border-border p-4"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium">
                            {session.name || session.session_id}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {session.session_id}
                          </div>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <Tag color={statusColor(session.status)}>
                            {session.status}
                          </Tag>
                          {session.agent_type && <Tag>{session.agent_type}</Tag>}
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                        <span>
                          {formatCountLabel(
                            session.message_count ?? 0,
                            "message",
                            "messages"
                          )}
                        </span>
                        <span>
                          {formatCountLabel(
                            mcpCount,
                            "MCP server",
                            "MCP servers"
                          )}
                        </span>
                        <span>
                          {session.workspace_context?.verification_level ||
                            session.workspace_context?.support_state ||
                            "unverified"}
                        </span>
                      </div>

                      <div className="mt-3 flex flex-wrap gap-2">
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() =>
                            openDirectSessionRoute(session.session_id)
                          }
                        >
                          {t("playground:workspace.openSession", "Open session")}
                        </Button>
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() =>
                            openDirectSessionRoute(
                              session.session_id,
                              "diagnostics"
                            )
                          }
                        >
                          {t(
                            "playground:workspace.openDiagnostics",
                            "Open diagnostics"
                          )}
                        </Button>
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() => openDirectSessionRoute(session.session_id, "artifacts")}
                        >
                          {t("playground:workspace.openArtifacts", "Open artifacts")}
                        </Button>
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() => openDirectSessionRoute(session.session_id, "audit")}
                        >
                          {t("playground:workspace.openAudit", "Open audit")}
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}
