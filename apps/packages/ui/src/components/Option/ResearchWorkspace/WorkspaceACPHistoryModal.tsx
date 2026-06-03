import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { Button, Empty, Modal, Spin, Tag } from "antd"
import { ExternalLink } from "lucide-react"

import { Alert as DSAlert } from "@/components/ui/primitives"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { buildACPAuthHeaders } from "@/services/acp/connection"
import type {
  ACPSessionListItem,
  ACPSessionListResponse
} from "@/services/acp/types"
import {
  resolveBrowserRequestTransport,
  type BrowserRequestTransport
} from "@/services/tldw/request-core"

const AGENT_ORCHESTRATION_BASE_PATH = "/api/v1/agent-orchestration"
const PROJECTS_PATH = `${AGENT_ORCHESTRATION_BASE_PATH}/projects`
const ACP_SESSIONS_PATH = "/api/v1/acp/sessions"
const CANONICAL_WORKSPACE_SOURCE = "research_workspace"
const AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE =
  "Agent orchestration is not available on this server."
const AGENT_ORCHESTRATION_UNSUPPORTED_CODE =
  "AGENT_ORCHESTRATION_UNSUPPORTED"
const MAX_TASK_DETAILS = 12
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

type WorkspaceRunHistoryEntry = {
  projectId: number
  projectName: string
  taskId: number
  taskTitle: string
  run: RunItem
}

type WorkspaceACPHistoryError =
  | { kind: "message"; message: string }
  | { kind: "unsupported" }

export interface WorkspaceACPHistoryModalProps {
  open: boolean
  workspaceId?: string | null
  workspaceName?: string | null
  onCancel: () => void
  onOpenAgentTasks: () => void
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

const buildCanonicalProjectsPath = (canonicalWorkspaceId: string): string => {
  const params = new URLSearchParams({
    canonical_workspace_id: canonicalWorkspaceId,
    canonical_workspace_source: CANONICAL_WORKSPACE_SOURCE
  })
  return `${PROJECTS_PATH}?${params.toString()}`
}

const buildWorkspaceSessionsPath = (canonicalWorkspaceId: string): string => {
  const params = new URLSearchParams({
    workspace_id: canonicalWorkspaceId,
    limit: String(MAX_RECENT_RUNS)
  })
  return `${ACP_SESSIONS_PATH}?${params.toString()}`
}

const readApiErrorMessage = async (response: Response): Promise<string> => {
  const payload = await response.json().catch(() => null)
  const detail =
    payload && typeof payload === "object"
      ? (payload as { detail?: unknown }).detail
      : null

  if (typeof detail === "string" && detail.trim()) {
    return detail
  }
  if (detail && typeof detail === "object") {
    const detailRecord = detail as Record<string, unknown>
    if (typeof detailRecord.message === "string" && detailRecord.message.trim()) {
      return detailRecord.message
    }
    if (typeof detailRecord.code === "string" && detailRecord.code.trim()) {
      return detailRecord.code
    }
  }
  return `HTTP ${response.status}`
}

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

export const WorkspaceACPHistoryModal: React.FC<
  WorkspaceACPHistoryModalProps
> = ({ open, workspaceId, workspaceName, onCancel, onOpenAgentTasks }) => {
  const { t } = useTranslation(["playground", "common"])
  const navigate = useNavigate()
  const {
    config: connectionConfig,
    loading: connectionConfigLoading
  } = useCanonicalConnectionConfig()
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<WorkspaceACPHistoryError | null>(null)
  const [entries, setEntries] = React.useState<WorkspaceRunHistoryEntry[]>([])
  const [directSessions, setDirectSessions] = React.useState<ACPSessionListItem[]>([])

  const buildRequestTransport = React.useCallback(
    (path: string): BrowserRequestTransport | null => {
      if (!connectionConfig) return null
      return resolveBrowserRequestTransport({
        config: connectionConfig,
        path
      })
    },
    [connectionConfig]
  )

  const getHeaders = React.useCallback(
    (transport: BrowserRequestTransport | null) => {
      if (transport?.mode === "hosted") {
        return { "Content-Type": "application/json" }
      }
      return buildACPAuthHeaders(connectionConfig)
    },
    [connectionConfig]
  )

  const fetchJson = React.useCallback(
    async <T,>(path: string, signal?: AbortSignal): Promise<T> => {
      const transport = buildRequestTransport(path)
      if (!transport) {
        throw new Error(
          connectionConfigLoading
            ? "Backend connection is still loading."
            : "Backend connection is not configured."
        )
      }

      const response = await fetch(transport.url, {
        headers: getHeaders(transport),
        signal
      })
      if (!response.ok) {
        const message = await readApiErrorMessage(response)
        if (
          response.status === 404 &&
          (path === PROJECTS_PATH || path.startsWith(`${PROJECTS_PATH}?`)) &&
          message === "Not Found"
        ) {
          throw createUnsupportedError()
        }
        throw new Error(message)
      }
      return (await response.json()) as T
    },
    [buildRequestTransport, connectionConfigLoading, getHeaders]
  )

  React.useEffect(() => {
    if (!open) return

    let cancelled = false
    const abortController = new AbortController()

    const loadHistory = async () => {
      const canonicalWorkspaceId = workspaceId?.trim()
      setLoading(true)
      setError(null)
      setEntries([])
      setDirectSessions([])

      try {
        if (!canonicalWorkspaceId) {
          throw new Error(
            "Select or save a workspace before loading ACP run history."
          )
        }

        const agentTaskHistoryPromise = (async (): Promise<
          WorkspaceRunHistoryEntry[]
        > => {
          const projectsPayload = await fetchJson<unknown>(
            buildCanonicalProjectsPath(canonicalWorkspaceId),
            abortController.signal
          )
          const projects = normalizeListPayload<ProjectSummary>(
            projectsPayload,
            "projects"
          )
          const matchingProjects = projects.filter(
            (project) => getCanonicalWorkspaceId(project) === canonicalWorkspaceId
          )

          const taskRows = (
            await Promise.all(
              matchingProjects.map(async (project) => {
                const tasksPayload = await fetchJson<unknown>(
                  `${PROJECTS_PATH}/${project.id}/tasks`,
                  abortController.signal
                )
                const tasks = normalizeListPayload<TaskSummary>(
                  tasksPayload,
                  "tasks"
                )
                return tasks.map((task) => ({ project, task }))
              })
            )
          ).flat()

          const taskDetails = await Promise.all(
            [...taskRows]
              .sort(
                (left, right) =>
                  getTaskTimestamp(right.task) - getTaskTimestamp(left.task)
              )
              .slice(0, MAX_TASK_DETAILS)
              .map(async ({ project, task }) => {
                const detail = await fetchJson<TaskDetail>(
                  `${AGENT_ORCHESTRATION_BASE_PATH}/tasks/${task.id}`,
                  abortController.signal
                )
                return { project, task: detail }
              })
          )

          return taskDetails
            .flatMap(({ project, task }) =>
              (task.runs || []).map((run) => ({
                projectId: project.id,
                projectName: project.name,
                taskId: task.id,
                taskTitle: task.title,
                run
              }))
            )
            .sort((left, right) => getRunTimestamp(right.run) - getRunTimestamp(left.run))
            .slice(0, MAX_RECENT_RUNS)
        })()

        const directSessionsPromise = (async (): Promise<ACPSessionListItem[]> => {
          const sessionsPayload = await fetchJson<ACPSessionListResponse | unknown>(
            buildWorkspaceSessionsPath(canonicalWorkspaceId),
            abortController.signal
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

        const [agentTaskHistoryResult, directSessionsResult] =
          await Promise.allSettled([agentTaskHistoryPromise, directSessionsPromise])

        if (
          agentTaskHistoryResult.status === "rejected" &&
          directSessionsResult.status === "rejected"
        ) {
          const errors = [
            agentTaskHistoryResult.reason,
            directSessionsResult.reason
          ]
          throw errors.find(isUnsupportedError) || errors[0]
        }
        if (
          agentTaskHistoryResult.status === "rejected" &&
          !isUnsupportedError(agentTaskHistoryResult.reason) &&
          directSessionsResult.status === "fulfilled" &&
          directSessionsResult.value.length === 0
        ) {
          throw agentTaskHistoryResult.reason
        }

        if (!cancelled) {
          setEntries(
            agentTaskHistoryResult.status === "fulfilled"
              ? agentTaskHistoryResult.value
              : []
          )
          setDirectSessions(
            directSessionsResult.status === "fulfilled"
              ? directSessionsResult.value
              : []
          )
        }
      } catch (err) {
        if (cancelled || abortController.signal.aborted) {
          return
        }
        if (!cancelled) {
          setError(
            isUnsupportedError(err)
              ? { kind: "unsupported" }
              : {
                  kind: "message",
                  message:
                    err instanceof Error
                      ? err.message
                      : "Failed to load ACP run history"
                }
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadHistory()

    return () => {
      cancelled = true
      abortController.abort()
    }
  }, [fetchJson, open, workspaceId])

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
  const hasHistory = entries.length > 0 || directSessions.length > 0

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
            {entries.map((entry) => {
              const { run } = entry
              const sessionId = getSessionId(run)
              const failureMessage =
                run.failure_context?.message || run.error || null
              const resultPreview =
                run.result_summary || run.history?.result?.preview || null

              return (
                <div
                  key={`${entry.taskId}-${run.id}`}
                  className="rounded-lg border border-border p-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">{entry.taskTitle}</div>
                      <div className="text-xs text-muted-foreground">
                        {entry.projectName}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag color={statusColor(run.status)}>{run.status}</Tag>
                      {run.agent_type && <Tag>{run.agent_type}</Tag>}
                    </div>
                  </div>

                  {sessionId && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      {sessionId}
                    </div>
                  )}

                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                    <span>{run.history?.artifact_count ?? 0} artifacts</span>
                    <span>{run.history?.diagnostic_count ?? 0} diagnostics</span>
                    <span>{run.history?.audit_event_count ?? 0} audit</span>
                  </div>

                  {failureMessage && (
                    <div className="mt-3 rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-300">
                      {failureMessage}
                    </div>
                  )}
                  {!failureMessage && resultPreview && (
                    <div className="mt-3 text-sm">{resultPreview}</div>
                  )}

                  {sessionId && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button
                        size="small"
                        icon={<ExternalLink className="h-3 w-3" />}
                        onClick={() => openSessionRoute(run)}
                      >
                        {t("playground:workspace.openSession", "Open session")}
                      </Button>
                      {run.session?.links?.diagnostics && (
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() => openSessionRoute(run, "diagnostics")}
                        >
                          {t(
                            "playground:workspace.openDiagnostics",
                            "Open diagnostics"
                          )}
                        </Button>
                      )}
                      {run.session?.links?.artifacts && (
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() => openSessionRoute(run, "artifacts")}
                        >
                          {t("playground:workspace.openArtifacts", "Open artifacts")}
                        </Button>
                      )}
                      {run.session?.links?.audit && (
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() => openSessionRoute(run, "audit")}
                        >
                          {t("playground:workspace.openAudit", "Open audit")}
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
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
                          onClick={() =>
                            openDirectSessionRoute(session.session_id, "artifacts")
                          }
                        >
                          {t("playground:workspace.openArtifacts", "Open artifacts")}
                        </Button>
                        <Button
                          size="small"
                          icon={<ExternalLink className="h-3 w-3" />}
                          onClick={() =>
                            openDirectSessionRoute(session.session_id, "audit")
                          }
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
