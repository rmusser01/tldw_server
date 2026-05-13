import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { Alert, Button, Empty, Modal, Spin, Tag } from "antd"
import { ExternalLink } from "lucide-react"

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { buildACPAuthHeaders } from "@/services/acp/connection"
import {
  resolveBrowserRequestTransport,
  type BrowserRequestTransport
} from "@/services/tldw/request-core"

const AGENT_ORCHESTRATION_BASE_PATH = "/api/v1/agent-orchestration"
const PROJECTS_PATH = `${AGENT_ORCHESTRATION_BASE_PATH}/projects`
const AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE =
  "Agent orchestration is not available on this server."
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

const getSessionId = (run: RunItem): string | null =>
  normalizeWorkspaceId(run.session?.session_id) ||
  normalizeWorkspaceId(run.session_id)

const statusColor = (status: string): string => {
  if (status === "completed" || status === "complete") return "success"
  if (status === "failed" || status === "triage") return "error"
  if (status === "running" || status === "inprogress") return "processing"
  return "default"
}

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
  const [error, setError] = React.useState<string | null>(null)
  const [entries, setEntries] = React.useState<WorkspaceRunHistoryEntry[]>([])

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
    async <T,>(path: string): Promise<T> => {
      const transport = buildRequestTransport(path)
      if (!transport) {
        throw new Error(
          connectionConfigLoading
            ? "Backend connection is still loading."
            : "Backend connection is not configured."
        )
      }

      const response = await fetch(transport.url, {
        headers: getHeaders(transport)
      })
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE)
        }
        throw new Error(await readApiErrorMessage(response))
      }
      return (await response.json()) as T
    },
    [buildRequestTransport, connectionConfigLoading, getHeaders]
  )

  React.useEffect(() => {
    if (!open) return

    let cancelled = false

    const loadHistory = async () => {
      const canonicalWorkspaceId = workspaceId?.trim()
      setLoading(true)
      setError(null)
      setEntries([])

      try {
        if (!canonicalWorkspaceId) {
          throw new Error(
            "Select or save a workspace before loading ACP run history."
          )
        }

        const projectsPayload = await fetchJson<unknown>(PROJECTS_PATH)
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
                `${PROJECTS_PATH}/${project.id}/tasks`
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
          taskRows.slice(0, MAX_TASK_DETAILS).map(async ({ project, task }) => {
            const detail = await fetchJson<TaskDetail>(
              `${AGENT_ORCHESTRATION_BASE_PATH}/tasks/${task.id}`
            )
            return { project, task: detail }
          })
        )

        const recentEntries = taskDetails
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

        if (!cancelled) {
          setEntries(recentEntries)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load ACP run history")
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
          <Alert
            type="error"
            showIcon
            title={t(
              "playground:workspace.acpRunHistoryLoadFailed",
              "Could not load ACP run history"
            )}
            description={error}
          />
        )}

        {!loading && !error && entries.length === 0 && (
          <Empty
            description={t(
              "playground:workspace.acpRunHistoryEmpty",
              "No ACP runs linked to this workspace yet"
            )}
          />
        )}

        {!loading && !error && entries.length > 0 && (
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
          </div>
        )}
      </div>
    </Modal>
  )
}
