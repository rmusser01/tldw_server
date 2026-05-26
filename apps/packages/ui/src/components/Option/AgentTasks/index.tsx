import React, { useEffect, useState, useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import {
  Button,
  Card,
  Empty,
  Input,
  Modal,
  Select,
  Spin,
  Tag,
  Tooltip,
  Form,
  Collapse,
} from "antd"
import {
  FolderPlus,
  ListTodo,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  RefreshCw,
  Plus,
  Trash2,
  ChevronRight,
  Search,
  ExternalLink,
} from "lucide-react"
import { useLocation, useNavigate } from "react-router-dom"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { RESEARCH_WORKSPACE_PATH } from "@/routes/route-paths"
import { buildACPAuthHeaders } from "@/services/acp/connection"
import { buildACPSetupIssues, normalizeACPHealthStatus, type ACPSetupIssue } from "@/services/acp/readiness"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { Alert } from "@/components/ui/primitives/Alert"
import { Badge as DesignSystemBadge } from "@/components/ui/primitives/Badge"

// Types matching the backend orchestration API
type CanonicalWorkspaceLink = {
  acp_workspace_id?: number | null
  canonical_workspace_id?: string | null
  canonical_workspace_source?: string | null
  link_status?: string | null
}

type ProjectSummary = {
  id: number
  name: string
  description?: string
  workspace_id?: number | null
  user_id: number
  created_at: string
  metadata?: Record<string, unknown>
  canonical_workspace?: CanonicalWorkspaceLink | null
  task_summary?: {
    total_tasks: number
    status_counts: Record<string, number>
  }
}

type TaskItem = {
  id: number
  project_id: number
  title: string
  description?: string
  status: string
  agent_type?: string
  dependency_id?: number | null
  review_count: number
  max_review_attempts: number
  created_at: string
  updated_at: string
  metadata?: Record<string, unknown>
  canonical_workspace?: CanonicalWorkspaceLink | null
  runs?: RunItem[]
}

type RunItem = {
  id: number
  task_id: number
  session_id?: string
  agent_type?: string
  status: string
  result_summary?: string
  error?: string
  started_at: string
  completed_at?: string
  session?: {
    session_id: string
    available?: boolean
    links?: Record<string, string>
  } | null
  history?: {
    event_count?: number
    audit_event_count?: number
    artifact_count?: number
    diagnostic_count?: number
    tool_call_count?: number
    stop_reason?: string | null
    result?: {
      preview?: string
    } | null
  }
  failure_context?: {
    reason_code?: string | null
    message?: string | null
    source?: string | null
    diagnostic_uri?: string | null
  } | null
  review_decision?: {
    available?: boolean
    approved?: boolean
    reviewer?: string | null
    feedback_preview?: string | null
  } | null
}

type ReviewItem = {
  reviewer?: string | null
  approved?: boolean
  feedback?: string | null
  created_at?: string | null
}

type TaskDetailItem = TaskItem & {
  reviews?: ReviewItem[]
}

const AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE = "Agent orchestration unavailable"
const AGENT_ORCHESTRATION_UNSUPPORTED_DESCRIPTION =
  "This server does not expose agent orchestration endpoints."
const AGENT_ORCHESTRATION_UNSUPPORTED_CODE = "AGENT_ORCHESTRATION_UNSUPPORTED"
const AGENT_ORCHESTRATION_PROJECTS_PATH = "/api/v1/agent-orchestration/projects"
const AGENT_ORCHESTRATION_BASE_PATH = "/api/v1/agent-orchestration"
const ALL_WORKSPACES_FILTER_VALUE = "__all_workspaces__"
const LINKED_CANONICAL_WORKSPACE_STATUS = "linked"

const STATUS_COLORS: Record<string, string> = {
  todo: "default",
  inprogress: "processing",
  review: "warning",
  complete: "success",
  triage: "error",
}

const STATUS_ICONS: Record<string, React.ReactNode> = {
  todo: <Clock className="h-3.5 w-3.5" />,
  inprogress: <Play className="h-3.5 w-3.5" />,
  review: <AlertTriangle className="h-3.5 w-3.5" />,
  complete: <CheckCircle className="h-3.5 w-3.5" />,
  triage: <XCircle className="h-3.5 w-3.5" />,
}

const navigateOptionRoute = (path: string) => {
  if (typeof window === "undefined") {
    return
  }
  window.location.hash = path
}

const normalizeWorkspaceFilterId = (value: unknown): string | null => {
  if (typeof value !== "string") {
    return null
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const readWorkspaceFilterFromSearch = (search: string): string | null => {
  const params = new URLSearchParams(search)
  return (
    normalizeWorkspaceFilterId(params.get("workspace")) ||
    normalizeWorkspaceFilterId(params.get("workspace_id")) ||
    normalizeWorkspaceFilterId(params.get("canonical_workspace_id"))
  )
}

const getCanonicalWorkspaceId = (
  projectOrTask: ProjectSummary | TaskItem
): string | null => {
  return (
    normalizeWorkspaceFilterId(projectOrTask.canonical_workspace?.canonical_workspace_id) ||
    normalizeWorkspaceFilterId(projectOrTask.metadata?.canonical_workspace_id)
  )
}

const getCanonicalWorkspaceLinkStatus = (
  project: ProjectSummary
): string | null => {
  return normalizeWorkspaceFilterId(project.canonical_workspace?.link_status)?.toLowerCase() ?? null
}

const normalizeListPayload = <T,>(payload: unknown, key: string): T[] => {
  if (Array.isArray(payload)) {
    return payload as T[]
  }
  if (payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>)[key])) {
    return (payload as Record<string, T[]>)[key]
  }
  return []
}

const createUnsupportedError = (): Error & { code: string } =>
  Object.assign(new Error(AGENT_ORCHESTRATION_UNSUPPORTED_CODE), {
    code: AGENT_ORCHESTRATION_UNSUPPORTED_CODE,
  })

const isUnsupportedError = (error: unknown): boolean =>
  Boolean(
    error &&
      typeof error === "object" &&
      "code" in error &&
      (error as { code?: string }).code === AGENT_ORCHESTRATION_UNSUPPORTED_CODE
  )

const readApiErrorMessage = async (response: Response): Promise<string> => {
  const payload = await response.json().catch(() => null)
  if (payload && typeof payload === "object" && typeof (payload as { detail?: unknown }).detail === "string") {
    return (payload as { detail: string }).detail
  }
  return `HTTP ${response.status}`
}

const ensureOrchestrationResponse = async (response: Response): Promise<void> => {
  if (response.ok) {
    return
  }
  if (response.status === 404) {
    throw createUnsupportedError()
  }
  throw new Error(await readApiErrorMessage(response))
}

export const AgentTasksPage: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const { config: connectionConfig } = useCanonicalConnectionConfig()
  const location = useLocation()
  const navigate = useNavigate()

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null)
  const [tasks, setTasks] = useState<TaskItem[]>([])
  const [loading, setLoading] = useState(true)
  const [tasksLoading, setTasksLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isUnsupported, setIsUnsupported] = useState(false)
  const [setupIssues, setSetupIssues] = useState<ACPSetupIssue[]>([])
  const [setupLoading, setSetupLoading] = useState(false)
  const [taskDetail, setTaskDetail] = useState<TaskDetailItem | null>(null)
  const [taskDetailLoading, setTaskDetailLoading] = useState(false)
  const [projectsLoadedSuccessfully, setProjectsLoadedSuccessfully] = useState(false)
  const workspaceFilterId = useMemo(
    () => readWorkspaceFilterFromSearch(location.search),
    [location.search]
  )

  // Modal states
  const [showProjectModal, setShowProjectModal] = useState(false)
  const [showTaskModal, setShowTaskModal] = useState(false)
  const [projectForm] = Form.useForm()
  const [taskForm] = Form.useForm()
  const orchestrationSupportRef = React.useRef<boolean | null>(null)
  const taskDetailRequestRef = React.useRef<AbortController | null>(null)

  const buildRequestTransport = useCallback(
    (path: string) => {
      if (!connectionConfig) return null
      return resolveBrowserRequestTransport({
        config: connectionConfig,
        path
      })
    },
    [connectionConfig]
  )

  const getHeaders = useCallback((transport?: { mode?: string } | null) => {
    if (transport?.mode === "hosted") {
      return { "Content-Type": "application/json" }
    }
    return buildACPAuthHeaders(connectionConfig, { includeContentType: true })
  }, [connectionConfig])

  const buildRequestUrl = useCallback(
    (path: string) => {
      return buildRequestTransport(path)?.url ?? null
    },
    [buildRequestTransport]
  )

  const apiTransport = useMemo(
    () => buildRequestTransport(AGENT_ORCHESTRATION_BASE_PATH),
    [buildRequestTransport]
  )

  const apiBase = useMemo(
    () => apiTransport?.url ?? null,
    [apiTransport]
  )

  React.useEffect(() => {
    orchestrationSupportRef.current = null
  }, [connectionConfig])

  const markUnsupported = useCallback(() => {
    setIsUnsupported(true)
    setError(null)
    setProjects([])
    setProjectsLoadedSuccessfully(false)
    setTasks([])
    setSelectedProjectId(null)
  }, [])

  const handleCloseTaskDetail = useCallback(() => {
    taskDetailRequestRef.current?.abort()
    taskDetailRequestRef.current = null
    setTaskDetailLoading(false)
    setTaskDetail(null)
  }, [])

  useEffect(() => {
    return () => {
      taskDetailRequestRef.current?.abort()
    }
  }, [])

  const hasOrchestrationSupport = useCallback(async (): Promise<boolean> => {
    if (!connectionConfig) return true
    if (orchestrationSupportRef.current != null) {
      return orchestrationSupportRef.current
    }
    try {
      const openApiUrl = buildRequestUrl("/openapi.json")
      if (!openApiUrl) {
        return true
      }
      const res = await fetch(openApiUrl)
      if (!res.ok) {
        return true
      }
      const spec = await res.json()
      const hasProjectsPath = Boolean(
        spec &&
          typeof spec === "object" &&
          spec.paths &&
          typeof spec.paths === "object" &&
          AGENT_ORCHESTRATION_PROJECTS_PATH in spec.paths
      )
      orchestrationSupportRef.current = hasProjectsPath
      return hasProjectsPath
    } catch {
      return true
    }
  }, [buildRequestUrl, connectionConfig])

  const fetchACPReadiness = useCallback(async () => {
    if (!connectionConfig) {
      setSetupIssues([])
      return
    }
    const healthTransport = buildRequestTransport("/api/v1/acp/health")
    if (!healthTransport) {
      return
    }
    setSetupLoading(true)
    try {
      const res = await fetch(healthTransport.url, {
        headers: getHeaders(healthTransport)
      })
      if (!res.ok) {
        setSetupIssues(buildACPSetupIssues(null, `ACP health returned HTTP ${res.status}`))
        return
      }
      setSetupIssues(buildACPSetupIssues(normalizeACPHealthStatus(await res.json())))
    } catch (err) {
      setSetupIssues(
        buildACPSetupIssues(
          null,
          err instanceof Error ? err.message : "Failed to reach ACP health"
        )
      )
    } finally {
      setSetupLoading(false)
    }
  }, [buildRequestTransport, connectionConfig, getHeaders])

  const fetchProjects = useCallback(async () => {
    if (!apiBase) return
    setLoading(true)
    setError(null)
    setProjectsLoadedSuccessfully(false)
    try {
      const supported = await hasOrchestrationSupport()
      if (!supported) {
        markUnsupported()
        return
      }
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/projects`, { headers })
      await ensureOrchestrationResponse(res)
      const data = await res.json()
      setIsUnsupported(false)
      setProjects(normalizeListPayload<ProjectSummary>(data, "projects"))
      setProjectsLoadedSuccessfully(true)
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setProjectsLoadedSuccessfully(false)
        setError(err instanceof Error ? err.message : "Failed to load projects")
      }
    } finally {
      setLoading(false)
    }
  }, [apiBase, apiTransport, getHeaders, hasOrchestrationSupport, markUnsupported])

  const fetchTasks = useCallback(
    async (projectId: number) => {
      if (!apiBase) return
      setTasksLoading(true)
      try {
        const headers = getHeaders(apiTransport)
        const res = await fetch(`${apiBase}/projects/${projectId}/tasks`, { headers })
        await ensureOrchestrationResponse(res)
        const data = await res.json()
        setIsUnsupported(false)
        setTasks(normalizeListPayload<TaskItem>(data, "tasks"))
      } catch (err) {
        if (isUnsupportedError(err)) {
          markUnsupported()
        } else {
          setError(err instanceof Error ? err.message : "Failed to load tasks")
        }
      } finally {
        setTasksLoading(false)
      }
    },
    [apiBase, apiTransport, getHeaders, markUnsupported]
  )

  useEffect(() => {
    if (!connectionConfig) return
    void fetchProjects()
  }, [connectionConfig, fetchProjects])

  useEffect(() => {
    if (!connectionConfig) return
    void fetchACPReadiness()
  }, [connectionConfig, fetchACPReadiness])

  useEffect(() => {
    if (connectionConfig && selectedProjectId !== null) {
      void fetchTasks(selectedProjectId)
    } else {
      setTasks([])
    }
  }, [connectionConfig, selectedProjectId, fetchTasks])

  const handleCreateProject = async (values: { name: string; description?: string }) => {
    try {
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/projects`, {
        method: "POST",
        headers,
        body: JSON.stringify(values),
      })
      await ensureOrchestrationResponse(res)
      setShowProjectModal(false)
      projectForm.resetFields()
      void fetchProjects()
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to create project")
      }
    }
  }

  const handleCreateTask = async (values: {
    title: string
    description?: string
    agent_type?: string
    dependency_id?: number
    max_review_attempts?: number
  }) => {
    if (selectedProjectId === null) return
    try {
      const headers = getHeaders(apiTransport)
      const body = {
        ...values,
        dependency_id: values.dependency_id || undefined,
        max_review_attempts: values.max_review_attempts || 3,
      }
      const res = await fetch(`${apiBase}/projects/${selectedProjectId}/tasks`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      })
      await ensureOrchestrationResponse(res)
      setShowTaskModal(false)
      taskForm.resetFields()
      void fetchTasks(selectedProjectId)
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to create task")
      }
    }
  }

  const handleDispatchRun = async (taskId: number) => {
    try {
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/tasks/${taskId}/run`, {
        method: "POST",
        headers,
        body: JSON.stringify({}),
      })
      if (res.status === 404) {
        throw createUnsupportedError()
      }
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }
      if (selectedProjectId !== null) {
        void fetchTasks(selectedProjectId)
      }
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to dispatch run")
      }
    }
  }

  const handleSubmitReview = async (taskId: number, approved: boolean) => {
    try {
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/tasks/${taskId}/review`, {
        method: "POST",
        headers,
        body: JSON.stringify({ approved }),
      })
      await ensureOrchestrationResponse(res)
      if (selectedProjectId !== null) {
        void fetchTasks(selectedProjectId)
      }
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to submit review")
      }
    }
  }

  const handleInspectTask = async (taskId: number) => {
    if (!apiBase) return
    taskDetailRequestRef.current?.abort()
    const controller = new AbortController()
    taskDetailRequestRef.current = controller
    setTaskDetail(null)
    setTaskDetailLoading(true)
    setError(null)
    try {
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/tasks/${taskId}`, {
        headers,
        signal: controller.signal
      })
      await ensureOrchestrationResponse(res)
      const data = await res.json()
      if (taskDetailRequestRef.current !== controller || controller.signal.aborted) {
        return
      }
      setTaskDetail(data as TaskDetailItem)
    } catch (err) {
      if (taskDetailRequestRef.current !== controller || controller.signal.aborted) {
        return
      }
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to load task diagnostics")
      }
    } finally {
      if (taskDetailRequestRef.current === controller) {
        taskDetailRequestRef.current = null
        setTaskDetailLoading(false)
      }
    }
  }

  const handleDeleteProject = async (projectId: number) => {
    try {
      const headers = getHeaders(apiTransport)
      const res = await fetch(`${apiBase}/projects/${projectId}`, {
        method: "DELETE",
        headers,
      })
      await ensureOrchestrationResponse(res)
      if (selectedProjectId === projectId) {
        setSelectedProjectId(null)
      }
      void fetchProjects()
    } catch (err) {
      if (isUnsupportedError(err)) {
        markUnsupported()
      } else {
        setError(err instanceof Error ? err.message : "Failed to delete project")
      }
    }
  }

  const workspaceOptions = useMemo(() => {
    const options = new Map<string, string>()
    if (workspaceFilterId) {
      options.set(workspaceFilterId, workspaceFilterId)
    }
    for (const project of projects) {
      const canonicalWorkspaceId = getCanonicalWorkspaceId(project)
      if (canonicalWorkspaceId) {
        options.set(canonicalWorkspaceId, canonicalWorkspaceId)
      }
    }
    return Array.from(options, ([value, label]) => ({ value, label }))
  }, [projects, workspaceFilterId])

  const workspaceSelectOptions = useMemo(
    () => [
      { value: ALL_WORKSPACES_FILTER_VALUE, label: "All workspaces" },
      ...workspaceOptions
    ],
    [workspaceOptions]
  )

  const filteredProjects = useMemo(() => {
    if (!workspaceFilterId) {
      return projects
    }
    return projects.filter(
      (project) => getCanonicalWorkspaceId(project) === workspaceFilterId
    )
  }, [projects, workspaceFilterId])

  const visibleTasks = useMemo(() => {
    if (!workspaceFilterId) {
      return tasks
    }
    return tasks.filter((task) => {
      const taskWorkspaceId = getCanonicalWorkspaceId(task)
      return !taskWorkspaceId || taskWorkspaceId === workspaceFilterId
    })
  }, [tasks, workspaceFilterId])

  const workspaceSetupIssues = useMemo<ACPSetupIssue[]>(() => {
    if (!workspaceFilterId || loading || isUnsupported || !projectsLoadedSuccessfully) {
      return []
    }

    const matchingProjects = filteredProjects
    if (matchingProjects.length === 0) {
      return [
        {
          code: "canonical_workspace_bridge_missing",
          title: `No ACP execution workspace is linked to ${workspaceFilterId}`,
          description:
            "Create an agent task from Research Workspace so the execution root, environment, and MCP readiness can be validated before dispatch."
        }
      ]
    }

    const hasLinkedProject = matchingProjects.some(
      (project) =>
        getCanonicalWorkspaceLinkStatus(project) ===
        LINKED_CANONICAL_WORKSPACE_STATUS
    )
    const unlinkedProject = matchingProjects.find((project) => {
      const status = getCanonicalWorkspaceLinkStatus(project)
      return status !== null && status !== LINKED_CANONICAL_WORKSPACE_STATUS
    })
    if (!hasLinkedProject && unlinkedProject) {
      const status = getCanonicalWorkspaceLinkStatus(unlinkedProject) || "unknown"
      return [
        {
          code: "canonical_workspace_bridge_unlinked",
          title: `ACP execution workspace link is ${status}`,
          description:
            "Recreate the workspace handoff from Research Workspace so root, environment, and MCP readiness are checked before dispatch."
        }
      ]
    }

    return []
  }, [
    filteredProjects,
    isUnsupported,
    loading,
    projectsLoadedSuccessfully,
    workspaceFilterId
  ])

  const updateWorkspaceFilter = useCallback(
    (nextWorkspaceFilterId: string | null) => {
      const params = new URLSearchParams(location.search)
      params.delete("workspace")
      params.delete("workspace_id")
      params.delete("canonical_workspace_id")
      if (nextWorkspaceFilterId) {
        params.set("workspace", nextWorkspaceFilterId)
      }
      const search = params.toString()
      navigate(
        {
          pathname: location.pathname,
          search: search ? `?${search}` : ""
        },
        { replace: true }
      )
    },
    [location.pathname, location.search, navigate]
  )

  useEffect(() => {
    if (
      selectedProjectId !== null &&
      !filteredProjects.some((project) => project.id === selectedProjectId)
    ) {
      setSelectedProjectId(null)
      setTasks([])
    }
  }, [filteredProjects, selectedProjectId])

  const selectedProject = filteredProjects.find((p) => p.id === selectedProjectId)

  return (
    <div className="space-y-6">
      {isUnsupported && (
        <Alert
          variant="warning"
          title={AGENT_ORCHESTRATION_UNSUPPORTED_MESSAGE}
        >
          <AgentTasksSetupDescription
            body={AGENT_ORCHESTRATION_UNSUPPORTED_DESCRIPTION}
            issues={[
              {
                code: "orchestration_routes_missing",
                title: "Agent task routes are missing",
                description: "Upgrade or enable the agent orchestration API before creating tasks."
              }
            ]}
          />
        </Alert>
      )}
      {!isUnsupported && setupIssues.length > 0 && (
        <Alert
          variant="warning"
          title="ACP setup needs attention"
        >
          <AgentTasksSetupDescription
            body={
              setupLoading
                ? "Checking ACP setup state..."
                : "Resolve these ACP setup items before dispatching production task runs."
            }
            issues={setupIssues}
          />
        </Alert>
      )}
      {!isUnsupported && workspaceSetupIssues.length > 0 && (
        <Alert
          variant="warning"
          title="Workspace setup needs attention"
        >
          <AgentTasksSetupDescription
            body="Resolve these workspace setup items before dispatching task runs."
            issues={workspaceSetupIssues}
            showAgentRegistry={false}
            showResearchWorkspace
          />
        </Alert>
      )}
      {error && (
        <Alert
          variant="error"
          title={error}
          dismissible
          onDismiss={() => setError(null)}
        />
      )}

      {(workspaceOptions.length > 0 || workspaceFilterId) && (
        <div className="flex flex-wrap items-center gap-3 border border-border bg-surface px-4 py-3">
          <span className="text-sm font-medium">Canonical workspace</span>
          <Select
            aria-label="Workspace filter"
            value={workspaceFilterId ?? ALL_WORKSPACES_FILTER_VALUE}
            options={workspaceSelectOptions}
            onChange={(value) => {
              const nextValue = String(value || "")
              updateWorkspaceFilter(
                nextValue === ALL_WORKSPACES_FILTER_VALUE
                  ? null
                  : normalizeWorkspaceFilterId(nextValue)
              )
            }}
            style={{ minWidth: 240 }}
          />
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Projects Panel */}
        <Card
          title={
            <span className="flex items-center gap-2">
              <FolderPlus className="h-4 w-4" />
              Projects
            </span>
          }
          extra={
            <div className="flex gap-1">
              <Button
                size="small"
                icon={<RefreshCw className="h-3.5 w-3.5" />}
                onClick={() => void fetchProjects()}
              />
              <Button
                size="small"
                type="primary"
                icon={<Plus className="h-3.5 w-3.5" />}
                onClick={() => setShowProjectModal(true)}
              >
                New
              </Button>
            </div>
          }
          className="lg:col-span-1"
          styles={{ body: { padding: 0 } }}
        >
          {loading ? (
            <div className="flex justify-center py-8">
              <Spin />
            </div>
          ) : filteredProjects.length === 0 ? (
            <Empty
              description={
                workspaceFilterId
                  ? "No projects linked to this workspace yet"
                  : "No projects yet"
              }
              className="py-8"
            >
              <Button type="primary" onClick={() => setShowProjectModal(true)}>
                Create Project
              </Button>
            </Empty>
          ) : (
            <div className="divide-y divide-border">
              {filteredProjects.map((project) => (
                <div
                  key={project.id}
                  className={`flex cursor-pointer items-center gap-2 px-4 py-3 transition-colors hover:bg-surface-hover ${
                    selectedProjectId === project.id ? "bg-surface-hover" : ""
                  }`}
                  onClick={() => setSelectedProjectId(project.id)}
                >
                  <ChevronRight
                    className={`h-4 w-4 shrink-0 transition-transform ${
                      selectedProjectId === project.id ? "rotate-90" : ""
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{project.name}</div>
                    {getCanonicalWorkspaceId(project) && (
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        Workspace: {getCanonicalWorkspaceId(project)}
                      </div>
                    )}
                    {project.task_summary && (
                      <div className="flex gap-1 text-xs text-muted-foreground mt-0.5">
                        <span>{project.task_summary.total_tasks} tasks</span>
                        {project.task_summary.status_counts?.complete > 0 && (
                          <span className="text-green-600">
                            {project.task_summary.status_counts.complete} done
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <Tooltip title="Delete project">
                    <Button
                      size="small"
                      type="text"
                      danger
                      icon={<Trash2 className="h-3.5 w-3.5" />}
                      onClick={(e) => {
                        e.stopPropagation()
                        void handleDeleteProject(project.id)
                      }}
                    />
                  </Tooltip>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Tasks Panel */}
        <Card
          title={
            <span className="flex items-center gap-2">
              <ListTodo className="h-4 w-4" />
              Tasks
              {selectedProject && (
                <span className="text-sm font-normal text-muted-foreground">
                  — {selectedProject.name}
                </span>
              )}
            </span>
          }
          extra={
            selectedProjectId !== null && (
              <div className="flex gap-1">
                <Button
                  size="small"
                  icon={<RefreshCw className="h-3.5 w-3.5" />}
                  onClick={() => void fetchTasks(selectedProjectId)}
                />
                <Button
                  size="small"
                  type="primary"
                  icon={<Plus className="h-3.5 w-3.5" />}
                  onClick={() => setShowTaskModal(true)}
                >
                  New Task
                </Button>
              </div>
            )
          }
          className="lg:col-span-2"
        >
          {selectedProjectId === null ? (
            <Empty description="Select a project to view tasks" className="py-8" />
          ) : tasksLoading ? (
            <div className="flex justify-center py-8">
              <Spin />
            </div>
          ) : visibleTasks.length === 0 ? (
            <Empty description="No tasks yet" className="py-8">
              <Button type="primary" onClick={() => setShowTaskModal(true)}>
                Create Task
              </Button>
            </Empty>
          ) : (
            <div className="space-y-3">
              {visibleTasks.map((task) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  allTasks={visibleTasks}
                  onDispatchRun={handleDispatchRun}
                  onReview={handleSubmitReview}
                  onInspectTask={handleInspectTask}
                />
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Create Project Modal */}
      <Modal
        title="Create Project"
        open={showProjectModal}
        onCancel={() => setShowProjectModal(false)}
        onOk={() => projectForm.submit()}
        okText="Create"
      >
        <Form form={projectForm} layout="vertical" onFinish={handleCreateProject}>
          <Form.Item
            name="name"
            label="Project Name"
            rules={[{ required: true, message: "Project name is required" }]}
          >
            <Input placeholder="My Agent Project" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Optional description..." rows={3} />
          </Form.Item>
        </Form>
      </Modal>

      {/* Create Task Modal */}
      <Modal
        title="Create Task"
        open={showTaskModal}
        onCancel={() => setShowTaskModal(false)}
        onOk={() => taskForm.submit()}
        okText="Create"
      >
        <Form form={taskForm} layout="vertical" onFinish={handleCreateTask}>
          <Form.Item
            name="title"
            label="Task Title"
            rules={[{ required: true, message: "Task title is required" }]}
          >
            <Input placeholder="Implement feature X" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea placeholder="Detailed task description..." rows={3} />
          </Form.Item>
          <Form.Item name="agent_type" label="Agent Type">
            <Select
              placeholder="Default agent"
              allowClear
              options={[
                { value: "claude_code", label: "Claude Code" },
                { value: "codex", label: "Codex CLI" },
                { value: "opencode", label: "OpenCode" },
              ]}
            />
          </Form.Item>
          <Form.Item name="dependency_id" label="Depends On">
            <Select
              placeholder="No dependency"
              allowClear
              options={visibleTasks.map((t) => ({
                value: t.id,
                label: `#${t.id}: ${t.title}`,
              }))}
            />
          </Form.Item>
          <Form.Item name="max_review_attempts" label="Max Review Attempts">
            <Select
              defaultValue={3}
              options={[
                { value: 1, label: "1" },
                { value: 2, label: "2" },
                { value: 3, label: "3" },
                { value: 5, label: "5" },
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="Task diagnostics"
        open={Boolean(taskDetail) || taskDetailLoading}
        onCancel={handleCloseTaskDetail}
        footer={null}
        width={820}
      >
        {taskDetailLoading ? (
          <div className="flex justify-center py-8">
            <Spin />
          </div>
        ) : taskDetail ? (
          <TaskDiagnostics task={taskDetail} />
        ) : null}
      </Modal>
    </div>
  )
}

const AgentTasksSetupDescription: React.FC<{
  body: string
  issues: ACPSetupIssue[]
  showAgentRegistry?: boolean
  showResearchWorkspace?: boolean
}> = ({
  body,
  issues,
  showAgentRegistry = true,
  showResearchWorkspace = false
}) => (
  <div className="space-y-3">
    <div>{body}</div>
    <ul className="m-0 space-y-2 pl-4">
      {issues.map((issue) => (
        <li key={issue.code}>
          <div className="font-medium">{issue.title}</div>
          <div className="text-sm">{issue.description}</div>
        </li>
      ))}
    </ul>
    <div className="flex flex-wrap gap-2">
      {showAgentRegistry && (
        <Button
          size="small"
          icon={<ExternalLink className="h-3 w-3" />}
          onClick={() => navigateOptionRoute("/agents")}
        >
          Open Agent Registry
        </Button>
      )}
      {showResearchWorkspace && (
        <Button
          size="small"
          icon={<ExternalLink className="h-3 w-3" />}
          onClick={() => navigateOptionRoute(RESEARCH_WORKSPACE_PATH)}
        >
          Open Research Workspace
        </Button>
      )}
      <Button
        size="small"
        icon={<ExternalLink className="h-3 w-3" />}
        onClick={() => navigateOptionRoute("/acp-playground")}
      >
        Open ACP Playground
      </Button>
    </div>
  </div>
)

const TaskDiagnostics: React.FC<{ task: TaskDetailItem }> = ({ task }) => {
  const runs = task.runs ?? []
  const reviews = task.reviews ?? []
  const runReviewFeedback = new Set(
    runs
      .map((run) => run.review_decision?.feedback_preview)
      .filter((feedback): feedback is string => Boolean(feedback))
  )
  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs uppercase tracking-wide text-muted-foreground">Task</div>
        <div className="text-base font-medium">{task.title}</div>
        <div className="mt-1 flex flex-wrap gap-2">
          <Tag color={STATUS_COLORS[task.status] ?? "default"}>{task.status}</Tag>
          <Tag>#{task.id}</Tag>
          <Tag>
            Reviews: {task.review_count}/{task.max_review_attempts}
          </Tag>
        </div>
      </div>

      {runs.length === 0 ? (
        <Empty description="No runs recorded for this task" />
      ) : (
        <div className="space-y-3">
          {runs.map((run) => (
            <RunDiagnostics key={run.id} run={run} />
          ))}
        </div>
      )}

      {reviews.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm font-medium">Reviews</div>
          {reviews.map((review, index) => (
            <div key={`${review.reviewer || "review"}-${index}`} className="rounded border border-border p-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <Tag color={review.approved ? "success" : "error"}>
                  {review.approved ? "Approved" : "Rejected"}
                </Tag>
                {review.reviewer && <span>{review.reviewer}</span>}
              </div>
              {review.feedback && !runReviewFeedback.has(review.feedback) && (
                <div className="mt-2 text-muted-foreground">{review.feedback}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const openRunSessionRoute = (run: RunItem, view?: string) => {
  const sessionId = run.session?.session_id || run.session_id
  if (!sessionId) return
  const params = new URLSearchParams({ session: sessionId })
  if (view) params.set("view", view)
  navigateOptionRoute(`/acp-playground?${params.toString()}`)
}

const RunDiagnostics: React.FC<{ run: RunItem }> = ({ run }) => {
  const sessionId = run.session?.session_id || run.session_id
  const failureContext = run.failure_context
  return (
    <div className="rounded-lg border border-border p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">Run #{run.id}</span>
          <Tag color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : "processing"}>
            {run.status}
          </Tag>
          {run.agent_type && <Tag>{run.agent_type}</Tag>}
        </div>
        {sessionId && <span className="text-xs text-muted-foreground">{sessionId}</span>}
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
        <span>{run.history?.event_count ?? 0} events</span>
        <span>{run.history?.audit_event_count ?? 0} audit</span>
        <span>{run.history?.artifact_count ?? 0} artifacts</span>
        <span>{run.history?.diagnostic_count ?? 0} diagnostics</span>
      </div>

      {failureContext && (
        <div className="mt-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-300">
          {failureContext.reason_code && (
            <div className="font-medium">{failureContext.reason_code}</div>
          )}
          {failureContext.message && <div>{failureContext.message}</div>}
        </div>
      )}

      {!failureContext?.message && run.error && (
        <div className="mt-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-300">
          {run.error}
        </div>
      )}

      {run.result_summary && (
        <div className="mt-3 text-sm">{run.result_summary}</div>
      )}
      {run.history?.result?.preview && (
        <div className="mt-3 text-sm text-muted-foreground">{run.history.result.preview}</div>
      )}
      {run.review_decision?.feedback_preview && (
        <div className="mt-3 text-sm">{run.review_decision.feedback_preview}</div>
      )}

      {sessionId && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            size="small"
            icon={<ExternalLink className="h-3 w-3" />}
            onClick={() => openRunSessionRoute(run)}
          >
            Open session
          </Button>
          {run.session?.links?.diagnostics && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => openRunSessionRoute(run, "diagnostics")}
            >
              Open diagnostics
            </Button>
          )}
          {run.session?.links?.artifacts && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => openRunSessionRoute(run, "artifacts")}
            >
              Open artifacts
            </Button>
          )}
          {run.session?.links?.audit && (
            <Button
              size="small"
              icon={<ExternalLink className="h-3 w-3" />}
              onClick={() => openRunSessionRoute(run, "audit")}
            >
              Open audit
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

const TaskCard: React.FC<{
  task: TaskItem
  allTasks: TaskItem[]
  onDispatchRun: (taskId: number) => Promise<void>
  onReview: (taskId: number, approved: boolean) => Promise<void>
  onInspectTask: (taskId: number) => Promise<void>
}> = ({ task, allTasks, onDispatchRun, onReview, onInspectTask }) => {
  const depTask = task.dependency_id
    ? allTasks.find((t) => t.id === task.dependency_id)
    : null

  return (
    <div className="rounded-lg border border-border p-4">
      <div className="mb-2 flex items-start justify-between">
        <div className="flex items-center gap-2">
          {STATUS_ICONS[task.status] ?? <Clock className="h-3.5 w-3.5" />}
          <h4 className="font-medium">{task.title}</h4>
          <Tag color={STATUS_COLORS[task.status] ?? "default"}>
            {task.status}
          </Tag>
        </div>
        <span className="text-xs text-muted-foreground">#{task.id}</span>
      </div>

      {task.description && (
        <p className="mb-2 text-sm text-muted-foreground">{task.description}</p>
      )}

      <div className="mb-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
        {task.agent_type && <Tag>{task.agent_type}</Tag>}
        {depTask && (
          <Tag color="blue">
            Depends on: #{depTask.id} ({depTask.status})
          </Tag>
        )}
        {task.review_count > 0 && (
          <Tag>
            Reviews: {task.review_count}/{task.max_review_attempts}
          </Tag>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button
          size="small"
          icon={<Search className="h-3 w-3" />}
          onClick={() => void onInspectTask(task.id)}
        >
          Inspect
        </Button>
        {task.status === "todo" && (
          <Button
            size="small"
            type="primary"
            icon={<Play className="h-3 w-3" />}
            onClick={() => void onDispatchRun(task.id)}
          >
            Run
          </Button>
        )}
        {task.status === "review" && (
          <>
            <Button
              size="small"
              type="primary"
              icon={<CheckCircle className="h-3 w-3" />}
              onClick={() => void onReview(task.id, true)}
            >
              Approve
            </Button>
            <Button
              size="small"
              danger
              icon={<XCircle className="h-3 w-3" />}
              onClick={() => void onReview(task.id, false)}
            >
              Reject
            </Button>
          </>
        )}
        {task.status === "inprogress" && (
          <DesignSystemBadge variant="info" size="sm">
            Running...
          </DesignSystemBadge>
        )}
        {task.status === "triage" && (
          <DesignSystemBadge variant="danger" size="sm">
            Needs human attention
          </DesignSystemBadge>
        )}
      </div>
    </div>
  )
}

export default AgentTasksPage
