import React from "react"
import { useTranslation } from "react-i18next"
import { Alert, Button, Input, Modal } from "antd"
import { ExternalLink } from "lucide-react"

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { buildACPAuthHeaders } from "@/services/acp/connection"
import {
  resolveBrowserRequestTransport,
  type BrowserRequestTransport
} from "@/services/tldw/request-core"

const AGENT_ORCHESTRATION_BASE_PATH = "/api/v1/agent-orchestration"
const CANONICAL_BRIDGE_PATH =
  `${AGENT_ORCHESTRATION_BASE_PATH}/workspaces/canonical-bridge`
const PROJECTS_PATH = `${AGENT_ORCHESTRATION_BASE_PATH}/projects`

type CanonicalBridgeResponse = {
  id?: number
  canonical_workspace?: {
    acp_workspace_id?: number
  } | null
}

type ProjectResponse = {
  id?: number
}

type TaskResponse = {
  id?: number
}

type CreatedAgentTask = {
  acpWorkspaceId: number
  projectId: number
  taskId: number
}

export interface WorkspaceAgentTaskHandoffModalProps {
  open: boolean
  workspaceId?: string | null
  workspaceName?: string | null
  workspaceTag?: string | null
  onBeforeSubmit?: () => Promise<void> | void
  onCancel: () => void
  onOpenAgentTasks: () => void
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

const getAcpWorkspaceId = (payload: CanonicalBridgeResponse): number | null => {
  const candidate =
    typeof payload.id === "number"
      ? payload.id
      : payload.canonical_workspace?.acp_workspace_id
  return typeof candidate === "number" && Number.isFinite(candidate)
    ? candidate
    : null
}

export const WorkspaceAgentTaskHandoffModal: React.FC<
  WorkspaceAgentTaskHandoffModalProps
> = ({
  open,
  workspaceId,
  workspaceName,
  workspaceTag,
  onBeforeSubmit,
  onCancel,
  onOpenAgentTasks
}) => {
  const { t } = useTranslation(["playground", "common"])
  const {
    config: connectionConfig,
    loading: connectionConfigLoading
  } = useCanonicalConnectionConfig()
  const [rootPath, setRootPath] = React.useState("")
  const [taskTitle, setTaskTitle] = React.useState("")
  const [taskDescription, setTaskDescription] = React.useState("")
  const [agentType, setAgentType] = React.useState("codex")
  const [submitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [createdTask, setCreatedTask] = React.useState<CreatedAgentTask | null>(
    null
  )
  const wasOpenRef = React.useRef(false)

  const workspaceDisplayName = workspaceName?.trim() || "Workspace"

  React.useEffect(() => {
    if (!open) {
      wasOpenRef.current = false
      return
    }
    if (wasOpenRef.current) return
    wasOpenRef.current = true
    setRootPath("")
    setTaskTitle(
      t("playground:workspace.agentTaskDefaultTitle", {
        defaultValue: "Continue {{workspace}} work",
        workspace: workspaceDisplayName
      })
    )
    setTaskDescription("")
    setAgentType("codex")
    setSubmitting(false)
    setError(null)
    setCreatedTask(null)
  }, [open, t, workspaceDisplayName])

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
      return buildACPAuthHeaders(connectionConfig, { includeContentType: true })
    },
    [connectionConfig]
  )

  const postJson = React.useCallback(
    async <T,>(path: string, payload: unknown): Promise<T> => {
      const transport = buildRequestTransport(path)
      if (!transport) {
        throw new Error(
          connectionConfigLoading
            ? t(
                "playground:workspace.agentTaskConnectionLoading",
                "Backend connection is still loading."
              )
            : t(
                "playground:workspace.agentTaskConnectionMissing",
                "Backend connection is not configured."
              )
        )
      }

      const response = await fetch(transport.url, {
        method: "POST",
        headers: getHeaders(transport),
        body: JSON.stringify(payload)
      })
      if (!response.ok) {
        throw new Error(await readApiErrorMessage(response))
      }
      return (await response.json()) as T
    },
    [buildRequestTransport, connectionConfigLoading, getHeaders, t]
  )

  const deleteProject = React.useCallback(
    async (projectId: number): Promise<void> => {
      const transport = buildRequestTransport(`${PROJECTS_PATH}/${projectId}`)
      if (!transport) {
        throw new Error(
          connectionConfigLoading
            ? t(
                "playground:workspace.agentTaskConnectionLoading",
                "Backend connection is still loading."
              )
            : t(
                "playground:workspace.agentTaskConnectionMissing",
                "Backend connection is not configured."
              )
        )
      }

      const response = await fetch(transport.url, {
        method: "DELETE",
        headers: getHeaders(transport)
      })
      if (!response.ok) {
        throw new Error(await readApiErrorMessage(response))
      }
    },
    [buildRequestTransport, connectionConfigLoading, getHeaders, t]
  )

  const handleSubmit = async () => {
    const canonicalWorkspaceId = workspaceId?.trim()
    const trimmedRootPath = rootPath.trim()
    const trimmedTitle = taskTitle.trim()

    if (connectionConfigLoading) {
      setError(
        t(
          "playground:workspace.agentTaskConnectionLoading",
          "Backend connection is still loading."
        )
      )
      return
    }
    if (!canonicalWorkspaceId) {
      setError(
        t(
          "playground:workspace.agentTaskMissingWorkspace",
          "Select or save a workspace before creating an agent task."
        )
      )
      return
    }
    if (!trimmedRootPath) {
      setError(
        t(
          "playground:workspace.agentTaskMissingRoot",
          "Enter an execution root path."
        )
      )
      return
    }
    if (!trimmedTitle) {
      setError(
        t("playground:workspace.agentTaskMissingTitle", "Enter a task title.")
      )
      return
    }

    setSubmitting(true)
    setError(null)
    setCreatedTask(null)

    const baseMetadata = {
      created_from: "workspace_playground",
      canonical_workspace_id: canonicalWorkspaceId,
      canonical_workspace_source: "workspace_playground",
      workspace_name: workspaceName?.trim() || null,
      workspace_tag: workspaceTag?.trim() || null
    }

    let createdProjectId: number | null = null
    try {
      await onBeforeSubmit?.()
      const bridge = await postJson<CanonicalBridgeResponse>(
        CANONICAL_BRIDGE_PATH,
        {
          canonical_workspace_id: canonicalWorkspaceId,
          canonical_workspace_source: "workspace_playground",
          root_path: trimmedRootPath,
          name: `${workspaceDisplayName} execution`,
          description: `ACP execution workspace linked from ${workspaceDisplayName}.`,
          metadata: baseMetadata
        }
      )
      const acpWorkspaceId = getAcpWorkspaceId(bridge)
      if (acpWorkspaceId == null) {
        throw new Error(
          t(
            "playground:workspace.agentTaskBridgeMissingId",
            "Canonical bridge did not return an ACP workspace ID."
          )
        )
      }

      const metadata = {
        ...baseMetadata,
        acp_workspace_id: acpWorkspaceId
      }
      const project = await postJson<ProjectResponse>(PROJECTS_PATH, {
        name: `${workspaceDisplayName} agent work`,
        description: `Agent tasks created from ${workspaceDisplayName}.`,
        workspace_id: acpWorkspaceId,
        metadata
      })
      if (typeof project.id !== "number") {
        throw new Error(
          t(
            "playground:workspace.agentTaskProjectMissingId",
            "Agent project creation did not return a project ID."
          )
        )
      }
      createdProjectId = project.id

      const task = await postJson<TaskResponse>(
        `${PROJECTS_PATH}/${project.id}/tasks`,
        {
          title: trimmedTitle,
          description: taskDescription.trim(),
          agent_type: agentType.trim() || undefined,
          max_review_attempts: 3,
          metadata
        }
      )
      if (typeof task.id !== "number") {
        throw new Error(
          t(
            "playground:workspace.agentTaskTaskMissingId",
            "Agent task creation did not return a task ID."
          )
        )
      }
      createdProjectId = null

      setCreatedTask({
        acpWorkspaceId,
        projectId: project.id,
        taskId: task.id
      })
    } catch (err) {
      if (createdProjectId != null) {
        try {
          await deleteProject(createdProjectId)
        } catch (rollbackError) {
          console.warn(
            "[workspace-agent-task-handoff] Failed to roll back ACP project after task creation failure",
            rollbackError
          )
        }
      }
      setError(
        err instanceof Error
          ? err.message
          : t(
              "playground:workspace.agentTaskCreateFailed",
              "Failed to create agent task."
            )
      )
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = React.useCallback(() => {
    if (submitting) return
    onCancel()
  }, [onCancel, submitting])

  const createTaskDisabled =
    submitting || Boolean(createdTask) || connectionConfigLoading

  return (
    <Modal
      title={t(
        "playground:workspace.agentTaskModalTitle",
        "Create agent task"
      )}
      open={open}
      onCancel={handleCancel}
      closable={!submitting}
      destroyOnHidden
      footer={[
        <Button key="cancel" onClick={handleCancel} disabled={submitting}>
          {t("common:cancel", "Cancel")}
        </Button>,
        createdTask ? (
          <Button
            key="open-agent-tasks"
            type="primary"
            icon={<ExternalLink className="h-4 w-4" />}
            onClick={onOpenAgentTasks}
          >
            {t("playground:workspace.openAgentTasks", "Open Agent Tasks")}
          </Button>
        ) : (
          <Button
            key="create-task"
            type="primary"
            loading={submitting}
            disabled={createTaskDisabled}
            onClick={() => void handleSubmit()}
          >
            {t("playground:workspace.createAgentTaskSubmit", "Create task")}
          </Button>
        )
      ]}
      centered
      maskClosable={!submitting}
      keyboard={!submitting}
    >
      <div className="space-y-4">
        {error && (
          <Alert
            type="error"
            title={error}
            showIcon
          />
        )}
        {createdTask && (
          <Alert
            type="success"
            title={t(
              "playground:workspace.agentTaskCreated",
              "Agent task created"
            )}
            description={
              <div className="flex flex-wrap gap-2 text-sm">
                <span>ACP workspace #{createdTask.acpWorkspaceId}</span>
                <span>Project #{createdTask.projectId}</span>
                <span>Task #{createdTask.taskId}</span>
              </div>
            }
            showIcon
          />
        )}

        <div className="space-y-1.5">
          <label
            htmlFor="workspace-agent-task-root-path"
            className="block text-sm font-medium text-text"
          >
            {t(
              "playground:workspace.agentTaskRootPath",
              "Execution root path"
            )}
          </label>
          <Input
            id="workspace-agent-task-root-path"
            value={rootPath}
            onChange={(event) => setRootPath(event.target.value)}
            placeholder="/Users/name/src/project"
            disabled={submitting || Boolean(createdTask)}
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="workspace-agent-task-title"
            className="block text-sm font-medium text-text"
          >
            {t("playground:workspace.agentTaskTitle", "Task title")}
          </label>
          <Input
            id="workspace-agent-task-title"
            value={taskTitle}
            onChange={(event) => setTaskTitle(event.target.value)}
            disabled={submitting || Boolean(createdTask)}
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="workspace-agent-task-description"
            className="block text-sm font-medium text-text"
          >
            {t(
              "playground:workspace.agentTaskDescription",
              "Task description"
            )}
          </label>
          <Input.TextArea
            id="workspace-agent-task-description"
            value={taskDescription}
            onChange={(event) => setTaskDescription(event.target.value)}
            autoSize={{ minRows: 3, maxRows: 6 }}
            disabled={submitting || Boolean(createdTask)}
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="workspace-agent-task-agent-type"
            className="block text-sm font-medium text-text"
          >
            {t("playground:workspace.agentTaskAgentType", "Agent type")}
          </label>
          <Input
            id="workspace-agent-task-agent-type"
            value={agentType}
            onChange={(event) => setAgentType(event.target.value)}
            disabled={submitting || Boolean(createdTask)}
          />
        </div>
      </div>
    </Modal>
  )
}
