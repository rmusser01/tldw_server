import React, { useState, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  Modal,
  Form,
  Input,
  Radio,
  Collapse,
  notification,
  Alert,
  Steps,
  Tag,
  Tooltip,
} from "antd"
import {
  Folder,
  Bot,
  Server,
  ChevronDown,
  AlertCircle,
  Check,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/Common/Button"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { ACPRestClient } from "@/services/acp/client"
import { buildACPClientConfig } from "@/services/acp/connection"
import type {
  ACPAgentType,
  ACPAgentInfo,
  ACPMCPServerConfig,
  ACPSessionCreationStep,
  ACPStructuredError,
} from "@/services/acp/types"
import { useACPSessionsStore, type CreateSessionOptions } from "@/store/acp-sessions"
import { useWorkspaceStore } from "@/store/workspace"
import { MCPServerConfigList } from "./MCPServerConfigList"

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

interface ACPSessionCreateModalProps {
  open: boolean
  onClose: () => void
  onSuccess?: (sessionId: string) => void
}

interface FormValues {
  name: string
  cwd: string
  agentType: ACPAgentType
  tags: string
}

// -----------------------------------------------------------------------------
// Agent Type Cards
// -----------------------------------------------------------------------------

interface AgentCardProps {
  agent: ACPAgentInfo
  selected: boolean
  onSelect: () => void
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, selected, onSelect }) => {
  const { t } = useTranslation("playground")
  const disabled = !agent.is_configured

  return (
    <div
      className={`relative flex cursor-pointer items-start gap-3 rounded-lg border-2 p-3 transition-all ${
        selected
          ? "border-primary bg-primary/5"
          : disabled
            ? "cursor-not-allowed border-border bg-surface2/50 opacity-60"
            : "border-border hover:border-primary/50 hover:bg-surface2"
      }`}
      onClick={() => !disabled && onSelect()}
    >
      <div
        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${
          selected ? "bg-primary/20" : "bg-surface2"
        }`}
      >
        <Bot className={`h-5 w-5 ${selected ? "text-primary" : "text-text-muted"}`} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-text">{agent.name}</span>
          {!agent.is_configured && (
            <Tooltip
              title={t("acp.create.requiresApiKey", {
                key: agent.requires_api_key,
                defaultValue: `Requires {{key}}. Set it in your shell (export {{key}}=...) or in the [ACP] runner_env in config.txt.`,
              })}
            >
              <AlertCircle className="h-4 w-4 text-warning" />
            </Tooltip>
          )}
        </div>
        <p className="mt-0.5 text-xs text-text-muted">{agent.description}</p>
      </div>
      {selected && (
        <div className="absolute right-2 top-2">
          <Check className="h-4 w-4 text-primary" />
        </div>
      )}
    </div>
  )
}

// -----------------------------------------------------------------------------
// Creation Progress Steps
// -----------------------------------------------------------------------------

interface CreationProgressProps {
  step: ACPSessionCreationStep
  error?: ACPStructuredError
}

const CreationProgress: React.FC<CreationProgressProps> = ({ step, error }) => {
  const { t } = useTranslation("playground")

  const stepIndex = {
    idle: -1,
    creating: 0,
    starting_agent: 1,
    connecting: 2,
    ready: 3,
    error: -1,
  }[step]

  const stepStatus = step === "error" ? "error" : stepIndex === 3 ? "finish" : "process"

  if (step === "idle") return null

  return (
    <div className="mb-4">
      <Steps
        size="small"
        current={stepIndex}
        status={stepStatus}
        items={[
          { title: t("acp.create.steps.creating", "Creating") },
          { title: t("acp.create.steps.startingAgent", "Starting Agent") },
          { title: t("acp.create.steps.connecting", "Connecting") },
          { title: t("acp.create.steps.ready", "Ready") },
        ]}
      />
      {error && (
        <Alert
          className="mt-3"
          type="error"
          showIcon
          title={error.message}
          description={
            error.suggestions.length > 0 && (
              <ul className="mt-2 list-inside list-disc text-sm">
                {error.suggestions.map((s, i) => (
                  <li key={i}>
                    <strong>{s.action}</strong>
                    {s.description && `: ${s.description}`}
                  </li>
                ))}
              </ul>
            )
          }
        />
      )}
    </div>
  )
}

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

export const ACPSessionCreateModal: React.FC<ACPSessionCreateModalProps> = ({
  open,
  onClose,
  onSuccess,
}) => {
  const { t } = useTranslation(["playground", "common"])
  const [form] = Form.useForm<FormValues>()
  const { config: connectionConfig, loading: isConnectionConfigLoading } = useCanonicalConnectionConfig()

  // Server config from storage
  const workspaceId = useWorkspaceStore((state) => state.workspaceId)

  // Local state
  const [mcpServers, setMcpServers] = useState<ACPMCPServerConfig[]>([])
  const [creationStep, setCreationStep] = useState<ACPSessionCreationStep>("idle")
  const [creationError, setCreationError] = useState<ACPStructuredError | undefined>()

  // Store
  const createSession = useACPSessionsStore((s) => s.createSession)
  const replaceSessionId = useACPSessionsStore((s) => s.replaceSessionId)
  const closeSession = useACPSessionsStore((s) => s.closeSession)

  // Create REST client
  const restClient = React.useMemo(() => {
    return connectionConfig
      ? new ACPRestClient(buildACPClientConfig(connectionConfig))
      : null
  }, [connectionConfig])

  // Fetch available agents
  const {
    data: agentsData,
    isLoading: isLoadingAgents,
  } = useQuery({
    queryKey: ["acp", "agents"],
    queryFn: () => restClient!.getAvailableAgents(),
    enabled: open && !!restClient,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  })

  const agents = agentsData?.agents ?? []
  const defaultAgent = agentsData?.default_agent ?? "claude_code"

  // Set default agent type when loaded
  useEffect(() => {
    if (open && agents.length > 0) {
      const configuredAgent = agents.find((a) => a.is_configured)
      form.setFieldValue("agentType", configuredAgent?.type ?? defaultAgent)
    }
  }, [open, agents, defaultAgent, form])

  // Reset form when modal opens
  useEffect(() => {
    if (open) {
      form.resetFields()
      setMcpServers([])
      setCreationStep("idle")
      setCreationError(undefined)
    }
  }, [open, form])

  // Create session mutation
  const createMutation = useMutation({
    mutationFn: async (values: FormValues) => {
      if (!restClient) {
        throw new Error("Missing ACP connection configuration")
      }
      setCreationStep("creating")
      setCreationError(undefined)

      // Create local session entry
      const options: CreateSessionOptions = {
        cwd: values.cwd.trim(),
        name: values.name.trim() || undefined,
        agentType: values.agentType,
        tags: values.tags
          ? values.tags.split(",").map((t) => t.trim()).filter(Boolean)
          : undefined,
        mcpServers: mcpServers.length > 0 ? mcpServers : undefined,
        workspaceId: workspaceId || undefined,
      }

      const localSessionId = createSession(options)

      try {
        setCreationStep("starting_agent")

        // Call backend to create the actual session
        const response = await restClient.createSession({
          cwd: options.cwd,
          name: options.name,
          agent_type: options.agentType,
          tags: options.tags,
          mcp_servers: options.mcpServers,
          workspace_id: options.workspaceId,
        })

        setCreationStep("connecting")

        // Small delay to show connecting step
        await new Promise((resolve) => setTimeout(resolve, 500))

        setCreationStep("ready")

        return {
          localSessionId,
          serverSessionId: response.session_id,
          name: response.name,
          metadata: {
            sandboxSessionId: response.sandbox_session_id ?? null,
            sandboxRunId: response.sandbox_run_id ?? null,
            sshWsUrl: response.ssh_ws_url ?? null,
            sshUser: response.ssh_user ?? null,
            personaId: response.persona_id ?? null,
            workspaceId: response.workspace_id ?? options.workspaceId ?? null,
            workspaceGroupId: response.workspace_group_id ?? null,
            scopeSnapshotId: response.scope_snapshot_id ?? null,
          },
        }
      } catch (error) {
        // Clean up local session on failure
        closeSession(localSessionId)
        throw error
      }
    },
    onSuccess: (result) => {
      replaceSessionId(result.localSessionId, result.serverSessionId, {
        name: result.name,
        ...result.metadata,
      })
      notification.success({
        message: t("acp.create.success", "Session created"),
        description: t("acp.create.successDesc", {
          name: result.name,
          defaultValue: `"${result.name}" is ready to use.`,
        }),
      })
      onSuccess?.(result.serverSessionId)
      onClose()
    },
    onError: (
      error: Error &
        Partial<ACPStructuredError> & {
          data?: Partial<ACPStructuredError>
        }
    ) => {
      setCreationStep("error")

      // Read structured ACP errors from top level first, then fall back to error.data
      const errorCode = error.code ?? error.data?.code
      const errorMessage = error.message ?? error.data?.message
      const errorSuggestions = error.suggestions ?? error.data?.suggestions

      // Prefer structured error from backend if available
      if (errorCode && errorMessage) {
        setCreationError({
          code: errorCode,
          message: errorMessage,
          suggestions: errorSuggestions ?? [],
        })
        notification.error({
          message: t("common:error", "Error"),
          description: errorMessage,
        })
        return
      }

      // Fall back to client-side string matching
      const structuredError: ACPStructuredError = {
        code: "creation_failed",
        message: error.message || t("acp.create.error", "Failed to create session"),
        suggestions: [],
      }

      // Add suggestions based on error message
      if (error.message.includes("not found") || error.message.includes("ENOENT")) {
        structuredError.suggestions.push({
          action: t("acp.create.suggestion.checkPath", "Check the path"),
          description: t(
            "acp.create.suggestion.checkPathDesc",
            "Make sure the working directory exists and is accessible."
          ),
        })
      }
      if (error.message.includes("permission") || error.message.includes("EACCES")) {
        structuredError.suggestions.push({
          action: t("acp.create.suggestion.checkPermissions", "Check permissions"),
          description: t(
            "acp.create.suggestion.checkPermissionsDesc",
            "Ensure the server has read/write access to the directory."
          ),
        })
      }
      if (error.message.includes("API key") || error.message.includes("unauthorized")) {
        structuredError.suggestions.push({
          action: t("acp.create.suggestion.checkApiKey", "Configure API key"),
          description: t(
            "acp.create.suggestion.checkApiKeyDesc",
            "Set the required API key in your environment variables."
          ),
        })
      }

      // Add generic suggestion if none matched
      if (structuredError.suggestions.length === 0) {
        structuredError.suggestions.push({
          action: t("acp.create.suggestion.retry", "Try again"),
          description: t(
            "acp.create.suggestion.retryDesc",
            "If the problem persists, check the server logs for details."
          ),
        })
      }

      setCreationError(structuredError)

      notification.error({
        message: t("common:error", "Error"),
        description: structuredError.message,
      })
    },
  })

  const handleSubmit = (values: FormValues) => {
    createMutation.mutate(values)
  }

  const isPending = createMutation.isPending
  const isInProgress = creationStep !== "idle" && creationStep !== "error"
  const isAgentSelectionLoading = isConnectionConfigLoading || isLoadingAgents

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={t("acp.create.title", "Create ACP Session")}
      footer={null}
      width={560}
      destroyOnHidden
      maskClosable={!isPending}
      closable={!isPending}
    >
      <CreationProgress step={creationStep} error={creationError} />

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          name: "",
          cwd: "",
          agentType: defaultAgent,
          tags: "",
        }}
        disabled={isInProgress}
      >
        {/* Working Directory - Required */}
        <Form.Item
          name="cwd"
          label={
            <span className="flex items-center gap-2">
              <Folder className="h-4 w-4" />
              {t("acp.create.workingDirectory", "Working Directory")}
            </span>
          }
          rules={[
            {
              required: true,
              message: t("acp.create.cwdRequired", "Please enter a working directory"),
            },
            {
              pattern: /^\/|^[A-Z]:\\/i,
              message: t("acp.create.cwdAbsolute", "Please enter an absolute path"),
            },
          ]}
          extra={t(
            "acp.create.cwdHelp",
            "Absolute path on the tldw_server machine where the agent will read and write files."
          )}
        >
          <Input
            placeholder="/path/to/your/project"
            autoFocus
          />
        </Form.Item>

        {/* Session Name - Optional */}
        <Form.Item
          name="name"
          label={t("acp.create.sessionName", "Session Name")}
          extra={t(
            "acp.create.nameHelp",
            "Optional. Auto-generated from directory name if not provided."
          )}
        >
          <Input placeholder={t("acp.create.namePlaceholder", "My Project Session")} />
        </Form.Item>

        {/* Agent Type Selection */}
        <Form.Item
          name="agentType"
          label={
            <span className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              {t("acp.create.agentType", "Agent Type")}
            </span>
          }
        >
          {isAgentSelectionLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-5 w-5 animate-spin text-text-muted" />
            </div>
          ) : (
            <Radio.Group className="w-full">
              <div className="flex flex-col gap-2">
                {agents.map((agent) => (
                  <AgentCard
                    key={agent.type}
                    agent={agent}
                    selected={form.getFieldValue("agentType") === agent.type}
                    onSelect={() => form.setFieldValue("agentType", agent.type)}
                  />
                ))}
              </div>
            </Radio.Group>
          )}
        </Form.Item>

        {/* Advanced Section - Collapsible */}
        <Collapse
          ghost
          expandIconPlacement="end"
          expandIcon={({ isActive }) => (
            <ChevronDown
              className={`h-4 w-4 transition-transform ${isActive ? "rotate-180" : ""}`}
            />
          )}
          items={[
            {
              key: "advanced",
              label: (
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Server className="h-4 w-4" />
                  {t("acp.create.advanced", "Advanced Options")}
                </span>
              ),
              children: (
                <div className="space-y-4">
                  {/* Tags */}
                  <Form.Item
                    name="tags"
                    label={t("acp.create.tags", "Tags")}
                    extra={t("acp.create.tagsHelp", "Comma-separated tags for organizing sessions.")}
                  >
                    <Input placeholder="project, frontend, experiment" />
                  </Form.Item>

                  {/* MCP Servers */}
                  <div>
                    <label className="mb-2 block text-sm font-medium text-text">
                      {t("acp.create.mcpServers", "MCP Servers")}
                    </label>
                    <MCPServerConfigList
                      servers={mcpServers}
                      onChange={setMcpServers}
                      disabled={isInProgress}
                    />
                  </div>
                </div>
              ),
            },
          ]}
        />

        {/* Actions */}
        <div className="mt-6 flex justify-end gap-2">
          <Button
            type="secondary"
            onClick={onClose}
            disabled={isPending}
          >
            {t("common:cancel", "Cancel")}
          </Button>
          <Button
            type="primary"
            htmlType="submit"
            loading={isPending}
            disabled={isAgentSelectionLoading || !restClient}
          >
            {isPending ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {t("acp.create.creating", "Creating...")}
              </span>
            ) : (
              t("acp.create.submit", "Create Session")
            )}
          </Button>
        </div>
      </Form>
    </Modal>
  )
}

export default ACPSessionCreateModal
