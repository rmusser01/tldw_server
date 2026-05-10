import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Alert, Button, Card, Empty, Space, Tag, Typography } from "antd"
import { useQueryClient } from "@tanstack/react-query"

import {
  describeExternalServerDiscoveryRefreshFailure,
  getToolRegistrySummary,
  listExternalServers,
  refreshExternalServerDiscovery,
  type McpHubExternalServer,
  type McpHubToolRegistryEntry,
  type McpHubToolRegistryModule
} from "@/services/tldw/mcp-hub"
import { useMcpTools } from "@/hooks/useMcpTools"

import { getManagedExternalServers, getPathScopeLabel, getToolEntriesByModule } from "./policyHelpers"
import { invalidateMcpRuntimeQueries } from "./runtimeRefresh"

type ToolCatalogsTabProps = {
  onAddServer?: () => void
}

const describeLoadFailure = (reason: unknown, fallback: string) => {
  if (reason instanceof Error && reason.message.trim()) {
    return reason.message
  }
  if (typeof reason === "string" && reason.trim()) {
    return reason
  }
  return fallback
}

export const ToolCatalogsTab = ({ onAddServer }: ToolCatalogsTabProps = {}) => {
  const queryClient = useQueryClient()
  const latestLoadRequestId = useRef(0)
  const [entries, setEntries] = useState<McpHubToolRegistryEntry[]>([])
  const [modules, setModules] = useState<McpHubToolRegistryModule[]>([])
  const [externalServers, setExternalServers] = useState<McpHubExternalServer[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [serverInventoryError, setServerInventoryError] = useState<string | null>(null)
  const groupedModules = useMemo(() => getToolEntriesByModule(entries, modules), [entries, modules])
  const managedExternalServers = useMemo(
    () => getManagedExternalServers(externalServers),
    [externalServers]
  )
  const mcpTools = useMcpTools()
  const chatEnabledToolCount = Number(mcpTools.toolCounts?.chatEnabled ?? 0)
  const hasExecutableChatTools =
    mcpTools.toolsAvailable === null
      ? null
      : chatEnabledToolCount > 0 || mcpTools.chatTools.length > 0
  const showExecutableAccessGuidance =
    groupedModules.length > 0 && hasExecutableChatTools === false
  const serverInventoryUnavailable = Boolean(serverInventoryError)

  const loadRegistrySummary = useCallback(async (options: { clearOnError?: boolean } = {}) => {
    const requestId = ++latestLoadRequestId.current
    const clearOnError = options.clearOnError !== false
    setLoading(true)
    setErrorMessage(null)
    try {
      const [summaryResult, serverResult] = await Promise.allSettled([
        getToolRegistrySummary(),
        listExternalServers()
      ])
      if (requestId !== latestLoadRequestId.current) return
      if (summaryResult.status === "fulfilled") {
        setEntries(Array.isArray(summaryResult.value?.entries) ? summaryResult.value.entries : [])
        setModules(Array.isArray(summaryResult.value?.modules) ? summaryResult.value.modules : [])
      } else {
        if (clearOnError) {
          setEntries([])
          setModules([])
        }
        setErrorMessage("Failed to load tool registry metadata.")
      }
      if (serverResult.status === "fulfilled") {
        setExternalServers(Array.isArray(serverResult.value) ? serverResult.value : [])
        setServerInventoryError(null)
      } else {
        setServerInventoryError(
          describeLoadFailure(serverResult.reason, "Failed to load managed server inventory.")
        )
      }
    } catch {
      if (requestId !== latestLoadRequestId.current) return
      if (clearOnError) {
        setEntries([])
        setModules([])
      }
      setErrorMessage("Failed to load tool registry metadata.")
    } finally {
      if (requestId === latestLoadRequestId.current) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    void loadRegistrySummary()
  }, [loadRegistrySummary])

  const handleRefreshTools = async () => {
    setRefreshing(true)
    setErrorMessage(null)
    try {
      const refreshResult = await refreshExternalServerDiscovery()
      await invalidateMcpRuntimeQueries(queryClient)
      await loadRegistrySummary({ clearOnError: false })
      if (!refreshResult.ok) {
        setErrorMessage(`Failed to refresh tool discovery: ${describeExternalServerDiscoveryRefreshFailure(refreshResult)}`)
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to refresh tool discovery: ${msg}`)
    } finally {
      setRefreshing(false)
    }
  }
  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Space wrap style={{ width: "100%", justifyContent: "space-between" }}>
        <Typography.Text type="secondary">
          Registry-backed tool metadata powers both the catalog view and the guided policy editor.
        </Typography.Text>
        <Button onClick={handleRefreshTools} loading={refreshing}>
          Refresh Tools
        </Button>
      </Space>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {showExecutableAccessGuidance ? (
        <Alert
          type="warning"
          showIcon
          title="Tools are registered but not executable in chat"
          description="Review profile assignments and disabled tool settings before testing these tools in Chat."
        />
      ) : null}
      {serverInventoryError ? (
        <Alert
          type="warning"
          showIcon
          title="Could not load server inventory"
          description={`${serverInventoryError} Tool Catalog guidance may be incomplete until the server list loads.`}
        />
      ) : null}

      {groupedModules.length > 0 ? (
        groupedModules.map((module) => (
          <Card
            key={module.module}
            title={
              <Space wrap>
                <Typography.Text strong>{module.display_name}</Typography.Text>
                <Tag>{`${module.tool_count} tools`}</Tag>
                {Object.entries(module.risk_summary)
                  .filter(([, count]) => Number(count) > 0)
                  .map(([riskClass, count]) => (
                    <Tag key={`${module.module}-${riskClass}`}>{`${riskClass}:${count}`}</Tag>
                  ))}
              </Space>
            }
            loading={loading}
          >
            <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
              {module.metadata_warnings.length > 0 ? (
                <Alert type="warning" showIcon message={module.metadata_warnings.join(" ")} />
              ) : null}
              {module.tools.map((tool) => (
                <Card key={tool.tool_name} size="small">
                  <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                    <Space wrap>
                      <Typography.Text strong>{tool.display_name}</Typography.Text>
                      <Tag>{tool.category}</Tag>
                      <Tag
                        color={
                          tool.risk_class === "high"
                            ? "red"
                            : tool.risk_class === "medium"
                              ? "gold"
                              : "green"
                        }
                      >
                        {tool.risk_class}
                      </Tag>
                      <Tag>{tool.metadata_source}</Tag>
                      {tool.mutates_state ? <Tag color="volcano">mutates</Tag> : null}
                      {tool.uses_network ? <Tag color="purple">network</Tag> : null}
                      {tool.uses_processes ? <Tag color="magenta">process</Tag> : null}
                      {tool.uses_filesystem && tool.path_boundable ? (
                        <Tag color="cyan">path-enforceable</Tag>
                      ) : null}
                      {tool.uses_filesystem && !tool.path_boundable ? (
                        <Tag color="orange">approval fallback</Tag>
                      ) : null}
                    </Space>
                    <Typography.Text type="secondary">
                      {tool.description || "No description"}
                    </Typography.Text>
                    <Space wrap>
                      {tool.capabilities.map((capability) => (
                        <Tag key={`${tool.tool_name}-${capability}`}>{capability}</Tag>
                      ))}
                      {tool.path_boundable && tool.path_argument_hints.length > 0 ? (
                        <Tag color="cyan">{`hints:${tool.path_argument_hints.join(", ")}`}</Tag>
                      ) : null}
                    </Space>
                    {tool.uses_filesystem && !tool.path_boundable ? (
                      <Alert
                        type="info"
                        showIcon
                        message={`Path-scoped profiles fall back to approval for ${tool.display_name}.`}
                        description={`This tool touches local files but is not yet marked as ${getPathScopeLabel("workspace_root")?.toLowerCase()} enforceable.`}
                      />
                    ) : null}
                    {tool.metadata_warnings.length > 0 ? (
                      <Alert type="warning" showIcon message={tool.metadata_warnings.join(" ")} />
                    ) : null}
                  </Space>
                </Card>
              ))}
            </Space>
          </Card>
        ))
      ) : (
        <Card loading={loading}>
          <Empty
            description={
              <Space orientation="vertical" size={4}>
                <Typography.Text type="secondary">
                  {serverInventoryUnavailable
                    ? "Server inventory unavailable"
                    : managedExternalServers.length > 0
                      ? "No tools discovered yet"
                      : "No managed MCP servers yet"}
                </Typography.Text>
                <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                  {serverInventoryUnavailable ? (
                    "Retry loading server inventory before deciding whether to add a server or refresh discovery."
                  ) : managedExternalServers.length > 0 ? (
                    "Managed servers are configured, but no registry tools are available yet. Refresh discovery after creating or editing a server."
                  ) : (
                    "Add a managed server before looking for tool catalog entries."
                  )}
                </Typography.Text>
              </Space>
            }
          >
            {serverInventoryUnavailable ? (
              <Button
                type="primary"
                onClick={() => void loadRegistrySummary({ clearOnError: false })}
                loading={loading}
              >
                Retry server inventory
              </Button>
            ) : managedExternalServers.length > 0 ? (
              <Button type="primary" onClick={handleRefreshTools} loading={refreshing}>
                Refresh discovery
              </Button>
            ) : (
              <Button type="primary" onClick={onAddServer}>
                Add Server
              </Button>
            )}
          </Empty>
        </Card>
      )}
    </Space>
  )
}
