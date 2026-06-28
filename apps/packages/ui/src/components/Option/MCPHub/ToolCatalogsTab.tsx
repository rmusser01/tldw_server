import { useCallback, useEffect, useMemo, useState } from "react"
import { Card, Empty, Modal, Space, Tag, Typography } from "antd"
import { useQueryClient } from "@tanstack/react-query"
import { StatePanel } from "@/components/ui/state"

import {
  getMcpHubReadiness,
  getToolRegistrySummary,
  listExternalServers,
  refreshExternalServerDiscovery,
  type McpHubExternalServer,
  type McpHubReadiness,
  type McpHubServerReadiness,
  type McpHubToolRegistryEntry,
  type McpHubToolRegistryModule
} from "@/services/tldw/mcp-hub"

import { getPathScopeLabel, getToolEntriesByModule } from "./policyHelpers"
import { invalidateMcpRuntimeQueries } from "./runtimeRefresh"

type ToolCatalogsTabProps = {
  onOpenServerSetup?: () => void
  onOpenServerCredentials?: (serverId: string) => void
  onOpenServerConfig?: (serverId: string) => void
}

export const ToolCatalogsTab = ({
  onOpenServerSetup,
  onOpenServerCredentials,
  onOpenServerConfig
}: ToolCatalogsTabProps) => {
  const queryClient = useQueryClient()
  const [entries, setEntries] = useState<McpHubToolRegistryEntry[]>([])
  const [modules, setModules] = useState<McpHubToolRegistryModule[]>([])
  const [servers, setServers] = useState<McpHubExternalServer[]>([])
  const [readiness, setReadiness] = useState<McpHubReadiness | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshingServerId, setRefreshingServerId] = useState<string | null>(null)
  const [detailsReadiness, setDetailsReadiness] = useState<McpHubServerReadiness | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [metadataWarningMessage, setMetadataWarningMessage] = useState<string | null>(null)
  const groupedModules = useMemo(() => getToolEntriesByModule(entries, modules), [entries, modules])
  const readinessServers = Array.isArray(readiness?.servers) ? readiness.servers : []
  const visibleServerCount = Math.max(readinessServers.length, servers.length)
  const hasNoExternalServers = !metadataWarningMessage && visibleServerCount === 0
  const recoveryReadiness =
    readinessServers.find((server) => server.display_state !== "ready") ??
    readinessServers[0] ??
    null
  const recoveryServer = recoveryReadiness
    ? servers.find((server) => server.id === recoveryReadiness.server_id) ?? null
    : null
  const catalogWarningMessage =
    groupedModules.length > 0 && readiness?.reason_codes?.includes("partial_capability")
      ? readiness.message
      : null

  const loadCatalog = useCallback(
    async ({
      isCancelled = () => false,
      suppressLoading = false
    }: {
      isCancelled?: () => boolean
      suppressLoading?: boolean
    } = {}) => {
      if (!suppressLoading) {
        setLoading(true)
      }
      setErrorMessage(null)
      setMetadataWarningMessage(null)
      try {
        const [summaryResult, rowsResult, readinessResult] = await Promise.allSettled([
          getToolRegistrySummary(),
          listExternalServers(),
          getMcpHubReadiness()
        ])
        if (summaryResult.status === "rejected") {
          throw summaryResult.reason
        }
        if (!isCancelled()) {
          const summary = summaryResult.value
          const metadataWarnings: string[] = []

          setEntries(Array.isArray(summary?.entries) ? summary.entries : [])
          setModules(Array.isArray(summary?.modules) ? summary.modules : [])

          if (rowsResult.status === "fulfilled") {
            setServers(Array.isArray(rowsResult.value) ? rowsResult.value : [])
          } else {
            setServers([])
            metadataWarnings.push("server inventory")
          }

          if (readinessResult.status === "fulfilled") {
            setReadiness(readinessResult.value)
          } else {
            setReadiness(null)
            metadataWarnings.push("readiness")
          }

          setMetadataWarningMessage(
            metadataWarnings.length > 0
              ? "Tools are still listed, but catalog recovery details are limited because server inventory or readiness metadata could not be fully loaded."
              : null
          )
        }
      } catch {
        if (!isCancelled()) {
          setEntries([])
          setModules([])
          setServers([])
          setReadiness(null)
          setMetadataWarningMessage(null)
          setErrorMessage("Failed to load tool registry metadata.")
        }
      } finally {
        if (!isCancelled()) {
          setLoading(false)
        }
      }
    },
    []
  )

  useEffect(() => {
    let cancelled = false
    void loadCatalog({
      isCancelled: () => cancelled
    })
    return () => {
      cancelled = true
    }
  }, [loadCatalog])

  const handleRefreshDiscovery = async (serverId: string) => {
    setRefreshingServerId(serverId)
    setErrorMessage(null)
    try {
      await refreshExternalServerDiscovery(serverId)
      await invalidateMcpRuntimeQueries(queryClient)
      await loadCatalog({ suppressLoading: true })
    } catch {
      setErrorMessage("Failed to refresh tool discovery.")
    } finally {
      setRefreshingServerId(null)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Registry-backed tool metadata powers both the catalog view and the guided policy editor.
      </Typography.Text>
      {errorMessage ? (
        <StatePanel state="unavailable" title={errorMessage} role="alert" aria-live="polite" />
      ) : null}

      {groupedModules.length > 0 ? (
        <>
          {metadataWarningMessage ? (
            <StatePanel
              state="degraded"
              title="Catalog recovery details are limited"
              message={metadataWarningMessage}
            />
          ) : catalogWarningMessage ? (
            <StatePanel
              state="degraded"
              title="Some catalog capabilities need review"
              message={catalogWarningMessage}
            />
          ) : null}
          {groupedModules.map((module) => (
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
                  <StatePanel
                    state="degraded"
                    title="Module metadata needs review"
                    message={module.metadata_warnings.join(" ")}
                  />
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
                        <StatePanel
                          state="degraded"
                          title={`Path-scoped profiles fall back to approval for ${tool.display_name}.`}
                          message={`This tool touches local files but is not yet marked as ${getPathScopeLabel("workspace_root")?.toLowerCase()} enforceable.`}
                        />
                      ) : null}
                      {tool.metadata_warnings.length > 0 ? (
                        <StatePanel
                          state="degraded"
                          title="Tool metadata needs review"
                          message={tool.metadata_warnings.join(" ")}
                        />
                      ) : null}
                    </Space>
                  </Card>
                ))}
              </Space>
            </Card>
          ))}
        </>
      ) : (
        <Card loading={loading}>
          {metadataWarningMessage ? (
            <StatePanel
              state="degraded"
              title="Catalog recovery details are limited"
              message={metadataWarningMessage}
            />
          ) : hasNoExternalServers ? (
            <StatePanel
              state="setup_required"
              title="No MCP servers connected"
              message="Add a managed server before the catalog can discover tools."
              primaryAction={{
                label: "Add server",
                onClick: onOpenServerSetup
              }}
            />
          ) : recoveryReadiness?.display_state === "checking" ? (
            <StatePanel
              state="loading"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} discovery is running`}
              message={recoveryReadiness.message}
              secondaryActions={[
                {
                  label: "View details",
                  onClick: () => setDetailsReadiness(recoveryReadiness)
                }
              ]}
            />
          ) : recoveryReadiness?.primary_reason_code === "discovery_not_run" ? (
            <StatePanel
              state="setup_required"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} is saved, but tool discovery has not run`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Refresh discovery",
                loading: refreshingServerId === recoveryReadiness.server_id,
                onClick: () => void handleRefreshDiscovery(recoveryReadiness.server_id)
              }}
            />
          ) : recoveryReadiness?.primary_reason_code === "auth_missing" ? (
            <StatePanel
              state="auth_required"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} needs credentials`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Fix credentials",
                onClick: () => onOpenServerCredentials?.(recoveryReadiness.server_id)
              }}
            />
          ) : recoveryReadiness?.primary_reason_code === "runtime_unavailable" ? (
            <StatePanel
              state="unavailable"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} runtime is unavailable`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Open server config",
                onClick: () => onOpenServerConfig?.(recoveryReadiness.server_id)
              }}
            />
          ) : recoveryReadiness?.primary_reason_code === "preflight_failed" ? (
            <StatePanel
              state="degraded"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} failed preflight validation`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Open server config",
                onClick: () => onOpenServerConfig?.(recoveryReadiness.server_id)
              }}
              secondaryActions={[
                {
                  label: "View details",
                  onClick: () => setDetailsReadiness(recoveryReadiness)
                }
              ]}
            />
          ) : recoveryReadiness?.primary_reason_code === "unreachable" ? (
            <StatePanel
              state="unavailable"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} is unreachable`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Open server config",
                onClick: () => onOpenServerConfig?.(recoveryReadiness.server_id)
              }}
              secondaryActions={[
                {
                  label: "Refresh discovery",
                  loading: refreshingServerId === recoveryReadiness.server_id,
                  onClick: () => void handleRefreshDiscovery(recoveryReadiness.server_id)
                },
                {
                  label: "View details",
                  onClick: () => setDetailsReadiness(recoveryReadiness)
                }
              ]}
            />
          ) : recoveryReadiness?.primary_reason_code === "discovery_failed" ? (
            <StatePanel
              state="degraded"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} discovery failed`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Refresh discovery",
                loading: refreshingServerId === recoveryReadiness.server_id,
                onClick: () => void handleRefreshDiscovery(recoveryReadiness.server_id)
              }}
              secondaryActions={[
                {
                  label: "View details",
                  onClick: () => setDetailsReadiness(recoveryReadiness)
                }
              ]}
            />
          ) : recoveryReadiness?.primary_reason_code === "config_changed" ? (
            <StatePanel
              state="degraded"
              title={`${recoveryServer?.name ?? recoveryReadiness.server_name} catalog is stale`}
              message={recoveryReadiness.message}
              primaryAction={{
                label: "Refresh discovery",
                loading: refreshingServerId === recoveryReadiness.server_id,
                onClick: () => void handleRefreshDiscovery(recoveryReadiness.server_id)
              }}
              secondaryActions={[
                {
                  label: "Open server config",
                  onClick: () => onOpenServerConfig?.(recoveryReadiness.server_id)
                }
              ]}
            />
          ) : (
            <Empty
              description={
                <Space orientation="vertical" size={4}>
                  <Typography.Text type="secondary">No tools registered yet</Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                    Tools are discovered automatically when you connect external MCP servers.
                    Add a server in the <Typography.Text strong>Servers &amp; Credentials</Typography.Text> tab to get started.
                  </Typography.Text>
                </Space>
              }
            />
          )}
        </Card>
      )}
      <Modal
        title={detailsReadiness ? `${detailsReadiness.server_name} details` : "Server details"}
        open={Boolean(detailsReadiness)}
        footer={null}
        onCancel={() => setDetailsReadiness(null)}
      >
        {detailsReadiness ? (
          <Space orientation="vertical" size="small" style={{ width: "100%" }}>
            <Typography.Text>{`Server ID: ${detailsReadiness.server_id}`}</Typography.Text>
            <Typography.Text>{`Reason codes: ${detailsReadiness.reason_codes.join(", ") || "none"}`}</Typography.Text>
            {detailsReadiness.last_error_message ? (
              <Typography.Text>{`Last error: ${detailsReadiness.last_error_message}`}</Typography.Text>
            ) : null}
          </Space>
        ) : null}
      </Modal>
    </Space>
  )
}
