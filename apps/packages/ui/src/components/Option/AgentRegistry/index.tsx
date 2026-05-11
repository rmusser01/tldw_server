import React, { useEffect, useState, useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { Alert, Button, Card, Spin, Tag, Tooltip, Empty, Badge } from "antd"
import {
  Bot,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Play,
  Settings,
  Heart,
} from "lucide-react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { ACPRestClient } from "@/services/acp/client"
import { buildACPAuthHeaders, buildACPClientConfig } from "@/services/acp/connection"
import { normalizeACPHealthStatus, type ACPHealthStatus } from "@/services/acp/readiness"
import type { ACPSupportState, ACPVerificationLevel } from "@/services/acp/types"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"

type AgentEntry = {
  type: string
  name: string
  description: string
  status: "available" | "unavailable" | "requires_setup"
  reason?: string
  is_default?: boolean
  support_state: ACPSupportState
  verification_level: ACPVerificationLevel
  compatibility_notes?: string
  compatibility_docs_url?: string | null
}

const COMPATIBILITY_COLOR: Record<ACPSupportState, "success" | "warning" | "error" | "processing" | "default"> = {
  supported: "success",
  supported_with_caveats: "processing",
  experimental: "warning",
  documented_unverified: "warning",
  unsupported: "error"
}

export const AgentRegistryPage: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const { config: connectionConfig } = useCanonicalConnectionConfig()

  const [agents, setAgents] = useState<AgentEntry[]>([])
  const [health, setHealth] = useState<ACPHealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [healthLoading, setHealthLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const restClient = useMemo(
    () =>
      connectionConfig
        ? new ACPRestClient(buildACPClientConfig(connectionConfig))
        : null,
    [connectionConfig]
  )

  const fetchAgents = useCallback(async () => {
    if (!restClient) return
    setLoading(true)
    setError(null)
    try {
      const response = await restClient.getAvailableAgents()
      setAgents(
        (response.agents ?? []).map((agent) => ({
          type: agent.type,
          name: agent.name,
          description: agent.description,
          status: agent.is_configured ? "available" : "requires_setup",
          support_state: agent.support_state ?? "documented_unverified",
          verification_level: agent.verification_level ?? "documented_only",
          compatibility_notes: agent.compatibility_notes,
          compatibility_docs_url: agent.compatibility_docs_url
        }))
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load agents")
    } finally {
      setLoading(false)
    }
  }, [restClient])

  const fetchHealth = useCallback(async () => {
    if (!connectionConfig) return
    setHealthLoading(true)
    try {
      const transport = resolveBrowserRequestTransport({
        config: connectionConfig,
        path: "/api/v1/acp/health"
      })
      const headers =
        transport.mode === "hosted"
          ? { "Content-Type": "application/json" }
          : buildACPAuthHeaders(connectionConfig, { includeContentType: true })
      const res = await fetch(transport.url, { headers })
      if (res.ok) {
        setHealth(normalizeACPHealthStatus(await res.json()))
      } else {
        setHealth(null)
      }
    } catch {
      // Health check failure is not critical
      setHealth(null)
    } finally {
      setHealthLoading(false)
    }
  }, [connectionConfig])

  useEffect(() => {
    if (!connectionConfig) return
    void fetchAgents()
    void fetchHealth()
  }, [connectionConfig, fetchAgents, fetchHealth])

  const statusIcon = (status: string) => {
    switch (status) {
      case "available":
      case "ok":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "unavailable":
      case "missing":
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    }
  }

  const statusColor = (status: string): "success" | "error" | "warning" => {
    switch (status) {
      case "available":
      case "ok":
        return "success"
      case "unavailable":
      case "missing":
      case "error":
        return "error"
      default:
        return "warning"
    }
  }

  return (
    <div className="space-y-6">
      {/* Health Status */}
      <Card
        title={
          <span className="flex items-center gap-2">
            <Heart className="h-4 w-4" />
            ACP System Health
          </span>
        }
        extra={
          <Button
            size="small"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            onClick={() => {
              void fetchHealth()
              void fetchAgents()
            }}
          >
            Refresh
          </Button>
        }
      >
        {healthLoading ? (
          <div className="flex justify-center py-4">
            <Spin size="small" />
          </div>
        ) : health ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="flex items-center gap-2 rounded-lg border border-border p-3">
              {statusIcon(health.runner)}
              <div>
                <div className="text-xs text-muted-foreground">Runner Binary</div>
                <Tag color={statusColor(health.runner)}>{health.runner}</Tag>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border p-3">
              {statusIcon(health.agent)}
              <div>
                <div className="text-xs text-muted-foreground">Agent Status</div>
                <Tag color={statusColor(health.agent)}>{health.agent}</Tag>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border p-3">
              {statusIcon(health.api_keys)}
              <div>
                <div className="text-xs text-muted-foreground">API Keys</div>
                <Tag color={statusColor(health.api_keys)}>{health.api_keys}</Tag>
              </div>
            </div>
          </div>
        ) : (
          <Alert
            type="warning"
            message="Health check unavailable"
            description="Could not reach the ACP health endpoint. Ensure the server is running."
            showIcon
          />
        )}
        {health?.details && (
          <div className="mt-2 text-xs text-muted-foreground">{health.details}</div>
        )}
      </Card>

      {/* Error */}
      {error && (
        <Alert type="error" message={error} closable onClose={() => setError(null)} />
      )}

      {/* Agent List */}
      <Card
        title={
          <span className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Registered Agents
            {!loading && (
              <Badge
                count={agents.length}
                style={{ backgroundColor: "var(--primary)" }}
              />
            )}
          </span>
        }
      >
        {loading ? (
          <div className="flex justify-center py-8">
            <Spin />
          </div>
        ) : agents.length === 0 ? (
          <Empty description="No agents registered" />
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {agents.map((agent) => (
              <AgentCard key={agent.type} agent={agent} />
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

const AgentCard: React.FC<{ agent: AgentEntry }> = ({ agent }) => {
  const statusColor =
    agent.status === "available"
      ? "success"
      : agent.status === "requires_setup"
        ? "warning"
        : "error"

  const statusLabel =
    agent.status === "available"
      ? "Ready"
      : agent.status === "requires_setup"
        ? "Setup Required"
        : "Unavailable"
  const showUnverifiedWarning =
    agent.status === "available" && agent.support_state === "documented_unverified"

  return (
    <div className="rounded-lg border border-border p-4 transition-shadow hover:shadow-md">
      <div className="mb-2 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          <h3 className="font-medium">{agent.name}</h3>
        </div>
        <div className="flex items-center gap-1">
          {agent.is_default && (
            <Tag color="blue" className="text-xs">
              Default
            </Tag>
          )}
          <Tag color={statusColor}>{statusLabel}</Tag>
        </div>
      </div>

      <p className="mb-3 text-sm text-muted-foreground">
        {agent.description || `Agent type: ${agent.type}`}
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <Tag color={COMPATIBILITY_COLOR[agent.support_state]}>
          {agent.support_state}
        </Tag>
        <Tag>{agent.verification_level}</Tag>
        {agent.compatibility_docs_url && (
          <a
            className="text-primary hover:underline"
            href={agent.compatibility_docs_url}
            target="_blank"
            rel="noreferrer"
          >
            Compatibility matrix
          </a>
        )}
      </div>

      {agent.compatibility_notes && (
        <div className="mb-3 rounded bg-muted px-2 py-1.5 text-xs text-muted-foreground">
          {agent.compatibility_notes}
        </div>
      )}

      {showUnverifiedWarning && (
        <div className="mb-3 rounded bg-yellow-50 p-2 text-xs text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400">
          Configured but compatibility is documented_unverified. Run the ACP certification checklist before release claims.
        </div>
      )}

      {agent.reason && (
        <div className="mb-3 rounded bg-yellow-50 p-2 text-xs text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400">
          {agent.reason}
        </div>
      )}

      <div className="flex items-center gap-2">
        <Tooltip title={agent.status === "available" ? "Launch session" : statusLabel}>
          <Button
            size="small"
            type="primary"
            icon={<Play className="h-3 w-3" />}
            disabled={agent.status !== "available"}
            onClick={() => {
              // Navigate to ACP playground with this agent pre-selected
              window.location.hash = `/acp-playground?agent=${agent.type}`
            }}
          >
            Launch
          </Button>
        </Tooltip>
        <span className="text-xs text-muted-foreground">{agent.type}</span>
      </div>
    </div>
  )
}

export default AgentRegistryPage
