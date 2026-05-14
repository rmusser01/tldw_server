import React, { useEffect, useState, useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { Button, Card, Spin, Tag, Tooltip, Empty, Badge } from "antd"
import {
  Bot,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Play,
  Settings,
  Heart,
  Activity,
} from "lucide-react"
import { Alert as DSAlert, Badge as DSBadge } from "@/components/ui/primitives"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { ACPRestClient } from "@/services/acp/client"
import { buildACPAuthHeaders, buildACPClientConfig } from "@/services/acp/connection"
import { normalizeACPHealthStatus, type ACPHealthStatus } from "@/services/acp/readiness"
import type {
  ACPExecutionHealthFailureBuckets,
  ACPExecutionHealthSetupSummary,
  ACPExecutionHealthSummaryResponse,
  ACPSupportState,
  ACPVerificationLevel
} from "@/services/acp/types"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { DESIGN_SYSTEM_STATES, getDesignSystemState, type DesignSystemStateKey } from "@/design-system"

const ACP_EXECUTION_HEALTH_SUMMARY_PATH =
  "/api/v1/admin/acp/execution-health/summary?range_days=30"

const FAILURE_BUCKET_LABELS: Array<[keyof ACPExecutionHealthFailureBuckets, string]> = [
  ["setup_blockers", "Setup blockers"],
  ["runner_session_failures", "Runner/session failures"],
  ["reviewer_rejections", "Reviewer rejections"],
  ["reviewer_failures", "Reviewer failures"],
  ["governance_denials", "Governance denials"],
  ["structured_completion_failures", "Structured completion failures"],
  ["sandbox_runtime_errors", "Sandbox runtime errors"],
  ["retention_redaction_actions", "Retention/redaction actions"]
]

const SETUP_DIMENSION_LABELS: Record<keyof ACPExecutionHealthSetupSummary, string> = {
  agent: "Agent",
  workspace: "Workspace",
  sandbox_runtime: "Sandbox runtime",
  mcp_injection: "MCP injection",
  scheduler_trigger_path: "Scheduler trigger path"
}
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
  const [executionHealth, setExecutionHealth] =
    useState<ACPExecutionHealthSummaryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [healthLoading, setHealthLoading] = useState(true)
  const [executionHealthLoading, setExecutionHealthLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const restClient = useMemo(
    () =>
      connectionConfig
        ? new ACPRestClient(buildACPClientConfig(connectionConfig))
        : null,
    [connectionConfig]
  )

  const getACPHeaders = useCallback(
    (transport: { mode: string }) =>
      transport.mode === "hosted"
        ? { "Content-Type": "application/json" }
        : buildACPAuthHeaders(connectionConfig, { includeContentType: true }),
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
      const res = await fetch(transport.url, { headers: getACPHeaders(transport) })
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
  }, [connectionConfig, getACPHeaders])

  const fetchExecutionHealth = useCallback(async () => {
    if (!connectionConfig) return
    setExecutionHealthLoading(true)
    try {
      const transport = resolveBrowserRequestTransport({
        config: connectionConfig,
        path: ACP_EXECUTION_HEALTH_SUMMARY_PATH
      })
      const res = await fetch(transport.url, { headers: getACPHeaders(transport) })
      if (res.ok) {
        setExecutionHealth(await res.json())
      } else {
        setExecutionHealth(null)
      }
    } catch {
      // The admin summary can be permission-gated; keep the registry usable.
      setExecutionHealth(null)
    } finally {
      setExecutionHealthLoading(false)
    }
  }, [connectionConfig, getACPHeaders])

  useEffect(() => {
    if (!connectionConfig) return
    void fetchAgents()
    void fetchHealth()
    void fetchExecutionHealth()
  }, [connectionConfig, fetchAgents, fetchHealth, fetchExecutionHealth])

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
              void fetchExecutionHealth()
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
          <DSAlert variant="warning" title="Health check unavailable">
            Could not reach the ACP health endpoint. Ensure the server is running.
          </DSAlert>
        )}
        {health?.details && (
          <div className="mt-2 text-xs text-muted-foreground">{health.details}</div>
        )}
      </Card>

      <Card
        title={
          <span className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            ACP Execution Health
          </span>
        }
      >
        {executionHealthLoading ? (
          <div className="flex justify-center py-4">
            <Spin size="small" />
          </div>
        ) : executionHealth ? (
          <ExecutionHealthSummary summary={executionHealth} />
        ) : (
          <DSAlert variant="warning" title="Execution health summary unavailable">
            The admin summary endpoint may require newer backend support or elevated permissions.
          </DSAlert>
        )}
      </Card>

      {/* Error */}
      {error && (
        <DSAlert variant="error" title={error} dismissible onDismiss={() => setError(null)}>
          Agent registry could not load.
        </DSAlert>
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

const ExecutionHealthSummary: React.FC<{
  summary: ACPExecutionHealthSummaryResponse
}> = ({ summary }) => {
  const sessionStatusEntries = Object.entries(summary.sessions.by_status ?? {})
    .filter(([, count]) => count > 0)
    .sort(([left], [right]) => left.localeCompare(right))

  const failureEntries = FAILURE_BUCKET_LABELS
    .map(([key, label]) => ({
      key,
      label,
      count: summary.failure_buckets[key] ?? 0
    }))
    .filter((entry) => entry.count > 0)

  const setupEntries = (Object.entries(summary.setup_health) as Array<
    [
      keyof ACPExecutionHealthSetupSummary,
      ACPExecutionHealthSetupSummary[keyof ACPExecutionHealthSetupSummary]
    ]
  >)
    .filter(([, dimension]) => {
      return (
        dimension.status === "blocked" ||
        dimension.status === "degraded" ||
        dimension.blockers.length > 0
      )
    })
    .map(([key, dimension]) => ({
      key,
      label: SETUP_DIMENSION_LABELS[key],
      status: dimension.status,
      blockers: dimension.blockers,
      evidenceCount: dimension.evidence_count
    }))

  const unverifiedAgents = summary.compatibility.documented_unverified_agents ?? []
  const redactionEnabled =
    summary.redaction.detail_events_artifacts_redacted_views &&
    summary.redaction.diagnostics_sanitized &&
    summary.redaction.audit_metadata_sanitized

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">Sessions</div>
          <div className="text-base font-medium">
            {summary.sessions.total} sessions in {summary.range_days}d
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {sessionStatusEntries.length > 0 ? (
              sessionStatusEntries.map(([status, count]) => (
                <DSBadge key={status} variant="secondary">{count} {status}</DSBadge>
              ))
            ) : (
              <span className="text-xs text-muted-foreground">No sessions recorded</span>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">Compatibility</div>
          <div className="mt-1 flex flex-wrap gap-1">
            {summary.compatibility.live_certification_required ? (
              <DSBadge variant="warning">Live certification required</DSBadge>
            ) : (
              <DSBadge variant="success">No live-certification blocker</DSBadge>
            )}
            {Object.entries(summary.compatibility.by_support_state ?? {}).map(
              ([state, count]) => (
                <DSBadge key={state} variant="secondary">{count} {state}</DSBadge>
              )
            )}
          </div>
          {unverifiedAgents.length > 0 ? (
            <div className="mt-2 text-xs text-yellow-700 dark:text-yellow-400">
              Unverified agents: {unverifiedAgents.join(", ")}
            </div>
          ) : (
            <div className="mt-2 text-xs text-muted-foreground">
              No documented-unverified agents
            </div>
          )}
          {summary.compatibility.docs_url && (
            <a
              className="mt-2 inline-block text-xs text-primary hover:underline"
              href={summary.compatibility.docs_url}
              target="_blank"
              rel="noreferrer"
            >
              Execution evidence docs
            </a>
          )}
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">Retention and redaction</div>
          <div className="mt-1 text-sm">
            Retention {summary.retention.session_retention_days}d sessions /{" "}
            {summary.retention.audit_retention_days}d audit
          </div>
          <div className="mt-2 text-xs text-muted-foreground">
            {redactionEnabled
              ? "Redacted drill-through enabled"
              : "Review redaction settings"}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-border p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            Failure buckets
          </div>
          {failureEntries.length > 0 ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {failureEntries.map((entry) => (
                <div
                  key={entry.key}
                  className="flex items-center justify-between rounded border border-border px-2 py-1.5 text-sm"
                >
                  <span>{entry.label}</span>
                  <DSBadge variant="warning">{entry.count}</DSBadge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No recent failure buckets</div>
          )}
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            Setup health
          </div>
          {setupEntries.length > 0 ? (
            <div className="space-y-2">
              {setupEntries.map((entry) => (
                <div key={entry.key} className="rounded border border-border px-2 py-1.5 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <DSBadge variant={entry.status === "blocked" ? "danger" : "warning"}>
                      {entry.label} {entry.status}
                    </DSBadge>
                    <span className="text-xs text-muted-foreground">
                      {entry.evidenceCount} evidence
                    </span>
                  </div>
                  {entry.blockers.length > 0 && (
                    <div className="mt-1 text-xs text-muted-foreground">
                      {entry.label} {entry.status}: {entry.blockers.join(", ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              No setup blockers in this window
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const AGENT_STATUS_STATE: Record<AgentEntry["status"], DesignSystemStateKey> = {
  available: "ready",
  requires_setup: "setup_required",
  unavailable: "unavailable"
}

const AGENT_STATUS_LABELS = Object.fromEntries(
  Object.entries(AGENT_STATUS_STATE).map(([status, stateKey]) => [
    status,
    (getDesignSystemState(stateKey) ?? DESIGN_SYSTEM_STATES[stateKey]).label
  ])
) as Record<AgentEntry["status"], string>

const AgentCard: React.FC<{ agent: AgentEntry }> = ({ agent }) => {
  const statusColor =
    agent.status === "available"
      ? "success"
      : agent.status === "requires_setup"
        ? "warning"
        : "error"

  const statusLabel = AGENT_STATUS_LABELS[agent.status]
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
