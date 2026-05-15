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
import {
  normalizeACPExecutionHealthSummary,
  normalizeACPHealthStatus,
  type ACPHealthStatus
} from "@/services/acp/readiness"
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

const FAILURE_BUCKET_LABELS: Array<{
  key: keyof ACPExecutionHealthFailureBuckets
  labelKey: string
  fallback: string
}> = [
  {
    key: "setup_blockers",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.setupBlockers",
    fallback: "Setup blockers"
  },
  {
    key: "runner_session_failures",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.runnerSessionFailures",
    fallback: "Runner/session failures"
  },
  {
    key: "reviewer_rejections",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.reviewerRejections",
    fallback: "Reviewer rejections"
  },
  {
    key: "reviewer_failures",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.reviewerFailures",
    fallback: "Reviewer failures"
  },
  {
    key: "governance_denials",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.governanceDenials",
    fallback: "Governance denials"
  },
  {
    key: "structured_completion_failures",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.structuredCompletionFailures",
    fallback: "Structured completion failures"
  },
  {
    key: "sandbox_runtime_errors",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.sandboxRuntimeErrors",
    fallback: "Sandbox runtime errors"
  },
  {
    key: "retention_redaction_actions",
    labelKey: "option:agentRegistry.executionHealth.failureBuckets.retentionRedactionActions",
    fallback: "Retention/redaction actions"
  }
]

const SETUP_DIMENSION_LABELS: Record<
  keyof ACPExecutionHealthSetupSummary,
  { labelKey: string; fallback: string }
> = {
  agent: {
    labelKey: "option:agentRegistry.executionHealth.setupDimensions.agent",
    fallback: "Agent"
  },
  workspace: {
    labelKey: "option:agentRegistry.executionHealth.setupDimensions.workspace",
    fallback: "Workspace"
  },
  sandbox_runtime: {
    labelKey: "option:agentRegistry.executionHealth.setupDimensions.sandboxRuntime",
    fallback: "Sandbox runtime"
  },
  mcp_injection: {
    labelKey: "option:agentRegistry.executionHealth.setupDimensions.mcpInjection",
    fallback: "MCP injection"
  },
  scheduler_trigger_path: {
    labelKey: "option:agentRegistry.executionHealth.setupDimensions.schedulerTriggerPath",
    fallback: "Scheduler trigger path"
  }
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
        setExecutionHealth(normalizeACPExecutionHealthSummary(await res.json()))
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
            {t("option:agentRegistry.health.title", "ACP System Health")}
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
            {t("common:refresh", "Refresh")}
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
                <div className="text-xs text-muted-foreground">
                  {t("option:agentRegistry.health.runnerBinary", "Runner Binary")}
                </div>
                <Tag color={statusColor(health.runner)}>{health.runner}</Tag>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border p-3">
              {statusIcon(health.agent)}
              <div>
                <div className="text-xs text-muted-foreground">
                  {t("option:agentRegistry.health.agentStatus", "Agent Status")}
                </div>
                <Tag color={statusColor(health.agent)}>{health.agent}</Tag>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border p-3">
              {statusIcon(health.api_keys)}
              <div>
                <div className="text-xs text-muted-foreground">
                  {t("option:agentRegistry.health.apiKeys", "API Keys")}
                </div>
                <Tag color={statusColor(health.api_keys)}>{health.api_keys}</Tag>
              </div>
            </div>
          </div>
        ) : (
          <DSAlert
            variant="warning"
            title={t(
              "option:agentRegistry.health.unavailableTitle",
              "Health check unavailable"
            )}
          >
            {t(
              "option:agentRegistry.health.unavailableBody",
              "Could not reach the ACP health endpoint. Ensure the server is running."
            )}
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
            {t("option:agentRegistry.executionHealth.title", "ACP Execution Health")}
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
          <DSAlert
            variant="warning"
            title={t(
              "option:agentRegistry.executionHealth.unavailableTitle",
              "Execution health summary unavailable"
            )}
          >
            {t(
              "option:agentRegistry.executionHealth.unavailableBody",
              "The admin summary endpoint may require newer backend support or elevated permissions."
            )}
          </DSAlert>
        )}
      </Card>

      {/* Error */}
      {error && (
        <DSAlert
          variant="error"
          title={t("option:agentRegistry.loadFailedTitle", "Agent registry could not load")}
          dismissible
          onDismiss={() => setError(null)}
        >
          {error}
        </DSAlert>
      )}

      {/* Agent List */}
      <Card
        title={
          <span className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            {t("option:agentRegistry.registeredAgents", "Registered Agents")}
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
          <Empty
            description={t("option:agentRegistry.empty", "No agents registered")}
          />
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
  const { t } = useTranslation(["option", "common"])
  const sessionStatusEntries = useMemo(
    () =>
      Object.entries(summary.sessions.by_status ?? {})
        .filter(([, count]) => count > 0)
        .sort(([left], [right]) => left.localeCompare(right)),
    [summary.sessions.by_status]
  )

  const failureEntries = useMemo(
    () =>
      FAILURE_BUCKET_LABELS
        .map(({ key, labelKey, fallback }) => ({
          key,
          label: t(labelKey, fallback),
          count: summary.failure_buckets[key] ?? 0
        }))
        .filter((entry) => entry.count > 0),
    [summary.failure_buckets, t]
  )

  const setupEntries = useMemo(
    () =>
      (Object.entries(summary.setup_health) as Array<
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
        .map(([key, dimension]) => {
          const label = SETUP_DIMENSION_LABELS[key]
          return {
            key,
            label: t(label.labelKey, label.fallback),
            status: dimension.status,
            blockers: dimension.blockers,
            evidenceCount: dimension.evidence_count
          }
        }),
    [summary.setup_health, t]
  )

  const unverifiedAgents = useMemo(
    () => summary.compatibility.documented_unverified_agents ?? [],
    [summary.compatibility.documented_unverified_agents]
  )
  const redactionEnabled = useMemo(
    () =>
      summary.redaction.detail_events_artifacts_redacted_views &&
      summary.redaction.diagnostics_sanitized &&
      summary.redaction.audit_metadata_sanitized,
    [summary.redaction]
  )

  const sessionWindowFallback =
    summary.sessions.total === 1
      ? "{{total}} session in {{days}}d"
      : "{{total}} sessions in {{days}}d"
  const sessionWindowKey =
    summary.sessions.total === 1
      ? "option:agentRegistry.executionHealth.sessionWindowSingular"
      : "option:agentRegistry.executionHealth.sessionWindowPlural"

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">
            {t("option:agentRegistry.executionHealth.sessions", "Sessions")}
          </div>
          <div className="text-base font-medium">
            {t(
              sessionWindowKey,
              sessionWindowFallback,
              { total: summary.sessions.total, days: summary.range_days }
            )}
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {sessionStatusEntries.length > 0 ? (
              sessionStatusEntries.map(([status, count]) => (
                <DSBadge key={status} variant="secondary">
                  {t(
                    "option:agentRegistry.executionHealth.statusCount",
                    "{{count}} {{status}}",
                    { count, status }
                  )}
                </DSBadge>
              ))
            ) : (
              <span className="text-xs text-muted-foreground">
                {t(
                  "option:agentRegistry.executionHealth.noSessions",
                  "No sessions recorded"
                )}
              </span>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">
            {t("option:agentRegistry.executionHealth.compatibility", "Compatibility")}
          </div>
          <div className="mt-1 flex flex-wrap gap-1">
            {summary.compatibility.live_certification_required ? (
              <DSBadge variant="warning">
                {t(
                  "option:agentRegistry.executionHealth.liveCertificationRequired",
                  "Live certification required"
                )}
              </DSBadge>
            ) : (
              <DSBadge variant="success">
                {t(
                  "option:agentRegistry.executionHealth.noLiveCertificationBlocker",
                  "No live-certification blocker"
                )}
              </DSBadge>
            )}
            {Object.entries(summary.compatibility.by_support_state ?? {}).map(
              ([state, count]) => (
                <DSBadge key={state} variant="secondary">
                  {t(
                    "option:agentRegistry.executionHealth.supportStateCount",
                    "{{count}} {{state}}",
                    { count, state }
                  )}
                </DSBadge>
              )
            )}
          </div>
          {unverifiedAgents.length > 0 ? (
            <div className="mt-2 text-xs text-yellow-700 dark:text-yellow-400">
              {t(
                "option:agentRegistry.executionHealth.unverifiedAgents",
                "Unverified agents: {{agents}}",
                { agents: unverifiedAgents.join(", ") }
              )}
            </div>
          ) : (
            <div className="mt-2 text-xs text-muted-foreground">
              {t(
                "option:agentRegistry.executionHealth.noDocumentedUnverifiedAgents",
                "No documented-unverified agents"
              )}
            </div>
          )}
          {summary.compatibility.docs_url && (
            <a
              className="mt-2 inline-block text-xs text-primary hover:underline"
              href={summary.compatibility.docs_url}
              target="_blank"
              rel="noreferrer"
            >
              {t(
                "option:agentRegistry.executionHealth.executionEvidenceDocs",
                "Execution evidence docs"
              )}
            </a>
          )}
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="text-xs text-muted-foreground">
            {t(
              "option:agentRegistry.executionHealth.retentionAndRedaction",
              "Retention and redaction"
            )}
          </div>
          <div className="mt-1 text-sm">
            {t(
              "option:agentRegistry.executionHealth.retentionSummary",
              "Retention {{sessionDays}}d sessions / {{auditDays}}d audit",
              {
                sessionDays: summary.retention.session_retention_days,
                auditDays: summary.retention.audit_retention_days
              }
            )}
          </div>
          <div className="mt-2 text-xs text-muted-foreground">
            {redactionEnabled
              ? t(
                  "option:agentRegistry.executionHealth.redactedDrillThroughEnabled",
                  "Redacted drill-through enabled"
                )
              : t(
                  "option:agentRegistry.executionHealth.reviewRedactionSettings",
                  "Review redaction settings"
                )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-border p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            {t(
              "option:agentRegistry.executionHealth.failureBucketsTitle",
              "Failure buckets"
            )}
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
            <div className="text-sm text-muted-foreground">
              {t(
                "option:agentRegistry.executionHealth.noRecentFailureBuckets",
                "No recent failure buckets"
              )}
            </div>
          )}
        </div>

        <div className="rounded-lg border border-border p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            {t("option:agentRegistry.executionHealth.setupHealthTitle", "Setup health")}
          </div>
          {setupEntries.length > 0 ? (
            <div className="space-y-2">
              {setupEntries.map((entry) => (
                <div key={entry.key} className="rounded border border-border px-2 py-1.5 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <DSBadge variant={entry.status === "blocked" ? "danger" : "warning"}>
                      {t(
                        "option:agentRegistry.executionHealth.setupStatus",
                        "{{label}} {{status}}",
                        { label: entry.label, status: entry.status }
                      )}
                    </DSBadge>
                    <span className="text-xs text-muted-foreground">
                      {t(
                        "option:agentRegistry.executionHealth.evidenceCount",
                        "{{count}} evidence",
                        { count: entry.evidenceCount }
                      )}
                    </span>
                  </div>
                  {entry.blockers.length > 0 && (
                    <div className="mt-1 text-xs text-muted-foreground">
                      {t(
                        "option:agentRegistry.executionHealth.setupBlockers",
                        "{{label}} {{status}}: {{blockers}}",
                        {
                          label: entry.label,
                          status: entry.status,
                          blockers: entry.blockers.join(", ")
                        }
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              {t(
                "option:agentRegistry.executionHealth.noSetupBlockers",
                "No setup blockers in this window"
              )}
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
