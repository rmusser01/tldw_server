import React from "react"
import { useTranslation } from "react-i18next"
import { Bot, Check, X, AlertCircle, Loader2 } from "lucide-react"

interface ACPHealthResponse {
  overall: string
  runner?: {
    status: string
    agent_type?: string
  }
  agents?: Array<{
    agent_type: string
    status: string
  }>
  [key: string]: unknown
}

interface ACPStatusCardProps {
  /** Health data from the parent's query (avoids a duplicate fetch). */
  healthData?: ACPHealthResponse | null
  /** Whether the health query is still loading. */
  isLoading?: boolean
  /** Whether the health query errored. */
  isError?: boolean
}

/**
 * ACPStatusCard - Displays ACP backend health status.
 *
 * Accepts health data as a prop from the parent rather than making its own
 * query, so that React Query can deduplicate the health check via the shared
 * `["acp", "health", serverUrl]` key.
 */
export const ACPStatusCard: React.FC<ACPStatusCardProps> = ({
  healthData,
  isLoading = false,
  isError = false,
}) => {
  const { t } = useTranslation(["playground", "common"])

  const isDegraded = healthData?.overall === "degraded"
  const agentCount = Array.isArray(healthData?.agents)
    ? healthData.agents.length
    : 0

  // Loading state
  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-surface p-4">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-text-muted" />
          <span className="text-sm text-text-muted">
            {t("playground:acp.statusCard.loading", "Checking ACP status...")}
          </span>
        </div>
      </div>
    )
  }

  // Error / unavailable state
  if (isError || !healthData || healthData.overall === "unavailable") {
    return (
      <div className="rounded-lg border border-border bg-surface p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-error/10">
            <X className="h-4 w-4 text-error" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-text">
              {t("playground:acp.statusCard.unavailable", "ACP Unavailable")}
            </div>
            <div className="mt-0.5 text-xs text-text-muted">
              {t(
                "playground:acp.statusCard.unavailableHelp",
                "The ACP runner is not reachable. Check that the backend is running and ACP is enabled in your configuration."
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Healthy / degraded state
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center gap-3">
        <div
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
            isDegraded ? "bg-warning/10" : "bg-success/10"
          }`}
        >
          {isDegraded ? (
            <AlertCircle className="h-4 w-4 text-warning" />
          ) : (
            <Check className="h-4 w-4 text-success" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-text">
            {isDegraded
              ? t("playground:acp.statusCard.degraded", "ACP Degraded")
              : t("playground:acp.statusCard.healthy", "ACP Healthy")}
          </div>
          <div className="mt-0.5 text-xs text-text-muted">
            {t(
              "playground:acp.statusCard.runnerStatus",
              "Runner: {{status}}",
              { status: healthData.runner?.status ?? healthData.overall }
            )}
            {agentCount > 0 && (
              <>
                {" "}
                &middot;{" "}
                {t(
                  "playground:acp.statusCard.agentCount",
                  "{{count}} agent(s) configured",
                  { count: agentCount }
                )}
              </>
            )}
          </div>
        </div>
        <Bot className="h-5 w-5 shrink-0 text-text-muted" />
      </div>
    </div>
  )
}

export default ACPStatusCard
