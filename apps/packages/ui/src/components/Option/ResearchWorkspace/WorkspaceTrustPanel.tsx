import React from "react"
import { AlertTriangle, Loader2, ShieldCheck } from "lucide-react"
import type {
  WorkspaceCapabilitiesResponse,
  WorkspaceCapabilityService,
  WorkspaceSourceStatusListResponse
} from "@/services/tldw/domains/workspace-api"

type WorkspaceTrustPanelProps = {
  sourceStatus: WorkspaceSourceStatusListResponse | null
  capabilities: WorkspaceCapabilitiesResponse | null
  loading?: boolean
  errorMessage?: string | null
  statusGuardrailsEnabled?: boolean
}

type ServiceRow = {
  key: string
  label: string
  service: WorkspaceCapabilityService | null
}

const SERVICE_LABELS: Record<string, string> = {
  mcp: "MCP Hub",
  acp: "ACP",
  sandbox: "Sandbox",
  provider: "Provider"
}

const formatCount = (value: number | null | undefined, label: string): string =>
  `${Math.max(0, value ?? 0)} ${label}`

const formatStateLabel = (state: string | null | undefined): string => {
  if (!state) return "unknown"
  return state.replace(/_/g, " ")
}

const getServiceToneClass = (
  service: WorkspaceCapabilityService | null
): string => {
  if (!service) return "border-border bg-surface2/50 text-text-muted"
  if (service.state === "available") {
    return "border-success/30 bg-success/10 text-success"
  }
  if (service.state === "degraded" || service.state === "unknown") {
    return "border-warning/30 bg-warning/10 text-warning"
  }
  if (service.state === "blocked") {
    return "border-error/30 bg-error/10 text-error"
  }
  return "border-border bg-surface2/60 text-text-muted"
}

const getActionReason = (
  capabilities: WorkspaceCapabilitiesResponse | null,
  action: string
): string | null => {
  const actionStatus = capabilities?.allowed_actions?.[action]
  if (!actionStatus || actionStatus.allowed) return null
  return actionStatus.reason_code || "not_allowed"
}

export const WorkspaceTrustPanel: React.FC<WorkspaceTrustPanelProps> = ({
  sourceStatus,
  capabilities,
  loading = false,
  errorMessage = null,
  statusGuardrailsEnabled = true
}) => {
  if (!statusGuardrailsEnabled) return null

  if (!sourceStatus && !capabilities && errorMessage) {
    return (
      <section
        data-testid="workspace-trust-panel"
        role="status"
        aria-live="polite"
        className="mx-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning"
      >
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <div className="min-w-0">
            <div className="font-semibold text-text">Workspace trust unavailable</div>
            <div className="mt-0.5 break-words text-warning">{errorMessage}</div>
          </div>
        </div>
      </section>
    )
  }

  if (!sourceStatus && !capabilities) {
    return (
      <section
        data-testid="workspace-trust-panel"
        role="status"
        aria-live="polite"
        aria-label="Workspace trust"
        className="mx-2 rounded-lg border border-border/80 bg-surface/95 px-3 py-2 text-xs text-text shadow-sm"
      >
        <div className="flex min-w-0 items-center gap-2">
          <ShieldCheck className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
          <div className="min-w-0">
            <div className="font-semibold text-text">Workspace trust</div>
            <div className="text-[11px] text-text-muted">
              <span className="inline-flex items-center gap-1">
                {loading && (
                  <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
                )}
                Checking workspace readiness
              </span>
            </div>
          </div>
        </div>
      </section>
    )
  }

  const summary = sourceStatus?.summary ?? capabilities?.source_summary
  const groundedBlockedReason = getActionReason(capabilities, "ask_grounded_questions")
  const serviceRows: ServiceRow[] = Object.entries(SERVICE_LABELS).map(
    ([key, label]) => ({
      key,
      label,
      service: capabilities?.workspace_services?.[key] ?? null
    })
  )

  return (
    <section
      data-testid="workspace-trust-panel"
      aria-label="Workspace trust"
      className="mx-2 rounded-lg border border-border/80 bg-surface/95 px-3 py-2 text-xs text-text shadow-sm"
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex min-w-[10rem] items-center gap-2">
          <ShieldCheck className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
          <div className="min-w-0">
            <div className="font-semibold text-text">Workspace trust</div>
            <div className="text-[11px] text-text-muted">
              {loading ? (
                <span className="inline-flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
                  Updating status
                </span>
              ) : (
                "Source and tool readiness"
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="rounded-full border border-success/30 bg-success/10 px-2 py-0.5 font-medium text-success">
            {formatCount(summary?.queryable, "queryable")}
          </span>
          <span className="rounded-full border border-warning/30 bg-warning/10 px-2 py-0.5 font-medium text-warning">
            {formatCount(summary?.processing, "processing")}
          </span>
          <span className="rounded-full border border-error/30 bg-error/10 px-2 py-0.5 font-medium text-error">
            {formatCount(summary?.missing, "missing")}
          </span>
        </div>

        {groundedBlockedReason && (
          <div className="inline-flex min-w-0 items-center gap-1.5 rounded border border-warning/30 bg-warning/10 px-2 py-1 text-warning">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span className="font-semibold">Grounded questions blocked</span>
            <code className="break-all rounded bg-surface px-1 py-0.5 text-[11px] text-warning">
              {groundedBlockedReason}
            </code>
          </div>
        )}

        <div className="flex min-w-0 flex-wrap items-center gap-1.5">
          {serviceRows.map(({ key, label, service }) => {
            const stateLabel = formatStateLabel(service?.state)
            const managementSurface = service?.management_surface
            const accessibleLabel = [
              label,
              stateLabel,
              service?.reason_code,
              managementSurface ? `managed in ${managementSurface}` : null
            ]
              .filter(Boolean)
              .join(", ")

            return (
              <span
                key={key}
                className={`inline-flex min-w-0 max-w-full items-center gap-1 rounded border px-2 py-1 ${getServiceToneClass(
                  service
                )}`}
                title={managementSurface ?? undefined}
                aria-label={accessibleLabel}
              >
                <span className="shrink-0 font-semibold">{label}</span>
                <span className="shrink-0 text-text-muted">{stateLabel}</span>
                {service?.reason_code && (
                  <code className="min-w-0 break-all rounded bg-surface px-1 py-0.5 text-[11px] text-current">
                    {service.reason_code}
                  </code>
                )}
                {managementSurface && (
                  <span className="sr-only">Managed in {managementSurface}</span>
                )}
              </span>
            )
          })}
        </div>

        {errorMessage && (
          <div className="inline-flex min-w-0 items-center gap-1 text-warning" role="status">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span className="break-words">{errorMessage}</span>
          </div>
        )}
      </div>
    </section>
  )
}
