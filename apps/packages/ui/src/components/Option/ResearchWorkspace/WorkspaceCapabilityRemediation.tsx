import React from "react"
import { AlertTriangle, ExternalLink } from "lucide-react"
import { Link } from "react-router-dom"
import type {
  WorkspaceCapabilitiesResponse,
  WorkspaceCapabilityService,
  WorkspaceCapabilityServiceState
} from "@/services/tldw/domains/workspace-api"

type WorkspaceCapabilityRemediationProps = {
  capabilities: WorkspaceCapabilitiesResponse | null | undefined
}

type RemediationSeverity = "warning" | "blocked" | "muted"

type RemediationLink = {
  label: string
  href: string
}

type RemediationItem = {
  id: string
  label: string
  stateLabel: string
  reasonLabel: string | null
  guidance: string
  impact: string
  severity: RemediationSeverity
  link: RemediationLink | null
}

type ServiceConfig = {
  label: string
  impact: string
}

const SERVICE_ORDER = ["provider", "mcp", "acp", "sandbox"] as const

const SERVICE_CONFIG: Record<(typeof SERVICE_ORDER)[number], ServiceConfig> = {
  provider: {
    label: "Model Provider",
    impact: "Grounded answers"
  },
  mcp: {
    label: "MCP Hub",
    impact: "Workspace tools"
  },
  acp: {
    label: "ACP Agents",
    impact: "Agent runs"
  },
  sandbox: {
    label: "Sandbox",
    impact: "Sandboxed actions"
  }
}

const MANAGEMENT_LINKS: Record<string, RemediationLink | undefined> = {
  mcp_hub: { label: "Open MCP Hub", href: "/mcp-hub" },
  acp_workspace: { label: "Open ACP Playground", href: "/acp-playground" },
  sandbox_settings: { label: "Open Runtime Config", href: "/admin/runtime-config" },
  model_settings: { label: "Open Model Settings", href: "/settings/model" },
  shared_workspaces: { label: "Open Shared", href: "/shared" }
}

const buildMcpHubWorkspaceSetHref = (
  workspaceId: string | null | undefined
): string => {
  const params = new URLSearchParams({
    workflow: "workspaces",
    view: "workspace-sets"
  })
  const normalizedWorkspaceId = String(workspaceId ?? "").trim()
  if (normalizedWorkspaceId) {
    params.set("workspace_id", normalizedWorkspaceId)
  }
  params.set("source", "research-workspace")
  return `/mcp-hub?${params.toString()}`
}

const READY_STATES = new Set<WorkspaceCapabilityServiceState | string>([
  "available",
  "private"
])

const STATE_LABELS: Record<string, string> = {
  available: "Available",
  private: "Private",
  not_configured: "Not configured",
  needs_approval: "Needs approval",
  unknown: "Unknown",
  blocked: "Blocked",
  degraded: "Degraded",
  external_provider_warning: "External provider warning"
}

const REASON_LABELS: Record<string, string> = {
  acp_agent_setup_blocked: "Agent setup blocked",
  acp_approval_required: "Approval required",
  acp_live_certification_required: "Live certification required",
  acp_no_agents_configured: "No ACP agents configured",
  acp_not_available: "ACP unavailable",
  acp_not_configured: "ACP not configured",
  acp_status_unknown: "ACP status unknown",
  external_provider_only: "External providers only",
  mcp_approval_required: "Approval required",
  mcp_capability_unresolved: "MCP capability unresolved",
  mcp_capability_warnings: "MCP capability warnings",
  mcp_not_available: "MCP unavailable",
  mcp_not_configured: "MCP not configured",
  mcp_policy_not_configured: "MCP policy not configured",
  mcp_policy_resolution_failed: "MCP policy check failed",
  mcp_tools_blocked: "MCP tools blocked",
  mcp_workspace_not_allowed: "Workspace outside MCP policy",
  no_queryable_sources: "No queryable sources",
  no_workspace_acp_binding: "No ACP workspace binding",
  no_workspace_mcp_binding: "No MCP workspace binding",
  no_workspace_sandbox_binding: "No sandbox workspace binding",
  provider_health_degraded: "Provider health degraded",
  provider_health_unknown: "Provider health unknown",
  provider_not_configured: "Provider not configured",
  provider_not_evaluated: "Provider not evaluated",
  provider_unavailable: "Provider unavailable",
  sandbox_no_runtimes_discovered: "No sandbox runtimes discovered",
  sandbox_not_available: "Sandbox unavailable",
  sandbox_not_configured: "Sandbox not configured",
  sandbox_runtime_unavailable: "Runtime unavailable",
  use_sandbox_not_allowed: "Sandbox unavailable"
}

const getStateLabel = (state: string | null | undefined): string =>
  STATE_LABELS[state ?? ""] ?? "Unknown"

const getReasonLabel = (reasonCode: string | null | undefined): string | null => {
  if (!reasonCode) return null
  return REASON_LABELS[reasonCode] ?? reasonCode.replace(/_/g, " ")
}

const getSeverity = (
  service: WorkspaceCapabilityService | null | undefined
): RemediationSeverity => {
  const state = service?.state ?? "unknown"
  if (state === "blocked" || state === "not_configured") return "blocked"
  if (state === "private") return "muted"
  return "warning"
}

const getServiceGuidance = (
  key: (typeof SERVICE_ORDER)[number],
  service: WorkspaceCapabilityService
): string => {
  const reason = service.reason_code ?? ""

  if (key === "provider") {
    if (
      service.state === "external_provider_warning" ||
      reason === "external_provider_only"
    ) {
      return "Only external providers are configured. Use a local provider for fully local answers."
    }
    if (
      service.state === "not_configured" ||
      reason === "provider_not_configured"
    ) {
      return "Configure a chat provider before generating grounded answers."
    }
    if (service.state === "blocked" || reason === "provider_unavailable") {
      return "Restore a healthy model provider before asking grounded questions."
    }
    if (service.state === "unknown") {
      return "Provider readiness could not be checked. Open model settings or retry status."
    }
    return "A configured model provider is degraded. Check provider health before a long run."
  }

  if (key === "mcp") {
    if (service.state === "needs_approval" || reason === "mcp_approval_required") {
      return "Approve workspace tool use before running MCP actions."
    }
    if (reason === "mcp_workspace_not_allowed") {
      return "Add this workspace to the selected MCP policy scope."
    }
    if (
      service.state === "not_configured" ||
      reason === "mcp_policy_not_configured"
    ) {
      return "Connect this workspace to an MCP Hub policy before tools can run."
    }
    if (service.state === "unknown") {
      return "MCP policy could not be checked. Review the MCP Hub policy state."
    }
    return "Allow at least one MCP tool and resolve missing policy capabilities."
  }

  if (key === "acp") {
    if (service.state === "needs_approval" || reason === "acp_approval_required") {
      return "Approve the agent run before starting workspace automation."
    }
    if (
      service.state === "not_configured" ||
      reason === "acp_no_agents_configured"
    ) {
      return "Configure an ACP agent before workspace agent runs."
    }
    if (service.state === "unknown") {
      return "ACP readiness could not be checked. Open the ACP workspace surface."
    }
    if (reason === "acp_live_certification_required") {
      return "Complete ACP live certification before relying on agent runs."
    }
    return "Resolve agent setup blockers before starting workspace agents."
  }

  if (
    service.state === "not_configured" ||
    reason === "sandbox_no_runtimes_discovered" ||
    reason === "sandbox_runtime_unavailable"
  ) {
    return "Enable a sandbox runtime before sandboxed actions can run."
  }
  if (service.state === "unknown") {
    return "Sandbox runtime discovery could not be checked. Open runtime config."
  }
  return "Fix sandbox runtime availability before running isolated actions."
}

const buildServiceItem = (
  key: (typeof SERVICE_ORDER)[number],
  service: WorkspaceCapabilityService | null | undefined,
  workspaceId: string | null | undefined
): RemediationItem | null => {
  if (!service || READY_STATES.has(service.state)) return null

  const config = SERVICE_CONFIG[key]
  const managementSurface = service.management_surface ?? ""
  const baseLink = MANAGEMENT_LINKS[managementSurface] ?? null
  const link =
    key === "mcp" && managementSurface === "mcp_hub" && baseLink
      ? { ...baseLink, href: buildMcpHubWorkspaceSetHref(workspaceId) }
      : baseLink

  return {
    id: key,
    label: config.label,
    stateLabel: getStateLabel(service.state),
    reasonLabel: getReasonLabel(service.reason_code),
    guidance: getServiceGuidance(key, service),
    impact: config.impact,
    severity: getSeverity(service),
    link
  }
}

const buildGroundedAnswersItem = (
  capabilities: WorkspaceCapabilitiesResponse
): RemediationItem | null => {
  const action = capabilities.allowed_actions?.ask_grounded_questions
  const sourceSummary = capabilities.source_summary
  const hasSources =
    (sourceSummary?.total ?? 0) > 0 || (sourceSummary?.selected ?? 0) > 0

  if (!hasSources || !action || action.allowed) return null

  const reasonLabel = getReasonLabel(action.reason_code) ?? "Not ready"
  const guidance =
    action.reason_code === "no_queryable_sources"
      ? "Wait for extraction and indexing to finish before asking grounded questions."
      : "Resolve source readiness and provider setup before asking grounded questions."

  return {
    id: "ask_grounded_questions",
    label: "Grounded answers",
    stateLabel: "Blocked",
    reasonLabel,
    guidance,
    impact: "Source-backed chat",
    severity: "blocked",
    link: null
  }
}

const buildRemediationItems = (
  capabilities: WorkspaceCapabilitiesResponse | null | undefined
): RemediationItem[] => {
  if (!capabilities) return []

  const items = SERVICE_ORDER.flatMap((key) => {
    const item = buildServiceItem(
      key,
      capabilities.workspace_services?.[key],
      capabilities.workspace_id
    )
    return item ? [item] : []
  })
  const groundedItem = buildGroundedAnswersItem(capabilities)
  if (groundedItem) items.unshift(groundedItem)
  return items
}

const itemToneClass = (severity: RemediationSeverity): string => {
  if (severity === "blocked") {
    return "border-error/30 bg-error/5 text-error"
  }
  if (severity === "warning") {
    return "border-warning/30 bg-warning/10 text-warning"
  }
  return "border-border bg-surface2/50 text-text-muted"
}

export const WorkspaceCapabilityRemediation: React.FC<
  WorkspaceCapabilityRemediationProps
> = ({ capabilities }) => {
  const items = buildRemediationItems(capabilities)
  if (items.length === 0) return null

  const blockedCount = items.filter((item) => item.severity === "blocked").length
  const summaryLabel =
    items.length === 1 ? "1 setup item" : `${items.length} setup items`

  return (
    <details
      data-testid="workspace-capability-remediation"
      className="mb-2 rounded-md border border-border/70 bg-surface2/45 text-xs text-text"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 marker:hidden">
        <span className="inline-flex min-w-0 items-center gap-2">
          <AlertTriangle
            className={`h-3.5 w-3.5 shrink-0 ${
              blockedCount > 0 ? "text-error" : "text-warning"
            }`}
            aria-hidden="true"
          />
          <span className="font-semibold text-text">Workspace readiness</span>
          <span className="truncate text-text-muted">
            {summaryLabel} affect tools, agents, sandbox, or grounded answers.
          </span>
        </span>
      </summary>
      <div className="grid gap-1 border-t border-border/60 p-2 sm:grid-cols-2">
        {items.map((item) => (
          <div
            key={item.id}
            className={`min-w-0 rounded border px-2.5 py-2 ${itemToneClass(
              item.severity
            )}`}
          >
            <div className="flex min-w-0 flex-wrap items-center gap-1.5">
              <span className="font-semibold text-text">{item.label}</span>
              <span className="rounded bg-surface px-1.5 py-0.5 text-[11px] font-medium text-current">
                {item.stateLabel}
              </span>
              {item.reasonLabel && (
                <span className="rounded bg-surface px-1.5 py-0.5 text-[11px] text-current">
                  {item.reasonLabel}
                </span>
              )}
            </div>
            <p className="mt-1 text-[11px] leading-4 text-text-muted">
              {item.guidance}
            </p>
            <div className="mt-1.5 flex min-w-0 flex-wrap items-center justify-between gap-2">
              <span className="truncate text-[11px] text-text-subtle">
                Affects: {item.impact}
              </span>
              {item.link && (
                <Link
                  to={item.link.href}
                  className="inline-flex shrink-0 items-center gap-1 rounded border border-current/20 px-1.5 py-0.5 text-[11px] font-semibold text-current hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
                >
                  {item.link.label}
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </Link>
              )}
            </div>
          </div>
        ))}
      </div>
    </details>
  )
}
