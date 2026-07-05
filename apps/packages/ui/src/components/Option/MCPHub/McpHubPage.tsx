import { useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useSearchParams } from "react-router-dom"
import { Button, Tabs, Typography } from "antd"
import { StatePanel } from "@/components/ui/state"
import { setupOnboardingMethods } from "@/services/tldw/domains/setup-onboarding"

import { ApprovalPoliciesTab } from "./ApprovalPoliciesTab"
import { CapabilityMappingsTab } from "./CapabilityMappingsTab"
import { GovernanceAuditTab } from "./GovernanceAuditTab"
import { GovernancePacksTab } from "./GovernancePacksTab"
import { PathScopesTab } from "./PathScopesTab"
import { PermissionProfilesTab } from "./PermissionProfilesTab"
import { PolicyAssignmentsTab } from "./PolicyAssignmentsTab"
import { SharedWorkspacesTab } from "./SharedWorkspacesTab"
import { ToolCatalogsTab } from "./ToolCatalogsTab"
import { ExternalServersTab } from "./ExternalServersTab"
import { WorkspaceSetsTab } from "./WorkspaceSetsTab"
import type {
  McpHubDrillAction,
  McpHubDrillTarget,
  McpHubGovernanceAuditNavigateTarget
} from "@/services/tldw/mcp-hub"
import {
  persistMcpHubExplainerDismissed,
  readMcpHubExplainerDismissed
} from "@/utils/ftux-storage"
import type {
  McpToolsRecoveryStatusResponse,
  McpToolsValidationState
} from "@/types/setup-onboarding"
import {
  MCP_HUB_VIEW_LABELS,
  MCP_HUB_WORKFLOW_ORDER,
  MCP_HUB_WORKFLOWS,
  resolveMcpHubRouteState,
  workflowForMcpHubView,
  type McpHubRouteState,
  type McpHubViewKey,
  type McpHubWorkflowKey
} from "./mcpHubWorkflowConfig"

type McpHubShortcutItem = {
  key: string
  title: string
  description: string
  workflow: McpHubWorkflowKey
  view: McpHubViewKey
  actionLabel: string
}

const MCP_HUB_SHORTCUT_ITEMS: McpHubShortcutItem[] = [
  {
    key: "servers-credentials",
    title: "Servers & Credentials",
    description: "Add external servers, manage credential slots, and check local setup.",
    workflow: "setup",
    view: "credentials",
    actionLabel: "Open Servers & Credentials"
  },
  {
    key: "policy-assignments",
    title: "Policy Assignments",
    description: "Review user, group, and default access assignments.",
    workflow: "access",
    view: "assignments",
    actionLabel: "Open Policy Assignments"
  },
  {
    key: "approvals",
    title: "Approvals",
    description: "Manage approval policies and governance pack controls.",
    workflow: "governance",
    view: "approvals",
    actionLabel: "Open Approvals"
  },
  {
    key: "workspace-boundaries",
    title: "Workspace Boundaries",
    description: "Define path scopes and shared workspace boundaries.",
    workflow: "workspaces",
    view: "path-scopes",
    actionLabel: "Open Workspace Boundaries"
  },
  {
    key: "audit-findings",
    title: "Audit Findings",
    description: "Inspect broken references and risky configuration findings.",
    workflow: "audit",
    view: "audit",
    actionLabel: "Open Audit Findings"
  }
]

const RECOVERABLE_MCP_TOOL_STATES = new Set<McpToolsValidationState>([
  "skipped",
  "failed",
  "not_run",
  "external_discovery_incomplete"
])

const mcpRecoveryStatusLabel = (status: McpToolsRecoveryStatusResponse) => {
  if (status.status === "profile_manually_changed") return "Profile manually changed"
  if (
    status.validation_state === "built_in_passed" ||
    status.validation_state === "external_tool_passed"
  ) {
    return "Validated during setup"
  }
  if (status.validation_state === "failed") return "Validation failed"
  if (status.validation_state === "external_discovery_incomplete") {
    return "External discovery incomplete"
  }
  return "Not validated during setup"
}

const mcpRecoveryPanelState = (status: McpToolsRecoveryStatusResponse) => {
  if (status.status === "profile_manually_changed" || status.validation_state === "failed") {
    return "error" as const
  }
  if (status.validation_state === "external_discovery_incomplete") {
    return "degraded" as const
  }
  if (
    status.validation_state === "built_in_passed" ||
    status.validation_state === "external_tool_passed"
  ) {
    return "ready" as const
  }
  return "empty" as const
}

export const McpHubPage = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [explainerDismissed, setExplainerDismissed] = useState(
    () => readMcpHubExplainerDismissed()
  )
  const [drillTarget, setDrillTarget] = useState<McpHubDrillTarget | null>(null)
  const [mcpToolsRecoveryStatus, setMcpToolsRecoveryStatus] =
    useState<McpToolsRecoveryStatusResponse | null>(null)
  const [mcpToolsRecoveryRunning, setMcpToolsRecoveryRunning] = useState(false)
  const requestIdRef = useRef(0)

  const routeState = useMemo(
    () =>
      resolveMcpHubRouteState({
        workflow: searchParams.get("workflow"),
        view: searchParams.get("view")
      }),
    [searchParams]
  )

  const updateRouteState = (nextState: McpHubRouteState) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set("workflow", nextState.workflow)
    nextParams.set("view", nextState.view)
    setSearchParams(nextParams, { replace: true })
  }

  const _deriveDrillAction = (
    target: McpHubGovernanceAuditNavigateTarget
  ): McpHubDrillAction => {
    if (
      target.tab === "assignments" ||
      target.tab === "workspace-sets" ||
      target.tab === "shared-workspaces" ||
      target.tab === "credentials"
    ) {
      return "edit"
    }
    return "focus"
  }

  const handleOpen = (target: McpHubGovernanceAuditNavigateTarget) => {
    requestIdRef.current += 1
    setDrillTarget({
      ...target,
      action: _deriveDrillAction(target),
      request_id: requestIdRef.current
    })
    updateRouteState({
      workflow: workflowForMcpHubView(target.tab),
      view: target.tab
    })
  }

  const handleDrillHandled = (requestId: number) => {
    setDrillTarget((current) => (current?.request_id === requestId ? null : current))
  }

  const handleExplainerClose = () => {
    setExplainerDismissed(true)
    persistMcpHubExplainerDismissed()
  }

  const handleWorkflowChange = (workflow: McpHubWorkflowKey) => {
    updateRouteState({
      workflow,
      view: MCP_HUB_WORKFLOWS[workflow].defaultView
    })
  }

  const handleViewChange = (view: string) => {
    const nextView = view as McpHubViewKey
    updateRouteState({
      workflow: workflowForMcpHubView(nextView),
      view: nextView
    })
  }

  const handleOpenToolCatalog = () => {
    updateRouteState({
      workflow: workflowForMcpHubView("tool-catalogs"),
      view: "tool-catalogs"
    })
  }

  const handleOpenServerSetup = () => {
    setDrillTarget(null)
    updateRouteState({
      workflow: workflowForMcpHubView("credentials"),
      view: "credentials"
    })
  }

  useEffect(() => {
    let active = true
    setupOnboardingMethods
      .getMcpToolsRecoveryStatus()
      .then((status) => {
        if (active) setMcpToolsRecoveryStatus(status)
      })
      .catch(() => {
        if (active) setMcpToolsRecoveryStatus(null)
      })
    return () => {
      active = false
    }
  }, [])

  const handleRunMcpToolsRecoveryValidation = async () => {
    setMcpToolsRecoveryRunning(true)
    try {
      setMcpToolsRecoveryStatus(await setupOnboardingMethods.validateMcpToolsRecovery())
    } catch {
      // Keep MCP Hub usable when recovery validation fails.
    } finally {
      setMcpToolsRecoveryRunning(false)
    }
  }

  const handleReviewFirstRunProfile = () => {
    if (mcpToolsRecoveryStatus?.profile_id != null) {
      requestIdRef.current += 1
      setDrillTarget({
        tab: "profiles",
        object_kind: "permission_profile",
        object_id: String(mcpToolsRecoveryStatus.profile_id),
        action: "edit",
        request_id: requestIdRef.current
      })
    }
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set("workflow", "access")
    nextParams.set("view", "profiles")
    if (mcpToolsRecoveryStatus?.profile_id != null) {
      nextParams.set("profile_id", String(mcpToolsRecoveryStatus.profile_id))
    }
    setSearchParams(nextParams, { replace: true })
  }

  const handleOpenExternalServer = (serverId: string, action: McpHubDrillAction) => {
    requestIdRef.current += 1
    setDrillTarget({
      tab: "credentials",
      object_kind: "external_server",
      object_id: serverId,
      action,
      request_id: requestIdRef.current
    })
    updateRouteState({
      workflow: workflowForMcpHubView("credentials"),
      view: "credentials"
    })
  }

  const tabContentByView: Record<McpHubViewKey, ReactNode> = {
    "tool-catalogs": (
      <ToolCatalogsTab
        onOpenServerSetup={handleOpenServerSetup}
        onOpenServerCredentials={(serverId) => handleOpenExternalServer(serverId, "focus")}
        onOpenServerConfig={(serverId) => handleOpenExternalServer(serverId, "edit")}
      />
    ),
    credentials: (
      <ExternalServersTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
        onOpenToolCatalog={handleOpenToolCatalog}
      />
    ),
    profiles: (
      <PermissionProfilesTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
      />
    ),
    assignments: (
      <PolicyAssignmentsTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
      />
    ),
    approvals: <ApprovalPoliciesTab />,
    "path-scopes": <PathScopesTab />,
    "capability-mappings": <CapabilityMappingsTab />,
    "workspace-sets": (
      <WorkspaceSetsTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
      />
    ),
    "shared-workspaces": (
      <SharedWorkspacesTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
      />
    ),
    "governance-packs": <GovernancePacksTab />,
    audit: <GovernanceAuditTab onOpen={handleOpen} />
  }

  const activeWorkflow = MCP_HUB_WORKFLOWS[routeState.workflow]
  const shouldShowMcpToolsRecovery =
    Boolean(mcpToolsRecoveryStatus) &&
    (searchParams.get("source") === "first-run" ||
      mcpToolsRecoveryStatus?.profile_id != null ||
      Boolean(mcpToolsRecoveryStatus?.selected_pack_ids.length))
  const mcpToolsRecoveryLabel = mcpToolsRecoveryStatus
    ? mcpRecoveryStatusLabel(mcpToolsRecoveryStatus)
    : null
  const mcpToolsProfileChanged =
    mcpToolsRecoveryStatus?.status === "profile_manually_changed"
  const canRunMcpToolsRecovery =
    Boolean(mcpToolsRecoveryStatus) &&
    !mcpToolsProfileChanged &&
    RECOVERABLE_MCP_TOOL_STATES.has(mcpToolsRecoveryStatus.validation_state)
  const childTabItems = activeWorkflow.views.map((view) => ({
    key: view,
    label: (
      <span data-testid={`mcp-hub-tab-${view}`}>
        {MCP_HUB_VIEW_LABELS[view]}
      </span>
    ),
    children: tabContentByView[view]
  }))

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 p-4" data-testid="mcp-hub-shell">
      <Typography.Title level={1} className="!mb-0 !text-2xl">
        MCP Hub
      </Typography.Title>
      <Typography.Text type="secondary">
        Manage external tool servers and governance policies for the Model Context Protocol (MCP).
      </Typography.Text>
      <nav
        aria-label="MCP Hub workflow shortcuts"
        className="grid gap-3 md:grid-cols-2 xl:grid-cols-5"
        data-testid="mcp-hub-workflow-shortcuts"
      >
        {MCP_HUB_SHORTCUT_ITEMS.map((item) => (
          <article
            key={item.key}
            className="rounded-lg border border-border bg-surface p-3 shadow-sm"
            data-testid={`mcp-hub-shortcut-${item.key}`}
          >
            <h2 className="text-sm font-semibold text-text">{item.title}</h2>
            <p className="mt-2 text-xs text-text-muted">{item.description}</p>
            <Button
              className="mt-3"
              size="small"
              onClick={() =>
                updateRouteState({
                  workflow: item.workflow,
                  view: item.view
                })
              }
            >
              {item.actionLabel}
            </Button>
          </article>
        ))}
      </nav>
      {!explainerDismissed && (
        <StatePanel
          data-testid="mcp-hub-explainer"
          state="empty"
          title="Getting Started with MCP Hub"
          message="MCP Hub lets you connect external tool servers, manage permissions, and govern how AI models interact with outside services. Start by adding or checking Servers & Credentials, then use the Tool Catalog to verify available tools."
          secondaryActions={[
            {
              label: "Dismiss",
              onClick: handleExplainerClose
            }
          ]}
        />
      )}
      {shouldShowMcpToolsRecovery && mcpToolsRecoveryStatus && mcpToolsRecoveryLabel && (
        <StatePanel
          data-testid="mcp-tools-recovery-status"
          state={mcpRecoveryPanelState(mcpToolsRecoveryStatus)}
          title="First-run MCP tools"
          message={mcpToolsRecoveryLabel}
          primaryAction={
            canRunMcpToolsRecovery
              ? {
                  label: "Run validation",
                  loading: mcpToolsRecoveryRunning,
                  onClick: handleRunMcpToolsRecoveryValidation
                }
              : mcpToolsProfileChanged
                ? {
                    label: "Review profile",
                    onClick: handleReviewFirstRunProfile
                  }
                : undefined
          }
          secondaryActions={
            mcpToolsRecoveryStatus.profile_id != null && !mcpToolsProfileChanged
              ? [
                  {
                    label: "Open profile",
                    onClick: handleReviewFirstRunProfile
                  }
                ]
              : []
          }
        />
      )}
      <div
        className="flex flex-wrap gap-2"
        data-testid="mcp-hub-workflows"
        role="group"
        aria-label="MCP Hub workflows"
      >
        {MCP_HUB_WORKFLOW_ORDER.map((workflow) => {
          const definition = MCP_HUB_WORKFLOWS[workflow]
          const isActive = workflow === routeState.workflow
          return (
            <Button
              key={workflow}
              type={isActive ? "primary" : "default"}
              aria-pressed={isActive}
              data-testid={`mcp-hub-workflow-${workflow}`}
              onClick={() => handleWorkflowChange(workflow)}
            >
              {definition.label}
            </Button>
          )
        })}
      </div>
      <Typography.Text
        type="secondary"
        data-testid="mcp-hub-workflow-description"
      >
        {activeWorkflow.description}
      </Typography.Text>
      <Tabs
        data-testid="mcp-hub-tabs"
        activeKey={routeState.view}
        onChange={handleViewChange}
        items={childTabItems}
      />
    </div>
  )
}
