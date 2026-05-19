import { useMemo, useRef, useState, type ReactNode } from "react"
import { useSearchParams } from "react-router-dom"
import { Alert, Button, Tabs, Typography } from "antd"

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

type McpHubStatusItem = {
  key: string
  title: string
  description: string
  statusLabel: string
  workflow: McpHubWorkflowKey
  view: McpHubViewKey
  actionLabel: string
}

const MCP_HUB_STATUS_ITEMS: McpHubStatusItem[] = [
  {
    key: "servers-credentials",
    title: "Servers & Credentials",
    description: "External servers and credential slots load in the setup workflow.",
    statusLabel: "Setup workflow",
    workflow: "setup",
    view: "credentials",
    actionLabel: "Open Servers & Credentials"
  },
  {
    key: "policy-assignments",
    title: "Policy Assignments",
    description: "User, group, and default access assignments live in the access workflow.",
    statusLabel: "Access workflow",
    workflow: "access",
    view: "assignments",
    actionLabel: "Open Policy Assignments"
  },
  {
    key: "approvals",
    title: "Approvals",
    description: "Approval policies and governance packs live in the governance workflow.",
    statusLabel: "Governance workflow",
    workflow: "governance",
    view: "approvals",
    actionLabel: "Open Approvals"
  },
  {
    key: "workspace-boundaries",
    title: "Workspace Boundaries",
    description: "Path scopes and shared workspace boundaries live in the workspaces workflow.",
    statusLabel: "Workspaces workflow",
    workflow: "workspaces",
    view: "path-scopes",
    actionLabel: "Open Workspace Boundaries"
  },
  {
    key: "audit-findings",
    title: "Audit Findings",
    description: "Broken references and risky configuration findings live in the audit workflow.",
    statusLabel: "Audit workflow",
    workflow: "audit",
    view: "audit",
    actionLabel: "Open Audit Findings"
  }
]

export const McpHubPage = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [explainerDismissed, setExplainerDismissed] = useState(
    () => readMcpHubExplainerDismissed()
  )
  const [drillTarget, setDrillTarget] = useState<McpHubDrillTarget | null>(null)
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

  const tabContentByView: Record<McpHubViewKey, ReactNode> = {
    "tool-catalogs": <ToolCatalogsTab />,
    credentials: (
      <ExternalServersTab
        drillTarget={drillTarget}
        onDrillHandled={handleDrillHandled}
      />
    ),
    profiles: <PermissionProfilesTab />,
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
      <Typography.Title level={3} style={{ margin: 0 }}>
        MCP Hub
      </Typography.Title>
      <Typography.Text type="secondary">
        Manage external tool servers and governance policies for the Model Context Protocol (MCP).
      </Typography.Text>
      <section
        aria-label="MCP Hub status summary"
        className="grid gap-3 md:grid-cols-2 xl:grid-cols-5"
        data-testid="mcp-hub-status-summary"
      >
        {MCP_HUB_STATUS_ITEMS.map((item) => (
          <article
            key={item.key}
            className="rounded-lg border border-border bg-surface p-3 shadow-sm"
            data-testid={`mcp-hub-status-${item.key}`}
          >
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-sm font-semibold text-text">{item.title}</h2>
              <span className="rounded-full border border-border bg-surface2 px-2 py-0.5 text-xs text-text-muted">
                {item.statusLabel}
              </span>
            </div>
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
      </section>
      {!explainerDismissed && (
        <Alert
          data-testid="mcp-hub-explainer"
          type="info"
          showIcon
          closable
          onClose={handleExplainerClose}
          title="Getting Started with MCP Hub"
          description="MCP Hub lets you connect external tool servers, manage permissions, and govern how AI models interact with outside services. Start by adding or checking Servers & Credentials, then use the Tool Catalog to verify available tools."
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
