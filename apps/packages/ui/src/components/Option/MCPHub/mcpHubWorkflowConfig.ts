import type { McpHubGovernanceAuditTabKey } from "@/services/tldw/mcp-hub"

export type McpHubWorkflowKey =
  | "setup"
  | "access"
  | "workspaces"
  | "governance"
  | "audit"

export type McpHubViewKey = McpHubGovernanceAuditTabKey

export type McpHubRouteState = {
  workflow: McpHubWorkflowKey
  view: McpHubViewKey
}

export type McpHubWorkflowDefinition = {
  key: McpHubWorkflowKey
  label: string
  description: string
  views: readonly McpHubViewKey[]
  defaultView: McpHubViewKey
}

export const MCP_HUB_VIEW_LABELS = {
  "tool-catalogs": "Tool Catalog",
  credentials: "Servers & Credentials",
  profiles: "Profiles",
  assignments: "Assignments",
  approvals: "Approvals",
  "path-scopes": "Path Scopes",
  "capability-mappings": "Capability Mappings",
  "workspace-sets": "Workspace Sets",
  "shared-workspaces": "Shared Workspaces",
  "governance-packs": "Governance Packs",
  audit: "Audit Findings"
} as const satisfies Record<McpHubViewKey, string>

export const MCP_HUB_VIEW_KEYS = Object.keys(
  MCP_HUB_VIEW_LABELS
) as McpHubViewKey[]

export const MCP_HUB_WORKFLOWS = {
  setup: {
    key: "setup",
    label: "Setup",
    description: "Connect MCP servers, configure credentials, and verify available tools.",
    views: ["credentials", "tool-catalogs"],
    defaultView: "credentials"
  },
  access: {
    key: "access",
    label: "Access",
    description: "Define profiles and assign tool access to users, groups, and defaults.",
    views: ["profiles", "assignments"],
    defaultView: "profiles"
  },
  workspaces: {
    key: "workspaces",
    label: "Workspaces",
    description: "Define trusted local paths and shared workspace boundaries.",
    views: ["path-scopes", "workspace-sets", "shared-workspaces"],
    defaultView: "path-scopes"
  },
  governance: {
    key: "governance",
    label: "Governance",
    description: "Manage approvals, governance packs, and capability adapters.",
    views: ["approvals", "governance-packs", "capability-mappings"],
    defaultView: "approvals"
  },
  audit: {
    key: "audit",
    label: "Audit",
    description: "Find and remediate risky or broken MCP Hub configuration.",
    views: ["audit"],
    defaultView: "audit"
  }
} as const satisfies Record<McpHubWorkflowKey, McpHubWorkflowDefinition>

export const MCP_HUB_WORKFLOW_ORDER = [
  "setup",
  "access",
  "workspaces",
  "governance",
  "audit"
] as const satisfies readonly McpHubWorkflowKey[]

export const DEFAULT_MCP_HUB_ROUTE_STATE: McpHubRouteState = {
  workflow: "setup",
  view: "credentials"
}

const workflowKeys = new Set<string>(MCP_HUB_WORKFLOW_ORDER)
const viewKeys = new Set<string>(MCP_HUB_VIEW_KEYS)

const viewToWorkflow = MCP_HUB_WORKFLOW_ORDER.reduce(
  (mapping, workflow) => {
    for (const view of MCP_HUB_WORKFLOWS[workflow].views) {
      mapping[view] = workflow
    }
    return mapping
  },
  {} as Partial<Record<McpHubViewKey, McpHubWorkflowKey>>
)

export const isMcpHubWorkflowKey = (
  value: string | null | undefined
): value is McpHubWorkflowKey => Boolean(value && workflowKeys.has(value))

export const isMcpHubViewKey = (
  value: string | null | undefined
): value is McpHubViewKey => Boolean(value && viewKeys.has(value))

export const workflowForMcpHubView = (
  view: McpHubViewKey
): McpHubWorkflowKey => {
  const workflow = viewToWorkflow[view]
  if (!workflow) {
    throw new Error(`MCP Hub view "${view}" is not assigned to a workflow.`)
  }
  return workflow
}

export const getDefaultMcpHubView = (
  workflow: McpHubWorkflowKey
): McpHubViewKey => MCP_HUB_WORKFLOWS[workflow].defaultView

export const resolveMcpHubRouteState = ({
  workflow,
  view
}: {
  workflow?: string | null
  view?: string | null
}): McpHubRouteState => {
  if (isMcpHubViewKey(view)) {
    return {
      workflow: workflowForMcpHubView(view),
      view
    }
  }

  if (isMcpHubWorkflowKey(workflow)) {
    return {
      workflow,
      view: getDefaultMcpHubView(workflow)
    }
  }

  return DEFAULT_MCP_HUB_ROUTE_STATE
}
