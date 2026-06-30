export const WORKSPACE_MANAGER_COPY = {
  workspace: "Workspace",
  researchWorkspace: "Research Workspace",
  projectWorkspace: "Project Workspace",
  hostLocalRoot: "Host-local root",
  sandboxManagedRoot: "Sandbox-managed root",
  mcpTrustedRootBinding: "MCP trusted root binding",
  mcpToolScope: "MCP tool scope",
  agentExecutionWorkspace: "agent execution workspace"
} as const

export type WorkspaceManagerCopyKey = keyof typeof WORKSPACE_MANAGER_COPY
export type WorkspaceManagerCanonicalLabel =
  (typeof WORKSPACE_MANAGER_COPY)[WorkspaceManagerCopyKey]

export const WORKSPACE_MANAGER_CANONICAL_LABELS = Object.values(
  WORKSPACE_MANAGER_COPY
) as WorkspaceManagerCanonicalLabel[]

export const isCanonicalWorkspaceManagerLabel = (
  label: string
): label is WorkspaceManagerCanonicalLabel =>
  WORKSPACE_MANAGER_CANONICAL_LABELS.includes(
    label as WorkspaceManagerCanonicalLabel
  )
