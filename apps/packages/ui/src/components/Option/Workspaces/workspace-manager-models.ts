import type {
  WorkspaceApiResponse,
  WorkspaceAttentionState,
  WorkspaceContextResponse,
  WorkspaceFileInventory,
  WorkspaceOperationResponse,
  WorkspaceProfile,
  WorkspaceProjectRoot,
  WorkspaceProjectRootBackend,
  WorkspaceProjectRootState
} from "@/services/tldw/domains/workspace-api"

export type WorkspaceManagerProfile = WorkspaceProfile | "unknown"
export type WorkspaceManagerAttention = WorkspaceAttentionState

export interface WorkspaceManagerFileInventory {
  state: string | null
  indexedFileCount: number | null
  totalFileCount: number | null
  updatedAt: string | null
  available: boolean
}

export interface WorkspaceManagerProjectRoot {
  state: WorkspaceProjectRootState
  rootId: string | null
  backend: WorkspaceProjectRootBackend | null
  displayName: string | null
  pathHint: string | null
  gitState: string | null
  fileInventoryState: string | null
  fileInventory: WorkspaceManagerFileInventory
  indexingState: string | null
  sandboxMountState: string | null
  mcpTrustState: string | null
}

export interface WorkspaceManagerItem {
  id: string
  name: string
  archived: boolean
  profile: WorkspaceManagerProfile
  attentionState: WorkspaceManagerAttention
  projectRoot: WorkspaceManagerProjectRoot
  fileInventoryAvailable: boolean
  activeOperations: WorkspaceOperationResponse[]
  updatedAt: string
  version: number
}

const WORKSPACE_ATTENTION_STATES = new Set<WorkspaceAttentionState>([
  "ready",
  "setup_pending",
  "working",
  "needs_attention",
  "blocked",
  "archived"
])

const WORKSPACE_PROFILES = new Set<WorkspaceProfile>(["research", "project"])

const DEFAULT_FILE_INVENTORY: WorkspaceManagerFileInventory = {
  state: "not_started",
  indexedFileCount: null,
  totalFileCount: null,
  updatedAt: null,
  available: false
}

const DEFAULT_PROJECT_ROOT: WorkspaceManagerProjectRoot = {
  state: "not_configured",
  rootId: null,
  backend: null,
  displayName: null,
  pathHint: null,
  gitState: null,
  fileInventoryState: "not_started",
  fileInventory: DEFAULT_FILE_INVENTORY,
  indexingState: null,
  sandboxMountState: null,
  mcpTrustState: null
}

export const normalizeWorkspaceManagerItem = (
  workspace: WorkspaceApiResponse,
  context?: WorkspaceContextResponse | null
): WorkspaceManagerItem => {
  const profile = normalizeWorkspaceProfile(workspace.workspace_profile)
  const projectRoot = normalizeProjectRoot(context?.project_root)
  const attentionState = normalizeAttentionState(workspace, context, profile)

  return {
    id: workspace.id,
    name: workspace.name || workspace.banner_title || workspace.id,
    archived: Boolean(workspace.archived),
    profile,
    attentionState,
    projectRoot,
    fileInventoryAvailable: projectRoot.fileInventory.available,
    activeOperations: Array.isArray(context?.active_operations)
      ? context.active_operations
      : [],
    updatedAt: workspace.last_modified,
    version: workspace.version
  }
}

const normalizeWorkspaceProfile = (
  profile: WorkspaceApiResponse["workspace_profile"]
): WorkspaceManagerProfile =>
  WORKSPACE_PROFILES.has(profile as WorkspaceProfile)
    ? (profile as WorkspaceProfile)
    : "unknown"

const normalizeAttentionState = (
  workspace: WorkspaceApiResponse,
  context: WorkspaceContextResponse | null | undefined,
  profile: WorkspaceManagerProfile
): WorkspaceManagerAttention => {
  if (
    context?.attention_state &&
    WORKSPACE_ATTENTION_STATES.has(context.attention_state)
  ) {
    return context.attention_state
  }
  if (workspace.archived) return "archived"
  if (profile === "project") return "setup_pending"
  if (profile === "research") return "ready"
  return "needs_attention"
}

const normalizeProjectRoot = (
  root: WorkspaceProjectRoot | null | undefined
): WorkspaceManagerProjectRoot => {
  if (!root) return { ...DEFAULT_PROJECT_ROOT, fileInventory: { ...DEFAULT_FILE_INVENTORY } }

  const fileInventory = normalizeFileInventory(root.file_inventory)
  return {
    state: root.state || "not_configured",
    rootId: root.root_id ?? null,
    backend: root.backend ?? null,
    displayName: root.display_name ?? null,
    pathHint: root.path_hint ?? null,
    gitState: root.git_state ?? null,
    fileInventoryState: root.file_inventory_state ?? fileInventory.state,
    fileInventory,
    indexingState: root.indexing_state ?? null,
    sandboxMountState: root.sandbox_mount_state ?? null,
    mcpTrustState: root.mcp_trust_state ?? null
  }
}

const normalizeFileInventory = (
  inventory: WorkspaceFileInventory | null | undefined
): WorkspaceManagerFileInventory => ({
  state: inventory?.state ?? "not_started",
  indexedFileCount: inventory?.indexed_file_count ?? null,
  totalFileCount: inventory?.total_file_count ?? null,
  updatedAt: inventory?.updated_at ?? null,
  available: Boolean(inventory?.available)
})
