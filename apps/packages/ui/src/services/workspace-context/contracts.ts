import type {
  WorkspaceAllowedAction,
  WorkspaceAttentionState,
  WorkspaceCapabilitiesResponse,
  WorkspaceContextPartialError,
  WorkspaceProjectRoot,
  WorkspaceProfile,
  WorkspaceResolution,
  WorkspaceSourceStatusSummary
} from "@/services/tldw/domains/workspace-api"

export type ActiveWorkspaceContextState =
  | "none"
  | "loading"
  | "ready"
  | "partial"
  | "missing"
  | "error"

export type WorkspaceRecoverySeverity = "info" | "warning" | "error"

export interface WorkspaceSummaryContract {
  id: string
  name: string | null
  label: string
  profile: WorkspaceProfile
  archived: boolean
  deleted: boolean
  studyMaterialsPolicy: "general" | "workspace"
  statusLabel: string
  version: number
  lastModified: string
}

export interface WorkspaceRecoveryContract {
  reasonCode: string
  severity: WorkspaceRecoverySeverity
  message: string
  nextStepLabel: string | null
  nextStepHref: string | null
}

export interface ActiveWorkspaceContextContract {
  state: ActiveWorkspaceContextState
  workspaceId: string | null
  workspace: WorkspaceSummaryContract | null
  attentionState: WorkspaceAttentionState | null
  resolution: WorkspaceResolution | null
  projectRoot: WorkspaceProjectRoot | null
  sourceSummary: WorkspaceSourceStatusSummary
  capabilities: WorkspaceCapabilitiesResponse | null
  allowedActions: Record<string, WorkspaceAllowedAction>
  partialErrors: WorkspaceContextPartialError[]
  recovery: WorkspaceRecoveryContract
}

export interface WorkspaceEligibilityDecision {
  action: string
  allowed: boolean
  reasonCode: string
  severity: WorkspaceRecoverySeverity
  primaryMessage: string
  nextStepLabel: string | null
  nextStepHref: string | null
  recovery: WorkspaceRecoveryContract
}

export type WorkspaceMembershipTone = "neutral" | "success" | "warning" | "error"

export interface WorkspaceMembershipLabel {
  workspaceId: string | null
  workspaceLabel: string
  membershipLabel: string
  tone: WorkspaceMembershipTone
  isAuthoritative: boolean
  reasonCode: string | null
}

export type ACPWorkspaceContextState =
  | "aligned"
  | "mismatch"
  | "session_only"
  | "active_only"
  | "missing"
  | "unknown"

export interface ACPWorkspaceContextInput {
  sessionWorkspaceId?: string | null
  activeWorkspaceId?: string | null
  sessionWorkspaceLabel?: string | null
  activeWorkspaceLabel?: string | null
}

export interface ACPWorkspaceContextContract {
  state: ACPWorkspaceContextState
  sessionWorkspaceId: string | null
  activeWorkspaceId: string | null
  sessionWorkspaceLabel: string | null
  activeWorkspaceLabel: string | null
  message: string
  recovery: WorkspaceRecoveryContract
}

export const EMPTY_WORKSPACE_SOURCE_SUMMARY: WorkspaceSourceStatusSummary = {
  total: 0,
  selected: 0,
  queryable: 0,
  partially_queryable: 0,
  processing: 0,
  failed: 0,
  missing: 0
}
