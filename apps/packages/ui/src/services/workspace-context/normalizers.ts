import type {
  WorkspaceAllowedAction,
  WorkspaceApiResponse,
  WorkspaceContextResponse
} from "@/services/tldw/domains/workspace-api"
import type {
  ACPWorkspaceContextContract,
  ACPWorkspaceContextInput,
  ActiveWorkspaceContextContract,
  WorkspaceEligibilityDecision,
  WorkspaceMembershipLabel,
  WorkspaceRecoveryContract,
  WorkspaceRecoverySeverity,
  WorkspaceSummaryContract
} from "./contracts"
import { EMPTY_WORKSPACE_SOURCE_SUMMARY } from "./contracts"

const WORKSPACES_MANAGER_HREF = "#/workspaces"

const normalizeOptionalId = (value?: string | null): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed || null
}

const fallbackWorkspaceLabel = (workspaceId: string): string =>
  `Workspace ${workspaceId}`

const workspaceLabel = (workspace: WorkspaceApiResponse): string => {
  const name = workspace.name?.trim()
  return name || fallbackWorkspaceLabel(workspace.id)
}

export const normalizeWorkspaceSummary = (
  workspace: WorkspaceApiResponse
): WorkspaceSummaryContract => ({
  id: workspace.id,
  name: workspace.name,
  label: workspaceLabel(workspace),
  profile: workspace.workspace_profile,
  archived: workspace.archived,
  deleted: workspace.deleted,
  studyMaterialsPolicy: workspace.study_materials_policy,
  statusLabel: workspace.deleted
    ? "Deleted"
    : workspace.archived
      ? "Archived"
      : "Active",
  version: workspace.version,
  lastModified: workspace.last_modified
})

export const normalizeWorkspaceMembershipLabel = (
  workspaceId?: string | null,
  options: {
    workspace?: WorkspaceApiResponse | null
  } = {}
): WorkspaceMembershipLabel => {
  const normalizedWorkspaceId = normalizeOptionalId(workspaceId)

  if (!normalizedWorkspaceId) {
    return {
      workspaceId: null,
      workspaceLabel: "Global",
      membershipLabel: "Global",
      tone: "neutral",
      isAuthoritative: true,
      reasonCode: null
    }
  }

  const workspace =
    options.workspace?.id === normalizedWorkspaceId ? options.workspace : null

  if (!workspace) {
    return {
      workspaceId: normalizedWorkspaceId,
      workspaceLabel: fallbackWorkspaceLabel(normalizedWorkspaceId),
      membershipLabel: "Unknown Workspace",
      tone: "warning",
      isAuthoritative: false,
      reasonCode: "workspace_missing"
    }
  }

  const summary = normalizeWorkspaceSummary(workspace)

  if (summary.deleted) {
    return {
      workspaceId: summary.id,
      workspaceLabel: summary.label,
      membershipLabel: `Deleted Workspace: ${summary.label}`,
      tone: "error",
      isAuthoritative: true,
      reasonCode: "workspace_deleted"
    }
  }

  if (summary.archived) {
    return {
      workspaceId: summary.id,
      workspaceLabel: summary.label,
      membershipLabel: `Archived Workspace: ${summary.label}`,
      tone: "warning",
      isAuthoritative: true,
      reasonCode: "workspace_archived"
    }
  }

  return {
    workspaceId: summary.id,
    workspaceLabel: summary.label,
    membershipLabel: `Workspace: ${summary.label}`,
    tone: "neutral",
    isAuthoritative: true,
    reasonCode: null
  }
}

export const createWorkspaceMembershipLookup = (
  workspaces: WorkspaceApiResponse[] = []
): ((workspaceId?: string | null) => WorkspaceMembershipLabel) => {
  const workspaceById = new Map(
    workspaces.map((workspace) => [workspace.id, workspace])
  )

  return (workspaceId?: string | null) => {
    const normalizedWorkspaceId = normalizeOptionalId(workspaceId)
    return normalizeWorkspaceMembershipLabel(normalizedWorkspaceId, {
      workspace: normalizedWorkspaceId
        ? workspaceById.get(normalizedWorkspaceId) ?? null
        : null
    })
  }
}

const recoveryMap: Record<string, WorkspaceRecoveryContract> = {
  allowed: {
    reasonCode: "allowed",
    severity: "info",
    message: "Workspace action is available.",
    nextStepLabel: null,
    nextStepHref: null
  },
  no_active_workspace: {
    reasonCode: "no_active_workspace",
    severity: "warning",
    message: "Select a server Workspace before using Workspace-specific actions.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  partial_context: {
    reasonCode: "partial_context",
    severity: "warning",
    message: "Server Workspace context partially resolved. Some Workspace actions may be unavailable.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  workspace_archived: {
    reasonCode: "workspace_archived",
    severity: "warning",
    message: "This server Workspace is archived. Restore it before making changes.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  workspace_missing: {
    reasonCode: "workspace_missing",
    severity: "error",
    message: "The selected server Workspace could not be found.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  workspace_context_error: {
    reasonCode: "workspace_context_error",
    severity: "error",
    message: "Server Workspace context is unavailable right now.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  workspace_project_root_missing: {
    reasonCode: "workspace_project_root_missing",
    severity: "warning",
    message: "Attach or provision a project root before using this Workspace action.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  workspace_mismatch: {
    reasonCode: "workspace_mismatch",
    severity: "warning",
    message: "This ACP session is attached to a different server Workspace than the active Workspace.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  aligned: {
    reasonCode: "aligned",
    severity: "info",
    message: "ACP session Workspace matches the active server Workspace.",
    nextStepLabel: null,
    nextStepHref: null
  },
  session_workspace_only: {
    reasonCode: "session_workspace_only",
    severity: "info",
    message: "This ACP session has a server Workspace, but no active server Workspace is selected.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  active_workspace_only: {
    reasonCode: "active_workspace_only",
    severity: "warning",
    message: "The active server Workspace is not attached to this ACP session.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  },
  acp_workspace_missing: {
    reasonCode: "acp_workspace_missing",
    severity: "warning",
    message: "No server Workspace is associated with this ACP session.",
    nextStepLabel: "Open Workspaces",
    nextStepHref: WORKSPACES_MANAGER_HREF
  }
}

export const resolveWorkspaceRecovery = (
  reasonCode: string | null | undefined,
  options: Partial<WorkspaceRecoveryContract> = {}
): WorkspaceRecoveryContract => {
  const normalizedReason = reasonCode || "workspace_context_error"
  const base =
    recoveryMap[normalizedReason] ?? {
      reasonCode: normalizedReason,
      severity: "warning" satisfies WorkspaceRecoverySeverity,
      message: "This Workspace action cannot complete until server Workspace context is resolved.",
      nextStepLabel: "Open Workspaces",
      nextStepHref: WORKSPACES_MANAGER_HREF
    }

  return {
    ...base,
    ...options,
    reasonCode: options.reasonCode ?? base.reasonCode
  }
}

export const normalizeActiveWorkspaceContext = (
  response: WorkspaceContextResponse | null,
  options: {
    state?: ActiveWorkspaceContextContract["state"]
    reasonCode?: string | null
  } = {}
): ActiveWorkspaceContextContract => {
  if (!response) {
    const state = options.state ?? "none"
    const reasonCode =
      options.reasonCode ??
      (state === "error"
        ? "workspace_context_error"
        : state === "missing"
          ? "workspace_missing"
          : "no_active_workspace")

    return {
      state,
      workspaceId: null,
      workspace: null,
      attentionState: null,
      resolution: null,
      projectRoot: null,
      sourceSummary: EMPTY_WORKSPACE_SOURCE_SUMMARY,
      capabilities: null,
      allowedActions: {},
      partialErrors: [],
      recovery: resolveWorkspaceRecovery(reasonCode)
    }
  }

  const resolutionStatus = response.resolution?.status
  const isArchived = response.workspace.archived || response.attention_state === "archived"
  const state: ActiveWorkspaceContextContract["state"] = isArchived
    ? "ready"
    : resolutionStatus === "partial"
      ? "partial"
      : resolutionStatus === "failed"
        ? "error"
        : "ready"

  const reasonCode =
    options.reasonCode ??
    (isArchived
      ? "workspace_archived"
      : state === "partial"
        ? "partial_context"
        : state === "error"
          ? "workspace_context_error"
          : "allowed")

  return {
    state: options.state ?? state,
    workspaceId: response.workspace_id,
    workspace: normalizeWorkspaceSummary(response.workspace),
    attentionState: response.attention_state,
    resolution: response.resolution,
    projectRoot: response.project_root,
    sourceSummary: response.sources.summary,
    capabilities: response.capabilities,
    allowedActions: response.allowed_actions,
    partialErrors: response.partial_errors,
    recovery: resolveWorkspaceRecovery(reasonCode)
  }
}

export const resolveWorkspaceActionEligibility = (
  action: string,
  allowedAction?: WorkspaceAllowedAction | null
): WorkspaceEligibilityDecision => {
  if (allowedAction?.allowed) {
    const recovery = resolveWorkspaceRecovery("allowed")
    return {
      action,
      allowed: true,
      reasonCode: "allowed",
      severity: recovery.severity,
      primaryMessage: recovery.message,
      nextStepLabel: recovery.nextStepLabel,
      nextStepHref: recovery.nextStepHref,
      recovery
    }
  }

  const reasonCode = allowedAction?.reason_code || "workspace_context_error"
  const recovery = resolveWorkspaceRecovery(reasonCode)
  return {
    action,
    allowed: false,
    reasonCode,
    severity: recovery.severity,
    primaryMessage: recovery.message,
    nextStepLabel: recovery.nextStepLabel,
    nextStepHref: recovery.nextStepHref,
    recovery
  }
}

export const compareACPWorkspaceContext = ({
  sessionWorkspaceId,
  activeWorkspaceId,
  sessionWorkspaceLabel,
  activeWorkspaceLabel
}: ACPWorkspaceContextInput): ACPWorkspaceContextContract => {
  const sessionId = normalizeOptionalId(sessionWorkspaceId)
  const activeId = normalizeOptionalId(activeWorkspaceId)
  const sessionLabel = sessionWorkspaceLabel?.trim() || sessionId
  const activeLabel = activeWorkspaceLabel?.trim() || activeId

  if (sessionId && activeId && sessionId === activeId) {
    return {
      state: "aligned",
      sessionWorkspaceId: sessionId,
      activeWorkspaceId: activeId,
      sessionWorkspaceLabel: sessionLabel,
      activeWorkspaceLabel: activeLabel,
      message: `ACP session is aligned with ${activeLabel ?? "the active server Workspace"}.`,
      recovery: resolveWorkspaceRecovery("aligned")
    }
  }

  if (sessionId && activeId) {
    return {
      state: "mismatch",
      sessionWorkspaceId: sessionId,
      activeWorkspaceId: activeId,
      sessionWorkspaceLabel: sessionLabel,
      activeWorkspaceLabel: activeLabel,
      message: `ACP session uses ${sessionLabel ?? sessionId}; active server Workspace is ${activeLabel ?? activeId}.`,
      recovery: resolveWorkspaceRecovery("workspace_mismatch")
    }
  }

  if (sessionId) {
    return {
      state: "session_only",
      sessionWorkspaceId: sessionId,
      activeWorkspaceId: null,
      sessionWorkspaceLabel: sessionLabel,
      activeWorkspaceLabel: null,
      message: `ACP session is attached to ${sessionLabel ?? sessionId}.`,
      recovery: resolveWorkspaceRecovery("session_workspace_only")
    }
  }

  if (activeId) {
    return {
      state: "active_only",
      sessionWorkspaceId: null,
      activeWorkspaceId: activeId,
      sessionWorkspaceLabel: null,
      activeWorkspaceLabel: activeLabel,
      message: `Active server Workspace is ${activeLabel ?? activeId}, but this ACP session is not attached to it.`,
      recovery: resolveWorkspaceRecovery("active_workspace_only")
    }
  }

  return {
    state: "missing",
    sessionWorkspaceId: null,
    activeWorkspaceId: null,
    sessionWorkspaceLabel: null,
    activeWorkspaceLabel: null,
    message: "No server Workspace is associated with this ACP session.",
    recovery: resolveWorkspaceRecovery("acp_workspace_missing")
  }
}
