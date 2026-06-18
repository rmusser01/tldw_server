export type WorkspaceIndexWarningSeverity = "info" | "warning" | "error"

export interface WorkspaceIndexCounts {
  total: number
  byResourceType: Record<string, number>
  byRole: Record<string, number>
}

export interface WorkspaceIndexOwnerSurface {
  label: string
  href: string
}

export interface WorkspaceIndexResourceSummary {
  title?: string
  subtitle?: string
  href?: string
  updatedAt?: string
  state: string
  metadata: Record<string, unknown>
}

export interface WorkspaceIndexResourceItem {
  workspaceId: string
  resourceType: string
  resourceId: string
  role: string
  label?: string
  transferPolicy: string
  provenance: Record<string, unknown>
  metadata: Record<string, unknown>
  summary?: WorkspaceIndexResourceSummary
  createdAt: string
  updatedAt: string
  version: number
  deleted: boolean
}

export interface WorkspaceIndexResourceGroup {
  resourceType: string
  count: number
  ownerSurface: WorkspaceIndexOwnerSurface
  items: WorkspaceIndexResourceItem[]
  nextCursor?: string
}

export interface WorkspaceIndexRuntimeSummary {
  total: number
  byKind: Record<string, number>
  byStatus: Record<string, number>
  bindings: Array<Record<string, unknown>>
}

export interface WorkspaceIndexWarning {
  severity: WorkspaceIndexWarningSeverity
  reasonCode: string
  message: string
  resourceType?: string
  resourceId?: string
  actionHref?: string
}

export interface WorkspaceActivityEvent {
  workspaceId: string
  eventId: string
  eventType: string
  category: string
  actorUserId?: string
  resourceType?: string
  resourceId?: string
  summary?: string
  metadata: Record<string, unknown>
  createdAt: string
  version: number
}

export interface WorkspaceIndexWorkspaceSummary {
  id: string
  name?: string
  profile?: string
  archived: boolean
  deleted: boolean
  version: number
}

export interface WorkspaceIndexSnapshot {
  workspaceId: string
  schemaVersion: number
  generatedAt: string
  workspace: WorkspaceIndexWorkspaceSummary
  membershipSummary: WorkspaceIndexCounts
  resourceGroups: WorkspaceIndexResourceGroup[]
  runtimeSummary: WorkspaceIndexRuntimeSummary
  warnings: WorkspaceIndexWarning[]
  recentActivity: WorkspaceActivityEvent[]
  partialErrors: Array<Record<string, unknown>>
}

export interface WorkspaceIndexPathOptions {
  groupLimit?: number
  activityLimit?: number
}
