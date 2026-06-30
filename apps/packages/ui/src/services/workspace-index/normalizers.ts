import type {
  WorkspaceActivityEvent,
  WorkspaceIndexCounts,
  WorkspaceIndexPathOptions,
  WorkspaceIndexResourceGroup,
  WorkspaceIndexResourceItem,
  WorkspaceIndexResourceSummary,
  WorkspaceIndexRuntimeSummary,
  WorkspaceIndexSnapshot,
  WorkspaceIndexWarning,
  WorkspaceIndexWarningSeverity,
  WorkspaceIndexWorkspaceSummary
} from "./contracts"

const warningSeverities = new Set<WorkspaceIndexWarningSeverity>([
  "info",
  "warning",
  "error"
])

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

const asRecord = (value: unknown): Record<string, unknown> =>
  isRecord(value) ? value : {}

const asString = (value: unknown, fallback = ""): string =>
  typeof value === "string" ? value : fallback

const asOptionalString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  return value.length > 0 ? value : undefined
}

const asNumber = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback

const asBoolean = (value: unknown): boolean =>
  value === true || value === 1 || value === "1" || value === "true" || value === "True"

const asArray = (value: unknown): unknown[] =>
  Array.isArray(value) ? value : []

const asCountRecord = (value: unknown): Record<string, number> => {
  const record = asRecord(value)
  const counts: Record<string, number> = {}
  for (const [key, rawCount] of Object.entries(record)) {
    const count = asNumber(rawCount, Number.NaN)
    if (key && Number.isFinite(count) && count >= 0) {
      counts[key] = count
    }
  }
  return counts
}

const normalizeCounts = (value: unknown): WorkspaceIndexCounts => {
  const record = asRecord(value)
  return {
    total: Math.max(0, asNumber(record.total, 0)),
    byResourceType: asCountRecord(record.by_resource_type ?? record.byResourceType),
    byRole: asCountRecord(record.by_role ?? record.byRole)
  }
}

const normalizeWorkspace = (
  value: unknown,
  fallbackWorkspaceId: string
): WorkspaceIndexWorkspaceSummary => {
  const record = asRecord(value)
  const id = asString(record.id, fallbackWorkspaceId)
  return {
    id,
    name: asOptionalString(record.name),
    profile: asOptionalString(record.workspace_profile ?? record.profile),
    archived: asBoolean(record.archived),
    deleted: asBoolean(record.deleted),
    version: Math.max(1, asNumber(record.version, 1))
  }
}

const normalizeSummary = (value: unknown): WorkspaceIndexResourceSummary | undefined => {
  if (!isRecord(value)) return undefined
  return {
    title: asOptionalString(value.title),
    subtitle: asOptionalString(value.subtitle),
    href: asOptionalString(value.href),
    updatedAt: asOptionalString(value.updated_at ?? value.updatedAt),
    state: asString(value.state, "unknown"),
    metadata: asRecord(value.metadata)
  }
}

const normalizeResourceItem = (value: unknown): WorkspaceIndexResourceItem => {
  const record = asRecord(value)
  return {
    workspaceId: asString(record.workspace_id ?? record.workspaceId),
    resourceType: asString(record.resource_type ?? record.resourceType),
    resourceId: asString(record.resource_id ?? record.resourceId),
    role: asString(record.role, "member"),
    label: asOptionalString(record.label),
    transferPolicy: asString(record.transfer_policy ?? record.transferPolicy, "link"),
    provenance: asRecord(record.provenance),
    metadata: asRecord(record.metadata),
    summary: normalizeSummary(record.summary),
    createdAt: asString(record.created_at ?? record.createdAt),
    updatedAt: asString(record.updated_at ?? record.updatedAt),
    version: Math.max(1, asNumber(record.version, 1)),
    deleted: asBoolean(record.deleted)
  }
}

const normalizeResourceGroup = (value: unknown): WorkspaceIndexResourceGroup => {
  const record = asRecord(value)
  const ownerSurface = asRecord(record.owner_surface ?? record.ownerSurface)
  return {
    resourceType: asString(record.resource_type ?? record.resourceType),
    count: Math.max(0, asNumber(record.count, 0)),
    ownerSurface: {
      label: asString(ownerSurface.label),
      href: asString(ownerSurface.href)
    },
    items: asArray(record.items).map(normalizeResourceItem),
    nextCursor: asOptionalString(record.next_cursor ?? record.nextCursor)
  }
}

const normalizeRuntimeSummary = (value: unknown): WorkspaceIndexRuntimeSummary => {
  const record = asRecord(value)
  return {
    total: Math.max(0, asNumber(record.total, 0)),
    byKind: asCountRecord(record.by_kind ?? record.byKind),
    byStatus: asCountRecord(record.by_status ?? record.byStatus),
    bindings: asArray(record.bindings)
      .map(asRecord)
      .filter((binding) => Object.keys(binding).length > 0)
  }
}

const normalizeWarningSeverity = (value: unknown): WorkspaceIndexWarningSeverity => {
  const severity = asString(value) as WorkspaceIndexWarningSeverity
  return warningSeverities.has(severity) ? severity : "warning"
}

const normalizeWarning = (value: unknown): WorkspaceIndexWarning => {
  const record = asRecord(value)
  const reasonCode = asString(record.reason_code ?? record.reasonCode, "unknown")
  return {
    severity: normalizeWarningSeverity(record.severity),
    reasonCode,
    message: asString(record.message, reasonCode),
    resourceType: asOptionalString(record.resource_type ?? record.resourceType),
    resourceId: asOptionalString(record.resource_id ?? record.resourceId),
    actionHref: asOptionalString(record.action_href ?? record.actionHref)
  }
}

const normalizeActivityEvent = (value: unknown): WorkspaceActivityEvent => {
  const record = asRecord(value)
  return {
    workspaceId: asString(record.workspace_id ?? record.workspaceId),
    eventId: asString(record.event_id ?? record.eventId),
    eventType: asString(record.event_type ?? record.eventType),
    category: asString(record.category),
    actorUserId: asOptionalString(record.actor_user_id ?? record.actorUserId),
    resourceType: asOptionalString(record.resource_type ?? record.resourceType),
    resourceId: asOptionalString(record.resource_id ?? record.resourceId),
    summary: asOptionalString(record.summary),
    metadata: asRecord(record.metadata),
    createdAt: asString(record.created_at ?? record.createdAt),
    version: Math.max(1, asNumber(record.version, 1))
  }
}

export const normalizeWorkspaceIndexResponse = (value: unknown): WorkspaceIndexSnapshot => {
  const record = asRecord(value)
  const workspaceId = asString(record.workspace_id ?? record.workspaceId)
  return {
    workspaceId,
    schemaVersion: Math.max(1, asNumber(record.schema_version ?? record.schemaVersion, 1)),
    generatedAt: asString(record.generated_at ?? record.generatedAt),
    workspace: normalizeWorkspace(record.workspace, workspaceId),
    membershipSummary: normalizeCounts(record.membership_summary ?? record.membershipSummary),
    resourceGroups: asArray(record.resource_groups ?? record.resourceGroups).map(normalizeResourceGroup),
    runtimeSummary: normalizeRuntimeSummary(record.runtime_summary ?? record.runtimeSummary),
    warnings: asArray(record.warnings).map(normalizeWarning),
    recentActivity: asArray(record.recent_activity ?? record.recentActivity).map(normalizeActivityEvent),
    partialErrors: asArray(record.partial_errors ?? record.partialErrors).map(asRecord)
  }
}

const boundedLimit = (value: number | undefined, fallback: number, max: number): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback
  return Math.max(1, Math.min(Math.trunc(value), max))
}

export const buildWorkspaceIndexPath = (
  workspaceId: string,
  options: WorkspaceIndexPathOptions = {}
): string => {
  const query = new URLSearchParams({
    group_limit: String(boundedLimit(options.groupLimit, 5, 25)),
    activity_limit: String(boundedLimit(options.activityLimit, 25, 100))
  })
  return `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/index?${query.toString()}`
}
