import {
  buildResearchWorkspaceSnapshotStorageKey,
  classifyResearchWorkspaceLegacyStorageSurface,
  RESEARCH_WORKSPACE_RECONCILIATION_MARKER_PREFIX,
  WORKSPACE_STORAGE_KEY
} from "@/store/research-workspace-legacy-storage-inventory"
import { buildResearchWorkspaceMigrationTombstoneKey } from "@/store/workspace-migration"
import type { WorkspaceProfile } from "@/services/tldw/domains/workspace-api"

const WORKSPACE_SPLIT_INDEX_SCHEMA = "workspace_split_v1"

export type WorkspaceReconciliationStatus =
  | "linked"
  | "metadata_promoted"
  | "conflict"

export interface WorkspaceReconciliationMarkerV1 {
  schemaVersion: 1
  serverWorkspaceId: string
  serverName: string
  serverProfile: WorkspaceProfile
  linkedAt: string
  status: WorkspaceReconciliationStatus
  conflictState?: string
}

export type WorkspaceReconciliationDryRunState =
  | "local_only"
  | "server_row_exists"
  | "name_conflict"
  | "possible_duplicate"
  | "unsupported_local_payload"
  | "ready_to_create_metadata"

export interface LocalResearchWorkspaceEntry {
  localWorkspaceId: string
  name: string
  sourceCount: number | null
  hasSavedMetadata: boolean
  storageSurfaceIds: string[]
  tombstoned: boolean
  marker: WorkspaceReconciliationMarkerV1 | null
  unsupportedReason: string | null
}

export interface WorkspaceReconciliationServerWorkspace {
  id: string
  name: string | null
  workspace_profile: WorkspaceProfile
}

export interface WorkspaceReconciliationDryRunItem {
  localWorkspaceId: string
  name: string
  sourceCount: number | null
  state: WorkspaceReconciliationDryRunState
  conflictServerWorkspaceId: string | null
  conflictServerName: string | null
  marker: WorkspaceReconciliationMarkerV1 | null
  tombstoned: boolean
  actionable: boolean
  reason: string | null
}

export interface WorkspaceReconciliationDryRun {
  items: WorkspaceReconciliationDryRunItem[]
  localOnlyCount: number
  actionableCount: number
}

type LocalStorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">

interface DiscoverLocalResearchWorkspaceEntriesInput {
  discoveredLocalStorageKeys: string[]
  readLocalStorageValue: (key: string) => string | null
  storage?: LocalStorageLike
}

interface WorkspaceReconciliationDryRunInput {
  localEntries: LocalResearchWorkspaceEntry[]
  serverWorkspaces: WorkspaceReconciliationServerWorkspace[]
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const safeParseJson = (raw: string | null | undefined): unknown => {
  if (typeof raw !== "string" || raw.length === 0) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

const normalizeName = (name: string): string =>
  name.trim().replace(/\s+/g, " ").toLowerCase()

const displayNameForId = (workspaceId: string): string =>
  workspaceId.trim() || "Untitled Workspace"

export const buildWorkspaceReconciliationMarkerStorageKey = (
  localWorkspaceId: string
): string =>
  `${RESEARCH_WORKSPACE_RECONCILIATION_MARKER_PREFIX}:${encodeURIComponent(
    localWorkspaceId
  )}`

const isWorkspaceReconciliationMarker = (
  value: unknown
): value is WorkspaceReconciliationMarkerV1 => {
  if (!isRecord(value)) return false
  return (
    value.schemaVersion === 1 &&
    typeof value.serverWorkspaceId === "string" &&
    typeof value.serverName === "string" &&
    (value.serverProfile === "research" || value.serverProfile === "project") &&
    typeof value.linkedAt === "string" &&
    (value.status === "linked" ||
      value.status === "metadata_promoted" ||
      value.status === "conflict") &&
    (value.conflictState === undefined || typeof value.conflictState === "string")
  )
}

export const readWorkspaceReconciliationMarker = ({
  storage,
  localWorkspaceId
}: {
  storage: Pick<Storage, "getItem">
  localWorkspaceId: string
}): WorkspaceReconciliationMarkerV1 | null => {
  const parsed = safeParseJson(
    storage.getItem(buildWorkspaceReconciliationMarkerStorageKey(localWorkspaceId))
  )
  return isWorkspaceReconciliationMarker(parsed) ? parsed : null
}

export const writeWorkspaceReconciliationMarker = ({
  storage,
  localWorkspaceId,
  marker
}: {
  storage: Pick<Storage, "setItem">
  localWorkspaceId: string
  marker: WorkspaceReconciliationMarkerV1
}): void => {
  storage.setItem(
    buildWorkspaceReconciliationMarkerStorageKey(localWorkspaceId),
    JSON.stringify(marker)
  )
}

const hasMigrationTombstone = (
  storage: Pick<Storage, "getItem"> | undefined,
  localWorkspaceId: string
): boolean => {
  const parsed = safeParseJson(
    storage?.getItem(buildResearchWorkspaceMigrationTombstoneKey(localWorkspaceId))
  )
  if (!isRecord(parsed)) return false
  return (
    parsed.contentRetained === false &&
    parsed.legacyWorkspaceId === localWorkspaceId &&
    typeof parsed.migrationId === "string" &&
    parsed.migrationId.trim().length > 0
  )
}

const extractStateRecord = (raw: string | null): Record<string, unknown> | null => {
  const parsed = safeParseJson(raw)
  if (!isRecord(parsed)) return null
  if (
    parsed.schema === WORKSPACE_SPLIT_INDEX_SCHEMA &&
    typeof parsed.version === "number" &&
    isRecord(parsed.state)
  ) {
    return parsed.state
  }
  if (isRecord(parsed.state)) return parsed.state
  return parsed
}

const toNumberOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null

const upsertEntry = (
  entries: Map<string, LocalResearchWorkspaceEntry>,
  next: LocalResearchWorkspaceEntry
): void => {
  const existing = entries.get(next.localWorkspaceId)
  if (!existing) {
    entries.set(next.localWorkspaceId, next)
    return
  }

  entries.set(next.localWorkspaceId, {
    ...existing,
    name:
      existing.name === displayNameForId(existing.localWorkspaceId)
        ? next.name
        : existing.name,
    sourceCount: existing.sourceCount ?? next.sourceCount,
    hasSavedMetadata: existing.hasSavedMetadata || next.hasSavedMetadata,
    storageSurfaceIds: Array.from(
      new Set([...existing.storageSurfaceIds, ...next.storageSurfaceIds])
    ),
    tombstoned: existing.tombstoned || next.tombstoned,
    marker: existing.marker ?? next.marker,
    unsupportedReason: existing.unsupportedReason ?? next.unsupportedReason
  })
}

const entryFromSavedWorkspace = (
  workspace: unknown,
  storageSurfaceId: string,
  storage: LocalStorageLike | undefined
): LocalResearchWorkspaceEntry | null => {
  if (!isRecord(workspace) || typeof workspace.id !== "string") return null

  const name =
    typeof workspace.name === "string" && workspace.name.trim()
      ? workspace.name.trim()
      : displayNameForId(workspace.id)

  return {
    localWorkspaceId: workspace.id,
    name,
    sourceCount: toNumberOrNull(workspace.sourceCount),
    hasSavedMetadata: true,
    storageSurfaceIds: [storageSurfaceId],
    tombstoned: hasMigrationTombstone(storage, workspace.id),
    marker: storage
      ? readWorkspaceReconciliationMarker({
          storage,
          localWorkspaceId: workspace.id
        })
      : null,
    unsupportedReason: null
  }
}

const entryFromSnapshot = ({
  workspaceId,
  raw,
  storageSurfaceId,
  storage
}: {
  workspaceId: string
  raw: string | null
  storageSurfaceId: string
  storage: LocalStorageLike | undefined
}): LocalResearchWorkspaceEntry => {
  const parsed = safeParseJson(raw)
  if (!isRecord(parsed)) {
    return {
      localWorkspaceId: workspaceId,
      name: displayNameForId(workspaceId),
      sourceCount: null,
      hasSavedMetadata: false,
      storageSurfaceIds: [storageSurfaceId],
      tombstoned: hasMigrationTombstone(storage, workspaceId),
      marker: storage
        ? readWorkspaceReconciliationMarker({ storage, localWorkspaceId: workspaceId })
        : null,
      unsupportedReason: "snapshot_parse_failed"
    }
  }

  const name =
    typeof parsed.workspaceName === "string" && parsed.workspaceName.trim()
      ? parsed.workspaceName.trim()
      : displayNameForId(workspaceId)

  return {
    localWorkspaceId: workspaceId,
    name,
    sourceCount: Array.isArray(parsed.sources) ? parsed.sources.length : null,
    hasSavedMetadata: false,
    storageSurfaceIds: [storageSurfaceId],
    tombstoned: hasMigrationTombstone(storage, workspaceId),
    marker: storage
      ? readWorkspaceReconciliationMarker({ storage, localWorkspaceId: workspaceId })
      : null,
    unsupportedReason: null
  }
}

export const discoverLocalResearchWorkspaceEntries = ({
  discoveredLocalStorageKeys,
  readLocalStorageValue,
  storage
}: DiscoverLocalResearchWorkspaceEntriesInput): LocalResearchWorkspaceEntry[] => {
  const entries = new Map<string, LocalResearchWorkspaceEntry>()
  const rootState = discoveredLocalStorageKeys.includes(WORKSPACE_STORAGE_KEY)
    ? extractStateRecord(readLocalStorageValue(WORKSPACE_STORAGE_KEY))
    : null
  const rootSurface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "local_storage",
    key: WORKSPACE_STORAGE_KEY
  })

  if (rootState && rootSurface) {
    for (const workspace of [
      ...(Array.isArray(rootState.savedWorkspaces)
        ? rootState.savedWorkspaces
        : []),
      ...(Array.isArray(rootState.archivedWorkspaces)
        ? rootState.archivedWorkspaces
        : [])
    ]) {
      const entry = entryFromSavedWorkspace(workspace, rootSurface.id, storage)
      if (entry) upsertEntry(entries, entry)
    }
  }

  for (const key of discoveredLocalStorageKeys) {
    const surface = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key
    })
    if (!surface?.workspaceId || !key.endsWith(":snapshot")) continue

    upsertEntry(
      entries,
      entryFromSnapshot({
        workspaceId: surface.workspaceId,
        raw: readLocalStorageValue(key),
        storageSurfaceId: surface.id,
        storage
      })
    )
  }

  return Array.from(entries.values()).sort((a, b) =>
    a.name.localeCompare(b.name)
  )
}

export const listResearchWorkspaceLocalStorageKeys = (
  storage: Pick<Storage, "key" | "length">
): string[] => {
  const keys: string[] = []
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index)
    if (key) keys.push(key)
  }
  return keys
}

export const discoverLocalResearchWorkspaceEntriesFromBrowser = (
  storage: Storage | undefined =
    typeof window !== "undefined" ? window.localStorage : undefined
): LocalResearchWorkspaceEntry[] => {
  if (!storage) return []
  return discoverLocalResearchWorkspaceEntries({
    discoveredLocalStorageKeys: listResearchWorkspaceLocalStorageKeys(storage),
    readLocalStorageValue: (key) => storage.getItem(key),
    storage
  })
}

const findServerById = (
  serverWorkspaces: WorkspaceReconciliationServerWorkspace[],
  localWorkspaceId: string
): WorkspaceReconciliationServerWorkspace | undefined =>
  serverWorkspaces.find((workspace) => workspace.id === localWorkspaceId)

const findExactNameConflict = (
  serverWorkspaces: WorkspaceReconciliationServerWorkspace[],
  entry: LocalResearchWorkspaceEntry
): WorkspaceReconciliationServerWorkspace | undefined =>
  serverWorkspaces.find(
    (workspace) =>
      workspace.id !== entry.localWorkspaceId && workspace.name === entry.name
  )

const findPossibleDuplicate = (
  serverWorkspaces: WorkspaceReconciliationServerWorkspace[],
  entry: LocalResearchWorkspaceEntry
): WorkspaceReconciliationServerWorkspace | undefined => {
  const localName = normalizeName(entry.name)
  return serverWorkspaces.find(
    (workspace) =>
      workspace.id !== entry.localWorkspaceId &&
      workspace.name != null &&
      workspace.name !== entry.name &&
      normalizeName(workspace.name) === localName
  )
}

export const buildWorkspaceReconciliationDryRun = ({
  localEntries,
  serverWorkspaces
}: WorkspaceReconciliationDryRunInput): WorkspaceReconciliationDryRun => {
  const items = localEntries.map<WorkspaceReconciliationDryRunItem>((entry) => {
    const sameIdServer = findServerById(serverWorkspaces, entry.localWorkspaceId)
    const exactNameConflict = findExactNameConflict(serverWorkspaces, entry)
    const possibleDuplicate = findPossibleDuplicate(serverWorkspaces, entry)
    const state: WorkspaceReconciliationDryRunState =
      entry.tombstoned || entry.unsupportedReason
        ? "unsupported_local_payload"
        : sameIdServer
          ? "server_row_exists"
          : exactNameConflict
            ? "name_conflict"
            : possibleDuplicate
              ? "possible_duplicate"
              : entry.hasSavedMetadata
                ? "ready_to_create_metadata"
                : "local_only"
    const conflict = sameIdServer ?? exactNameConflict ?? possibleDuplicate ?? null
    const actionable =
      state === "ready_to_create_metadata" ||
      state === "name_conflict" ||
      state === "possible_duplicate" ||
      state === "server_row_exists"

    return {
      localWorkspaceId: entry.localWorkspaceId,
      name: entry.name,
      sourceCount: entry.sourceCount,
      state,
      conflictServerWorkspaceId: conflict?.id ?? null,
      conflictServerName: conflict?.name ?? null,
      marker: entry.marker,
      tombstoned: entry.tombstoned,
      actionable,
      reason:
        entry.tombstoned
          ? "migration_tombstone_present"
          : entry.unsupportedReason
    }
  })

  return {
    items,
    localOnlyCount: items.filter((item) => item.state === "local_only").length,
    actionableCount: items.filter((item) => item.actionable).length
  }
}
