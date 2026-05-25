import {
  WORKSPACE_BROADCAST_SYNC_FLAG,
  WORKSPACE_STORAGE_KEY
} from "@/store/workspace-events"

export const RESEARCH_WORKSPACE_INDEXEDDB_NAME = "tldw-workspace-storage"
export const RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE = "workspace-chat-sessions"
export const RESEARCH_WORKSPACE_INDEXEDDB_ARTIFACT_STORE =
  "workspace-artifact-payloads"

export type ResearchWorkspaceLegacyStorageKind =
  | "local_storage"
  | "indexeddb_store"
  | "broadcast_channel"

export type ResearchWorkspaceLegacyStorageClassification =
  | "content"
  | "metadata"
  | "ui_only"
  | "derived"
  | "obsolete"
  | "unsupported"

export type ResearchWorkspaceLegacyDeletionPolicy =
  | "requires_server_receipt"
  | "retain_local"
  | "delete_after_import"
  | "unknown_blocks_deletion"

export interface ResearchWorkspaceLegacyStorageInventoryItem {
  id: string
  kind: ResearchWorkspaceLegacyStorageKind
  classification: ResearchWorkspaceLegacyStorageClassification
  deletionPolicy: ResearchWorkspaceLegacyDeletionPolicy
  description: string
  contentClasses: string[]
  serverDestination: string | null
  authoritativeForMigration: boolean
  key?: string
  keyPattern?: string
  databaseName?: string
  storeName?: string
}

export interface ResearchWorkspaceLegacyStorageSurface
  extends ResearchWorkspaceLegacyStorageInventoryItem {
  key?: string
  workspaceId?: string
}

export interface ResearchWorkspaceUnknownLegacyStorageSurface {
  id: string
  kind: "local_storage" | "indexeddb_store"
  key?: string
  databaseName?: string
  storeName?: string
  deletionPolicy: "unknown_blocks_deletion"
}

export type ResearchWorkspaceLegacyStorageSurfaceInput =
  | {
      kind: "local_storage"
      key: string
    }
  | {
      kind: "indexeddb_store"
      databaseName: string
      storeName: string
    }

export interface ResearchWorkspaceLegacyDeletionEligibilityInput {
  discoveredLocalStorageKeys: string[]
  discoveredIndexedDbStores?: Array<{
    databaseName: string
    storeName: string
  }>
  manifestCoveredSurfaceIds: string[]
}

export interface ResearchWorkspaceLegacyDeletionEligibility {
  eligible: boolean
  blockingSurfaces: ResearchWorkspaceLegacyStorageSurface[]
  coveredContentSurfaces: ResearchWorkspaceLegacyStorageSurface[]
  retainedLocalSurfaces: ResearchWorkspaceLegacyStorageSurface[]
  unknownSurfaces: ResearchWorkspaceUnknownLegacyStorageSurface[]
}

const workspaceSplitKeyPattern = /^tldw-workspace:workspace:([^:]+):(snapshot|chat)$/

const decodeWorkspaceId = (encodedWorkspaceId: string): string => {
  try {
    return decodeURIComponent(encodedWorkspaceId)
  } catch {
    return encodedWorkspaceId
  }
}

export const buildResearchWorkspaceSnapshotStorageKey = (
  workspaceId: string
): string =>
  `${WORKSPACE_STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`

export const buildResearchWorkspaceChatStorageKey = (workspaceId: string): string =>
  `${WORKSPACE_STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:chat`

const inventory = [
  {
    id: "localStorage:tldw-workspace",
    kind: "local_storage",
    key: WORKSPACE_STORAGE_KEY,
    classification: "content",
    deletionPolicy: "requires_server_receipt",
    description:
      "Primary workspace persistence key. It may contain a legacy monolithic workspace payload or a split-storage index with active snapshot/chat fallbacks.",
    contentClasses: [
      "workspace_identity",
      "workspace_list",
      "sources",
      "folders",
      "notes",
      "artifacts",
      "chat_messages",
      "metadata"
    ],
    serverDestination:
      "Workspace core, workspace sources, notes, artifacts, chats, and migration receipt.",
    authoritativeForMigration: true
  },
  {
    id: "localStorage:tldw-workspace:workspace:*:snapshot",
    kind: "local_storage",
    keyPattern: "tldw-workspace:workspace:<workspace_id>:snapshot",
    classification: "content",
    deletionPolicy: "requires_server_receipt",
    description:
      "Split-storage per-workspace snapshot containing sources, selected source state, folders, notes, generated artifacts, banner metadata, and local layout fields.",
    contentClasses: [
      "workspace_identity",
      "sources",
      "folders",
      "selected_sources",
      "notes",
      "artifacts",
      "metadata"
    ],
    serverDestination:
      "Workspace core, workspace sources, folders/tags, notes, artifacts, and migration receipt.",
    authoritativeForMigration: true
  },
  {
    id: "localStorage:tldw-workspace:workspace:*:chat",
    kind: "local_storage",
    keyPattern: "tldw-workspace:workspace:<workspace_id>:chat",
    classification: "content",
    deletionPolicy: "requires_server_receipt",
    description:
      "Split-storage per-workspace chat session or IndexedDB chat pointer.",
    contentClasses: ["chat_messages", "chat_session_metadata"],
    serverDestination: "Workspace chat history and migration receipt.",
    authoritativeForMigration: true
  },
  {
    id: "indexedDB:tldw-workspace-storage/workspace-chat-sessions",
    kind: "indexeddb_store",
    databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
    storeName: RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE,
    classification: "content",
    deletionPolicy: "requires_server_receipt",
    description:
      "IndexedDB offload store for large workspace chat sessions referenced from split localStorage chat keys.",
    contentClasses: ["chat_messages", "chat_session_metadata"],
    serverDestination: "Workspace chat history and migration receipt.",
    authoritativeForMigration: true
  },
  {
    id: "indexedDB:tldw-workspace-storage/workspace-artifact-payloads",
    kind: "indexeddb_store",
    databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
    storeName: RESEARCH_WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
    classification: "content",
    deletionPolicy: "requires_server_receipt",
    description:
      "IndexedDB offload store for large generated artifact content/data referenced from split workspace snapshots.",
    contentClasses: ["artifacts", "artifact_payloads"],
    serverDestination: "Workspace artifacts, outputs, and migration receipt.",
    authoritativeForMigration: true
  },
  {
    id: "localStorage:tldw:research-workspace:pinned-workspaces:v1",
    kind: "local_storage",
    key: "tldw:research-workspace:pinned-workspaces:v1",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description:
      "Pinned workspace preference used by the Research Workspace header.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:research-workspace:recent-output-types:v1",
    kind: "local_storage",
    key: "tldw:research-workspace:recent-output-types:v1",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description:
      "Local preference for recent Studio output types. It does not contain generated output payloads.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:research-workspace:add-source-tab-usage:v1",
    kind: "local_storage",
    key: "tldw:research-workspace:add-source-tab-usage:v1",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description:
      "Add Source modal tab-use preference used to choose the default intake tab.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:research-workspace:onboarding-dismissed:v1",
    kind: "local_storage",
    key: "tldw:research-workspace:onboarding-dismissed:v1",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description: "Dismissal flag for the Research Workspace first-run panel.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:research-workspace:telemetry",
    kind: "local_storage",
    key: "tldw:research-workspace:telemetry",
    classification: "metadata",
    deletionPolicy: "retain_local",
    description:
      "Local-only Research Workspace product/quality telemetry counters and recent events. Content-free by policy.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:workspace:playground:telemetry",
    kind: "local_storage",
    key: "tldw:workspace:playground:telemetry",
    classification: "metadata",
    deletionPolicy: "delete_after_import",
    description:
      "Legacy telemetry import source. It is behavior metadata, not workspace content, and is not authoritative for migration completion.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:workspace_migrated",
    kind: "local_storage",
    key: "workspace_migrated",
    classification: "obsolete",
    deletionPolicy: "retain_local",
    description:
      "Old one-time migration flag from an earlier non-receipted helper. The true-move migration must ignore it as an authority.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:feature-rollout:workspace_split_storage_v1:enabled",
    kind: "local_storage",
    key: "tldw:feature-rollout:workspace_split_storage_v1:enabled",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description: "Local rollout flag for split workspace persistence.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "localStorage:tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled",
    kind: "local_storage",
    key: "tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled",
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description: "Local rollout flag for workspace IndexedDB payload offload.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: `localStorage:${WORKSPACE_BROADCAST_SYNC_FLAG}`,
    kind: "local_storage",
    key: WORKSPACE_BROADCAST_SYNC_FLAG,
    classification: "ui_only",
    deletionPolicy: "retain_local",
    description: "Local flag enabling multi-tab workspace storage broadcasts.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  },
  {
    id: "broadcastChannel:tldw-workspace-sync",
    kind: "broadcast_channel",
    key: "tldw-workspace-sync",
    classification: "derived",
    deletionPolicy: "retain_local",
    description:
      "Runtime-only BroadcastChannel name for multi-tab storage notifications. It is not persisted content.",
    contentClasses: [],
    serverDestination: null,
    authoritativeForMigration: false
  }
] as const satisfies readonly ResearchWorkspaceLegacyStorageInventoryItem[]

export const RESEARCH_WORKSPACE_LEGACY_STORAGE_INVENTORY = inventory

const cloneInventoryItem = (
  item: ResearchWorkspaceLegacyStorageInventoryItem,
  overrides: Partial<ResearchWorkspaceLegacyStorageSurface> = {}
): ResearchWorkspaceLegacyStorageSurface => ({
  ...item,
  contentClasses: [...item.contentClasses],
  ...overrides
})

const findExactLocalStorageItem = (
  key: string
): ResearchWorkspaceLegacyStorageInventoryItem | undefined =>
  inventory.find(
    (item) => item.kind === "local_storage" && "key" in item && item.key === key
  )

const classifySplitWorkspaceKey = (
  key: string
): ResearchWorkspaceLegacyStorageSurface | null => {
  const match = workspaceSplitKeyPattern.exec(key)
  if (!match) return null

  const [, encodedWorkspaceId, payloadKind] = match
  const inventoryId =
    payloadKind === "snapshot"
      ? "localStorage:tldw-workspace:workspace:*:snapshot"
      : "localStorage:tldw-workspace:workspace:*:chat"
  const inventoryItem = inventory.find((item) => item.id === inventoryId)
  if (!inventoryItem) return null

  return cloneInventoryItem(inventoryItem, {
    id: `localStorage:${key}`,
    key,
    workspaceId: decodeWorkspaceId(encodedWorkspaceId)
  })
}

const classifyIndexedDbStore = (
  databaseName: string,
  storeName: string
): ResearchWorkspaceLegacyStorageSurface | null => {
  const inventoryItem = inventory.find(
    (item) =>
      item.kind === "indexeddb_store" &&
      item.databaseName === databaseName &&
      item.storeName === storeName
  )
  return inventoryItem ? cloneInventoryItem(inventoryItem) : null
}

export const classifyResearchWorkspaceLegacyStorageSurface = (
  input: ResearchWorkspaceLegacyStorageSurfaceInput
): ResearchWorkspaceLegacyStorageSurface | null => {
  if (input.kind === "indexeddb_store") {
    return classifyIndexedDbStore(input.databaseName, input.storeName)
  }

  const exact = findExactLocalStorageItem(input.key)
  if (exact) {
    return cloneInventoryItem(exact, { key: input.key })
  }

  return classifySplitWorkspaceKey(input.key)
}

const isUnknownWorkspaceLocalStorageKey = (key: string): boolean =>
  key === WORKSPACE_STORAGE_KEY ||
  key.startsWith(`${WORKSPACE_STORAGE_KEY}:`) ||
  key.startsWith("tldw:research-workspace:") ||
  key.startsWith("tldw:workspace:playground:")

const buildUnknownLocalStorageSurface = (
  key: string
): ResearchWorkspaceUnknownLegacyStorageSurface => ({
  id: `unknown:localStorage:${key}`,
  kind: "local_storage",
  key,
  deletionPolicy: "unknown_blocks_deletion"
})

const buildUnknownIndexedDbSurface = (
  databaseName: string,
  storeName: string
): ResearchWorkspaceUnknownLegacyStorageSurface => ({
  id: `unknown:indexedDB:${databaseName}/${storeName}`,
  kind: "indexeddb_store",
  databaseName,
  storeName,
  deletionPolicy: "unknown_blocks_deletion"
})

const isBlockingClassification = (
  classification: ResearchWorkspaceLegacyStorageClassification
): boolean =>
  classification === "content" || classification === "unsupported"

export const evaluateResearchWorkspaceLegacyDeletionEligibility = ({
  discoveredLocalStorageKeys,
  discoveredIndexedDbStores = [],
  manifestCoveredSurfaceIds
}: ResearchWorkspaceLegacyDeletionEligibilityInput): ResearchWorkspaceLegacyDeletionEligibility => {
  const coveredSurfaceIds = new Set(manifestCoveredSurfaceIds)
  const blockingSurfaces: ResearchWorkspaceLegacyStorageSurface[] = []
  const coveredContentSurfaces: ResearchWorkspaceLegacyStorageSurface[] = []
  const retainedLocalSurfaces: ResearchWorkspaceLegacyStorageSurface[] = []
  const unknownSurfaces: ResearchWorkspaceUnknownLegacyStorageSurface[] = []

  const visitSurface = (surface: ResearchWorkspaceLegacyStorageSurface) => {
    if (isBlockingClassification(surface.classification)) {
      if (coveredSurfaceIds.has(surface.id)) {
        coveredContentSurfaces.push(surface)
      } else {
        blockingSurfaces.push(surface)
      }
      return
    }

    if (surface.deletionPolicy === "retain_local") {
      retainedLocalSurfaces.push(surface)
    }
  }

  for (const key of discoveredLocalStorageKeys) {
    const surface = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key
    })
    if (surface) {
      visitSurface(surface)
      continue
    }
    if (isUnknownWorkspaceLocalStorageKey(key)) {
      unknownSurfaces.push(buildUnknownLocalStorageSurface(key))
    }
  }

  for (const { databaseName, storeName } of discoveredIndexedDbStores) {
    const surface = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "indexeddb_store",
      databaseName,
      storeName
    })
    if (surface) {
      visitSurface(surface)
      continue
    }
    if (databaseName === RESEARCH_WORKSPACE_INDEXEDDB_NAME) {
      unknownSurfaces.push(buildUnknownIndexedDbSurface(databaseName, storeName))
    }
  }

  return {
    eligible: blockingSurfaces.length === 0 && unknownSurfaces.length === 0,
    blockingSurfaces,
    coveredContentSurfaces,
    retainedLocalSurfaces,
    unknownSurfaces
  }
}
