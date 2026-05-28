import {
  classifyResearchWorkspaceLegacyStorageSurface,
  evaluateResearchWorkspaceLegacyDeletionEligibility,
  type ResearchWorkspaceLegacyDeletionEligibility
} from "@/store/research-workspace-legacy-storage-inventory"

export const RESEARCH_WORKSPACE_MIGRATION_SCHEMA_VERSION = 1
export const RESEARCH_WORKSPACE_MIGRATION_SOURCE_PRODUCT =
  "research-workspace-webui"
export const RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX =
  "tldw:research-workspace:migration:tombstone"
const RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFLIGHT_PREFIX =
  "tldw:research-workspace:migration:tombstone-preflight"

export interface ResearchWorkspaceIndexedDbStoreRef {
  databaseName: string
  storeName: string
}

export interface ResearchWorkspaceMigrationPlanInput {
  targetWorkspaceId: string
  targetWorkspaceName: string
  discoveredLocalStorageKeys: string[]
  discoveredIndexedDbStores?: ResearchWorkspaceIndexedDbStoreRef[]
  readLocalStorageValue: (key: string) => Promise<string | null>
  readIndexedDbStorePayload?: (
    store: ResearchWorkspaceIndexedDbStoreRef
  ) => Promise<unknown>
  sourceProduct?: string
  generatedAt?: string
}

export interface ResearchWorkspaceMigrationChunkDeclaration {
  id: string
  sha256: string
  byte_count: number
  chunk_kind: string
}

export interface ResearchWorkspaceMigrationChunkPlan
  extends ResearchWorkspaceMigrationChunkDeclaration {
  surfaceId: string
  storageKind: "local_storage" | "indexeddb_store"
  key?: string
  databaseName?: string
  storeName?: string
}

export interface ResearchWorkspaceMigrationManifest extends Record<string, unknown> {
  schema_version: typeof RESEARCH_WORKSPACE_MIGRATION_SCHEMA_VERSION
  generated_at: string
  target_workspace_id: string
  target_workspace_name: string
  source_product: string
  covered_surface_ids: string[]
  retained_local_surface_ids: string[]
  unknown_surface_ids: string[]
  chunks: ResearchWorkspaceMigrationChunkDeclaration[]
}

export interface ResearchWorkspaceMigrationPlan {
  migrationId: string
  idempotencyKey: string
  manifestHash: string
  manifest: ResearchWorkspaceMigrationManifest
  chunks: ResearchWorkspaceMigrationChunkPlan[]
  declaredChunks: ResearchWorkspaceMigrationChunkDeclaration[]
  localDeletionEligibility: ResearchWorkspaceLegacyDeletionEligibility
}

export interface ResearchWorkspaceMigrationTombstoneInput {
  legacyWorkspaceId: string
  serverWorkspaceId: string
  migrationId: string
  deletedAt: string
}

export interface ResearchWorkspaceMigrationTombstone
  extends ResearchWorkspaceMigrationTombstoneInput {
  contentRetained: false
}

export interface ResearchWorkspaceMigrationSessionResponse {
  id: string
  status: string
  client_delete_eligible: boolean
  chunks?: unknown[]
}

export interface ResearchWorkspaceMigrationApi {
  createWorkspaceMigration: (body: {
    id: string
    idempotency_key: string
    target_workspace_id: string
    target_workspace_name: string
    source_product: string
    manifest_hash: string
    declared_chunks: ResearchWorkspaceMigrationChunkDeclaration[]
    manifest: ResearchWorkspaceMigrationManifest
    diagnostics: Record<string, unknown>
  }) => Promise<ResearchWorkspaceMigrationSessionResponse>
  putWorkspaceMigrationChunk: (
    migrationId: string,
    chunkId: string,
    body: {
      sha256: string
      byte_count: number
      chunk_kind: string
      metadata: Record<string, unknown>
    }
  ) => Promise<unknown>
  finalizeWorkspaceMigration: (
    migrationId: string,
    body: { manifest_hash: string }
  ) => Promise<ResearchWorkspaceMigrationSessionResponse>
  getWorkspaceMigration: (
    migrationId: string
  ) => Promise<ResearchWorkspaceMigrationSessionResponse>
  ackWorkspaceMigrationClientDelete: (
    migrationId: string,
    body: { acknowledged_manifest_hash: string }
  ) => Promise<unknown>
}

export type ResearchWorkspaceMigrationRunStatus =
  | "not_needed"
  | "blocked"
  | "finalized_not_delete_eligible"
  | "deleted"
  | "failed"

export interface ResearchWorkspaceMigrationRunInput
  extends ResearchWorkspaceMigrationPlanInput {
  api: ResearchWorkspaceMigrationApi
  legacyWorkspaceId?: string
  deleteLocalStorageValue?: (key: string) => Promise<void> | void
  writeLocalStorageValue?: (key: string, value: string) => Promise<void> | void
  deleteIndexedDbStorePayload?: (
    store: ResearchWorkspaceIndexedDbStoreRef
  ) => Promise<void> | void
  now?: () => string
}

export interface ResearchWorkspaceMigrationRunResult {
  status: ResearchWorkspaceMigrationRunStatus
  migrationId: string | null
  manifestHash: string | null
  serverMigration: ResearchWorkspaceMigrationSessionResponse | null
  localDeletionEligibility: ResearchWorkspaceLegacyDeletionEligibility | null
  deletedSurfaceIds: string[]
  message: string
  error?: unknown
}

const textEncoder = new TextEncoder()

const bytesToHex = (bytes: ArrayBuffer): string =>
  Array.from(new Uint8Array(bytes))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")

export const byteLengthText = (value: string): number =>
  textEncoder.encode(value).byteLength

export const sha256Text = async (value: string): Promise<string> => {
  const digest = await globalThis.crypto?.subtle?.digest(
    "SHA-256",
    textEncoder.encode(value)
  )
  if (!digest) {
    throw new Error("workspace-migration-sha256-unavailable")
  }
  return bytesToHex(digest)
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const stableStringify = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`
  }
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
      .join(",")}}`
  }
  return JSON.stringify(value)
}

const buildChunkId = async (
  surfaceId: string,
  payload: string,
  ordinal: number
): Promise<string> => {
  const hash = await sha256Text(`${surfaceId}:${payload}`)
  return `chunk-${ordinal + 1}-${hash.slice(0, 16)}`
}

const buildManifest = ({
  targetWorkspaceId,
  targetWorkspaceName,
  sourceProduct,
  generatedAt,
  chunks,
  localDeletionEligibility
}: {
  targetWorkspaceId: string
  targetWorkspaceName: string
  sourceProduct: string
  generatedAt: string
  chunks: ResearchWorkspaceMigrationChunkPlan[]
  localDeletionEligibility: ResearchWorkspaceLegacyDeletionEligibility
}): ResearchWorkspaceMigrationManifest => ({
  schema_version: RESEARCH_WORKSPACE_MIGRATION_SCHEMA_VERSION,
  generated_at: generatedAt,
  target_workspace_id: targetWorkspaceId,
  target_workspace_name: targetWorkspaceName,
  source_product: sourceProduct,
  covered_surface_ids: chunks.map((chunk) => chunk.surfaceId),
  retained_local_surface_ids: localDeletionEligibility.retainedLocalSurfaces.map(
    (surface) => surface.id
  ),
  unknown_surface_ids: localDeletionEligibility.unknownSurfaces.map(
    (surface) => surface.id
  ),
  chunks: chunks.map(({ id, sha256, byte_count, chunk_kind }) => ({
    id,
    sha256,
    byte_count,
    chunk_kind
  }))
})

const createLocalStorageChunk = async (
  key: string,
  payload: string,
  ordinal: number
): Promise<ResearchWorkspaceMigrationChunkPlan | null> => {
  const surface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "local_storage",
    key
  })
  if (!surface || surface.classification !== "content") return null

  return {
    id: await buildChunkId(surface.id, payload, ordinal),
    surfaceId: surface.id,
    storageKind: "local_storage",
    key,
    sha256: await sha256Text(payload),
    byte_count: byteLengthText(payload),
    chunk_kind: "workspace_bundle"
  }
}

const createIndexedDbChunk = async (
  store: ResearchWorkspaceIndexedDbStoreRef,
  payload: unknown,
  ordinal: number
): Promise<ResearchWorkspaceMigrationChunkPlan | null> => {
  const surface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "indexeddb_store",
    databaseName: store.databaseName,
    storeName: store.storeName
  })
  if (!surface || surface.classification !== "content") return null

  const serializedPayload = stableStringify(payload)
  return {
    id: await buildChunkId(surface.id, serializedPayload, ordinal),
    surfaceId: surface.id,
    storageKind: "indexeddb_store",
    databaseName: store.databaseName,
    storeName: store.storeName,
    sha256: await sha256Text(serializedPayload),
    byte_count: byteLengthText(serializedPayload),
    chunk_kind: "indexeddb_store"
  }
}

export const buildResearchWorkspaceMigrationPlan = async ({
  targetWorkspaceId,
  targetWorkspaceName,
  discoveredLocalStorageKeys,
  discoveredIndexedDbStores = [],
  readLocalStorageValue,
  readIndexedDbStorePayload,
  sourceProduct = RESEARCH_WORKSPACE_MIGRATION_SOURCE_PRODUCT,
  generatedAt = new Date(0).toISOString()
}: ResearchWorkspaceMigrationPlanInput): Promise<ResearchWorkspaceMigrationPlan> => {
  const chunks: ResearchWorkspaceMigrationChunkPlan[] = []

  for (const key of discoveredLocalStorageKeys) {
    const payload = await readLocalStorageValue(key)
    if (payload == null) continue
    const chunk = await createLocalStorageChunk(key, payload, chunks.length)
    if (chunk) chunks.push(chunk)
  }

  if (readIndexedDbStorePayload) {
    for (const store of discoveredIndexedDbStores) {
      const payload = await readIndexedDbStorePayload(store)
      if (payload == null) continue
      const chunk = await createIndexedDbChunk(store, payload, chunks.length)
      if (chunk) chunks.push(chunk)
    }
  }

  const localDeletionEligibility =
    evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys,
      discoveredIndexedDbStores,
      manifestCoveredSurfaceIds: chunks.map((chunk) => chunk.surfaceId)
    })

  const manifest = buildManifest({
    targetWorkspaceId,
    targetWorkspaceName,
    sourceProduct,
    generatedAt,
    chunks,
    localDeletionEligibility
  })
  const manifestHash = await sha256Text(stableStringify(manifest))
  const migrationId = `research-workspace-${targetWorkspaceId}-${manifestHash.slice(
    0,
    16
  )}`

  return {
    migrationId,
    idempotencyKey: `${migrationId}:${manifestHash}`,
    manifestHash,
    manifest,
    chunks,
    declaredChunks: manifest.chunks,
    localDeletionEligibility
  }
}

export const buildResearchWorkspaceMigrationTombstoneKey = (
  legacyWorkspaceId: string
): string =>
  `${RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX}:${encodeURIComponent(
    legacyWorkspaceId
  )}`

export const buildResearchWorkspaceMigrationTombstone = (
  input: ResearchWorkspaceMigrationTombstoneInput
): ResearchWorkspaceMigrationTombstone => ({
  ...input,
  contentRetained: false
})

const buildChunkMetadata = (
  chunk: ResearchWorkspaceMigrationChunkPlan
): Record<string, unknown> => ({
  surface_id: chunk.surfaceId,
  storage_kind: chunk.storageKind,
  key: chunk.key,
  database_name: chunk.databaseName,
  store_name: chunk.storeName
})

const canDeleteCoveredLocalPayloads = ({
  chunks,
  deleteLocalStorageValue,
  deleteIndexedDbStorePayload
}: {
  chunks: ResearchWorkspaceMigrationChunkPlan[]
  deleteLocalStorageValue?: (key: string) => Promise<void> | void
  deleteIndexedDbStorePayload?: (
    store: ResearchWorkspaceIndexedDbStoreRef
  ) => Promise<void> | void
}): boolean => {
  for (const chunk of chunks) {
    if (chunk.storageKind === "local_storage") {
      if (!chunk.key || !deleteLocalStorageValue) return false
      continue
    }

    if (chunk.storageKind === "indexeddb_store") {
      if (
        !chunk.databaseName ||
        !chunk.storeName ||
        !deleteIndexedDbStorePayload
      ) {
        return false
      }
    }
  }

  return true
}

const deleteCoveredLocalPayloads = async ({
  chunks,
  deleteLocalStorageValue,
  deleteIndexedDbStorePayload
}: {
  chunks: ResearchWorkspaceMigrationChunkPlan[]
  deleteLocalStorageValue?: (key: string) => Promise<void> | void
  deleteIndexedDbStorePayload?: (
    store: ResearchWorkspaceIndexedDbStoreRef
  ) => Promise<void> | void
}): Promise<string[] | null> => {
  if (
    !canDeleteCoveredLocalPayloads({
      chunks,
      deleteLocalStorageValue,
      deleteIndexedDbStorePayload
    })
  ) {
    return null
  }

  const deletedSurfaceIds: string[] = []

  for (const chunk of chunks) {
    if (chunk.storageKind === "local_storage") {
      if (!chunk.key || !deleteLocalStorageValue) return null
      await deleteLocalStorageValue(chunk.key)
      deletedSurfaceIds.push(chunk.surfaceId)
      continue
    }

    if (
      chunk.storageKind === "indexeddb_store" &&
      chunk.databaseName &&
      chunk.storeName
    ) {
      if (!deleteIndexedDbStorePayload) return null
      await deleteIndexedDbStorePayload({
        databaseName: chunk.databaseName,
        storeName: chunk.storeName
      })
      deletedSurfaceIds.push(chunk.surfaceId)
    }
  }

  return deletedSurfaceIds
}

export const runResearchWorkspaceMigration = async ({
  api,
  legacyWorkspaceId,
  deleteLocalStorageValue,
  writeLocalStorageValue,
  deleteIndexedDbStorePayload,
  now = () => new Date().toISOString(),
  ...planInput
}: ResearchWorkspaceMigrationRunInput): Promise<ResearchWorkspaceMigrationRunResult> => {
  let plan: ResearchWorkspaceMigrationPlan | null = null
  try {
    plan = await buildResearchWorkspaceMigrationPlan(planInput)

    if (plan.chunks.length === 0) {
      return {
        status: plan.localDeletionEligibility.eligible ? "not_needed" : "blocked",
        migrationId: null,
        manifestHash: plan.manifestHash,
        serverMigration: null,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: plan.localDeletionEligibility.eligible
          ? "No legacy Research Workspace content was discovered."
          : "Legacy Research Workspace storage includes unknown or uncovered content."
      }
    }

    await api.createWorkspaceMigration({
      id: plan.migrationId,
      idempotency_key: plan.idempotencyKey,
      target_workspace_id: planInput.targetWorkspaceId,
      target_workspace_name: planInput.targetWorkspaceName,
      source_product:
        planInput.sourceProduct || RESEARCH_WORKSPACE_MIGRATION_SOURCE_PRODUCT,
      manifest_hash: plan.manifestHash,
      declared_chunks: plan.declaredChunks,
      manifest: plan.manifest,
      diagnostics: {}
    })

    for (const chunk of plan.chunks) {
      await api.putWorkspaceMigrationChunk(plan.migrationId, chunk.id, {
        sha256: chunk.sha256,
        byte_count: chunk.byte_count,
        chunk_kind: chunk.chunk_kind,
        metadata: buildChunkMetadata(chunk)
      })
    }

    await api.finalizeWorkspaceMigration(plan.migrationId, {
      manifest_hash: plan.manifestHash
    })
    const serverMigration = await api.getWorkspaceMigration(plan.migrationId)

    if (!plan.localDeletionEligibility.eligible) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server receipt was saved, but local deletion is blocked by the legacy inventory gate."
      }
    }

    if (!serverMigration.client_delete_eligible) {
      return {
        status: "finalized_not_delete_eligible",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server receipt was saved. Local data is retained until server deletion eligibility is available."
      }
    }

    if (!writeLocalStorageValue) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server deletion eligibility is available, but local deletion dependencies are not configured."
      }
    }

    const hasCoveredLocalPayloads = plan.chunks.length > 0
    if (hasCoveredLocalPayloads && !deleteLocalStorageValue) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server deletion eligibility is available, but local deletion dependencies are not configured."
      }
    }

    if (
      !canDeleteCoveredLocalPayloads({
        chunks: plan.chunks,
        deleteLocalStorageValue,
        deleteIndexedDbStorePayload
      })
    ) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server deletion eligibility is available, but local deletion dependencies are not configured."
      }
    }

    const tombstone = buildResearchWorkspaceMigrationTombstone({
      legacyWorkspaceId: legacyWorkspaceId || planInput.targetWorkspaceId,
      serverWorkspaceId: planInput.targetWorkspaceId,
      migrationId: plan.migrationId,
      deletedAt: now()
    })
    const tombstoneKey = buildResearchWorkspaceMigrationTombstoneKey(
      tombstone.legacyWorkspaceId
    )
    const tombstonePayload = JSON.stringify(tombstone)
    const preflightKey = `${RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFLIGHT_PREFIX}:${encodeURIComponent(
      tombstone.legacyWorkspaceId
    )}`
    if (hasCoveredLocalPayloads && deleteLocalStorageValue) {
      await writeLocalStorageValue(preflightKey, tombstonePayload)
    }

    const deletedSurfaceIds = await deleteCoveredLocalPayloads({
      chunks: plan.chunks,
      deleteLocalStorageValue,
      deleteIndexedDbStorePayload
    })

    if (!deletedSurfaceIds) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server deletion eligibility is available, but local deletion dependencies are not configured."
      }
    }

    await writeLocalStorageValue(tombstoneKey, tombstonePayload)
    if (hasCoveredLocalPayloads && deleteLocalStorageValue) {
      await deleteLocalStorageValue(preflightKey)
    }
    await api.ackWorkspaceMigrationClientDelete(plan.migrationId, {
      acknowledged_manifest_hash: plan.manifestHash
    })

    return {
      status: "deleted",
      migrationId: plan.migrationId,
      manifestHash: plan.manifestHash,
      serverMigration,
      localDeletionEligibility: plan.localDeletionEligibility,
      deletedSurfaceIds,
      message: "Legacy Research Workspace content was migrated and local content payloads were deleted."
    }
  } catch (error) {
    return {
      status: "failed",
      migrationId: plan?.migrationId ?? null,
      manifestHash: plan?.manifestHash ?? null,
      serverMigration: null,
      localDeletionEligibility: plan?.localDeletionEligibility ?? null,
      deletedSurfaceIds: [],
      message: "Research Workspace migration failed before local deletion.",
      error
    }
  }
}
