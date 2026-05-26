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

export interface ResearchWorkspaceMigrationManifest {
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
