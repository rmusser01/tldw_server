import {
  classifyResearchWorkspaceLegacyStorageSurface,
  evaluateResearchWorkspaceLegacyDeletionEligibility,
  type ResearchWorkspaceLegacyDeletionEligibility
} from "@/store/research-workspace-legacy-storage-inventory"
import { isWorkspaceChatSessionKeyForWorkspace } from "@/store/workspace-chat-session-key"

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
  serverWorkspace?: unknown
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
  eligibility: ResearchWorkspaceMigrationEligibility
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
  getCurrentWorkspace?: () => {
    workspaceId: string | null
    serverWorkspace: unknown
  }
  subscribeToWorkspaceChanges?: (listener: () => void) => () => void
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

const containsWorkspaceOffloadReference = (value: unknown): boolean => {
  if (Array.isArray(value)) return value.some(containsWorkspaceOffloadReference)
  if (!isRecord(value)) return false
  return value.offloadType === "workspace_chat_session_v1" ||
    value.offloadType === "workspace_artifact_payload_v1" ||
    Object.values(value).some(containsWorkspaceOffloadReference)
}

type ResearchWorkspaceMigrationEligibility = "legacy" | "canonical" | "blocked"

/** Canonical provenance excludes migration; it never grants owner authority. */
export const getResearchWorkspaceMigrationEligibility = (
  workspaceId: string,
  serverWorkspace: unknown
): ResearchWorkspaceMigrationEligibility => {
  if (!workspaceId.trim()) return "blocked"
  if (serverWorkspace == null) return "legacy"
  return isRecord(serverWorkspace) &&
    typeof serverWorkspace.scopeKey === "string" &&
    serverWorkspace.scopeKey.trim() &&
    isRecord(serverWorkspace.metadata) &&
    serverWorkspace.metadata.id === workspaceId
    ? "canonical"
    : "blocked"
}

const getLocalPayloadMigrationEligibility = (
  workspaceId: string,
  key: string,
  payload: string
): ResearchWorkspaceMigrationEligibility => {
  const surface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "local_storage",
    key
  })
  if (!surface || surface.classification !== "content") return "legacy"
  let parsed: unknown
  try {
    parsed = JSON.parse(payload)
  } catch {
    return "blocked"
  }
  if (!isRecord(parsed) || containsWorkspaceOffloadReference(parsed)) return "blocked"
  if (surface.workspaceId) {
    if (!isWorkspaceChatSessionKeyForWorkspace(surface.workspaceId, workspaceId)) {
      return "blocked"
    }
    if (key.endsWith(":snapshot") && parsed.workspaceId !== workspaceId) return "blocked"
    if (parsed.workspaceId != null && parsed.workspaceId !== workspaceId) return "blocked"
    return getResearchWorkspaceMigrationEligibility(workspaceId, parsed.serverWorkspace)
  }

  if ("state" in parsed && !isRecord(parsed.state)) return "blocked"
  const state = isRecord(parsed.state) ? parsed.state : parsed
  const legacyWorkspaces = state.workspaces
  for (const id of [state.workspaceId, state.activeWorkspaceId]) {
    if (id != null && id !== workspaceId) return "blocked"
  }
  const activeId = state.workspaceId ?? state.activeWorkspaceId ?? (
    Array.isArray(legacyWorkspaces) && legacyWorkspaces.length === 1 &&
    isRecord(legacyWorkspaces[0]) ? legacyWorkspaces[0].id : null
  )
  if (activeId !== workspaceId) return "blocked"
  for (const field of ["workspaceIds", "savedWorkspaces", "archivedWorkspaces", "workspaces"]) {
    const entries = state[field]
    if (entries != null && (!Array.isArray(entries) || entries.some((entry) =>
      (field === "workspaceIds" ? entry : isRecord(entry) ? entry.id : null) !== workspaceId))) {
      return "blocked"
    }
  }
  let eligibility = getResearchWorkspaceMigrationEligibility(workspaceId, state.serverWorkspace)
  for (const field of ["workspaceSnapshots", "workspaceChatSessions"]) {
    const entries = state[field]
    if (entries == null) continue
    if (!isRecord(entries)) return "blocked"
    for (const [id, entry] of Object.entries(entries)) {
      if (!isWorkspaceChatSessionKeyForWorkspace(id, workspaceId) || !isRecord(entry) ||
          (field === "workspaceSnapshots" && (id !== workspaceId || entry.workspaceId !== workspaceId)) ||
          (entry.workspaceId != null && entry.workspaceId !== workspaceId)) {
        return "blocked"
      }
      const entryEligibility = getResearchWorkspaceMigrationEligibility(workspaceId, entry.serverWorkspace)
      if (entryEligibility === "blocked") return "blocked"
      if (entryEligibility === "canonical" && eligibility === "legacy") eligibility = "canonical"
    }
  }
  return eligibility
}

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

export const buildResearchWorkspaceMigrationPlan = async ({
  targetWorkspaceId,
  targetWorkspaceName,
  serverWorkspace,
  discoveredLocalStorageKeys,
  discoveredIndexedDbStores = [],
  readLocalStorageValue,
  sourceProduct = RESEARCH_WORKSPACE_MIGRATION_SOURCE_PRODUCT,
  generatedAt = new Date(0).toISOString()
}: ResearchWorkspaceMigrationPlanInput): Promise<ResearchWorkspaceMigrationPlan> => {
  const chunks: ResearchWorkspaceMigrationChunkPlan[] = []
  let eligibility = getResearchWorkspaceMigrationEligibility(
    targetWorkspaceId,
    serverWorkspace
  )
  const payloads = new Map<string, string>()
  if (eligibility === "legacy") {
    for (const key of discoveredLocalStorageKeys) {
      const payload = await readLocalStorageValue(key)
      if (payload == null) continue
      payloads.set(key, payload)
      const payloadEligibility = getLocalPayloadMigrationEligibility(
        targetWorkspaceId,
        key,
        payload
      )
      if (payloadEligibility === "blocked") eligibility = "blocked"
      else if (payloadEligibility === "canonical" && eligibility === "legacy") eligibility = "canonical"
    }
  }
  // Store-level exports/deletes cannot isolate one workspace in shared offload stores.
  if (eligibility === "legacy" && discoveredIndexedDbStores.length > 0) {
    eligibility = "blocked"
  }
  if (eligibility === "legacy") {
    for (const [key, payload] of payloads) {
      const chunk = await createLocalStorageChunk(key, payload, chunks.length)
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
    eligibility,
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

export const runResearchWorkspaceMigration = async ({
  api,
  getCurrentWorkspace,
  subscribeToWorkspaceChanges,
  ...planInput
}: ResearchWorkspaceMigrationRunInput): Promise<ResearchWorkspaceMigrationRunResult> => {
  let plan: ResearchWorkspaceMigrationPlan | null = null
  let serverMigration: ResearchWorkspaceMigrationSessionResponse | null = null
  let attemptRevoked = false
  let unsubscribe: (() => void) | undefined
  const targetChanged = new Error("workspace-migration-target-no-longer-legacy")
  const observeWorkspace = () => {
    const current = getCurrentWorkspace?.()
    if (current && (current.workspaceId !== planInput.targetWorkspaceId ||
        getResearchWorkspaceMigrationEligibility(
          planInput.targetWorkspaceId,
          current.serverWorkspace
        ) !== "legacy")) {
      attemptRevoked = true
    }
  }
  const assertLegacyTarget = () => {
    observeWorkspace()
    if (attemptRevoked) throw targetChanged
  }
  try {
    const eligibility = getResearchWorkspaceMigrationEligibility(
      planInput.targetWorkspaceId,
      planInput.serverWorkspace
    )
    if (eligibility !== "legacy") {
      return {
        status: eligibility === "canonical" ? "not_needed" : "blocked",
        migrationId: null,
        manifestHash: null,
        serverMigration: null,
        localDeletionEligibility: null,
        deletedSurfaceIds: [],
        message: eligibility === "canonical"
          ? "Canonical workspace cache is not legacy migration content."
          : "Workspace provenance does not match the migration target. Local data retained."
      }
    }
    // Latch transitions, including A -> B -> A between awaited operations.
    unsubscribe = subscribeToWorkspaceChanges?.(observeWorkspace)
    assertLegacyTarget()
    plan = await buildResearchWorkspaceMigrationPlan(planInput)
    assertLegacyTarget()

    if (plan.chunks.length === 0) {
      return {
        status: plan.eligibility === "canonical" ||
          (plan.eligibility === "legacy" && plan.localDeletionEligibility.eligible) ? "not_needed" : "blocked",
        migrationId: null,
        manifestHash: plan.manifestHash,
        serverMigration: null,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: plan.eligibility === "canonical"
          ? "Canonical workspace cache is not legacy migration content."
          : plan.eligibility === "blocked"
          ? planInput.discoveredIndexedDbStores?.length
            ? "Shared IndexedDB offload stores cannot be migrated as one workspace. Local data retained."
            : "Workspace payload identity is mixed, mismatched, or unreadable. Local data retained."
          : plan.localDeletionEligibility.eligible
          ? "No legacy Research Workspace content was discovered."
          : "Legacy Research Workspace storage includes unknown or uncovered content."
      }
    }

    assertLegacyTarget()
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
      assertLegacyTarget()
      await api.putWorkspaceMigrationChunk(plan.migrationId, chunk.id, {
        sha256: chunk.sha256,
        byte_count: chunk.byte_count,
        chunk_kind: chunk.chunk_kind,
        metadata: buildChunkMetadata(chunk)
      })
    }

    assertLegacyTarget()
    await api.finalizeWorkspaceMigration(plan.migrationId, {
      manifest_hash: plan.manifestHash
    })
    assertLegacyTarget()
    serverMigration = await api.getWorkspaceMigration(plan.migrationId)
    assertLegacyTarget()

    if (!plan.localDeletionEligibility.eligible) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: "Server metadata receipt was saved. Local inventory includes unknown or uncovered content; automatic local cleanup is disabled and local data is retained."
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
        message: "Server metadata receipt was saved. Automatic local cleanup is disabled; writable legacy copies are retained."
      }
    }

    // Receipts cover declarations and hashes, not a durable import of content.
    // Writable local copies cannot safely be consumed by automatic cleanup.
    return {
      status: "blocked",
      migrationId: plan.migrationId,
      manifestHash: plan.manifestHash,
      serverMigration,
      localDeletionEligibility: plan.localDeletionEligibility,
      deletedSurfaceIds: [],
      message: "Server metadata receipt was saved. Automatic local cleanup is disabled; writable legacy copies are retained."
    }
  } catch (error) {
    return {
      status: error === targetChanged ? "blocked" : "failed",
      migrationId: plan?.migrationId ?? null,
      manifestHash: plan?.manifestHash ?? null,
      serverMigration,
      localDeletionEligibility: plan?.localDeletionEligibility ?? null,
      deletedSurfaceIds: [],
      message: error === targetChanged
        ? "The active workspace or canonical provenance changed. Further migration actions were stopped."
        : "Research Workspace metadata migration failed. Local data retained.",
      error
    }
  } finally {
    unsubscribe?.()
  }
}
