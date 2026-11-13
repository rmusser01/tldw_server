import {
  classifyResearchWorkspaceLegacyStorageSurface,
  RESEARCH_WORKSPACE_CHAT_REFERENCE_PREFIX,
  RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX,
  evaluateResearchWorkspaceLegacyDeletionEligibility,
  type ResearchWorkspaceLegacyDeletionEligibility,
} from "@/store/research-workspace-legacy-storage-inventory";

export const RESEARCH_WORKSPACE_MIGRATION_SCHEMA_VERSION = 1
export const RESEARCH_WORKSPACE_MIGRATION_SOURCE_PRODUCT =
  "research-workspace-webui"
export { RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX } from "@/store/research-workspace-legacy-storage-inventory";
import { extractWorkspaceIdFromChatSessionKey } from "@/store/workspace-chat-session-key";

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

export type ResearchWorkspaceChatReference = {
  referenceId: string
  serverChatId: string | null
}

export type ResearchWorkspaceChatReferenceOwner = {
  scopeKey: string
  isCurrent: () => boolean
  clear?: boolean
}

const isChatReferenceId = (value: unknown): value is string =>
  typeof value === "string" && /^[A-Za-z0-9._:-]{1,256}$/.test(value)

export const readResearchWorkspaceChatReference = (
  value: unknown
): ResearchWorkspaceChatReference | undefined => {
  if (value === undefined) return undefined
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Saved workspace chat reference is invalid")
  const { referenceId, serverChatId } = value as Record<string, unknown>
  const savedId =
    serverChatId === null
      ? null
      : isChatReferenceId(serverChatId)
        ? serverChatId
        : undefined
  if (!isChatReferenceId(referenceId) || savedId === undefined)
    throw new Error("Saved workspace chat reference is invalid")
  return { referenceId, serverChatId: savedId }
}

const chatReferenceKey = (workspaceId: string, scopeKey: string): string =>
  `${RESEARCH_WORKSPACE_CHAT_REFERENCE_PREFIX}:${encodeURIComponent(scopeKey)}:${encodeURIComponent(workspaceId)}`

export const readSavedResearchWorkspaceChatReference = (
  storage: Storage,
  workspaceId: string,
  scopeKey: string
): { key: string; raw: string | null; value: ResearchWorkspaceChatReference | undefined } => {
  const key = chatReferenceKey(workspaceId, scopeKey)
  const raw = storage.getItem(key)
  if (raw === null) return { key, raw, value: undefined }
  if (raw.length > 4096) throw new Error("Saved workspace chat reference is invalid")
  const value = readResearchWorkspaceChatReference(JSON.parse(raw))
  if (!value) throw new Error("Saved workspace chat reference is invalid")
  return { key, raw, value }
}

/** Retain only an untrusted location hint; restored content still requires scoped server reads. */
export const saveResearchWorkspaceChatReference = (
  workspaceId: string,
  referenceId: string,
  serverChatId: string | null,
  owner: ResearchWorkspaceChatReferenceOwner
): void => {
  if (!owner.isCurrent() || !owner.scopeKey || (!serverChatId && !owner.clear))
    return
  let match: { key: string; marker: Record<string, unknown> } | undefined
  let receipts = 0
  for (let index = 0; index < localStorage.length; index++) {
    const key = localStorage.key(index)
    if (!key?.startsWith(`${RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX}:`))
      continue
    if (++receipts > 128)
      throw new Error("Too many saved workspace migration markers")
    const raw = localStorage.getItem(key)
    if (!raw || raw.length > 65536) continue
    let marker: Record<string, unknown>
    try {
      marker = JSON.parse(raw)
    } catch {
      continue
    }
    if (
      !marker ||
      typeof marker !== "object" ||
      Array.isArray(marker) ||
      marker.contentRetained !== false ||
      marker.serverWorkspaceId !== workspaceId ||
      typeof marker.legacyWorkspaceId !== "string" ||
      key !==
        buildResearchWorkspaceMigrationTombstoneKey(marker.legacyWorkspaceId) ||
      typeof marker.migrationId !== "string" ||
      !marker.migrationId ||
      (marker.serverScopeKey && marker.serverScopeKey !== owner.scopeKey)
    )
      continue
    if (match)
      throw new Error("Saved workspace chat has ambiguous migration markers")
    match = { key, marker }
  }
  const chatReference = readResearchWorkspaceChatReference({
    referenceId,
    serverChatId
  })!
  if (!owner.isCurrent()) return
  const key = chatReferenceKey(workspaceId, owner.scopeKey)
  const raw = JSON.stringify(chatReference)
  if (localStorage.getItem(key) !== raw) localStorage.setItem(key, raw)
  if (match && JSON.stringify(match.marker.chatReference) !== raw)
    localStorage.setItem(match.key, JSON.stringify({ ...match.marker, chatReference }))
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

export interface ResearchWorkspaceMigrationRunInput extends ResearchWorkspaceMigrationPlanInput {
  api: ResearchWorkspaceMigrationApi;
  legacyWorkspaceId?: string;
  getChatReference?: () => ResearchWorkspaceChatReference | undefined;
  isCurrent?: () => boolean;
  deleteLocalStorageValue?: (key: string) => Promise<void> | void;
  writeLocalStorageValue?: (key: string, value: string) => Promise<void> | void;
  deleteIndexedDbStorePayload?: (
    store: ResearchWorkspaceIndexedDbStoreRef,
  ) => Promise<void> | void;
  now?: () => string;
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

// Shared storage can contain other workspaces even when its active identity matches.
const containsOnlyTargetWorkspace = (payload: unknown, target: string, keyedWorkspace?: string): boolean => {
  try {
    const value: unknown = typeof payload === "string" ? JSON.parse(payload) : payload
    const root = isRecord(value) && isRecord(value.state) ? value.state : value
    if (Array.isArray(root)) return root.length > 0 && root.every(row => containsOnlyTargetWorkspace(row, target))
    if (!isRecord(root)) return false
    const recordBound = Array.isArray(root.records)
      && root.records.length > 0 && root.records.every(row => containsOnlyTargetWorkspace(row, target))
    if (root.records !== undefined && !recordBound) return false
    const ids = new Set<string>(keyedWorkspace ? [keyedWorkspace] : [])
    let bound = Boolean(keyedWorkspace) || (recordBound
      && Object.keys(root).every(key => ["records", "version", "schema"].includes(key)))
    for (const name of ["workspaceId", "workspace_id", "activeWorkspaceId"]) {
      if (typeof root[name] === "string" && root[name]) bound = true
    }
    if (Array.isArray(root.workspaces) && root.workspaces.length
      && Object.keys(root).every(key => ["workspaces", "version", "schema"].includes(key))) bound = true
    const seen = new WeakSet<object>()
    const visit = (item: unknown): boolean => {
      if (!item || typeof item !== "object") return true
      if (seen.has(item)) return false
      seen.add(item)
      if (Array.isArray(item)) return item.every(visit)
      if (!isRecord(item)) return false
      for (const name of ["workspaceId", "workspace_id", "activeWorkspaceId"]) {
        const id = item[name]
        if (typeof id === "string" && id) ids.add(id)
        else if (id !== null && id !== undefined && id !== "") return false
      }
      if (item.workspaceIds !== undefined) {
        if (!Array.isArray(item.workspaceIds) || !item.workspaceIds.every(id => typeof id === "string" && id)) return false
        item.workspaceIds.forEach(id => ids.add(id))
      }
      for (const name of ["workspaces", "savedWorkspaces", "archivedWorkspaces"]) {
        if (item[name] === undefined) continue
        if (!Array.isArray(item[name]) || !item[name].every(row => isRecord(row) && typeof row.id === "string" && row.id)) return false
        item[name].forEach(row => ids.add(row.id as string))
      }
      for (const name of ["workspaceSnapshots", "workspaceChatSessions"]) {
        if (item[name] === undefined) continue
        if (!isRecord(item[name])) return false
        for (const key of Object.keys(item[name])) ids.add(name === "workspaceChatSessions" ? extractWorkspaceIdFromChatSessionKey(key) : key)
      }
      if (item.offloadType !== undefined && !["workspace_chat_session_v1", "workspace_artifact_payload_v1"].includes(String(item.offloadType))) return false
      if (item.offloadType !== undefined || (typeof item.key === "string" && item.key.startsWith("workspace:"))) {
        if (typeof item.key !== "string") return false
        const match = /^workspace:([^:]+):(?:chat|artifact:.+)$/.exec(item.key)
        if (!match) return false
        ids.add(decodeURIComponent(match[1]))
      }
      return Object.values(item).every(visit)
    }
    return bound && visit(value) && ids.size > 0 && [...ids].every(id => id === target)
  } catch { return false }
}

const createLocalStorageChunk = async (
  key: string,
  payload: string,
  ordinal: number,
  targetWorkspaceId: string,
): Promise<ResearchWorkspaceMigrationChunkPlan | null> => {
  const surface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "local_storage",
    key,
  });
  if (!surface || surface.classification !== "content") return null;
  if (
    !containsOnlyTargetWorkspace(
      payload,
      targetWorkspaceId,
      surface.workspaceId,
    )
  )
    return null;

  return {
    id: await buildChunkId(surface.id, payload, ordinal),
    surfaceId: surface.id,
    storageKind: "local_storage",
    key,
    sha256: await sha256Text(payload),
    byte_count: byteLengthText(payload),
    chunk_kind: "workspace_bundle",
  };
};

const createIndexedDbChunk = async (
  store: ResearchWorkspaceIndexedDbStoreRef,
  payload: unknown,
  ordinal: number,
  targetWorkspaceId: string,
): Promise<ResearchWorkspaceMigrationChunkPlan | null> => {
  const surface = classifyResearchWorkspaceLegacyStorageSurface({
    kind: "indexeddb_store",
    databaseName: store.databaseName,
    storeName: store.storeName,
  });
  if (!surface || surface.classification !== "content") return null;
  if (!containsOnlyTargetWorkspace(payload, targetWorkspaceId)) return null;

  const serializedPayload = stableStringify(payload);
  return {
    id: await buildChunkId(surface.id, serializedPayload, ordinal),
    surfaceId: surface.id,
    storageKind: "indexeddb_store",
    databaseName: store.databaseName,
    storeName: store.storeName,
    sha256: await sha256Text(serializedPayload),
    byte_count: byteLengthText(serializedPayload),
    chunk_kind: "indexeddb_store",
  };
};

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
    const chunk = await createLocalStorageChunk(
      key,
      payload,
      chunks.length,
      targetWorkspaceId,
    );
    if (chunk) chunks.push(chunk)
  }

  if (readIndexedDbStorePayload) {
    for (const store of discoveredIndexedDbStores) {
      const payload = await readIndexedDbStorePayload(store)
      if (payload == null) continue
      const chunk = await createIndexedDbChunk(
        store,
        payload,
        chunks.length,
        targetWorkspaceId,
      );
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

/** Compare the captured value and current operation immediately before local removal. */
export const deleteResearchWorkspaceMigrationLocalValue = (
  key: string, expected: string | null | undefined, isCurrent: () => boolean
): void => {
  if (!isCurrent() || typeof expected !== "string" || window.localStorage.getItem(key) !== expected)
    throw new Error("Local workspace changed before migration cleanup.")
  window.localStorage.removeItem(key)
}

export const runResearchWorkspaceMigration = async ({
  api,
  isCurrent,
  ...planInput
}: ResearchWorkspaceMigrationRunInput): Promise<ResearchWorkspaceMigrationRunResult> => {
  let plan: ResearchWorkspaceMigrationPlan | null = null;
  const deletedSurfaceIds: string[] = [];
  const assertCurrent = () => {
    if (isCurrent?.() === false)
      throw new Error(
        "Workspace migration owner changed before local cleanup.",
      );
  };
  try {
    assertCurrent();
    plan = await buildResearchWorkspaceMigrationPlan(planInput);
    assertCurrent();

    if (plan.chunks.length === 0) {
      return {
        status: plan.localDeletionEligibility.eligible
          ? "not_needed"
          : "blocked",
        migrationId: null,
        manifestHash: plan.manifestHash,
        serverMigration: null,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message: plan.localDeletionEligibility.eligible
          ? "No legacy Research Workspace content was discovered."
          : "Legacy Research Workspace storage includes unknown or uncovered content.",
      };
    }

    assertCurrent();
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
      diagnostics: {},
    });

    for (const chunk of plan.chunks) {
      assertCurrent();
      await api.putWorkspaceMigrationChunk(plan.migrationId, chunk.id, {
        sha256: chunk.sha256,
        byte_count: chunk.byte_count,
        chunk_kind: chunk.chunk_kind,
        metadata: buildChunkMetadata(chunk),
      });
    }

    assertCurrent();
    await api.finalizeWorkspaceMigration(plan.migrationId, {
      manifest_hash: plan.manifestHash,
    });
    assertCurrent();
    const serverMigration = await api.getWorkspaceMigration(plan.migrationId);
    assertCurrent();

    if (!plan.localDeletionEligibility.eligible) {
      return {
        status: "blocked",
        migrationId: plan.migrationId,
        manifestHash: plan.manifestHash,
        serverMigration,
        localDeletionEligibility: plan.localDeletionEligibility,
        deletedSurfaceIds: [],
        message:
          "Server receipt was saved, but local deletion is blocked by the legacy inventory gate.",
      };
    }

    // This protocol transfers hashes, not recoverable note/artifact/chat content.
    return {
      status: "finalized_not_delete_eligible",
      migrationId: plan.migrationId,
      manifestHash: plan.manifestHash,
      serverMigration,
      localDeletionEligibility: plan.localDeletionEligibility,
      deletedSurfaceIds: [],
      message:
        "Migration receipts were saved. Local content is retained until its server copy can be verified.",
    };
  } catch (error) {
    return {
      status: "failed",
      migrationId: plan?.migrationId ?? null,
      manifestHash: plan?.manifestHash ?? null,
      serverMigration: null,
      localDeletionEligibility: plan?.localDeletionEligibility ?? null,
      deletedSurfaceIds,
      message: "Research Workspace migration failed; local content is retained.",
      error,
    };
  }
};
