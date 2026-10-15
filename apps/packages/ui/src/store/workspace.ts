import {
  knowledgeNoteHead,
  validateKnowledgeNoteProvenance
} from "@/utils/knowledge-note-provenance"
/**
 * Workspace Zustand Store
 * Manages state for the NotebookLM-style three-pane research interface
 */

import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"
import type { ChatHistory, Message } from "@/store/option"
import {
  WORKSPACE_STORAGE_CHANNEL_NAME,
  WORKSPACE_STORAGE_KEY,
  WORKSPACE_STORAGE_QUOTA_EVENT,
  WORKSPACE_STORAGE_RECOVERY_EVENT,
  isWorkspaceBroadcastSyncEnabled,
  type WorkspaceBroadcastUpdateMessage,
  type WorkspaceStorageQuotaEventDetail,
  type WorkspaceStorageRecoveryAction,
  type WorkspaceStorageRecoveryEventDetail
} from "@/store/workspace-events"
import {
  WORKSPACE_EXPORT_BUNDLE_FORMAT,
  WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
  sanitizeImportedChatSession,
  type WorkspaceBundleChatSession,
  type WorkspaceBundleSnapshot,
  type WorkspaceExportBundle
} from "@/store/workspace-bundle"
import {
  buildWorkspaceChatSessionKey,
  extractWorkspaceIdFromChatSessionKey,
  isWorkspaceChatSessionKeyForWorkspace
} from "@/store/workspace-chat-session-key"
import { buildResearchWorkspaceMigrationTombstoneKey } from "@/store/workspace-migration"
import type {
  AddSourceModalState,
  AddSourceTab,
  ArtifactStatus,
  ArtifactType,
  AudioGenerationSettings,
  AudioTtsProvider,
  EffectiveWorkspaceAssistantDefault,
  GeneratedArtifact,
  StudyMaterialsPolicy,
  SavedWorkspace,
  WorkspaceAssistantDefaults,
  WorkspaceBanner,
  WorkspaceBannerImage,
  WorkspaceBannerImageMimeType,
  WorkspaceCollection,
  WorkspaceConfig,
  WorkspaceNote,
  WorkspaceSource,
  WorkspaceSourceReviewUpdate,
  WorkspaceSourceReadiness,
  WorkspaceSourceStatusDetails,
  WorkspaceSourceTransferConflictResolution,
  WorkspaceSourceTransferEmptyFolderPolicy,
  WorkspaceSourceTransferMode,
  WorkspaceSourceTransferResult,
  WorkspaceSourceFolder,
  WorkspaceSourceFolderMembership,
  WorkspaceSourceStatus,
  WorkspaceSourceType
} from "@/types/workspace"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_BANNER,
  DEFAULT_WORKSPACE_NOTE
} from "@/types/workspace"
import { normalizeWorkspaceAssistantDefaults } from "@/types/workspace-assistant-defaults"
import {
  collectDescendantFolderIds,
  createWorkspaceOrganizationIndex,
  deriveEffectiveSelectedSourceIds
} from "@/store/workspace-organization"
import { trackResearchWorkspaceTelemetry } from "@/utils/research-workspace-telemetry"
import { createSourcesSlice } from "./workspace-slices/sources-slice"
import { createStudioSlice } from "./workspace-slices/studio-slice"
import { createUISlice } from "./workspace-slices/ui-slice"
import { createWorkspaceListSlice } from "./workspace-slices/workspace-list-slice"
import { isWorkspaceSourceSelectable } from "./workspace-source-status"

export {
  isWorkspaceSourceSelectable,
  isWorkspaceSourcePartiallyQueryable
} from "./workspace-source-status"

// ─────────────────────────────────────────────────────────────────────────────
// Storage Configuration
// ─────────────────────────────────────────────────────────────────────────────

export const generateWorkspaceId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }

  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"))
    return [
      hex.slice(0, 4).join(""),
      hex.slice(4, 6).join(""),
      hex.slice(6, 8).join(""),
      hex.slice(8, 10).join(""),
      hex.slice(10, 16).join(""),
    ].join("-")
  }

  return `workspace-${Date.now().toString(36)}`
}

let workspaceBroadcastChannel: BroadcastChannel | null = null

const isQuotaExceededError = (error: unknown): boolean => {
  if (!error) return false

  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return (
      error.name === "QuotaExceededError" ||
      error.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      error.code === 22 ||
      error.code === 1014
    )
  }

  const candidate = error as {
    name?: string
    code?: number
    message?: string
  }
  return (
    candidate.name === "QuotaExceededError" ||
    candidate.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    candidate.code === 22 ||
    candidate.code === 1014 ||
    /quota/i.test(candidate.message || "")
  )
}

const emitWorkspaceQuotaExceeded = (key: string, error: unknown): void => {
  if (typeof window === "undefined") return

  const reason =
    error instanceof Error
      ? error.message
      : "Workspace data exceeded local storage quota."
  const detail: WorkspaceStorageQuotaEventDetail = { key, reason }
  window.dispatchEvent(
    new CustomEvent<WorkspaceStorageQuotaEventDetail>(
      WORKSPACE_STORAGE_QUOTA_EVENT,
      { detail }
    )
  )
}

const emitWorkspaceStorageRecoveryEvent = (
  detail: WorkspaceStorageRecoveryEventDetail
): void => {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent<WorkspaceStorageRecoveryEventDetail>(
      WORKSPACE_STORAGE_RECOVERY_EVENT,
      { detail }
    )
  )
}

type WorkspaceStorageRecoveryMutation = {
  action: WorkspaceStorageRecoveryAction
  workspaceId?: string
  beforeBytes: number
  afterBytes: number
  recoveredBytes: number
}

type WorkspaceStorageRecoveryAttempt = {
  value: string
  beforeBytes: number
  afterBytes: number
  mutations: WorkspaceStorageRecoveryMutation[]
}

const WORKSPACE_STORAGE_RECOVERY_MIN_RECLAIM_BYTES = 64 * 1024
const WORKSPACE_STORAGE_OVERSIZED_ARTIFACT_MIN_BYTES = 12 * 1024
const AUDIO_TTS_PROVIDERS = new Set<AudioTtsProvider>([
  "browser",
  "elevenlabs",
  "openai",
  "tldw"
])
const AUDIO_OUTPUT_FORMATS = new Set<AudioGenerationSettings["format"]>([
  "mp3",
  "wav",
  "opus",
  "aac",
  "flac"
])
const WORKSPACE_BANNER_IMAGE_MIME_TYPES = new Set<WorkspaceBannerImageMimeType>(
  ["image/jpeg", "image/png", "image/webp"]
)
const WORKSPACE_SPLIT_INDEX_SCHEMA = "workspace_split_v1"
const WORKSPACE_SPLIT_INDEX_VERSION = 1
export const WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY =
  "tldw:feature-rollout:workspace_split_storage_v1:enabled"
export const WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY =
  "tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled"
const WORKSPACE_STORAGE_SPLIT_KEY_FLAG_VITE_ENV =
  "VITE_WORKSPACE_SPLIT_STORAGE_V1_ENABLED"
const WORKSPACE_STORAGE_SPLIT_KEY_FLAG_NEXT_ENV =
  "NEXT_PUBLIC_WORKSPACE_SPLIT_STORAGE_V1_ENABLED"
const WORKSPACE_STORAGE_INDEXEDDB_FLAG_VITE_ENV =
  "VITE_WORKSPACE_INDEXEDDB_OFFLOAD_V1_ENABLED"
const WORKSPACE_STORAGE_INDEXEDDB_FLAG_NEXT_ENV =
  "NEXT_PUBLIC_WORKSPACE_INDEXEDDB_OFFLOAD_V1_ENABLED"
const WORKSPACE_STORAGE_SPLIT_KEY_DEFAULT_ENABLED = true
const WORKSPACE_STORAGE_INDEXEDDB_DEFAULT_ENABLED = true
const WORKSPACE_INDEXEDDB_NAME = "tldw-workspace-storage"
const WORKSPACE_INDEXEDDB_VERSION = 1
const WORKSPACE_INDEXEDDB_CHAT_STORE = "workspace-chat-sessions"
const WORKSPACE_INDEXEDDB_ARTIFACT_STORE = "workspace-artifact-payloads"
const WORKSPACE_CHAT_OFFLOAD_MIN_BYTES = 8 * 1024
const WORKSPACE_ARTIFACT_OFFLOAD_MIN_BYTES = 12 * 1024
const WORKSPACE_PERSIST_MAX_CHAT_MESSAGES_PER_SESSION = 250
const WORKSPACE_PERSIST_SERVER_ARTIFACT_CONTENT_MAX_BYTES = 24 * 1024
const WORKSPACE_PERSIST_SERVER_ARTIFACT_DATA_MAX_BYTES = 16 * 1024
const WORKSPACE_PERSIST_TRUNCATION_SUFFIX =
  "\n\n[Truncated in local persistence cache; open the server output for full content.]"
const WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY = "__tldwArtifactPayloadRef"
const WORKSPACE_CHAT_POINTER_KIND = "workspace_chat_session_v1"
const WORKSPACE_ARTIFACT_POINTER_KIND = "workspace_artifact_payload_v1"
// Internal per-snapshot persist metadata (WP-16 per-workspace diff). These
// fields ride ON the persisted snapshot object (not a wrapper envelope) so
// legacy readers keep working; they are stripped on every hydration read and
// never leak into store state.
const WORKSPACE_SNAPSHOT_REVISION_FIELD = "__tldwPersistRevision"
const WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD = "__tldwPersistUpdatedAt"

type RevisionedWorkspaceSnapshot = WorkspaceSnapshot & {
  [WORKSPACE_SNAPSHOT_REVISION_FIELD]?: number
  [WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD]?: number
}

const stripWorkspaceSnapshotPersistMetadata = <T>(snapshot: T): T => {
  if (
    !isRecord(snapshot) ||
    (!(WORKSPACE_SNAPSHOT_REVISION_FIELD in snapshot) &&
      !(WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD in snapshot))
  ) {
    return snapshot
  }
  const stripped: Record<string, unknown> = { ...snapshot }
  delete stripped[WORKSPACE_SNAPSHOT_REVISION_FIELD]
  delete stripped[WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD]
  return stripped as T
}

type WorkspaceIndexedDbChatPointer = {
  offloadType: typeof WORKSPACE_CHAT_POINTER_KIND
  key: string
  historyId: string | null
  serverChatId: string | null
  updatedAt: number
}

type WorkspaceIndexedDbArtifactPayloadPointer = {
  offloadType: typeof WORKSPACE_ARTIFACT_POINTER_KIND
  key: string
  fields: Array<"content" | "data">
  updatedAt: number
}

type PersistedWorkspaceArtifact = GeneratedArtifact & {
  [WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY]?: WorkspaceIndexedDbArtifactPayloadPointer
}

type PersistedWorkspaceChatSessionReference =
  | PersistedWorkspaceChatSession
  | WorkspaceIndexedDbChatPointer

type WorkspaceIndexedDbChatRecord = {
  key: string
  workspaceId: string
  session: PersistedWorkspaceChatSession
  updatedAt: number
}

type WorkspaceIndexedDbArtifactPayloadRecord = {
  key: string
  workspaceId: string
  artifactId: string
  payload: {
    content?: string
    data?: Record<string, unknown>
  }
  updatedAt: number
}

type WorkspaceIndexedDbAdapter = {
  isAvailable: () => boolean
  putChatRecord: (record: WorkspaceIndexedDbChatRecord) => Promise<boolean>
  getChatRecord: (key: string) => Promise<WorkspaceIndexedDbChatRecord | null>
  deleteChatRecord: (key: string) => Promise<boolean>
  putArtifactPayloadRecord: (
    record: WorkspaceIndexedDbArtifactPayloadRecord
  ) => Promise<boolean>
  getArtifactPayloadRecord: (
    key: string
  ) => Promise<WorkspaceIndexedDbArtifactPayloadRecord | null>
  deleteArtifactPayloadRecord: (key: string) => Promise<boolean>
}

type WorkspaceSplitIndexState = {
  workspaceId: string
  savedWorkspaces: SavedWorkspace[]
  archivedWorkspaces: SavedWorkspace[]
  workspaceCollections: WorkspaceCollection[]
  workspaceIds: string[]
  workspaceSnapshots: Record<string, WorkspaceSnapshot>
  workspaceChatSessions: Record<string, PersistedWorkspaceChatSessionReference>
}

type WorkspaceSplitIndexEnvelope = {
  schema: typeof WORKSPACE_SPLIT_INDEX_SCHEMA
  splitVersion: number
  version: number
  state: WorkspaceSplitIndexState
}

let workspaceIndexedDbConnectionPromise: Promise<IDBDatabase> | null = null
let workspaceIndexedDbAdapterSingleton: WorkspaceIndexedDbAdapter | null = null

const isWorkspaceIndexedDbRuntimeAvailable = (): boolean =>
  typeof window !== "undefined" && typeof indexedDB !== "undefined"

const openWorkspaceIndexedDbConnection = async (): Promise<IDBDatabase> => {
  if (!isWorkspaceIndexedDbRuntimeAvailable()) {
    throw new Error("IndexedDB is not available in this runtime.")
  }

  if (workspaceIndexedDbConnectionPromise) {
    return workspaceIndexedDbConnectionPromise
  }

  workspaceIndexedDbConnectionPromise = new Promise<IDBDatabase>(
    (resolve, reject) => {
      const request = indexedDB.open(
        WORKSPACE_INDEXEDDB_NAME,
        WORKSPACE_INDEXEDDB_VERSION
      )

      request.onupgradeneeded = () => {
        const database = request.result
        if (!database.objectStoreNames.contains(WORKSPACE_INDEXEDDB_CHAT_STORE)) {
          database.createObjectStore(WORKSPACE_INDEXEDDB_CHAT_STORE, {
            keyPath: "key"
          })
        }
        if (
          !database.objectStoreNames.contains(WORKSPACE_INDEXEDDB_ARTIFACT_STORE)
        ) {
          database.createObjectStore(WORKSPACE_INDEXEDDB_ARTIFACT_STORE, {
            keyPath: "key"
          })
        }
      }

      request.onsuccess = () => {
        const database = request.result
        database.onversionchange = () => {
          database.close()
          workspaceIndexedDbConnectionPromise = null
        }
        resolve(database)
      }
      request.onerror = () =>
        reject(request.error || new Error("Failed to open workspace IndexedDB."))
      request.onblocked = () =>
        reject(new Error("Workspace IndexedDB upgrade is blocked."))
    }
  ).catch((error) => {
    workspaceIndexedDbConnectionPromise = null
    throw error
  })

  return workspaceIndexedDbConnectionPromise
}

const withWorkspaceIndexedDbStore = async <TResult>(
  storeName:
    | typeof WORKSPACE_INDEXEDDB_CHAT_STORE
    | typeof WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<TResult>
): Promise<TResult | null> => {
  const database = await openWorkspaceIndexedDbConnection()
  return new Promise<TResult | null>((resolve, reject) => {
    const transaction = database.transaction(storeName, mode)
    const store = transaction.objectStore(storeName)

    let settled = false
    const settleResolve = (value: TResult | null) => {
      if (settled) return
      settled = true
      resolve(value)
    }
    const settleReject = (error: unknown) => {
      if (settled) return
      settled = true
      reject(error)
    }

    const request = operation(store)
    request.onsuccess = () => settleResolve(request.result ?? null)
    request.onerror = () =>
      settleReject(request.error || new Error("Workspace IndexedDB request failed."))
    transaction.onabort = () =>
      settleReject(
        transaction.error || new Error("Workspace IndexedDB transaction aborted.")
      )
    transaction.onerror = () =>
      settleReject(
        transaction.error || new Error("Workspace IndexedDB transaction failed.")
      )
  })
}

const createWorkspaceIndexedDbAdapter = (): WorkspaceIndexedDbAdapter => {
  let disabled = false
  const disableAdapter = () => {
    disabled = true
  }

  const run = async <T>(
    operation: () => Promise<T>,
    fallbackValue: T
  ): Promise<T> => {
    if (disabled || !isWorkspaceIndexedDbRuntimeAvailable()) {
      return fallbackValue
    }
    try {
      return await operation()
    } catch {
      disableAdapter()
      return fallbackValue
    }
  }

  return {
    isAvailable: () => !disabled && isWorkspaceIndexedDbRuntimeAvailable(),
    putChatRecord: (record) =>
      run(
        async () => {
          await withWorkspaceIndexedDbStore<IDBValidKey>(
            WORKSPACE_INDEXEDDB_CHAT_STORE,
            "readwrite",
            (store) => store.put(record)
          )
          return true
        },
        false
      ),
    getChatRecord: (key) =>
      run(
        () =>
          withWorkspaceIndexedDbStore<WorkspaceIndexedDbChatRecord>(
            WORKSPACE_INDEXEDDB_CHAT_STORE,
            "readonly",
            (store) => store.get(key)
          ),
        null
      ),
    deleteChatRecord: (key) =>
      run(
        async () => {
          await withWorkspaceIndexedDbStore<undefined>(
            WORKSPACE_INDEXEDDB_CHAT_STORE,
            "readwrite",
            (store) => store.delete(key)
          )
          return true
        },
        false
      ),
    putArtifactPayloadRecord: (record) =>
      run(
        async () => {
          await withWorkspaceIndexedDbStore<IDBValidKey>(
            WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
            "readwrite",
            (store) => store.put(record)
          )
          return true
        },
        false
      ),
    getArtifactPayloadRecord: (key) =>
      run(
        () =>
          withWorkspaceIndexedDbStore<WorkspaceIndexedDbArtifactPayloadRecord>(
            WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
            "readonly",
            (store) => store.get(key)
          ),
        null
      ),
    deleteArtifactPayloadRecord: (key) =>
      run(
        async () => {
          await withWorkspaceIndexedDbStore<undefined>(
            WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
            "readwrite",
            (store) => store.delete(key)
          )
          return true
        },
        false
      )
  }
}

const createNoopWorkspaceIndexedDbAdapter = (): WorkspaceIndexedDbAdapter => ({
  isAvailable: () => false,
  putChatRecord: async () => false,
  getChatRecord: async () => null,
  deleteChatRecord: async () => false,
  putArtifactPayloadRecord: async () => false,
  getArtifactPayloadRecord: async () => null,
  deleteArtifactPayloadRecord: async () => false
})

const noopWorkspaceIndexedDbAdapter = createNoopWorkspaceIndexedDbAdapter()

const getWorkspaceIndexedDbAdapter = (): WorkspaceIndexedDbAdapter => {
  if (!workspaceIndexedDbAdapterSingleton) {
    workspaceIndexedDbAdapterSingleton = createWorkspaceIndexedDbAdapter()
  }
  return workspaceIndexedDbAdapterSingleton
}

const isWorkspaceIndexedDbChatPointer = (
  candidate: unknown
): candidate is WorkspaceIndexedDbChatPointer => {
  if (!isRecord(candidate)) return false
  if (candidate.offloadType !== WORKSPACE_CHAT_POINTER_KIND) return false
  if (typeof candidate.key !== "string") return false
  const historyId = candidate.historyId
  if (historyId !== null && typeof historyId !== "string") return false
  const serverChatId = candidate.serverChatId
  if (serverChatId !== null && typeof serverChatId !== "string") return false
  return typeof candidate.updatedAt === "number"
}

const parseWorkspaceIndexedDbArtifactPayloadPointer = (
  candidate: unknown
): WorkspaceIndexedDbArtifactPayloadPointer | null => {
  if (!isRecord(candidate)) return null
  if (candidate.offloadType !== WORKSPACE_ARTIFACT_POINTER_KIND) return null
  if (typeof candidate.key !== "string") return null
  if (!Array.isArray(candidate.fields)) return null

  const fields = candidate.fields.filter(
    (field): field is "content" | "data" =>
      field === "content" || field === "data"
  )
  if (fields.length === 0) return null

  return {
    offloadType: WORKSPACE_ARTIFACT_POINTER_KIND,
    key: candidate.key,
    fields,
    updatedAt:
      typeof candidate.updatedAt === "number" && Number.isFinite(candidate.updatedAt)
        ? candidate.updatedAt
        : Date.now()
  }
}

const getWorkspaceArtifactPayloadPointer = (
  artifact: unknown
): WorkspaceIndexedDbArtifactPayloadPointer | null => {
  if (!isRecord(artifact)) return null
  return parseWorkspaceIndexedDbArtifactPayloadPointer(
    artifact[WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY]
  )
}

const hasWorkspaceArtifactPayloadPointers = (snapshot: unknown): boolean => {
  if (!isRecord(snapshot) || !Array.isArray(snapshot.generatedArtifacts)) {
    return false
  }
  return snapshot.generatedArtifacts.some(
    (artifact) => getWorkspaceArtifactPayloadPointer(artifact) !== null
  )
}

const buildWorkspaceIndexedDbChatRecordKey = (workspaceId: string): string =>
  `workspace:${encodeURIComponent(workspaceId)}:chat`

const buildWorkspaceIndexedDbArtifactRecordKey = (
  workspaceId: string,
  artifactId: string
): string =>
  `workspace:${encodeURIComponent(workspaceId)}:artifact:${encodeURIComponent(
    artifactId
  )}`

const collectWorkspaceArtifactIdsFromSnapshot = (snapshot: unknown): string[] => {
  if (!isRecord(snapshot) || !Array.isArray(snapshot.generatedArtifacts)) {
    return []
  }

  const ids = new Set<string>()
  for (const artifact of snapshot.generatedArtifacts) {
    if (!isRecord(artifact) || typeof artifact.id !== "string") continue
    ids.add(artifact.id)
  }
  return Array.from(ids)
}

const collectWorkspaceArtifactPointerKeysFromSnapshot = (
  snapshot: unknown
): string[] => {
  if (!isRecord(snapshot) || !Array.isArray(snapshot.generatedArtifacts)) {
    return []
  }

  const keys = new Set<string>()
  for (const artifact of snapshot.generatedArtifacts) {
    const pointer = getWorkspaceArtifactPayloadPointer(artifact)
    if (pointer?.key) {
      keys.add(pointer.key)
    }
  }
  return Array.from(keys)
}

const readWorkspaceSnapshotFromStorage = (
  workspaceId: string
): WorkspaceSnapshot | null => {
  const raw = localStorage.getItem(buildWorkspaceSnapshotStorageKey(workspaceId))
  const parsed = safeParseJson(raw)
  if (!isRecord(parsed)) return null
  // Accept both legacy (no-revision) and revision-carrying snapshots; the
  // internal diff metadata never leaves the persistence layer.
  return stripWorkspaceSnapshotPersistMetadata(
    parsed as unknown as WorkspaceSnapshot
  )
}

const isAudioTtsProvider = (value: unknown): value is AudioTtsProvider =>
  typeof value === "string" &&
  AUDIO_TTS_PROVIDERS.has(value as AudioTtsProvider)

const isAudioOutputFormat = (
  value: unknown
): value is AudioGenerationSettings["format"] =>
  typeof value === "string" &&
  AUDIO_OUTPUT_FORMATS.has(value as AudioGenerationSettings["format"])

const isWorkspaceBannerImageMimeType = (
  value: unknown
): value is WorkspaceBannerImageMimeType =>
  typeof value === "string" &&
  WORKSPACE_BANNER_IMAGE_MIME_TYPES.has(value as WorkspaceBannerImageMimeType)

const parseWorkspaceTimestamp = (value: unknown): number => {
  if (value instanceof Date) return value.getTime()
  if (typeof value === "string" || typeof value === "number") {
    const parsed = new Date(value).getTime()
    if (Number.isFinite(parsed)) return parsed
  }
  return Number.POSITIVE_INFINITY
}

const attemptWorkspaceStorageRecovery = (
  key: string,
  serializedValue: string
): WorkspaceStorageRecoveryAttempt | null => {
  if (key !== WORKSPACE_STORAGE_KEY) return null
  if (typeof serializedValue !== "string" || serializedValue.length === 0) {
    return null
  }

  let parsedPayload: unknown
  try {
    parsedPayload = JSON.parse(serializedValue)
  } catch {
    return null
  }

  if (!isRecord(parsedPayload)) return null

  const hasStateEnvelope =
    isRecord(parsedPayload) &&
    "state" in parsedPayload &&
    isRecord(parsedPayload.state)
  const mutableState = hasStateEnvelope
    ? (parsedPayload.state as Record<string, unknown>)
    : (parsedPayload as Record<string, unknown>)

  const beforeBytes = estimateSerializedByteLength(mutableState)
  if (beforeBytes <= 0) return null

  const targetBytes = Math.max(
    beforeBytes - WORKSPACE_STORAGE_RECOVERY_MIN_RECLAIM_BYTES,
    0
  )

  const workspaceId =
    typeof mutableState.workspaceId === "string" ? mutableState.workspaceId : ""
  const savedWorkspaces = Array.isArray(mutableState.savedWorkspaces)
    ? (mutableState.savedWorkspaces as Array<Record<string, unknown>>)
    : []
  const archivedWorkspaces = Array.isArray(mutableState.archivedWorkspaces)
    ? (mutableState.archivedWorkspaces as Array<Record<string, unknown>>)
    : []
  const workspaceSnapshots = isRecord(mutableState.workspaceSnapshots)
    ? (mutableState.workspaceSnapshots as Record<string, unknown>)
    : {}
  const workspaceChatSessions = isRecord(mutableState.workspaceChatSessions)
    ? (mutableState.workspaceChatSessions as Record<string, unknown>)
    : {}

  const workspaceLastAccess = new Map<string, number>()
  for (const workspace of [...savedWorkspaces, ...archivedWorkspaces]) {
    if (!isRecord(workspace)) continue
    const id = typeof workspace.id === "string" ? workspace.id : null
    if (!id) continue
    const timestamp = parseWorkspaceTimestamp(workspace.lastAccessedAt)
    workspaceLastAccess.set(id, timestamp)
  }

  const mutations: WorkspaceStorageRecoveryMutation[] = []
  let currentBytes = beforeBytes
  const applyMutation = (
    action: WorkspaceStorageRecoveryAction,
    nextWorkspaceId?: string
  ) => {
    const nextBytes = estimateSerializedByteLength(mutableState)
    if (nextBytes >= currentBytes) return
    mutations.push({
      action,
      workspaceId: nextWorkspaceId,
      beforeBytes: currentBytes,
      afterBytes: nextBytes,
      recoveredBytes: currentBytes - nextBytes
    })
    currentBytes = nextBytes
  }
  const reachedTarget = () => currentBytes <= targetBytes

  // Evict archived workspace snapshots/sessions first (least recently used).
  const archivedIds = archivedWorkspaces
    .map((workspace) => (typeof workspace.id === "string" ? workspace.id : null))
    .filter((id): id is string => Boolean(id))
    .sort(
      (a, b) =>
        (workspaceLastAccess.get(a) ?? Number.POSITIVE_INFINITY) -
        (workspaceLastAccess.get(b) ?? Number.POSITIVE_INFINITY)
    )

  // Trim archived banner images before removing entire workspaces.
  for (const archivedId of archivedIds) {
    if (reachedTarget()) break
    if (!isRecord(mutableState.workspaceSnapshots)) break

    const snapshotsMap = mutableState.workspaceSnapshots as Record<string, unknown>
    const snapshot = snapshotsMap[archivedId]
    if (!isRecord(snapshot) || !isRecord(snapshot.workspaceBanner)) continue

    const workspaceBanner = snapshot.workspaceBanner as Record<string, unknown>
    if (!workspaceBanner.image) continue

    workspaceBanner.image = null
    applyMutation("banner_image_removed", archivedId)
  }

  for (const archivedId of archivedIds) {
    if (reachedTarget()) break

    mutableState.archivedWorkspaces = (
      Array.isArray(mutableState.archivedWorkspaces)
        ? (mutableState.archivedWorkspaces as Array<Record<string, unknown>>)
        : []
    ).filter((workspace) => workspace.id !== archivedId)
    if (isRecord(mutableState.workspaceSnapshots)) {
      delete (mutableState.workspaceSnapshots as Record<string, unknown>)[archivedId]
    }
    if (isRecord(mutableState.workspaceChatSessions)) {
      delete (mutableState.workspaceChatSessions as Record<string, unknown>)[archivedId]
    }

    applyMutation("archived_workspace_removed", archivedId)
  }

  // Evict oldest non-active chat sessions next.
  const sessionIds = Object.keys(workspaceChatSessions)
    .filter((id) => id !== workspaceId)
    .sort(
      (a, b) =>
        (workspaceLastAccess.get(a) ?? Number.POSITIVE_INFINITY) -
        (workspaceLastAccess.get(b) ?? Number.POSITIVE_INFINITY)
    )

  for (const sessionId of sessionIds) {
    if (reachedTarget()) break
    if (!isRecord(mutableState.workspaceChatSessions)) break
    const sessionsMap = mutableState.workspaceChatSessions as Record<string, unknown>
    if (!(sessionId in sessionsMap)) continue
    delete sessionsMap[sessionId]
    applyMutation("chat_session_removed", sessionId)
  }

  // Evict oversized artifacts from least-recently-used snapshots.
  const snapshotIds = Object.keys(workspaceSnapshots).sort((a, b) => {
    const aScore =
      a === workspaceId
        ? Number.MAX_SAFE_INTEGER
        : workspaceLastAccess.get(a) ?? Number.POSITIVE_INFINITY
    const bScore =
      b === workspaceId
        ? Number.MAX_SAFE_INTEGER
        : workspaceLastAccess.get(b) ?? Number.POSITIVE_INFINITY
    return aScore - bScore
  })

  for (const snapshotId of snapshotIds) {
    if (reachedTarget()) break
    if (!isRecord(mutableState.workspaceSnapshots)) break
    const snapshotsMap = mutableState.workspaceSnapshots as Record<string, unknown>
    const snapshot = snapshotsMap[snapshotId]
    if (!isRecord(snapshot) || !Array.isArray(snapshot.generatedArtifacts)) continue

    const artifacts = snapshot.generatedArtifacts as Array<Record<string, unknown>>
    const oversizedArtifacts = artifacts
      .filter(
        (artifact) =>
          isRecord(artifact) &&
          estimateSerializedByteLength(artifact) >=
            WORKSPACE_STORAGE_OVERSIZED_ARTIFACT_MIN_BYTES
      )
      .sort(
        (a, b) =>
          estimateSerializedByteLength(b) - estimateSerializedByteLength(a)
      )

    for (const artifact of oversizedArtifacts) {
      if (reachedTarget()) break
      const artifactIndex = artifacts.indexOf(artifact)
      if (artifactIndex < 0) continue
      artifacts.splice(artifactIndex, 1)
      applyMutation("artifact_removed", snapshotId)
    }
  }

  if (mutations.length === 0) return null

  const nextSerializedValue = JSON.stringify(parsedPayload)
  return {
    value: nextSerializedValue,
    beforeBytes,
    afterBytes: currentBytes,
    mutations
  }
}

const normalizeWorkspaceStorageIds = (
  ...candidates: Array<string | null | undefined>
): string[] => {
  const ids = new Set<string>()
  for (const candidate of candidates) {
    if (!candidate) continue
    const trimmed = candidate.trim()
    if (!trimmed) continue
    ids.add(trimmed)
  }
  return Array.from(ids)
}

const buildWorkspaceSnapshotStorageKey = (workspaceId: string): string =>
  `${WORKSPACE_STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`

const buildWorkspaceChatStorageKey = (workspaceId: string): string =>
  `${WORKSPACE_STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:chat`

const safeParseJson = (raw: string | null | undefined): unknown => {
  if (typeof raw !== "string" || raw.length === 0) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

export const hasResearchWorkspaceMigrationTombstone = (workspaceId: string): boolean => {
  const trimmedWorkspaceId = workspaceId.trim()
  if (!trimmedWorkspaceId || typeof localStorage === "undefined") return false

  let tombstoneRaw: string | null
  try {
    tombstoneRaw = localStorage.getItem(
      buildResearchWorkspaceMigrationTombstoneKey(trimmedWorkspaceId)
    )
  } catch {
    return false
  }

  const tombstone = safeParseJson(tombstoneRaw)
  if (!isRecord(tombstone)) return false
  return (
    tombstone.contentRetained === false &&
    tombstone.legacyWorkspaceId === trimmedWorkspaceId &&
    typeof tombstone.migrationId === "string" &&
    tombstone.migrationId.trim().length > 0
  )
}

const cleanedUpMigrationTombstoneWorkspaceIds = new Set<string>()

const parseWorkspaceFeatureFlagCandidate = (
  candidate: unknown
): boolean | null => {
  if (typeof candidate === "boolean") return candidate
  if (typeof candidate === "number") {
    if (!Number.isFinite(candidate)) return null
    if (candidate === 1) return true
    if (candidate === 0) return false
    return null
  }
  if (typeof candidate !== "string") return null

  const normalized = candidate.trim().toLowerCase()
  if (!normalized) return null
  if (["1", "true", "on", "yes", "enabled"].includes(normalized)) return true
  if (["0", "false", "off", "no", "disabled"].includes(normalized)) return false
  return null
}

const readWorkspaceLocalStorageValue = (key: string): string | null => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

const resolveWorkspaceStorageFeatureFlag = ({
  localStorageKey,
  viteEnvKey,
  nextEnvKey,
  defaultEnabled
}: {
  localStorageKey: string
  viteEnvKey: string
  nextEnvKey: string
  defaultEnabled: boolean
}): boolean => {
  const persistedValue = readWorkspaceLocalStorageValue(localStorageKey)
  const persistedDecision = parseWorkspaceFeatureFlagCandidate(persistedValue)
  if (persistedDecision != null) return persistedDecision

  const viteEnv = (import.meta as unknown as { env?: Record<string, unknown> }).env
  const viteDecision = parseWorkspaceFeatureFlagCandidate(viteEnv?.[viteEnvKey])
  if (viteDecision != null) return viteDecision

  const nextCandidate =
    typeof process !== "undefined"
      ? (process as { env?: Record<string, string | undefined> }).env?.[
          nextEnvKey
        ]
      : undefined
  const nextDecision = parseWorkspaceFeatureFlagCandidate(nextCandidate)
  if (nextDecision != null) return nextDecision

  return defaultEnabled
}

const isWorkspaceSplitKeyStorageEnabled = (): boolean =>
  resolveWorkspaceStorageFeatureFlag({
    localStorageKey: WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY,
    viteEnvKey: WORKSPACE_STORAGE_SPLIT_KEY_FLAG_VITE_ENV,
    nextEnvKey: WORKSPACE_STORAGE_SPLIT_KEY_FLAG_NEXT_ENV,
    defaultEnabled: WORKSPACE_STORAGE_SPLIT_KEY_DEFAULT_ENABLED
  })

const isWorkspaceIndexedDbOffloadEnabled = (): boolean =>
  resolveWorkspaceStorageFeatureFlag({
    localStorageKey: WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY,
    viteEnvKey: WORKSPACE_STORAGE_INDEXEDDB_FLAG_VITE_ENV,
    nextEnvKey: WORKSPACE_STORAGE_INDEXEDDB_FLAG_NEXT_ENV,
    defaultEnabled: WORKSPACE_STORAGE_INDEXEDDB_DEFAULT_ENABLED
  })

const isPromiseLike = <T>(candidate: T | Promise<T>): candidate is Promise<T> =>
  Boolean(candidate) && typeof (candidate as Promise<T>).then === "function"

const buildIndexedDbArtifactPayload = (
  artifact: GeneratedArtifact
): {
  payload: WorkspaceIndexedDbArtifactPayloadRecord["payload"]
  fields: Array<"content" | "data">
} => {
  const payload: WorkspaceIndexedDbArtifactPayloadRecord["payload"] = {}
  const fields: Array<"content" | "data"> = []

  if (typeof artifact.content === "string") {
    payload.content = artifact.content
    fields.push("content")
  }

  if (isRecord(artifact.data)) {
    payload.data = cloneWorkspaceValue(artifact.data as Record<string, unknown>)
    fields.push("data")
  }

  return { payload, fields }
}

// 64-bit-flavored string hash (two independent 32-bit lanes + length): used
// to recognize unchanged artifact payloads cheaply. A collision would need to
// match both lanes AND the length; the fallback failure mode is one redundant
// IndexedDB put, never data loss (a changed payload with a colliding hash
// would skip a put whose stored record already holds different bytes — the
// length lane makes that practically impossible for real payloads).
const hashWorkspacePersistString = (value: string): string => {
  let laneOne = 5381
  let laneTwo = 52711
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    laneOne = (Math.imul(laneOne, 33) ^ code) >>> 0
    laneTwo = (Math.imul(laneTwo, 31) + code) >>> 0
  }
  return `${value.length}:${laneOne}:${laneTwo}`
}

const hashWorkspaceArtifactPayload = (
  payload: WorkspaceIndexedDbArtifactPayloadRecord["payload"]
): string => {
  const dataJson =
    payload.data === undefined ? "" : JSON.stringify(payload.data)
  return `${hashWorkspacePersistString(payload.content ?? "")}|${hashWorkspacePersistString(dataJson)}`
}

// Offload dedup registry (WP-16): adapter -> (record key -> payload hash +
// the updatedAt the stored pointer carries). Scoped PER ADAPTER INSTANCE: the
// production adapter is a module singleton, so its stamps faithfully describe
// its IndexedDB records, while a freshly created adapter (tests, rollout
// toggles) starts with an empty namespace instead of inheriting stamps for
// records it does not hold. Entries are invalidated whenever the
// corresponding IndexedDB record is deleted, so a removed-then-re-added
// artifact with identical content is re-put rather than pointed at a deleted
// record.
const workspaceArtifactOffloadStampsByAdapter = new WeakMap<
  WorkspaceIndexedDbAdapter,
  Map<string, { hash: string; updatedAt: number }>
>()

const getWorkspaceArtifactOffloadStamps = (
  adapter: WorkspaceIndexedDbAdapter
): Map<string, { hash: string; updatedAt: number }> => {
  let stamps = workspaceArtifactOffloadStampsByAdapter.get(adapter)
  if (!stamps) {
    stamps = new Map()
    workspaceArtifactOffloadStampsByAdapter.set(adapter, stamps)
  }
  return stamps
}

const invalidateWorkspaceArtifactOffloadStamps = (
  adapter: WorkspaceIndexedDbAdapter,
  keys: Iterable<string>
): void => {
  const stamps = workspaceArtifactOffloadStampsByAdapter.get(adapter)
  if (!stamps) return
  for (const key of keys) {
    stamps.delete(key)
  }
}

const offloadWorkspaceSnapshotArtifacts = async (
  workspaceId: string,
  snapshot: WorkspaceSnapshot,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): Promise<{
  snapshot: WorkspaceSnapshot
  offloadedArtifactIds: Set<string>
}> => {
  if (!indexedDbAdapter.isAvailable()) {
    return { snapshot, offloadedArtifactIds: new Set<string>() }
  }

  const updatedAt = Date.now()

  type ArtifactOffloadOutcome =
    | { kind: "inline"; artifact: GeneratedArtifact }
    | {
        kind: "pointer"
        artifact: GeneratedArtifact
        key: string
        fields: Array<"content" | "data">
        pointerUpdatedAt: number
      }

  const outcomes: ArtifactOffloadOutcome[] = []
  const pendingPuts: Array<{
    index: number
    artifact: GeneratedArtifact
    key: string
    payload: WorkspaceIndexedDbArtifactPayloadRecord["payload"]
    fields: Array<"content" | "data">
    hash: string
  }> = []

  const artifacts = snapshot.generatedArtifacts || []
  for (let index = 0; index < artifacts.length; index += 1) {
    const artifact = artifacts[index]
    if (!artifact || typeof artifact.id !== "string") {
      outcomes.push({ kind: "inline", artifact })
      continue
    }

    const { payload, fields } = buildIndexedDbArtifactPayload(artifact)
    if (
      fields.length === 0 ||
      estimateSerializedByteLength(payload) < WORKSPACE_ARTIFACT_OFFLOAD_MIN_BYTES
    ) {
      outcomes.push({ kind: "inline", artifact })
      continue
    }

    const key = buildWorkspaceIndexedDbArtifactRecordKey(workspaceId, artifact.id)
    const hash = hashWorkspaceArtifactPayload(payload)
    const stamp = getWorkspaceArtifactOffloadStamps(indexedDbAdapter).get(key)
    if (stamp && stamp.hash === hash) {
      // Already offloaded with identical bytes: keep the existing record and
      // the original pointer timestamp so the persisted snapshot value stays
      // byte-stable for the per-workspace diff.
      outcomes.push({
        kind: "pointer",
        artifact,
        key,
        fields,
        pointerUpdatedAt: stamp.updatedAt
      })
      continue
    }

    pendingPuts.push({ index, artifact, key, payload, fields, hash })
    outcomes.push({ kind: "inline", artifact })
  }

  // Parallelize the remaining puts: one IndexedDB transaction per artifact,
  // issued together instead of sequentially.
  await Promise.all(
    pendingPuts.map(async (entry) => {
      const didPersist = await indexedDbAdapter.putArtifactPayloadRecord({
        key: entry.key,
        workspaceId,
        artifactId: entry.artifact.id,
        payload: entry.payload,
        updatedAt
      })
      if (!didPersist) return

      getWorkspaceArtifactOffloadStamps(indexedDbAdapter).set(entry.key, {
        hash: entry.hash,
        updatedAt
      })
      outcomes[entry.index] = {
        kind: "pointer",
        artifact: entry.artifact,
        key: entry.key,
        fields: entry.fields,
        pointerUpdatedAt: updatedAt
      }
    })
  )

  const persistedArtifacts: GeneratedArtifact[] = []
  const offloadedArtifactIds = new Set<string>()
  for (const outcome of outcomes) {
    if (outcome.kind === "inline") {
      persistedArtifacts.push({ ...outcome.artifact })
      continue
    }

    const nextArtifact: PersistedWorkspaceArtifact = {
      ...outcome.artifact,
      [WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY]: {
        offloadType: WORKSPACE_ARTIFACT_POINTER_KIND,
        key: outcome.key,
        fields: outcome.fields,
        updatedAt: outcome.pointerUpdatedAt
      }
    }
    if (outcome.fields.includes("content")) {
      delete nextArtifact.content
    }
    if (outcome.fields.includes("data")) {
      delete nextArtifact.data
    }

    persistedArtifacts.push(nextArtifact)
    if (typeof outcome.artifact.id === "string") {
      offloadedArtifactIds.add(outcome.artifact.id)
    }
  }

  return {
    snapshot: {
      ...snapshot,
      generatedArtifacts: persistedArtifacts as GeneratedArtifact[]
    },
    offloadedArtifactIds
  }
}

const rehydrateWorkspaceSnapshotArtifacts = async (
  snapshot: WorkspaceSnapshot,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): Promise<WorkspaceSnapshot> => {
  if (!Array.isArray(snapshot.generatedArtifacts)) {
    return snapshot
  }

  const hydratedArtifacts: GeneratedArtifact[] = []
  for (const artifact of snapshot.generatedArtifacts) {
    const clonedArtifact: PersistedWorkspaceArtifact = { ...artifact }
    const pointer = getWorkspaceArtifactPayloadPointer(clonedArtifact)

    if (pointer && indexedDbAdapter.isAvailable()) {
      const payloadRecord = await indexedDbAdapter.getArtifactPayloadRecord(
        pointer.key
      )
      if (payloadRecord?.payload) {
        if (
          pointer.fields.includes("content") &&
          typeof payloadRecord.payload.content === "string"
        ) {
          clonedArtifact.content = payloadRecord.payload.content
        }
        if (
          pointer.fields.includes("data") &&
          isRecord(payloadRecord.payload.data)
        ) {
          clonedArtifact.data = cloneWorkspaceValue(payloadRecord.payload.data)
        }
      }
    }

    if (WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY in clonedArtifact) {
      delete clonedArtifact[WORKSPACE_ARTIFACT_PAYLOAD_POINTER_KEY]
    }
    hydratedArtifacts.push(clonedArtifact)
  }

  return {
    ...snapshot,
    generatedArtifacts: hydratedArtifacts
  }
}

const rehydrateWorkspaceChatSessionReference = async (
  reference: PersistedWorkspaceChatSessionReference,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): Promise<PersistedWorkspaceChatSession | null> => {
  if (isWorkspaceIndexedDbChatPointer(reference)) {
    if (!indexedDbAdapter.isAvailable()) {
      return {
        messages: [],
        historyId: reference.historyId,
        serverChatId: reference.serverChatId
      }
    }
    const chatRecord = await indexedDbAdapter.getChatRecord(reference.key)
    if (chatRecord?.session && isRecord(chatRecord.session)) {
      return chatRecord.session as PersistedWorkspaceChatSession
    }
    return {
      messages: [],
      historyId: reference.historyId,
      serverChatId: reference.serverChatId
    }
  }

  if (!isRecord(reference)) return null
  return reference as PersistedWorkspaceChatSession
}

const cleanupWorkspaceIndexedDbRecords = async (
  workspaceId: string,
  snapshot: unknown,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): Promise<void> => {
  if (!indexedDbAdapter.isAvailable()) return

  const artifactIds = collectWorkspaceArtifactIdsFromSnapshot(snapshot)
  const artifactPointerKeys = collectWorkspaceArtifactPointerKeysFromSnapshot(snapshot)
  const artifactKeys = new Set<string>()
  for (const artifactId of artifactIds) {
    artifactKeys.add(buildWorkspaceIndexedDbArtifactRecordKey(workspaceId, artifactId))
  }
  for (const key of artifactPointerKeys) {
    artifactKeys.add(key)
  }

  await indexedDbAdapter.deleteChatRecord(
    buildWorkspaceIndexedDbChatRecordKey(workspaceId)
  )
  invalidateWorkspaceArtifactOffloadStamps(indexedDbAdapter, artifactKeys)
  await Promise.all(
    Array.from(artifactKeys, (key) =>
      indexedDbAdapter.deleteArtifactPayloadRecord(key)
    )
  )
}

const removeWorkspaceLocalPersistenceForId = async (
  workspaceId: string,
  indexedDbAdapter: WorkspaceIndexedDbAdapter,
  durability: WorkspacePersistDurability = "coalesced"
): Promise<void> => {
  const snapshot = readWorkspaceSnapshotFromStorage(workspaceId)
  const chatKey = buildWorkspaceChatStorageKey(workspaceId)
  const chatCandidate = safeParseJson(localStorage.getItem(chatKey))
  const chatPointerKey = isWorkspaceIndexedDbChatPointer(chatCandidate)
    ? chatCandidate.key
    : null

  localStorage.removeItem(buildWorkspaceSnapshotStorageKey(workspaceId))
  localStorage.removeItem(chatKey)
  invalidateWorkspacePersistStamp(workspaceId)
  invalidateWorkspacePersistIndexStamp()

  if (!indexedDbAdapter.isAvailable()) return

  const cleanupIndexedDbRecords = cleanupWorkspaceIndexedDbRecords(
    workspaceId,
    snapshot,
    indexedDbAdapter
  ).then(async () => {
    if (chatPointerKey) {
      await indexedDbAdapter.deleteChatRecord(chatPointerKey)
    }
  })
  if (durability === "lifecycle") {
    // Best-effort on lifecycle flush: the localStorage removals above already
    // landed synchronously; awaiting IndexedDB here would put the split-index
    // write of the calling pass at risk of page death.
    void cleanupIndexedDbRecords.catch(() => {})
    return
  }
  await cleanupIndexedDbRecords
}

const omitTombstonedWorkspaceRecords = <T>(
  records: Record<string, T>,
  tombstonedWorkspaceIds: Set<string>
): Record<string, T> => {
  if (tombstonedWorkspaceIds.size === 0) return records

  const filtered: Record<string, T> = {}
  for (const [workspaceId, value] of Object.entries(records)) {
    if (!tombstonedWorkspaceIds.has(workspaceId)) {
      filtered[workspaceId] = value
    }
  }
  return filtered
}

const omitTombstonedSavedWorkspaces = (
  workspaces: SavedWorkspace[],
  tombstonedWorkspaceIds: Set<string>
): SavedWorkspace[] => {
  if (tombstonedWorkspaceIds.size === 0) return workspaces
  return workspaces.filter(
    (workspace) => !tombstonedWorkspaceIds.has(workspace.id)
  )
}

const collectTombstonedSavedWorkspaceIds = (workspaces: unknown[]): Set<string> =>
  new Set(
    workspaces
      .map((workspace) => (isRecord(workspace) ? workspace.id : null))
      .filter(
        (workspaceId): workspaceId is string =>
          typeof workspaceId === "string" &&
          hasResearchWorkspaceMigrationTombstone(workspaceId)
      )
  )

const hasPersistableWorkspaceIndexState = (
  state: PersistedWorkspaceState
): boolean =>
  Boolean(state.workspaceId) ||
  state.savedWorkspaces.length > 0 ||
  state.archivedWorkspaces.length > 0 ||
  state.workspaceCollections.length > 0 ||
  Object.keys(state.workspaceSnapshots).length > 0 ||
  Object.keys(state.workspaceChatSessions).length > 0

const parsePersistedWorkspaceEnvelope = (
  serializedValue: string
): { state: Record<string, unknown>; version: number } | null => {
  const parsed = safeParseJson(serializedValue)
  if (!isRecord(parsed)) return null

  if (
    parsed.schema === WORKSPACE_SPLIT_INDEX_SCHEMA &&
    typeof parsed.version === "number" &&
    isRecord(parsed.state)
  ) {
    return {
      state: parsed.state,
      version: parsed.version
    }
  }

  const candidateState =
    isRecord(parsed.state) ? parsed.state : parsed
  if (!isRecord(candidateState)) return null

  const version =
    typeof parsed.version === "number" && Number.isFinite(parsed.version)
      ? parsed.version
      : 1

  return { state: candidateState, version }
}

const shouldSuppressInitialEmptyWorkspaceWrite = (
  name: string,
  serializedValue: string
): boolean => {
  if (
    name !== WORKSPACE_STORAGE_KEY ||
    localStorage.getItem(name) !== null
  ) {
    return false
  }

  const envelope = parsePersistedWorkspaceEnvelope(serializedValue)
  if (!envelope) return false

  return !hasPersistableWorkspaceIndexState(
    migratePersistedWorkspaceState(envelope.state)
  )
}

const isWorkspaceSplitIndexEnvelope = (
  candidate: unknown
): candidate is WorkspaceSplitIndexEnvelope => {
  if (!isRecord(candidate)) return false
  if (candidate.schema !== WORKSPACE_SPLIT_INDEX_SCHEMA) return false
  if (typeof candidate.version !== "number") return false
  if (!isRecord(candidate.state)) return false
  return true
}

const getWorkspaceIdsFromStoredValue = (raw: string | null): string[] => {
  const parsed = safeParseJson(raw)
  if (!parsed) return []

  if (isWorkspaceSplitIndexEnvelope(parsed)) {
    const ids = Array.isArray(parsed.state.workspaceIds)
      ? parsed.state.workspaceIds.filter(
          (workspaceId): workspaceId is string => typeof workspaceId === "string"
        )
      : []
    return normalizeWorkspaceStorageIds(...ids, parsed.state.workspaceId)
  }

  const envelope = parsePersistedWorkspaceEnvelope(raw || "")
  if (!envelope) return []

  const workspaceSnapshots = isRecord(envelope.state.workspaceSnapshots)
    ? envelope.state.workspaceSnapshots
    : {}
  const workspaceChatSessions = isRecord(envelope.state.workspaceChatSessions)
    ? envelope.state.workspaceChatSessions
    : {}
  const workspaceId =
    typeof envelope.state.workspaceId === "string"
      ? envelope.state.workspaceId
      : null

  return normalizeWorkspaceStorageIds(
    ...Object.keys(workspaceSnapshots),
    ...Object.keys(workspaceChatSessions),
    workspaceId
  )
}

const buildWorkspaceSplitIndexEnvelope = (
  persistedState: PersistedWorkspaceState,
  version: number,
  options?: {
    workspaceSnapshots?: Record<string, WorkspaceSnapshot>
    workspaceChatSessions?: Record<string, PersistedWorkspaceChatSessionReference>
  }
): WorkspaceSplitIndexEnvelope => {
  const workspaceSnapshots =
    options?.workspaceSnapshots || persistedState.workspaceSnapshots
  const workspaceChatSessions =
    options?.workspaceChatSessions || persistedState.workspaceChatSessions
  const workspaceIds = normalizeWorkspaceStorageIds(
    ...Object.keys(workspaceSnapshots || {}),
    ...Object.keys(workspaceChatSessions || {}),
    persistedState.workspaceId
  )
  const activeWorkspaceId = persistedState.workspaceId
  const activeSnapshot =
    activeWorkspaceId && workspaceSnapshots[activeWorkspaceId]
      ? { [activeWorkspaceId]: workspaceSnapshots[activeWorkspaceId] }
      : {}
  const activeChatSession =
    activeWorkspaceId && workspaceChatSessions[activeWorkspaceId]
      ? { [activeWorkspaceId]: workspaceChatSessions[activeWorkspaceId] }
      : {}

  return {
    schema: WORKSPACE_SPLIT_INDEX_SCHEMA,
    splitVersion: WORKSPACE_SPLIT_INDEX_VERSION,
    version,
    state: {
      workspaceId: persistedState.workspaceId,
      savedWorkspaces: persistedState.savedWorkspaces,
      archivedWorkspaces: persistedState.archivedWorkspaces,
      workspaceCollections: persistedState.workspaceCollections,
      workspaceIds,
      workspaceSnapshots: activeSnapshot,
      workspaceChatSessions: activeChatSession
    }
  }
}

const reconstructPersistedWorkspaceStateFromSplitIndex = (
  envelope: WorkspaceSplitIndexEnvelope,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): PersistedWorkspaceState | Promise<PersistedWorkspaceState> => {
  const stateCandidate = envelope.state
  const candidateWorkspaceId =
    typeof stateCandidate.workspaceId === "string"
      ? stateCandidate.workspaceId
      : ""
  const workspaceId =
    candidateWorkspaceId &&
    !hasResearchWorkspaceMigrationTombstone(candidateWorkspaceId)
      ? candidateWorkspaceId
      : ""
  const savedWorkspaces = Array.isArray(stateCandidate.savedWorkspaces)
    ? omitTombstonedSavedWorkspaces(
        stateCandidate.savedWorkspaces as SavedWorkspace[],
        collectTombstonedSavedWorkspaceIds(stateCandidate.savedWorkspaces)
      )
    : []
  const archivedWorkspaces = Array.isArray(stateCandidate.archivedWorkspaces)
    ? omitTombstonedSavedWorkspaces(
        stateCandidate.archivedWorkspaces as SavedWorkspace[],
        collectTombstonedSavedWorkspaceIds(stateCandidate.archivedWorkspaces)
      )
    : []
  const workspaceCollections = Array.isArray(stateCandidate.workspaceCollections)
    ? (stateCandidate.workspaceCollections as WorkspaceCollection[])
    : []
  const workspaceIds = Array.isArray(stateCandidate.workspaceIds)
    ? stateCandidate.workspaceIds.filter(
        (workspaceStorageId): workspaceStorageId is string =>
          typeof workspaceStorageId === "string" &&
          !hasResearchWorkspaceMigrationTombstone(workspaceStorageId)
      )
    : []

  const snapshots: Record<string, WorkspaceSnapshot> = {}
  const chatSessionReferences: Record<
    string,
    PersistedWorkspaceChatSessionReference
  > = {}
  let requiresIndexedDbHydration = false

  for (const workspaceStorageId of workspaceIds) {
    const snapshotRaw = localStorage.getItem(
      buildWorkspaceSnapshotStorageKey(workspaceStorageId)
    )
    const parsedSnapshot = safeParseJson(snapshotRaw)
    if (isRecord(parsedSnapshot)) {
      snapshots[workspaceStorageId] = stripWorkspaceSnapshotPersistMetadata(
        parsedSnapshot as unknown as WorkspaceSnapshot
      )
      if (hasWorkspaceArtifactPayloadPointers(parsedSnapshot)) {
        requiresIndexedDbHydration = true
      }
    }

    const chatRaw = localStorage.getItem(
      buildWorkspaceChatStorageKey(workspaceStorageId)
    )
    const parsedChat = safeParseJson(chatRaw)
    if (isRecord(parsedChat)) {
      if (isWorkspaceIndexedDbChatPointer(parsedChat)) {
        requiresIndexedDbHydration = true
      }
      chatSessionReferences[workspaceStorageId] =
        parsedChat as PersistedWorkspaceChatSessionReference
    }
  }

  if (
    workspaceId &&
    !snapshots[workspaceId] &&
    isRecord(stateCandidate.workspaceSnapshots) &&
    isRecord(stateCandidate.workspaceSnapshots[workspaceId])
  ) {
    snapshots[workspaceId] = stripWorkspaceSnapshotPersistMetadata(
      stateCandidate.workspaceSnapshots[workspaceId] as unknown as WorkspaceSnapshot
    )
    if (hasWorkspaceArtifactPayloadPointers(snapshots[workspaceId])) {
      requiresIndexedDbHydration = true
    }
  }

  if (
    workspaceId &&
    !chatSessionReferences[workspaceId] &&
    isRecord(stateCandidate.workspaceChatSessions) &&
    isRecord(stateCandidate.workspaceChatSessions[workspaceId])
  ) {
    const reference = stateCandidate.workspaceChatSessions[
      workspaceId
    ] as PersistedWorkspaceChatSessionReference
    if (isWorkspaceIndexedDbChatPointer(reference)) {
      requiresIndexedDbHydration = true
    }
    chatSessionReferences[workspaceId] =
      stateCandidate.workspaceChatSessions[
        workspaceId
      ] as PersistedWorkspaceChatSessionReference
  }

  const baseState = {
    workspaceId,
    savedWorkspaces: savedWorkspaces as SavedWorkspace[],
    archivedWorkspaces: archivedWorkspaces as SavedWorkspace[],
    workspaceCollections
  }

  if (!requiresIndexedDbHydration) {
    const chatSessions: Record<string, PersistedWorkspaceChatSession> = {}
    for (const [workspaceStorageId, reference] of Object.entries(
      chatSessionReferences
    )) {
      if (isWorkspaceIndexedDbChatPointer(reference)) continue
      if (!isRecord(reference)) continue
      chatSessions[workspaceStorageId] = reference as PersistedWorkspaceChatSession
    }
    return {
      ...baseState,
      workspaceSnapshots: snapshots,
      workspaceChatSessions: chatSessions
    }
  }

  return (async () => {
    const hydratedSnapshots: Record<string, WorkspaceSnapshot> = {}
    for (const [workspaceStorageId, snapshot] of Object.entries(snapshots)) {
      hydratedSnapshots[workspaceStorageId] = await rehydrateWorkspaceSnapshotArtifacts(
        snapshot,
        indexedDbAdapter
      )
    }

    const hydratedChatSessions: Record<string, PersistedWorkspaceChatSession> = {}
    for (const [workspaceStorageId, reference] of Object.entries(
      chatSessionReferences
    )) {
      const hydratedSession = await rehydrateWorkspaceChatSessionReference(
        reference,
        indexedDbAdapter
      )
      if (hydratedSession) {
        hydratedChatSessions[workspaceStorageId] = hydratedSession
      }
    }

    return {
      ...baseState,
      workspaceSnapshots: hydratedSnapshots,
      workspaceChatSessions: hydratedChatSessions
    }
  })()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-workspace persist diff state (WP-16)
// ─────────────────────────────────────────────────────────────────────────────
// Two module-level registries make a persist pass O(changed workspaces)
// instead of O(all workspaces):
//
//  - `workspacePersistPartializeRecords` is written by `partialize`, where the
//    RAW snapshot/chat references from store state are still identifiable. An
//    unchanged raw reference means the persisted (sanitized/bounded) build is
//    reusable — unchanged workspaces skip their sanitize pass entirely.
//  - `workspacePersistStamps` is written by the split-key write path when a
//    workspace's snapshot/chat is actually serialized and written. A stamp
//    carries the raw reference it last persisted, the metadata-free content
//    signature it produced (for the legacy-compat first pass and direct
//    adapter writes that bypass partialize), and the cached previous
//    artifact-id/pointer-key lists so unchanged workspaces perform ZERO
//    per-workspace storage reads on subsequent passes.
//
// Correctness rules:
//  - "No revision known" (missing stamp — legacy storage or first write after
//    load) is DIRTY: serialize once, write revision-carrying, then track.
//  - Identity is authoritative only together with the partialize record; a
//    write whose envelope did not come through partialize (migration path,
//    direct adapter writes) always re-serializes and compares the content
//    signature, so equal content still produces zero writes.
//  - `key in localStorage` existence guards keep an out-of-band clear (e.g.
//    tests calling localStorage.clear() directly) from being mistaken for
//    "already persisted".

type WorkspacePersistPartializeRecord<TPersisted> = {
  rawRef: unknown
  persisted: TPersisted
}

type WorkspacePersistSnapshotStamp = {
  rawRef: unknown
  persistedSnapshot: WorkspaceSnapshot | null
  contentSignature: string | null
  revision: number
  previousArtifactIds: string[]
  previousArtifactPointerKeys: string[]
}

type WorkspacePersistChatStamp = {
  rawRef: unknown
  persistedReference: PersistedWorkspaceChatSessionReference | null
  serializedValue: string | null
  previousPointerKey: string | null
}

const workspacePersistPartializeRecords = new Map<
  string,
  {
    snapshot?: WorkspacePersistPartializeRecord<WorkspaceSnapshot>
    chat?: WorkspacePersistPartializeRecord<PersistedWorkspaceChatSession>
  }
>()

// Set by `partialize` and consumed by the NEXT `storage.setItem` call: zustand
// persist builds the persisted value as
// `JSON.stringify({ state: partialize(get()), version })` and hands it to
// setItem synchronously, so a setItem observing this flag true knows the
// module-level partialize records describe its own value (identity anchors are
// valid). Any other setItem (direct adapter writes, the migration path) must
// NOT trust the records — its envelope never passed through partialize, and a
// stale record would make the differ skip genuinely different content.
let workspacePersistPartializeRecordsFresh = false

const setWorkspacePersistPartializeRecord = (
  workspaceId: string,
  next: {
    snapshot?: WorkspacePersistPartializeRecord<WorkspaceSnapshot>
    chat?: WorkspacePersistPartializeRecord<PersistedWorkspaceChatSession>
  }
): void => {
  const merged = workspacePersistPartializeRecords.get(workspaceId) || {}
  workspacePersistPartializeRecords.set(workspaceId, { ...merged, ...next })
}

const workspacePersistStamps = new Map<
  string,
  { snapshot?: WorkspacePersistSnapshotStamp; chat?: WorkspacePersistChatStamp }
>()

let lastWrittenWorkspaceIndexValue: string | null = null

const getWorkspacePersistStamp = (
  workspaceId: string
): { snapshot?: WorkspacePersistSnapshotStamp; chat?: WorkspacePersistChatStamp } =>
  workspacePersistStamps.get(workspaceId) || {}

const setWorkspacePersistStamp = (
  workspaceId: string,
  next: {
    snapshot?: WorkspacePersistSnapshotStamp
    chat?: WorkspacePersistChatStamp
  }
): void => {
  const merged = workspacePersistStamps.get(workspaceId) || {}
  workspacePersistStamps.set(workspaceId, { ...merged, ...next })
}

const invalidateWorkspacePersistStamp = (workspaceId: string): void => {
  workspacePersistStamps.delete(workspaceId)
}

const invalidateWorkspacePersistIndexStamp = (): void => {
  lastWrittenWorkspaceIndexValue = null
}

const clearWorkspacePersistStamps = (
  indexedDbAdapter?: WorkspaceIndexedDbAdapter
): void => {
  workspacePersistStamps.clear()
  if (indexedDbAdapter) {
    workspaceArtifactOffloadStampsByAdapter.delete(indexedDbAdapter)
  }
  lastWrittenWorkspaceIndexValue = null
}

// Prefix-concatenated snapshot serialization: the internal revision metadata
// is prepended to the metadata-free content JSON, so one stringify covers both
// the diff signature and (on change) the written value.
const serializeWorkspaceSnapshotForPersist = (
  snapshotForStorage: WorkspaceSnapshot,
  revision: number,
  updatedAt: number
): { contentSignature: string; persistedValue: string } => {
  const contentSignature = JSON.stringify(snapshotForStorage)
  const persistedValue = `{"${WORKSPACE_SNAPSHOT_REVISION_FIELD}":${revision},"${WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD}":${updatedAt},${contentSignature.slice(1)}`
  return { contentSignature, persistedValue }
}

const parsePersistedSnapshotRevision = (candidate: unknown): number | null => {
  if (!isRecord(candidate)) return null
  const revision = candidate[WORKSPACE_SNAPSHOT_REVISION_FIELD]
  return typeof revision === "number" && Number.isFinite(revision)
    ? revision
    : null
}

const writeSplitWorkspacePersistence = async (
  name: string,
  serializedValue: string,
  indexedDbAdapter: WorkspaceIndexedDbAdapter,
  options?: {
    /** Trust the module-level partialize records as identity anchors. */
    trustPartializeRecords?: boolean
    /**
     * "lifecycle": this pass was flushed by pagehide/beforeunload/tab-hidden.
     * Every IndexedDB await is skipped (artifact/chat offload and record
     * reconciliation) so all localStorage writes land synchronously — page
     * death cannot lose the buffered state. Oversized artifacts stay INLINE
     * for this pass; the next coalesced pass re-offloads them.
     */
    durability?: WorkspacePersistDurability
  }
): Promise<boolean> => {
  if (name !== WORKSPACE_STORAGE_KEY) return false

  const recordsTrusted = options?.trustPartializeRecords === true
  const lifecycleFlush = options?.durability === "lifecycle"
  const envelope = parsePersistedWorkspaceEnvelope(serializedValue)
  if (!envelope) return false

  const migrated = migratePersistedWorkspaceState(envelope.state)
  const version =
    typeof envelope.version === "number" && Number.isFinite(envelope.version)
      ? envelope.version
      : 1
  const nextWorkspaceIds = normalizeWorkspaceStorageIds(
    ...Object.keys(migrated.workspaceSnapshots || {}),
    ...Object.keys(migrated.workspaceChatSessions || {}),
    migrated.workspaceId
  )
  const candidateWorkspaceIds = normalizeWorkspaceStorageIds(
    ...nextWorkspaceIds,
    ...migrated.savedWorkspaces.map((workspace) => workspace.id),
    ...migrated.archivedWorkspaces.map((workspace) => workspace.id)
  )
  const tombstonedWorkspaceIds = new Set(
    candidateWorkspaceIds.filter(hasResearchWorkspaceMigrationTombstone)
  )
  const migratedForIndex: PersistedWorkspaceState =
    tombstonedWorkspaceIds.size === 0
      ? migrated
      : {
          ...migrated,
          workspaceId: tombstonedWorkspaceIds.has(migrated.workspaceId)
            ? ""
            : migrated.workspaceId,
          savedWorkspaces: omitTombstonedSavedWorkspaces(
            migrated.savedWorkspaces,
            tombstonedWorkspaceIds
          ),
          archivedWorkspaces: omitTombstonedSavedWorkspaces(
            migrated.archivedWorkspaces,
            tombstonedWorkspaceIds
          ),
          workspaceSnapshots: omitTombstonedWorkspaceRecords(
            migrated.workspaceSnapshots,
            tombstonedWorkspaceIds
          ),
          workspaceChatSessions: omitTombstonedWorkspaceRecords(
            migrated.workspaceChatSessions,
            tombstonedWorkspaceIds
          )
        }
  const persistedWorkspaceIds = nextWorkspaceIds.filter(
    (workspaceId) => !tombstonedWorkspaceIds.has(workspaceId)
  )
  const existingWorkspaceIds = getWorkspaceIdsFromStoredValue(
    localStorage.getItem(name)
  )
  const persistedSnapshotsForIndex: Record<string, WorkspaceSnapshot> = {}
  const persistedChatReferencesForIndex: Record<
    string,
    PersistedWorkspaceChatSessionReference
  > = {}

  for (const workspaceStorageId of nextWorkspaceIds) {
    if (tombstonedWorkspaceIds.has(workspaceStorageId)) {
      if (!cleanedUpMigrationTombstoneWorkspaceIds.has(workspaceStorageId)) {
        await removeWorkspaceLocalPersistenceForId(
          workspaceStorageId,
          indexedDbAdapter,
          lifecycleFlush ? "lifecycle" : "coalesced"
        )
        cleanedUpMigrationTombstoneWorkspaceIds.add(workspaceStorageId)
      }
      continue
    }

    const snapshotKey = buildWorkspaceSnapshotStorageKey(workspaceStorageId)
    const snapshot = migrated.workspaceSnapshots[workspaceStorageId]
    const partializeSnapshotRecord =
      workspacePersistPartializeRecords.get(workspaceStorageId)?.snapshot
    const snapshotStamp = getWorkspacePersistStamp(workspaceStorageId).snapshot
    // Identity anchors are only meaningful when this envelope was produced by
    // the partialize call that wrote the records; otherwise fall back to the
    // (always-fresh, never-matching) parsed snapshot reference so the content
    // signature decides.
    const snapshotRawRef =
      recordsTrusted && partializeSnapshotRecord
        ? partializeSnapshotRecord.rawRef
        : snapshot
    const snapshotUnchanged =
      Boolean(snapshotStamp) &&
      snapshotStamp.rawRef === snapshotRawRef &&
      (snapshot
        ? snapshotStamp.persistedSnapshot !== null &&
          snapshotStamp.contentSignature !== null &&
          snapshotKey in localStorage
        : snapshotStamp.persistedSnapshot === null)

    if (snapshotUnchanged && snapshotStamp) {
      // Unchanged workspace: skip re-serialization, the per-workspace storage
      // read, the offload pass, and the artifact-key reconciliation — the
      // last persisted pass already wrote exactly this content.
      if (snapshotStamp.persistedSnapshot) {
        persistedSnapshotsForIndex[workspaceStorageId] =
          snapshotStamp.persistedSnapshot
      }
    } else {
      // Dirty: no stamp (legacy no-revision storage or first write after
      // load — always serialize once, then track), the raw reference moved,
      // or the storage key vanished out-of-band.
      const previousSnapshot = snapshotStamp
        ? snapshotStamp.persistedSnapshot
        : readWorkspaceSnapshotFromStorage(workspaceStorageId)
      const previousArtifactIds = collectWorkspaceArtifactIdsFromSnapshot(
        previousSnapshot
      )
      const previousArtifactPointerKeys =
        collectWorkspaceArtifactPointerKeysFromSnapshot(previousSnapshot)

      const nextArtifactIds = snapshot
        ? collectWorkspaceArtifactIdsFromSnapshot(snapshot)
        : []
      const retainedArtifactKeys = new Set<string>()

      if (snapshot) {
        let snapshotForStorage = snapshot
        let offloadedArtifactIds = new Set<string>()
        if (indexedDbAdapter.isAvailable() && !lifecycleFlush) {
          const offloadResult = await offloadWorkspaceSnapshotArtifacts(
            workspaceStorageId,
            snapshot,
            indexedDbAdapter
          )
          snapshotForStorage = offloadResult.snapshot
          offloadedArtifactIds = offloadResult.offloadedArtifactIds
          for (const artifactId of offloadedArtifactIds) {
            retainedArtifactKeys.add(
              buildWorkspaceIndexedDbArtifactRecordKey(
                workspaceStorageId,
                artifactId
              )
            )
          }
        }

        const previousRevision = snapshotStamp ? snapshotStamp.revision : 0
        const updatedAt = Date.now()
        const { contentSignature, persistedValue } =
          serializeWorkspaceSnapshotForPersist(
            snapshotForStorage,
            previousRevision + 1,
            updatedAt
          )
        const contentChanged =
          !snapshotStamp ||
          snapshotStamp.contentSignature !== contentSignature
        const snapshotKeyMissing = !(snapshotKey in localStorage)
        const mustWriteSnapshot = contentChanged || snapshotKeyMissing

        let persistedSnapshotForIndex: WorkspaceSnapshot
        if (mustWriteSnapshot) {
          localStorage.setItem(snapshotKey, persistedValue)
          persistedSnapshotForIndex = {
            ...snapshotForStorage,
            [WORKSPACE_SNAPSHOT_REVISION_FIELD]: previousRevision + 1,
            [WORKSPACE_SNAPSHOT_UPDATED_AT_FIELD]: updatedAt
          } as RevisionedWorkspaceSnapshot
        } else {
          persistedSnapshotForIndex =
            snapshotStamp?.persistedSnapshot ?? snapshotForStorage
        }
        persistedSnapshotsForIndex[workspaceStorageId] =
          persistedSnapshotForIndex

        setWorkspacePersistStamp(workspaceStorageId, {
          snapshot: {
            rawRef: snapshotRawRef ?? null,
            persistedSnapshot: persistedSnapshotForIndex,
            contentSignature,
            revision: mustWriteSnapshot
              ? previousRevision + 1
              : previousRevision,
            previousArtifactIds: nextArtifactIds,
            // Lifecycle flushes write artifacts INLINE, so the freshly written
            // snapshot holds no pointers — carry the previous pointer-key cache
            // forward instead so the next coalesced pass can still reconcile
            // (delete) the records this pass deliberately left in place.
            previousArtifactPointerKeys: lifecycleFlush
              ? previousArtifactPointerKeys
              : collectWorkspaceArtifactPointerKeysFromSnapshot(
                  persistedSnapshotForIndex
                )
          }
        })
      } else {
        if (snapshotKey in localStorage) {
          localStorage.removeItem(snapshotKey)
        }
        setWorkspacePersistStamp(workspaceStorageId, {
          snapshot: {
            rawRef: null,
            persistedSnapshot: null,
            contentSignature: null,
            revision: 0,
            previousArtifactIds: [],
            previousArtifactPointerKeys: []
          }
        })
      }

      if (indexedDbAdapter.isAvailable() && !lifecycleFlush) {
        const artifactPayloadKeysToDelete = new Set<string>(
          previousArtifactPointerKeys
        )
        for (const artifactId of previousArtifactIds) {
          const payloadKey = buildWorkspaceIndexedDbArtifactRecordKey(
            workspaceStorageId,
            artifactId
          )
          if (!nextArtifactIds.includes(artifactId)) {
            artifactPayloadKeysToDelete.add(payloadKey)
            continue
          }
          if (!retainedArtifactKeys.has(payloadKey)) {
            artifactPayloadKeysToDelete.add(payloadKey)
          }
        }
        for (const key of retainedArtifactKeys) {
          artifactPayloadKeysToDelete.delete(key)
        }
        invalidateWorkspaceArtifactOffloadStamps(
          indexedDbAdapter,
          artifactPayloadKeysToDelete
        )
        await Promise.all(
          Array.from(artifactPayloadKeysToDelete, (key) =>
            indexedDbAdapter.deleteArtifactPayloadRecord(key)
          )
        )
      }
    }

    const chatKey = buildWorkspaceChatStorageKey(workspaceStorageId)
    const chatSession = migrated.workspaceChatSessions[workspaceStorageId]
    const partializeChatRecord =
      workspacePersistPartializeRecords.get(workspaceStorageId)?.chat
    const chatStamp = getWorkspacePersistStamp(workspaceStorageId).chat
    const chatRawRef =
      recordsTrusted && partializeChatRecord
        ? partializeChatRecord.rawRef
        : chatSession
    const chatUnchanged =
      Boolean(chatStamp) &&
      chatStamp.rawRef === chatRawRef &&
      (chatSession
        ? chatStamp.persistedReference !== null && chatKey in localStorage
        : chatStamp.persistedReference === null)

    if (chatUnchanged && chatStamp) {
      if (chatStamp.persistedReference) {
        persistedChatReferencesForIndex[workspaceStorageId] =
          chatStamp.persistedReference
      }
    } else {
      const previousChatCandidate = chatStamp
        ? null
        : safeParseJson(localStorage.getItem(chatKey))
      const previousChatPointerKey = chatStamp
        ? chatStamp.previousPointerKey
        : isWorkspaceIndexedDbChatPointer(previousChatCandidate)
          ? previousChatCandidate.key
          : null
      let retainedChatRecordKey: string | null = null
      let nextChatStamp: WorkspacePersistChatStamp

      if (chatSession) {
        let chatReference: PersistedWorkspaceChatSessionReference = chatSession
        if (
          indexedDbAdapter.isAvailable() &&
          !lifecycleFlush &&
          estimateSerializedByteLength(chatSession) >=
            WORKSPACE_CHAT_OFFLOAD_MIN_BYTES
        ) {
          const chatRecordKey = buildWorkspaceIndexedDbChatRecordKey(
            workspaceStorageId
          )
          const updatedAt = Date.now()
          const didPersist = await indexedDbAdapter.putChatRecord({
            key: chatRecordKey,
            workspaceId: workspaceStorageId,
            session: chatSession,
            updatedAt
          })
          if (didPersist) {
            retainedChatRecordKey = chatRecordKey
            chatReference = {
              offloadType: WORKSPACE_CHAT_POINTER_KIND,
              key: chatRecordKey,
              historyId: chatSession.historyId,
              serverChatId: chatSession.serverChatId,
              updatedAt
            }
          }
        }

        const nextChatValue = JSON.stringify(chatReference)
        if (
          !chatStamp ||
          chatStamp.serializedValue !== nextChatValue ||
          !(chatKey in localStorage)
        ) {
          localStorage.setItem(chatKey, nextChatValue)
        }
        persistedChatReferencesForIndex[workspaceStorageId] = chatReference
        nextChatStamp = {
          rawRef: chatRawRef ?? null,
          persistedReference: chatReference,
          serializedValue: nextChatValue,
          // Lifecycle flushes skip the chat offload, so no record was
          // retained this pass — carry the previous pointer forward so the
          // next coalesced pass can still reconcile the record left in place.
          previousPointerKey: lifecycleFlush
            ? previousChatPointerKey
            : retainedChatRecordKey
        }
      } else {
        if (chatKey in localStorage) {
          localStorage.removeItem(chatKey)
        }
        nextChatStamp = {
          rawRef: null,
          persistedReference: null,
          serializedValue: null,
          previousPointerKey: null
        }
      }

      if (indexedDbAdapter.isAvailable() && !lifecycleFlush) {
        const chatKeysToDelete = new Set<string>()
        chatKeysToDelete.add(
          buildWorkspaceIndexedDbChatRecordKey(workspaceStorageId)
        )
        if (previousChatPointerKey) {
          chatKeysToDelete.add(previousChatPointerKey)
        }
        if (retainedChatRecordKey) {
          chatKeysToDelete.delete(retainedChatRecordKey)
        }
        await Promise.all(
          Array.from(chatKeysToDelete, (key) =>
            indexedDbAdapter.deleteChatRecord(key)
          )
        )
      }

      setWorkspacePersistStamp(workspaceStorageId, { chat: nextChatStamp })
    }
  }

  for (const staleWorkspaceId of existingWorkspaceIds) {
    if (persistedWorkspaceIds.includes(staleWorkspaceId)) continue

    const staleSnapshot = readWorkspaceSnapshotFromStorage(staleWorkspaceId)
    const staleChatRaw = safeParseJson(
      localStorage.getItem(buildWorkspaceChatStorageKey(staleWorkspaceId))
    )
    const staleChatPointerKey = isWorkspaceIndexedDbChatPointer(staleChatRaw)
      ? staleChatRaw.key
      : null

    localStorage.removeItem(buildWorkspaceSnapshotStorageKey(staleWorkspaceId))
    localStorage.removeItem(buildWorkspaceChatStorageKey(staleWorkspaceId))
    invalidateWorkspacePersistStamp(staleWorkspaceId)
    invalidateWorkspacePersistIndexStamp()

    if (indexedDbAdapter.isAvailable()) {
      const staleRecordCleanup = cleanupWorkspaceIndexedDbRecords(
        staleWorkspaceId,
        staleSnapshot,
        indexedDbAdapter
      ).then(async () => {
        if (staleChatPointerKey) {
          await indexedDbAdapter.deleteChatRecord(staleChatPointerKey)
        }
      })
      if (lifecycleFlush) {
        // Best-effort on lifecycle flush: localStorage removals above already
        // landed; do not gate the split-index write on IndexedDB deletes.
        void staleRecordCleanup.catch(() => {})
      } else {
        await staleRecordCleanup
      }
    }
  }

  if (
    tombstonedWorkspaceIds.size > 0 &&
    !hasPersistableWorkspaceIndexState(migratedForIndex)
  ) {
    localStorage.removeItem(name)
    invalidateWorkspacePersistIndexStamp()
    return true
  }

  const splitIndex = buildWorkspaceSplitIndexEnvelope(migratedForIndex, version, {
    workspaceSnapshots: {
      ...migratedForIndex.workspaceSnapshots,
      ...persistedSnapshotsForIndex
    },
    workspaceChatSessions: {
      ...migratedForIndex.workspaceChatSessions,
      ...persistedChatReferencesForIndex
    }
  })
  const indexValue = JSON.stringify(splitIndex)
  if (
    indexValue !== lastWrittenWorkspaceIndexValue ||
    localStorage.getItem(name) !== indexValue
  ) {
    localStorage.setItem(name, indexValue)
  }
  lastWrittenWorkspaceIndexValue = indexValue

  return true
}

const rebuildWorkspaceEnvelopeFromStorage = (
  name: string,
  indexedDbAdapter: WorkspaceIndexedDbAdapter
): string | null | Promise<string | null> => {
  const raw = localStorage.getItem(name)
  if (raw === null) return null

  const parsed = safeParseJson(raw)
  if (!isWorkspaceSplitIndexEnvelope(parsed)) {
    const envelope = parsePersistedWorkspaceEnvelope(raw)
    if (!envelope) {
      return raw
    }
    const migrated = migratePersistedWorkspaceState(envelope.state)
    const migratedEnvelope = JSON.stringify({
      state: migrated,
      version: envelope.version
    })
    // Best-effort migration to split-key storage on first read.
    void writeSplitWorkspacePersistence(
      name,
      migratedEnvelope,
      indexedDbAdapter
    ).catch(() => {
      // Ignore migration failures and continue with in-memory rehydrate value.
    })
    return migratedEnvelope
  }

  const reconstructed = reconstructPersistedWorkspaceStateFromSplitIndex(
    parsed,
    indexedDbAdapter
  )
  if (isPromiseLike(reconstructed)) {
    return reconstructed.then((state) =>
      JSON.stringify({
        state,
        version: parsed.version
      })
    )
  }

  return JSON.stringify({
    state: reconstructed,
    version: parsed.version
  })
}

const getWorkspaceBroadcastChannel = (): BroadcastChannel | null => {
  if (typeof window === "undefined" || typeof BroadcastChannel === "undefined") {
    return null
  }

  if (!workspaceBroadcastChannel) {
    workspaceBroadcastChannel = new BroadcastChannel(
      WORKSPACE_STORAGE_CHANNEL_NAME
    )
  }

  return workspaceBroadcastChannel
}

const broadcastWorkspaceStorageUpdate = (key: string): void => {
  if (!isWorkspaceBroadcastSyncEnabled()) return

  const channel = getWorkspaceBroadcastChannel()
  if (!channel) return

  const payload: WorkspaceBroadcastUpdateMessage = {
    type: "workspace-storage-updated",
    key,
    updatedAt: Date.now()
  }

  try {
    channel.postMessage(payload)
  } catch (error) {
    console.warn("Workspace broadcast sync unavailable", error)
  }
}

/**
 * Creates a memory storage fallback for SSR environments
 */
const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

// ─────────────────────────────────────────────────────────────────────────────
// Coalescing persist writer (WP-16)
// ─────────────────────────────────────────────────────────────────────────────
// Zustand persist calls `storage.setItem` synchronously on EVERY `set()`, so a
// burst of mutations each ran the whole split-key persist chain (partialize ->
// stringify -> parse -> per-workspace stringify/compare). This writer coalesces
// them using the serialized-writer pattern from
// `entries/background-session-store.ts` (`createSerializedSessionStateWriter`).
// Pending jobs are keyed BY STORAGE KEY: a write for a different key arriving
// inside the coalescing window (e.g. the Research onboarding sidecar) can
// never displace the buffered workspace envelope, while a same-key write
// replaces the pending job for that key and inherits its waiters (the newest
// value supersedes theirs too). While a write is in flight further calls for
// that key just replace the pending job, and one final trailing write runs
// with the newest value once the in-flight write settles. Callers are never
// blocked and an older snapshot can never overwrite a newer one.
//
// A buffered write is delivered on the FIRST of:
//   - `requestIdleCallback`, where available (a delivery hint that keeps the
//     write off the keystroke's critical path),
//   - the microtask queue in environments without `requestIdleCallback`
//     (jsdom), so single mutations still land promptly in tests,
//   - the trailing timer: WORKSPACE_PERSIST_TRAILING_FLUSH_MS after the write
//     was buffered. This is the worst-case landing bound — the plan's "300ms
//     trailing" debounce — and it is unconditional: a busy main thread can
//     delay idle delivery, never the timer.
//
// Lifecycle flushes (pagehide/beforeunload/tab hidden) drain immediately in
// "lifecycle" durability mode: IndexedDB offload is skipped for that pass so
// every localStorage write (snapshots, chat references, split index) lands
// synchronously before the flush call returns — page death cannot lose the
// buffered state. The next coalesced pass re-offloads oversized artifacts and
// pointer-izes the snapshot again.

export const WORKSPACE_PERSIST_TRAILING_FLUSH_MS = 300

type WorkspacePersistWriteJob = {
  name: string
  value: string
  perform: (name: string, value: string) => Promise<void>
  waiters: WorkspacePersistWaiter[]
}

type WorkspacePersistWaiter = {
  resolve: () => void
  reject: (error: unknown) => void
}

type WorkspacePersistDurability = "coalesced" | "lifecycle"

const pendingWorkspacePersistJobs = new Map<string, WorkspacePersistWriteJob>()
let workspacePersistDrainInFlight: Promise<void> | null = null
let workspacePersistDrainCompleted = true
let workspacePersistDrainScheduled = false
let workspacePersistIdleHandle: number | null = null
let workspacePersistTrailingHandle: ReturnType<typeof setTimeout> | null = null
let workspacePersistLifecycleBound = false
// True while the current drain was initiated by a lifecycle flush (or overlaps
// one): jobs performed under it skip IndexedDB awaits so their localStorage
// writes land synchronously. Cleared when the drain settles.
let workspacePersistLifecycleFlushActive = false

const hasPendingWorkspacePersistWrite = (): boolean =>
  pendingWorkspacePersistJobs.size > 0
const settleWorkspacePersistWaiters = (
  waiters: WorkspacePersistWaiter[],
  error?: unknown
): void => {
  for (const waiter of waiters) {
    if (error) {
      waiter.reject(error)
    } else {
      waiter.resolve()
    }
  }
}

const drainWorkspacePersistWrites = async (): Promise<void> => {
  workspacePersistDrainCompleted = false
  try {
    // Keep writing while newer jobs arrived. There is no `await` between the
    // loop's pending check and the final `inFlight = null`, so a caller can
    // only ever observe `inFlight` set (we are suspended on the write below)
    // or fully idle — never in between (same ordering argument as the
    // background session writer). Jobs drain in insertion order; a same-key
    // write that replaced a pending job keeps that key's original position.
    while (pendingWorkspacePersistJobs.size > 0) {
      const nextJob = pendingWorkspacePersistJobs.entries().next()
      if (nextJob.done) break
      const [jobKey, job] = nextJob.value
      pendingWorkspacePersistJobs.delete(jobKey)
      try {
        await job.perform(job.name, job.value)
        settleWorkspacePersistWaiters(job.waiters)
      } catch (error) {
        settleWorkspacePersistWaiters(job.waiters, error)
      }
    }
  } finally {
    workspacePersistDrainCompleted = true
    workspacePersistDrainInFlight = null
    workspacePersistLifecycleFlushActive = false
  }
}

const startWorkspacePersistDrain = (): void => {
  if (workspacePersistDrainInFlight) return
  const drainPromise = drainWorkspacePersistWrites()
  // A drain whose queue was already empty (or whose writes completed without
  // ever suspending) runs to completion synchronously and releases the
  // in-flight slot itself — before the assignment above would land. Re-claiming
  // the slot with that settled promise would wedge the writer forever, so only
  // take ownership when the drain is genuinely still running.
  if (workspacePersistDrainCompleted) return
  workspacePersistDrainInFlight = drainPromise
}

const cancelScheduledWorkspacePersistDrain = (): void => {
  if (
    workspacePersistIdleHandle !== null &&
    typeof cancelIdleCallback === "function"
  ) {
    cancelIdleCallback(workspacePersistIdleHandle)
  }
  workspacePersistIdleHandle = null
  if (workspacePersistTrailingHandle !== null) {
    clearTimeout(workspacePersistTrailingHandle)
    workspacePersistTrailingHandle = null
  }
  workspacePersistDrainScheduled = false
}

const scheduleWorkspacePersistDrain = (): void => {
  if (workspacePersistDrainScheduled || workspacePersistDrainInFlight) return
  workspacePersistDrainScheduled = true
  // First delivery path to fire wins; deliver() cancels the others. The
  // trailing timer is unconditional — it is the worst-case landing bound, not
  // a fallback for missing idle callbacks.
  const deliverPendingWorkspacePersistWrite = (): void => {
    if (!workspacePersistDrainScheduled) return
    cancelScheduledWorkspacePersistDrain()
    startWorkspacePersistDrain()
  }
  workspacePersistTrailingHandle = setTimeout(
    deliverPendingWorkspacePersistWrite,
    WORKSPACE_PERSIST_TRAILING_FLUSH_MS
  )
  if (typeof requestIdleCallback === "function") {
    workspacePersistIdleHandle = requestIdleCallback(
      deliverPendingWorkspacePersistWrite,
      { timeout: WORKSPACE_PERSIST_TRAILING_FLUSH_MS }
    )
    return
  }
  // No `requestIdleCallback` (jsdom): deliver on the microtask queue. The
  // trailing timer above still bounds the write if even microtask delivery
  // were starved.
  queueMicrotask(deliverPendingWorkspacePersistWrite)
}

/**
 * Flush any pending persist write NOW. The drain starts synchronously. In
 * "lifecycle" durability mode (pagehide/beforeunload/tab hidden) IndexedDB
 * offload is skipped for the drained pass, so every localStorage write lands
 * before this returns — page unload cannot lose the buffered state.
 */
const flushWorkspacePersistWrites = (
  durability: WorkspacePersistDurability = "coalesced"
): void => {
  if (durability === "lifecycle") {
    workspacePersistLifecycleFlushActive = true
  }
  cancelScheduledWorkspacePersistDrain()
  startWorkspacePersistDrain()
}

/**
 * Read-side drain: awaits any in-flight write chain and drains whatever is
 * still buffered (bounded loop — each cycle either awaits a drain or starts
 * one, and a drain's loop consumes everything pending). Production readers
 * that must observe their own just-buffered write (use-research-workspace-
 * prefill verifying an import persisted) call this before reading, keeping
 * getItem itself storage-wins for rehydration.
 */
export const flushWorkspacePersistence = async (): Promise<void> => {
  for (let cycle = 0; cycle < 10; cycle += 1) {
    if (
      !hasPendingWorkspacePersistWrite() &&
      !workspacePersistDrainInFlight
    ) {
      return
    }
    flushWorkspacePersistWrites()
    if (workspacePersistDrainInFlight) {
      await workspacePersistDrainInFlight.catch(() => {})
    }
  }
}

const enqueueWorkspacePersistWrite = (
  job: {
    name: string
    value: string
    perform: (name: string, value: string) => Promise<void>
  }
): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    // Per-key single slot: a write for a DIFFERENT key must never displace a
    // buffered job for another key; a same-key write replaces the pending job
    // and carries its waiters forward (the newest value satisfies them).
    const previous = pendingWorkspacePersistJobs.get(job.name)
    pendingWorkspacePersistJobs.set(job.name, {
      ...job,
      waiters: [...(previous?.waiters ?? []), { resolve, reject }]
    })
    bindWorkspacePersistLifecycleFlush()
    scheduleWorkspacePersistDrain()
  })
}

const handleWorkspacePersistLifecycleFlush = (): void => {
  flushWorkspacePersistWrites("lifecycle")
}

const onWorkspacePersistVisibilityChange = (): void => {
  if (
    typeof document !== "undefined" &&
    document.visibilityState === "hidden"
  ) {
    handleWorkspacePersistLifecycleFlush()
  }
}

const bindWorkspacePersistLifecycleFlush = (): void => {
  if (
    workspacePersistLifecycleBound ||
    typeof window === "undefined" ||
    typeof document === "undefined"
  ) {
    return
  }
  window.addEventListener("pagehide", handleWorkspacePersistLifecycleFlush)
  window.addEventListener("beforeunload", handleWorkspacePersistLifecycleFlush)
  document.addEventListener(
    "visibilitychange",
    onWorkspacePersistVisibilityChange
  )
  workspacePersistLifecycleBound = true
}

/**
 * Test/inspection seam for the coalescing writer (module-scoped state cannot
 * be observed through the storage adapter alone).
 */
export const workspacePersistWriterInternals = {
  flush: flushWorkspacePersistWrites,
  hasPendingWrite: hasPendingWorkspacePersistWrite,
  cancelPendingWrite: () => {
    cancelScheduledWorkspacePersistDrain()
    const displaced = Array.from(pendingWorkspacePersistJobs.values())
    pendingWorkspacePersistJobs.clear()
    settleWorkspacePersistWaiters(
      displaced.flatMap((job) => job.waiters)
    )
  }
}

/**
 * Custom storage adapter for localStorage with SSR-safe fallback.
 * Date revival is handled in `onRehydrateStorage`.
 */
type WorkspaceStorageOptions = {
  indexedDbAdapter?: WorkspaceIndexedDbAdapter
}

export const createWorkspaceStorage = (
  options: WorkspaceStorageOptions = {}
): StateStorage => {
  if (typeof window === "undefined") {
    return createMemoryStorage()
  }

  const splitStorageEnabled = isWorkspaceSplitKeyStorageEnabled()
  const indexedDbOffloadEnabled =
    splitStorageEnabled && isWorkspaceIndexedDbOffloadEnabled()
  const indexedDbAdapter = indexedDbOffloadEnabled
    ? options.indexedDbAdapter || getWorkspaceIndexedDbAdapter()
    : noopWorkspaceIndexedDbAdapter

  return {
    getItem: (name: string): string | null | Promise<string | null> => {
      // Rehydration observes what is actually AT REST, not what is buffered:
      // a pending coalesced write reflects pre-mutation in-memory state and
      // flushing it here would clobber storage written out-of-band after the
      // write was buffered — a cross-tab write, or a caller that seeded
      // storage and rehydrates from it (the reload pattern; storage must win
      // over our buffered envelope). Store creation is therefore safe (it
      // runs before any set()), and so is zustand rehydrate.
      //
      // Production readers that need READ-YOUR-WRITES semantics — verifying a
      // just-buffered write landed, like use-research-workspace-prefill —
      // must flush first via `flushWorkspacePersistence()`; that drains (and,
      // when a write chain is mid-drain, awaits it) before they read.
      if (name === WORKSPACE_STORAGE_KEY) {
        if (!splitStorageEnabled) {
          return localStorage.getItem(name)
        }
        return rebuildWorkspaceEnvelopeFromStorage(name, indexedDbAdapter)
      }
      return localStorage.getItem(name)
    },
    setItem: (name: string, value: string): Promise<void> => {
      // Consume the partialize freshness handshake NOW (synchronously with
      // the caller): zustand persist runs partialize -> stringify -> setItem
      // in one synchronous sequence, so a fresh flag here means the module
      // records describe this exact value.
      const trustPartializeRecords = workspacePersistPartializeRecordsFresh
      workspacePersistPartializeRecordsFresh = false
      return enqueueWorkspacePersistWrite({
        name,
        value,
        perform: async (jobName, jobValue): Promise<void> => {
          try {
            if (shouldSuppressInitialEmptyWorkspaceWrite(jobName, jobValue)) {
              return
            }

            const handledBySplitStorage = splitStorageEnabled
              ? await writeSplitWorkspacePersistence(
                  jobName,
                  jobValue,
                  indexedDbAdapter,
                  {
                    trustPartializeRecords,
                    durability: workspacePersistLifecycleFlushActive
                      ? "lifecycle"
                      : "coalesced"
                  }
                )
              : false
            if (!handledBySplitStorage) {
              localStorage.setItem(jobName, jobValue)
            }
            broadcastWorkspaceStorageUpdate(jobName)
          } catch (error) {
            if (isQuotaExceededError(error)) {
              const recoveryAttempt = attemptWorkspaceStorageRecovery(
                jobName,
                jobValue
              )
              if (!recoveryAttempt) {
                emitWorkspaceStorageRecoveryEvent({
                  key: jobName,
                  action: "retry_skipped",
                  beforeBytes: estimateSerializedByteLength(jobValue),
                  afterBytes: estimateSerializedByteLength(jobValue),
                  recoveredBytes: 0,
                  reason:
                    error instanceof Error
                      ? error.message
                      : "Quota exceeded and no recoverable payload sections were found."
                })
                emitWorkspaceQuotaExceeded(jobName, error)
                return
              }

              for (const mutation of recoveryAttempt.mutations) {
                emitWorkspaceStorageRecoveryEvent({
                  key: jobName,
                  action: mutation.action,
                  workspaceId: mutation.workspaceId,
                  beforeBytes: mutation.beforeBytes,
                  afterBytes: mutation.afterBytes,
                  recoveredBytes: mutation.recoveredBytes
                })
              }

              try {
                const handledBySplitStorage = splitStorageEnabled
                  ? await writeSplitWorkspacePersistence(
                      jobName,
                      recoveryAttempt.value,
                      indexedDbAdapter,
                      {
                        durability: workspacePersistLifecycleFlushActive
                          ? "lifecycle"
                          : "coalesced"
                      }
                    )
                  : false
                if (!handledBySplitStorage) {
                  localStorage.setItem(jobName, recoveryAttempt.value)
                }
                broadcastWorkspaceStorageUpdate(jobName)
                emitWorkspaceStorageRecoveryEvent({
                  key: jobName,
                  action: "retry_success",
                  beforeBytes: recoveryAttempt.beforeBytes,
                  afterBytes: recoveryAttempt.afterBytes,
                  recoveredBytes:
                    recoveryAttempt.beforeBytes - recoveryAttempt.afterBytes
                })
                return
              } catch (retryError) {
                if (!isQuotaExceededError(retryError)) {
                  throw retryError
                }
                emitWorkspaceStorageRecoveryEvent({
                  key: jobName,
                  action: "retry_failed",
                  beforeBytes: recoveryAttempt.beforeBytes,
                  afterBytes: recoveryAttempt.afterBytes,
                  recoveredBytes:
                    recoveryAttempt.beforeBytes - recoveryAttempt.afterBytes,
                  reason:
                    retryError instanceof Error
                      ? retryError.message
                      : "Quota exceeded after recovery retry."
                })
                emitWorkspaceQuotaExceeded(jobName, retryError)
                return
              }
            }
            throw error
          }
        }
      })
    },
    removeItem: async (name: string): Promise<void> => {
      // Removal must not race a pending coalesced write (the write would
      // resurrect the removed key after this returns), and the per-workspace
      // diff stamps describe storage that is about to be wiped.
      flushWorkspacePersistWrites()
      clearWorkspacePersistStamps(indexedDbAdapter)
      if (name === WORKSPACE_STORAGE_KEY) {
        const workspaceIds = getWorkspaceIdsFromStoredValue(
          localStorage.getItem(name)
        )
        for (const workspaceStorageId of workspaceIds) {
          const snapshot = readWorkspaceSnapshotFromStorage(workspaceStorageId)
          const chatCandidate = safeParseJson(
            localStorage.getItem(buildWorkspaceChatStorageKey(workspaceStorageId))
          )
          const chatPointerKey = isWorkspaceIndexedDbChatPointer(chatCandidate)
            ? chatCandidate.key
            : null

          localStorage.removeItem(
            buildWorkspaceSnapshotStorageKey(workspaceStorageId)
          )
          localStorage.removeItem(
            buildWorkspaceChatStorageKey(workspaceStorageId)
          )

          if (indexedDbAdapter.isAvailable()) {
            await cleanupWorkspaceIndexedDbRecords(
              workspaceStorageId,
              snapshot,
              indexedDbAdapter
            )
            if (chatPointerKey) {
              await indexedDbAdapter.deleteChatRecord(chatPointerKey)
            }
          }
        }
      }
      localStorage.removeItem(name)
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// State Types
// ─────────────────────────────────────────────────────────────────────────────

interface WorkspaceIdentityState {
  workspaceId: string
  workspaceName: string
  workspaceTag: string // Format: "workspace:<slug>"
  studyMaterialsPolicy: StudyMaterialsPolicy | null
  assistantDefaults: WorkspaceAssistantDefaults | null
  effectiveAssistantDefault: EffectiveWorkspaceAssistantDefault | null
  workspaceCreatedAt: Date | null
  workspaceChatReferenceId: string
}

interface SourcesState {
  sources: WorkspaceSource[]
  selectedSourceIds: string[]
  sourceFolders: WorkspaceSourceFolder[]
  sourceFolderMemberships: WorkspaceSourceFolderMembership[]
  selectedSourceFolderIds: string[]
  activeFolderId: string | null
  sourceSearchQuery: string
  sourceFocusTarget: { sourceId: string; token: number } | null
  sourcesLoading: boolean
  sourcesError: string | null
}

interface StudioState {
  generatedArtifacts: GeneratedArtifact[]
  notes: string // Legacy simple notes field
  currentNote: WorkspaceNote // Full note with title, keywords, versioning
  workspaceBanner: WorkspaceBanner
  isGeneratingOutput: boolean
  generatingOutputType: ArtifactType | null
}

interface UIState {
  storeHydrated: boolean
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean
  addSourceModalOpen: boolean
  addSourceModalTab: AddSourceTab
  addSourceProcessing: boolean
  addSourceError: string | null
  chatFocusTarget: { messageId: string; token: number } | null
  noteFocusTarget: { field: "title" | "content"; token: number } | null
}

interface AudioSettingsState {
  audioSettings: AudioGenerationSettings
}

interface WorkspaceListState {
  savedWorkspaces: SavedWorkspace[]
  archivedWorkspaces: SavedWorkspace[]
  workspaceCollections: WorkspaceCollection[]
}

interface WorkspaceSnapshot {
  workspaceId: string
  workspaceName: string
  workspaceTag: string
  studyMaterialsPolicy: StudyMaterialsPolicy | null
  assistantDefaults: WorkspaceAssistantDefaults | null
  workspaceCreatedAt: Date | null
  workspaceChatReferenceId: string
  sources: WorkspaceSource[]
  selectedSourceIds: string[]
  sourceFolders: WorkspaceSourceFolder[]
  sourceFolderMemberships: WorkspaceSourceFolderMembership[]
  selectedSourceFolderIds: string[]
  activeFolderId: string | null
  generatedArtifacts: GeneratedArtifact[]
  notes: string
  currentNote: WorkspaceNote
  workspaceBanner: WorkspaceBanner
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean
  audioSettings: AudioGenerationSettings
}

interface WorkspaceSnapshotsState {
  workspaceSnapshots: Record<string, WorkspaceSnapshot>
}

export interface WorkspaceChatSession {
  messages: Message[]
  history: ChatHistory
  historyId: string | null
  serverChatId: string | null
}

interface PersistedWorkspaceChatSession {
  messages: Message[]
  historyId: string | null
  serverChatId: string | null
  // Legacy fallback persisted by older versions.
  history?: ChatHistory
}

interface WorkspaceChatSessionsState {
  workspaceChatSessions: Record<string, WorkspaceChatSession>
}

export interface WorkspaceUndoSnapshot {
  workspaceId: string
  workspaceName: string
  workspaceTag: string
  studyMaterialsPolicy: StudyMaterialsPolicy | null
  assistantDefaults: WorkspaceAssistantDefaults | null
  effectiveAssistantDefault: EffectiveWorkspaceAssistantDefault | null
  workspaceCreatedAt: Date | null
  workspaceChatReferenceId: string
  sources: WorkspaceSource[]
  selectedSourceIds: string[]
  sourceFolders: WorkspaceSourceFolder[]
  sourceFolderMemberships: WorkspaceSourceFolderMembership[]
  selectedSourceFolderIds: string[]
  activeFolderId: string | null
  generatedArtifacts: GeneratedArtifact[]
  notes: string
  currentNote: WorkspaceNote
  workspaceBanner: WorkspaceBanner
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean
  audioSettings: AudioGenerationSettings
  savedWorkspaces: SavedWorkspace[]
  archivedWorkspaces: SavedWorkspace[]
  workspaceCollections: WorkspaceCollection[]
  workspaceSnapshots: Record<string, WorkspaceSnapshot>
  workspaceChatSessions: Record<string, WorkspaceChatSession>
}

type CaptureNoteMode = "append" | "replace"

interface CaptureToNoteInput {
  provenance?: WorkspaceNote["pendingKnowledgeProvenance"]
  title?: string
  content: string
  mode?: CaptureNoteMode
}

export type WorkspaceSourceTransferDestination =
  | {
      kind: "existing"
      workspaceId: string
    }
  | {
      kind: "new"
      name: string
    }

export interface WorkspaceSourceTransferRequest {
  mode: WorkspaceSourceTransferMode
  destination: WorkspaceSourceTransferDestination
  selectedSourceIds: string[]
  conflictResolutions?: Record<
    number,
    WorkspaceSourceTransferConflictResolution
  >
  emptyFolderPolicy?: WorkspaceSourceTransferEmptyFolderPolicy
  sourceFolderFallbackName?: string
  switchToDestinationOnComplete?: boolean
}

export interface WorkspaceSourceTransferExecutionResult
  extends WorkspaceSourceTransferResult {
  originWorkspaceId: string
  destinationWorkspaceId: string
  destinationWasCreated: boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// Action Types
// ─────────────────────────────────────────────────────────────────────────────

interface WorkspaceIdentityActions {
  initializeWorkspace: (name?: string) => string
  restoreServerWorkspace: (snapshot: WorkspaceSnapshot) => void
  setWorkspaceName: (name: string) => void
  loadWorkspace: (config: WorkspaceConfig) => void
}

interface SourcesActions {
  createSourceFolder: (
    name: string,
    parentFolderId?: string | null
  ) => WorkspaceSourceFolder
  renameSourceFolder: (folderId: string, name: string) => void
  moveSourceFolder: (folderId: string, parentFolderId: string | null) => void
  deleteSourceFolder: (folderId: string) => void
  assignSourceToFolders: (sourceId: string, folderIds: string[]) => void
  removeSourceFromFolder: (sourceId: string, folderId: string) => void
  toggleSourceFolderSelection: (folderId: string) => void
  setActiveFolder: (folderId: string | null) => void
  addSource: (
    source: Omit<WorkspaceSource, "id" | "addedAt">
  ) => WorkspaceSource
  addSources: (
    sources: Omit<WorkspaceSource, "id" | "addedAt">[]
  ) => WorkspaceSource[]
  removeSource: (id: string) => void
  removeSources: (ids: string[]) => void
  mergeSourceReviewUpdates: (updates: WorkspaceSourceReviewUpdate[]) => void
  reorderSource: (sourceId: string, targetIndex: number) => void
  toggleSourceSelection: (id: string) => void
  selectAllSources: () => void
  deselectAllSources: () => void
  setSelectedSourceIds: (ids: string[]) => void
  setSourceSearchQuery: (query: string) => void
  setSourceStatusById: (
    sourceId: string,
    status: WorkspaceSourceStatus,
    statusMessage?: string,
    readiness?: WorkspaceSourceReadiness,
    statusDetails?: WorkspaceSourceStatusDetails
  ) => void
  setSourceStatusByMediaId: (
    mediaId: number,
    status: WorkspaceSourceStatus,
    statusMessage?: string,
    readiness?: WorkspaceSourceReadiness,
    statusDetails?: WorkspaceSourceStatusDetails
  ) => void
  focusSourceById: (id: string) => boolean
  focusSourceByMediaId: (mediaId: number) => boolean
  clearSourceFocusTarget: () => void
  setSourcesLoading: (loading: boolean) => void
  setSourcesError: (error: string | null) => void
  restoreSource: (
    source: WorkspaceSource,
    options?: { index?: number; select?: boolean }
  ) => void
  getSelectedSources: () => WorkspaceSource[]
  getSelectedMediaIds: () => number[]
  getEffectiveSelectedSources: () => WorkspaceSource[]
  getEffectiveSelectedMediaIds: () => number[]
}

interface StudioActions {
  addArtifact: (
    artifact: Omit<GeneratedArtifact, "id" | "createdAt">
  ) => GeneratedArtifact
  updateArtifactStatus: (
    id: string,
    status: ArtifactStatus,
    updates?: Partial<GeneratedArtifact>
  ) => void
  removeArtifact: (id: string) => void
  restoreArtifact: (
    artifact: GeneratedArtifact,
    options?: { index?: number }
  ) => void
  clearArtifacts: () => void
  setNotes: (notes: string) => void
  setWorkspaceBanner: (banner: Partial<WorkspaceBanner>) => void
  clearWorkspaceBannerImage: () => void
  resetWorkspaceBanner: () => void
  setIsGeneratingOutput: (
    isGenerating: boolean,
    outputType?: ArtifactType | null
  ) => void
  // Note management actions
  setCurrentNote: (note: WorkspaceNote | null) => void
  updateNoteContent: (content: string) => void
  updateNoteTitle: (title: string) => void
  updateNoteKeywords: (keywords: string[]) => void
  clearCurrentNote: () => void
  captureToCurrentNote: (input: CaptureToNoteInput) => void
  loadNote: (note: Omit<WorkspaceNote, "isDirty" | "keywords"> & { id: string | number; keywords?: string[] }) => void
}

interface UIActions {
  toggleLeftPane: () => void
  toggleRightPane: () => void
  setLeftPaneCollapsed: (collapsed: boolean) => void
  setRightPaneCollapsed: (collapsed: boolean) => void
  openAddSourceModal: (tab?: AddSourceTab) => void
  closeAddSourceModal: () => void
  setAddSourceModalTab: (tab: AddSourceTab) => void
  setAddSourceProcessing: (processing: boolean) => void
  setAddSourceError: (error: string | null) => void
  focusChatMessageById: (messageId: string) => boolean
  clearChatFocusTarget: () => void
  focusWorkspaceNote: (field?: "title" | "content") => void
  clearNoteFocusTarget: () => void
}

interface AudioSettingsActions {
  setAudioSettings: (settings: Partial<AudioGenerationSettings>) => void
  resetAudioSettings: () => void
}

interface WorkspaceListActions {
  createWorkspaceCollection: (
    name: string,
    description?: string | null
  ) => WorkspaceCollection
  renameWorkspaceCollection: (
    collectionId: string,
    name: string,
    description?: string | null
  ) => void
  deleteWorkspaceCollection: (collectionId: string) => void
  assignWorkspaceToCollection: (
    workspaceId: string,
    collectionId: string | null
  ) => void
  /** Save current workspace state to the saved workspaces list */
  saveCurrentWorkspace: () => void
  /** Export a workspace snapshot bundle (defaults to current workspace) */
  exportWorkspaceBundle: (id?: string) => WorkspaceExportBundle | null
  /** Import a workspace snapshot bundle and switch to it */
  importWorkspaceBundle: (bundle: WorkspaceExportBundle) => string | null
  /** Switch to a different workspace by ID */
  switchWorkspace: (id: string) => void
  /** Create a new workspace (optionally with a name), saving current first */
  createNewWorkspace: (name?: string) => void
  /** Duplicate a workspace (defaults to current) and switch to the duplicate */
  duplicateWorkspace: (id?: string) => string | null
  /** Transfer selected sources from the current workspace into another workspace */
  transferSourcesBetweenWorkspaces: (
    request: WorkspaceSourceTransferRequest
  ) => WorkspaceSourceTransferExecutionResult | null
  /** Archive a workspace from active saved list */
  archiveWorkspace: (id: string) => void
  /** Restore a workspace from archive back into saved list */
  restoreArchivedWorkspace: (id: string) => void
  /** Delete a workspace from the saved list */
  deleteWorkspace: (id: string) => void
  /** Get the list of saved workspaces sorted by last accessed */
  getSavedWorkspaces: () => SavedWorkspace[]
  /** Get archived workspaces sorted by last accessed */
  getArchivedWorkspaces: () => SavedWorkspace[]
  /** Save chat session state for a workspace */
  saveWorkspaceChatSession: (
    workspaceSessionKey: string,
    session: WorkspaceChatSession
  ) => void
  /** Retrieve chat session state for a workspace */
  getWorkspaceChatSession: (
    workspaceSessionKey: string
  ) => WorkspaceChatSession | null
  /** Clear chat session state for a workspace */
  clearWorkspaceChatSession: (workspaceSessionKey: string) => void
}

interface UndoActions {
  captureUndoSnapshot: () => WorkspaceUndoSnapshot
  restoreUndoSnapshot: (snapshot: WorkspaceUndoSnapshot) => void
}

interface ResetActions {
  reset: () => void
  resetSources: () => void
  resetStudio: () => void
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined State & Actions
// ─────────────────────────────────────────────────────────────────────────────

export type WorkspaceState = WorkspaceIdentityState &
  SourcesState &
  StudioState &
  UIState &
  AudioSettingsState &
  WorkspaceListState &
  WorkspaceSnapshotsState &
  WorkspaceChatSessionsState &
  WorkspaceIdentityActions &
  SourcesActions &
  StudioActions &
  UIActions &
  AudioSettingsActions &
  WorkspaceListActions &
  UndoActions &
  ResetActions

// ─────────────────────────────────────────────────────────────────────────────
// Initial State
// ─────────────────────────────────────────────────────────────────────────────

export const createSlug = (name: string): string => {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 32)
}

const initialIdentityState: WorkspaceIdentityState = {
  workspaceId: "",
  workspaceName: "",
  workspaceTag: "",
  studyMaterialsPolicy: null,
  assistantDefaults: null,
  effectiveAssistantDefault: null,
  workspaceCreatedAt: null,
  workspaceChatReferenceId: ""
}

export const initialSourcesState: SourcesState = {
  sources: [],
  selectedSourceIds: [],
  sourceFolders: [],
  sourceFolderMemberships: [],
  selectedSourceFolderIds: [],
  activeFolderId: null,
  sourceSearchQuery: "",
  sourceFocusTarget: null,
  sourcesLoading: false,
  sourcesError: null
}

export const initialStudioState: StudioState = {
  generatedArtifacts: [],
  notes: "",
  currentNote: { ...DEFAULT_WORKSPACE_NOTE },
  workspaceBanner: { ...DEFAULT_WORKSPACE_BANNER },
  isGeneratingOutput: false,
  generatingOutputType: null
}

const initialUIState: UIState = {
  storeHydrated: false,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  addSourceModalOpen: false,
  addSourceModalTab: "upload",
  addSourceProcessing: false,
  addSourceError: null,
  chatFocusTarget: null,
  noteFocusTarget: null
}

const initialAudioSettingsState: AudioSettingsState = {
  audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
}

const initialWorkspaceListState: WorkspaceListState = {
  savedWorkspaces: [],
  archivedWorkspaces: [],
  workspaceCollections: []
}

const initialWorkspaceSnapshotsState: WorkspaceSnapshotsState = {
  workspaceSnapshots: {}
}

const initialWorkspaceChatSessionsState: WorkspaceChatSessionsState = {
  workspaceChatSessions: {}
}

export const initialState = {
  ...initialIdentityState,
  ...initialSourcesState,
  ...initialStudioState,
  ...initialUIState,
  ...initialAudioSettingsState,
  ...initialWorkspaceListState,
  ...initialWorkspaceSnapshotsState,
  ...initialWorkspaceChatSessionsState
}

// ─────────────────────────────────────────────────────────────────────────────
// Persisted State Type (subset of state that gets saved)
// ─────────────────────────────────────────────────────────────────────────────

interface PersistedWorkspaceState {
  // Active workspace identity
  workspaceId: string

  // Saved workspaces list
  savedWorkspaces: SavedWorkspace[]
  archivedWorkspaces: SavedWorkspace[]
  workspaceCollections: WorkspaceCollection[]

  // Workspace snapshots keyed by workspace ID
  workspaceSnapshots: Record<string, WorkspaceSnapshot>

  // Workspace chat sessions keyed by workspace ID
  workspaceChatSessions: Record<string, PersistedWorkspaceChatSession>

  // Legacy fields supported during migration/rehydration.
  workspaceName?: string
  workspaceTag?: string
  workspaceCreatedAt?: Date | null
  workspaceChatReferenceId?: string
  sources?: WorkspaceSource[]
  selectedSourceIds?: string[]
  sourceFolders?: WorkspaceSourceFolder[]
  sourceFolderMemberships?: WorkspaceSourceFolderMembership[]
  selectedSourceFolderIds?: string[]
  activeFolderId?: string | null
  generatedArtifacts?: GeneratedArtifact[]
  notes?: string
  currentNote?: WorkspaceNote
  leftPaneCollapsed?: boolean
  rightPaneCollapsed?: boolean
  audioSettings?: AudioGenerationSettings
}

export type WorkspacePersistenceSectionKey =
  | "workspaceSnapshots"
  | "workspaceChatSessions"
  | "workspaceBanner"
  | "generatedArtifacts"
  | "notes"
  | "sources"
  | "selectedSourceIds"
  | "savedWorkspaces"
  | "archivedWorkspaces"
  | "other"

export type WorkspacePersistenceSectionBytes = Record<
  WorkspacePersistenceSectionKey,
  number
>

export interface WorkspacePersistenceMetricsSnapshot {
  totalBytes: number
  sections: WorkspacePersistenceSectionBytes
}

export interface WorkspacePersistenceDiagnosticsSnapshot
  extends WorkspacePersistenceMetricsSnapshot {
  key: string
  writeCount: number
  maxTotalBytes: number
  updatedAt: number
}

type WorkspacePersistedCandidate =
  | PersistedWorkspaceState
  | {
      state?: PersistedWorkspaceState | Record<string, unknown> | null
      version?: number
    }
  | Record<string, unknown>

declare global {
  interface Window {
    __tldwWorkspacePersistenceMetrics?: WorkspacePersistenceDiagnosticsSnapshot
  }
}

const MAX_ARCHIVED_WORKSPACES = 50
const INTERRUPTED_GENERATION_ERROR_MESSAGE =
  "Generation was interrupted. Click regenerate to try again."
const WORKSPACE_BANNER_TITLE_MAX_CHARS = 80
const WORKSPACE_BANNER_SUBTITLE_MAX_CHARS = 180
let workspacePersistenceWriteCount = 0
let workspacePersistenceMaxBytes = 0

export const cloneWorkspaceValue = <T>(value: T): T => {
  if (typeof structuredClone === "function") {
    return structuredClone(value)
  }
  return JSON.parse(JSON.stringify(value)) as T
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const estimateUtf8ByteLength = (value: string): number => {
  if (typeof TextEncoder !== "undefined") {
    return new TextEncoder().encode(value).length
  }
  return encodeURIComponent(value).replace(/%[A-F\d]{2}/g, "x").length
}

const estimateSerializedByteLength = (value: unknown): number => {
  try {
    return estimateUtf8ByteLength(JSON.stringify(value))
  } catch {
    return 0
  }
}

const unwrapPersistedWorkspaceCandidate = (
  candidate: WorkspacePersistedCandidate | null | undefined
): Record<string, unknown> => {
  if (!candidate) return {}
  if (!isRecord(candidate)) return {}

  const nestedState = candidate.state
  if (isRecord(nestedState)) {
    return nestedState
  }

  return candidate
}

export const estimateWorkspacePersistenceMetrics = (
  candidate: WorkspacePersistedCandidate | null | undefined
): WorkspacePersistenceMetricsSnapshot => {
  const payload = unwrapPersistedWorkspaceCandidate(candidate)
  const totalBytes = estimateSerializedByteLength(payload)
  const rawWorkspaceSnapshotsBytes = estimateSerializedByteLength(
    payload.workspaceSnapshots
  )
  const workspaceBannerBytes: number = isRecord(payload.workspaceSnapshots)
    ? Object.values(payload.workspaceSnapshots as Record<string, unknown>).reduce<number>(
        (accumulator, snapshot) => {
          if (!isRecord(snapshot)) return accumulator
          return (
            accumulator + estimateSerializedByteLength(snapshot.workspaceBanner)
          )
        },
        0
      )
    : 0

  const sections: WorkspacePersistenceSectionBytes = {
    workspaceSnapshots: Math.max(
      0,
      rawWorkspaceSnapshotsBytes - workspaceBannerBytes
    ),
    workspaceChatSessions: estimateSerializedByteLength(payload.workspaceChatSessions),
    workspaceBanner: workspaceBannerBytes,
    generatedArtifacts: estimateSerializedByteLength(payload.generatedArtifacts),
    notes: estimateSerializedByteLength(payload.notes),
    sources: estimateSerializedByteLength(payload.sources),
    selectedSourceIds: estimateSerializedByteLength(payload.selectedSourceIds),
    savedWorkspaces: estimateSerializedByteLength(payload.savedWorkspaces),
    archivedWorkspaces: estimateSerializedByteLength(payload.archivedWorkspaces),
    other: 0
  }

  const knownSectionBytes = Object.entries(sections)
    .filter(([key]) => key !== "other")
    .reduce((accumulator, [, bytes]) => accumulator + bytes, 0)
  sections.other = Math.max(0, totalBytes - knownSectionBytes)

  return {
    totalBytes,
    sections
  }
}

const shouldCaptureWorkspacePersistenceDiagnostics = (): boolean =>
  typeof window !== "undefined" && process.env.NODE_ENV !== "production"

const recordWorkspacePersistenceDiagnostics = (
  key: string,
  candidate: WorkspacePersistedCandidate
): void => {
  if (!shouldCaptureWorkspacePersistenceDiagnostics()) return

  const metrics = estimateWorkspacePersistenceMetrics(candidate)
  workspacePersistenceWriteCount += 1
  workspacePersistenceMaxBytes = Math.max(
    workspacePersistenceMaxBytes,
    metrics.totalBytes
  )

  window.__tldwWorkspacePersistenceMetrics = {
    key,
    writeCount: workspacePersistenceWriteCount,
    maxTotalBytes: workspacePersistenceMaxBytes,
    updatedAt: Date.now(),
    totalBytes: metrics.totalBytes,
    sections: metrics.sections
  }
}

const truncateTextForPersistence = (
  text: string,
  maxBytes: number,
  suffix: string
): string => {
  if (!text || maxBytes <= 0) return ""
  const originalBytes = estimateUtf8ByteLength(text)
  if (originalBytes <= maxBytes) return text

  const suffixBytes = estimateUtf8ByteLength(suffix)
  if (suffixBytes >= maxBytes) {
    return suffix.slice(0, Math.max(0, Math.floor(maxBytes / 2)))
  }

  const targetBytes = maxBytes - suffixBytes

  if (typeof TextEncoder !== "undefined" && typeof TextDecoder !== "undefined") {
    const encoder = new TextEncoder()
    const decoder = new TextDecoder()
    const encoded = encoder.encode(text).slice(0, targetBytes)
    let decoded = decoder.decode(encoded).replace(/\uFFFD+$/g, "").trimEnd()
    while (
      decoded.length > 0 &&
      estimateUtf8ByteLength(decoded) > targetBytes
    ) {
      decoded = decoded.slice(0, -1)
    }
    if (decoded.length === 0) return suffix.trim()
    return `${decoded}${suffix}`
  }

  let end = text.length
  let truncated = text.slice(0, end)
  while (
    end > 0 &&
    estimateUtf8ByteLength(truncated) > targetBytes
  ) {
    end -= 1
    truncated = text.slice(0, end)
  }
  truncated = truncated.trimEnd()
  if (!truncated) return suffix.trim()
  return `${truncated}${suffix}`
}

const sanitizeArtifactForPersistence = (
  artifact: GeneratedArtifact
): GeneratedArtifact => {
  const sanitizedArtifact: GeneratedArtifact = {
    ...artifact,
    audioUrl: undefined
  }

  const hasServerBackedPayload =
    sanitizedArtifact.serverId !== null &&
    sanitizedArtifact.serverId !== undefined
  if (!hasServerBackedPayload) {
    return sanitizedArtifact
  }

  if (
    typeof sanitizedArtifact.content === "string" &&
    estimateUtf8ByteLength(sanitizedArtifact.content) >
      WORKSPACE_PERSIST_SERVER_ARTIFACT_CONTENT_MAX_BYTES
  ) {
    sanitizedArtifact.content = truncateTextForPersistence(
      sanitizedArtifact.content,
      WORKSPACE_PERSIST_SERVER_ARTIFACT_CONTENT_MAX_BYTES,
      WORKSPACE_PERSIST_TRUNCATION_SUFFIX
    )
  }

  if (
    isRecord(sanitizedArtifact.data) &&
    estimateSerializedByteLength(sanitizedArtifact.data) >
      WORKSPACE_PERSIST_SERVER_ARTIFACT_DATA_MAX_BYTES
  ) {
    sanitizedArtifact.data = undefined
  }

  return sanitizedArtifact
}

const sanitizeArtifactsForPersistence = (
  artifacts: GeneratedArtifact[]
): GeneratedArtifact[] => artifacts.map((artifact) => sanitizeArtifactForPersistence(artifact))

export const reviveDateOrNull = (value: Date | string | null | undefined): Date | null => {
  if (!value) return null
  if (value instanceof Date) return value
  if (typeof value === "string") {
    const parsed = new Date(value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  }
  return null
}

export const reviveDateOrUndefined = (
  value: Date | string | null | undefined
): Date | undefined => {
  const revived = reviveDateOrNull(value)
  return revived ?? undefined
}

const reviveSourceStatusDetails = (
  details: WorkspaceSourceStatusDetails | undefined
): WorkspaceSourceStatusDetails | undefined =>
  details
    ? {
        ...details,
        updatedAt: reviveDateOrUndefined(
          details.updatedAt as Date | string | null | undefined
        )
      }
    : undefined

export const reviveSources = (sources: WorkspaceSource[]): WorkspaceSource[] =>
  sources.map((source) => ({
    ...source,
    status: source.status || "ready",
    statusMessage: source.statusMessage || undefined,
    statusDetails: reviveSourceStatusDetails(source.statusDetails),
    addedAt: reviveDateOrNull(source.addedAt) || new Date(),
    sourceCreatedAt: reviveDateOrUndefined(source.sourceCreatedAt)
  }))

export const reviveSourceFolders = (
  folders: WorkspaceSourceFolder[],
  workspaceId: string
): WorkspaceSourceFolder[] =>
  folders.map((folder) => ({
    ...folder,
    workspaceId: folder.workspaceId || workspaceId,
    parentFolderId: folder.parentFolderId || null,
    createdAt: reviveDateOrNull(folder.createdAt) || new Date(),
    updatedAt: reviveDateOrNull(folder.updatedAt) || new Date()
  }))

export const reviveSourceFolderMemberships = (
  memberships: WorkspaceSourceFolderMembership[],
  sourceIds: Set<string>,
  folderIds: Set<string>
): WorkspaceSourceFolderMembership[] =>
  memberships.filter(
    (membership) =>
      sourceIds.has(membership.sourceId) && folderIds.has(membership.folderId)
  )

export const reviveWorkspaceCollections = (
  collections: WorkspaceCollection[]
): WorkspaceCollection[] =>
  collections.map((collection) => ({
    ...collection,
    description: collection.description || null,
    createdAt: reviveDateOrNull(collection.createdAt) || new Date(),
    updatedAt: reviveDateOrNull(collection.updatedAt) || new Date()
  }))

export const getWorkspaceSourceStatus = (
  source: WorkspaceSource
): WorkspaceSourceStatus => source.status || "ready"

const normalizeSourceFolderName = (name: string): string => {
  const trimmedName = name.trim()
  return trimmedName || "Untitled Folder"
}

export const getUniqueSourceFolderName = (
  folders: WorkspaceSourceFolder[],
  name: string,
  parentFolderId: string | null,
  excludeFolderId?: string
): string => {
  const normalizedName = normalizeSourceFolderName(name)
  const siblingNames = new Set(
    folders
      .filter(
        (folder) =>
          folder.parentFolderId === parentFolderId &&
          folder.id !== excludeFolderId
      )
      .map((folder) => folder.name.trim().toLowerCase())
  )
  if (!siblingNames.has(normalizedName.toLowerCase())) {
    return normalizedName
  }

  let suffix = 2
  let candidate = `${normalizedName} (${suffix})`
  while (siblingNames.has(candidate.toLowerCase())) {
    suffix += 1
    candidate = `${normalizedName} (${suffix})`
  }
  return candidate
}

export const createWorkspaceOrganizationStateIndex = (
  state: Pick<
    WorkspaceState,
    "sources" | "sourceFolders" | "sourceFolderMemberships"
  >
) =>
  createWorkspaceOrganizationIndex({
    sources: state.sources,
    sourceFolders: state.sourceFolders,
    sourceFolderMemberships: state.sourceFolderMemberships
  })

const normalizeWorkspaceCollectionName = (name: string): string => {
  const trimmedName = name.trim()
  return trimmedName || "Untitled Collection"
}

export const getUniqueWorkspaceCollectionName = (
  collections: WorkspaceCollection[],
  name: string,
  excludeCollectionId?: string
): string => {
  const normalizedName = normalizeWorkspaceCollectionName(name)
  const existingNames = new Set(
    collections
      .filter((collection) => collection.id !== excludeCollectionId)
      .map((collection) => collection.name.trim().toLowerCase())
  )
  if (!existingNames.has(normalizedName.toLowerCase())) {
    return normalizedName
  }

  let suffix = 2
  let candidate = `${normalizedName} (${suffix})`
  while (existingNames.has(candidate.toLowerCase())) {
    suffix += 1
    candidate = `${normalizedName} (${suffix})`
  }
  return candidate
}

export const reviveArtifacts = (artifacts: GeneratedArtifact[]): GeneratedArtifact[] =>
  artifacts.map((artifact) => {
    const revivedArtifact: GeneratedArtifact = {
      ...artifact,
      createdAt: reviveDateOrNull(artifact.createdAt) || new Date(),
      completedAt: reviveDateOrUndefined(artifact.completedAt)
    }

    if (revivedArtifact.status !== "generating") {
      return revivedArtifact
    }

    return {
      ...revivedArtifact,
      status: "failed",
      completedAt: revivedArtifact.completedAt || new Date(),
      errorMessage:
        revivedArtifact.errorMessage || INTERRUPTED_GENERATION_ERROR_MESSAGE
    }
  })

export const reviveSavedWorkspace = (workspace: SavedWorkspace): SavedWorkspace => ({
  ...workspace,
  collectionId: workspace.collectionId || null,
  createdAt: reviveDateOrNull(workspace.createdAt) || new Date(),
  lastAccessedAt: reviveDateOrNull(workspace.lastAccessedAt) || new Date()
})

export const coerceWorkspaceAssistantDefaultsForRehydrate =
  normalizeWorkspaceAssistantDefaults

export const reviveWorkspaceSnapshot = (
  workspaceId: string,
  snapshot: WorkspaceSnapshot
): WorkspaceSnapshot => {
  const resolvedStudyMaterialsPolicy =
    snapshot.studyMaterialsPolicy === "workspace"
      ? "workspace"
      : snapshot.studyMaterialsPolicy === "general"
        ? "general"
        : null
  const createdAt = reviveDateOrNull(snapshot.workspaceCreatedAt)
  const sources = reviveSources(snapshot.sources || [])
  const sourceIdSet = new Set(sources.map((source) => source.id))
  const sourceFolders = reviveSourceFolders(
    snapshot.sourceFolders || [],
    snapshot.workspaceId || workspaceId
  )
  const folderIdSet = new Set(sourceFolders.map((folder) => folder.id))
  const sourceFolderMemberships = reviveSourceFolderMemberships(
    snapshot.sourceFolderMemberships || [],
    sourceIdSet,
    folderIdSet
  )
  return {
    ...snapshot,
    workspaceId: snapshot.workspaceId || workspaceId,
    studyMaterialsPolicy: resolvedStudyMaterialsPolicy,
    assistantDefaults: coerceWorkspaceAssistantDefaultsForRehydrate(
      snapshot.assistantDefaults
    ),
    workspaceCreatedAt: createdAt,
    workspaceChatReferenceId:
      snapshot.workspaceChatReferenceId ||
      snapshot.workspaceId ||
      workspaceId,
    sources,
    selectedSourceIds: snapshot.selectedSourceIds || [],
    sourceFolders,
    sourceFolderMemberships,
    selectedSourceFolderIds: (snapshot.selectedSourceFolderIds || []).filter((id) =>
      folderIdSet.has(id)
    ),
    activeFolderId:
      snapshot.activeFolderId && folderIdSet.has(snapshot.activeFolderId)
        ? snapshot.activeFolderId
        : null,
    generatedArtifacts: reviveArtifacts(snapshot.generatedArtifacts || []),
    currentNote: snapshot.currentNote || { ...DEFAULT_WORKSPACE_NOTE },
    workspaceBanner: coerceWorkspaceBannerForRehydrate(snapshot.workspaceBanner),
    audioSettings: snapshot.audioSettings || { ...DEFAULT_AUDIO_SETTINGS }
  }
}

const normalizeChatHistoryEntries = (history: unknown): ChatHistory => {
  if (!Array.isArray(history)) return []

  return history
    .map((entry) => {
      if (!isRecord(entry)) return null
      const role =
        entry.role === "assistant" || entry.role === "system"
          ? entry.role
          : entry.role === "user"
            ? "user"
            : null
      const content = typeof entry.content === "string" ? entry.content : ""
      if (!role || !content) return null
      const normalized: ChatHistory[number] = { role, content }
      if (typeof entry.image === "string") {
        normalized.image = entry.image
      }
      if (typeof entry.messageType === "string") {
        normalized.messageType = entry.messageType
      }
      return normalized
    })
    .filter((entry): entry is ChatHistory[number] => Boolean(entry))
}

const normalizeWorkspaceSessionMessages = (messages: unknown): Message[] => {
  if (!Array.isArray(messages)) return []
  return messages
    .filter((message): message is Message => isRecord(message))
    .map((message) => ({ ...message }))
}

const deriveHistoryFromMessages = (messages: Message[]): ChatHistory =>
  messages
    .map((message) => {
      const content = typeof message.message === "string" ? message.message : ""
      if (!content) return null
      const role =
        message.role && ["assistant", "system", "user"].includes(message.role)
          ? message.role
          : message.isBot
            ? "assistant"
            : "user"
      return { role, content }
    })
    .filter((entry): entry is ChatHistory[number] => Boolean(entry))

const normalizeWorkspaceChatSession = (
  workspaceId: string,
  sessionCandidate: unknown
): WorkspaceChatSession | null => {
  if (!isRecord(sessionCandidate)) return null

  const messages = normalizeWorkspaceSessionMessages(sessionCandidate.messages)
  const normalizedHistory = normalizeChatHistoryEntries(sessionCandidate.history)
  const history =
    normalizedHistory.length > 0
      ? normalizedHistory
      : deriveHistoryFromMessages(messages)
  const historyId =
    typeof sessionCandidate.historyId === "string"
      ? sessionCandidate.historyId
      : null
  const serverChatId =
    typeof sessionCandidate.serverChatId === "string"
      ? sessionCandidate.serverChatId
      : null

  // Drop empty shells so we don't keep expanding persisted payloads with no data.
  if (
    messages.length === 0 &&
    history.length === 0 &&
    historyId === null &&
    serverChatId === null
  ) {
    return null
  }

  return {
    messages,
    history,
    historyId,
    serverChatId
  }
}

const normalizeWorkspaceChatSessionsForRehydrate = (
  candidate: unknown
): Record<string, WorkspaceChatSession> => {
  const normalized: Record<string, WorkspaceChatSession> = {}

  if (Array.isArray(candidate)) {
    for (const entry of candidate) {
      if (!isRecord(entry)) continue
      const workspaceId =
        typeof entry.workspaceId === "string"
          ? entry.workspaceId
          : typeof entry.id === "string"
            ? entry.id
            : null
      if (!workspaceId) continue
      const sessionCandidate = isRecord(entry.session)
        ? entry.session
        : entry
      const session = normalizeWorkspaceChatSession(workspaceId, sessionCandidate)
      if (session) {
        normalized[workspaceId] = session
      }
    }
    return normalized
  }

  if (!isRecord(candidate)) return normalized

  for (const [workspaceId, entry] of Object.entries(candidate)) {
    const session = normalizeWorkspaceChatSession(workspaceId, entry)
    if (session) {
      normalized[workspaceId] = session
    }
  }

  return normalized
}

const normalizeWorkspaceSnapshotsForRehydrate = (
  candidate: unknown
): Record<string, WorkspaceSnapshot> => {
  const normalized: Record<string, WorkspaceSnapshot> = {}

  if (Array.isArray(candidate)) {
    for (const entry of candidate) {
      if (!isRecord(entry)) continue
      const workspaceId =
        typeof entry.workspaceId === "string"
          ? entry.workspaceId
          : typeof entry.id === "string"
            ? entry.id
            : null
      if (!workspaceId) continue
      normalized[workspaceId] = reviveWorkspaceSnapshot(
        workspaceId,
        entry as unknown as WorkspaceSnapshot
      )
    }
    return normalized
  }

  if (!isRecord(candidate)) return normalized

  for (const [workspaceId, snapshot] of Object.entries(candidate)) {
    if (!isRecord(snapshot)) continue
    normalized[workspaceId] = reviveWorkspaceSnapshot(
      workspaceId,
      snapshot as unknown as WorkspaceSnapshot
    )
  }

  return normalized
}

const coerceWorkspaceNoteForRehydrate = (candidate: unknown): WorkspaceNote => {
  if (!isRecord(candidate)) {
    return { ...DEFAULT_WORKSPACE_NOTE }
  }

  return {
    id: typeof candidate.id === "number" ||
      (typeof candidate.id === "string" && candidate.id.trim())
        ? candidate.id
        : undefined,
    title: typeof candidate.title === "string" ? candidate.title : "",
    content: typeof candidate.content === "string" ? candidate.content : "",
    keywords: Array.isArray(candidate.keywords)
      ? candidate.keywords.filter(
          (keyword): keyword is string => typeof keyword === "string"
        )
      : [],
    ...knowledgeNoteHead(candidate),
    pendingKnowledgeProvenance:
      validateKnowledgeNoteProvenance(candidate.pendingKnowledgeProvenance) ||
      undefined,
    version: typeof candidate.version === "number" ? candidate.version : 1,
    isDirty: Boolean(candidate.isDirty)
  }
}

const coerceAudioSettingsForRehydrate = (
  candidate: unknown
): AudioGenerationSettings => {
  if (!isRecord(candidate)) {
    return { ...DEFAULT_AUDIO_SETTINGS }
  }

  return {
    provider:
      isAudioTtsProvider(candidate.provider)
        ? candidate.provider
        : DEFAULT_AUDIO_SETTINGS.provider,
    model:
      typeof candidate.model === "string"
        ? candidate.model
        : DEFAULT_AUDIO_SETTINGS.model,
    voice:
      typeof candidate.voice === "string"
        ? candidate.voice
        : DEFAULT_AUDIO_SETTINGS.voice,
    speed:
      typeof candidate.speed === "number"
        ? candidate.speed
        : DEFAULT_AUDIO_SETTINGS.speed,
    format:
      isAudioOutputFormat(candidate.format)
        ? candidate.format
        : DEFAULT_AUDIO_SETTINGS.format,
    backend:
      typeof candidate.backend === "string"
        ? candidate.backend.trim()
        : DEFAULT_AUDIO_SETTINGS.backend,
    allowFallback:
      typeof candidate.allowFallback === "boolean"
        ? candidate.allowFallback
        : DEFAULT_AUDIO_SETTINGS.allowFallback
  }
}

export const coerceWorkspaceBannerForRehydrate = (
  candidate: unknown
): WorkspaceBanner => {
  if (!isRecord(candidate)) {
    return { ...DEFAULT_WORKSPACE_BANNER }
  }

  const imageCandidate = isRecord(candidate.image) ? candidate.image : null
  const imageMimeType = isWorkspaceBannerImageMimeType(imageCandidate?.mimeType)
    ? imageCandidate.mimeType
    : null

  const image: WorkspaceBannerImage | null =
    imageCandidate &&
    typeof imageCandidate.dataUrl === "string" &&
    imageMimeType &&
    typeof imageCandidate.width === "number" &&
    Number.isFinite(imageCandidate.width) &&
    imageCandidate.width > 0 &&
    typeof imageCandidate.height === "number" &&
    Number.isFinite(imageCandidate.height) &&
    imageCandidate.height > 0 &&
    typeof imageCandidate.bytes === "number" &&
    Number.isFinite(imageCandidate.bytes) &&
    imageCandidate.bytes > 0
      ? {
          dataUrl: imageCandidate.dataUrl,
          mimeType: imageMimeType,
          width: imageCandidate.width,
          height: imageCandidate.height,
          bytes: imageCandidate.bytes,
          updatedAt:
            reviveDateOrNull(
              imageCandidate.updatedAt as Date | string | null | undefined
            ) || new Date()
        }
      : null

  return {
    title:
      typeof candidate.title === "string"
        ? candidate.title
        : DEFAULT_WORKSPACE_BANNER.title,
    subtitle:
      typeof candidate.subtitle === "string"
        ? candidate.subtitle
        : DEFAULT_WORKSPACE_BANNER.subtitle,
    image
  }
}

const sanitizeWorkspaceBannerText = (
  value: string,
  maxChars: number
): string => value.trim().slice(0, maxChars)

export const sanitizeWorkspaceBanner = (
  candidate: WorkspaceBanner
): WorkspaceBanner => {
  const normalized = coerceWorkspaceBannerForRehydrate(candidate)
  return {
    title: sanitizeWorkspaceBannerText(
      normalized.title,
      WORKSPACE_BANNER_TITLE_MAX_CHARS
    ),
    subtitle: sanitizeWorkspaceBannerText(
      normalized.subtitle,
      WORKSPACE_BANNER_SUBTITLE_MAX_CHARS
    ),
    image: normalized.image
  }
}

const buildLegacyTopLevelSnapshotForMigration = (
  workspaceId: string,
  persisted: Record<string, unknown>
): WorkspaceSnapshot | null => {
  if (!workspaceId) return null

  const sources = reviveSources(
    Array.isArray(persisted.sources) ? (persisted.sources as WorkspaceSource[]) : []
  )
  const selectedSourceIds = (
    Array.isArray(persisted.selectedSourceIds)
      ? (persisted.selectedSourceIds as string[])
      : []
  ).filter((sourceId) => sources.some((source) => source.id === sourceId))
  const generatedArtifacts = reviveArtifacts(
    Array.isArray(persisted.generatedArtifacts)
      ? (persisted.generatedArtifacts as GeneratedArtifact[])
      : []
  )
  const resolvedWorkspaceTag =
    typeof persisted.workspaceTag === "string" && persisted.workspaceTag.trim()
      ? persisted.workspaceTag
      : `workspace:${workspaceId.slice(0, 8)}`
  const resolvedWorkspaceName =
    typeof persisted.workspaceName === "string" && persisted.workspaceName.trim()
      ? persisted.workspaceName
      : "Untitled Workspace"
  const resolvedStudyMaterialsPolicy =
    persisted.studyMaterialsPolicy === "workspace"
      ? "workspace"
      : persisted.studyMaterialsPolicy === "general"
        ? "general"
        : null

  return {
    workspaceId,
    workspaceName: resolvedWorkspaceName,
    workspaceTag: resolvedWorkspaceTag,
    studyMaterialsPolicy: resolvedStudyMaterialsPolicy,
    assistantDefaults: coerceWorkspaceAssistantDefaultsForRehydrate(
      persisted.assistantDefaults
    ),
    workspaceCreatedAt: reviveDateOrNull(
      persisted.workspaceCreatedAt as Date | string | null | undefined
    ),
    workspaceChatReferenceId:
      typeof persisted.workspaceChatReferenceId === "string" &&
      persisted.workspaceChatReferenceId.trim()
        ? persisted.workspaceChatReferenceId
        : workspaceId,
    sources,
    selectedSourceIds,
    sourceFolders: reviveSourceFolders(
      Array.isArray(persisted.sourceFolders)
        ? (persisted.sourceFolders as WorkspaceSourceFolder[])
        : [],
      workspaceId
    ),
    sourceFolderMemberships: [],
    selectedSourceFolderIds: [],
    activeFolderId: null,
    generatedArtifacts,
    notes: typeof persisted.notes === "string" ? persisted.notes : "",
    currentNote: coerceWorkspaceNoteForRehydrate(persisted.currentNote),
    workspaceBanner: coerceWorkspaceBannerForRehydrate(
      persisted.workspaceBanner
    ),
    leftPaneCollapsed: Boolean(persisted.leftPaneCollapsed),
    rightPaneCollapsed: Boolean(persisted.rightPaneCollapsed),
    audioSettings: coerceAudioSettingsForRehydrate(persisted.audioSettings)
  }
}

const migratePersistedWorkspaceState = (
  candidate: unknown
): PersistedWorkspaceState => {
  const persisted = isRecord(candidate) ? candidate : {}

  const normalizedSnapshots = normalizeWorkspaceSnapshotsForRehydrate(
    persisted.workspaceSnapshots
  )
  const initialWorkspaceId =
    typeof persisted.workspaceId === "string" ? persisted.workspaceId : ""
  const legacySnapshot =
    initialWorkspaceId && !normalizedSnapshots[initialWorkspaceId]
      ? buildLegacyTopLevelSnapshotForMigration(initialWorkspaceId, persisted)
      : null
  if (legacySnapshot) {
    normalizedSnapshots[legacySnapshot.workspaceId] = legacySnapshot
  }

  const resolvedWorkspaceId =
    initialWorkspaceId || Object.keys(normalizedSnapshots)[0] || ""
  if (resolvedWorkspaceId && !normalizedSnapshots[resolvedWorkspaceId]) {
    normalizedSnapshots[resolvedWorkspaceId] = createEmptyWorkspaceSnapshot({
      id: resolvedWorkspaceId,
      name: "Untitled Workspace",
      tag: `workspace:${resolvedWorkspaceId.slice(0, 8)}`,
      createdAt: new Date()
    })
  }

  return {
    workspaceId: resolvedWorkspaceId,
    savedWorkspaces: Array.isArray(persisted.savedWorkspaces)
      ? (persisted.savedWorkspaces as SavedWorkspace[])
      : [],
    archivedWorkspaces: Array.isArray(persisted.archivedWorkspaces)
      ? (persisted.archivedWorkspaces as SavedWorkspace[])
      : [],
    workspaceCollections: Array.isArray(persisted.workspaceCollections)
      ? (persisted.workspaceCollections as WorkspaceCollection[])
      : [],
    workspaceSnapshots: normalizedSnapshots,
    workspaceChatSessions: buildPersistedWorkspaceChatSessions(
      normalizeWorkspaceChatSessionsForRehydrate(persisted.workspaceChatSessions)
    )
  }
}

export const createEmptyWorkspaceSnapshot = ({
  id,
  name,
  tag,
  createdAt,
  studyMaterialsPolicy = null,
  assistantDefaults = null
}: {
  id: string
  name: string
  tag: string
  createdAt: Date
  studyMaterialsPolicy?: StudyMaterialsPolicy | null
  assistantDefaults?: WorkspaceAssistantDefaults | null
}): WorkspaceSnapshot => ({
  workspaceId: id,
  workspaceName: name,
  workspaceTag: tag,
  studyMaterialsPolicy: studyMaterialsPolicy ?? null,
  assistantDefaults: assistantDefaults ?? null,
  workspaceCreatedAt: createdAt,
  workspaceChatReferenceId: id,
  sources: [],
  selectedSourceIds: [],
  sourceFolders: [],
  sourceFolderMemberships: [],
  selectedSourceFolderIds: [],
  activeFolderId: null,
  generatedArtifacts: [],
  notes: "",
  currentNote: { ...DEFAULT_WORKSPACE_NOTE },
  workspaceBanner: { ...DEFAULT_WORKSPACE_BANNER },
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
})

export const applyWorkspaceSnapshot = (
  snapshot: WorkspaceSnapshot
): Pick<
  WorkspaceState,
  | "workspaceId"
  | "workspaceName"
  | "workspaceTag"
  | "studyMaterialsPolicy"
  | "assistantDefaults"
  | "effectiveAssistantDefault"
  | "workspaceCreatedAt"
  | "workspaceChatReferenceId"
  | "sources"
  | "selectedSourceIds"
  | "sourceFolders"
  | "sourceFolderMemberships"
  | "selectedSourceFolderIds"
  | "activeFolderId"
  | "generatedArtifacts"
  | "notes"
  | "currentNote"
  | "workspaceBanner"
  | "leftPaneCollapsed"
  | "rightPaneCollapsed"
  | "audioSettings"
> => ({
  workspaceId: snapshot.workspaceId,
  workspaceName: snapshot.workspaceName,
  workspaceTag: snapshot.workspaceTag,
  studyMaterialsPolicy: snapshot.studyMaterialsPolicy ?? null,
  assistantDefaults: snapshot.assistantDefaults ?? null,
  effectiveAssistantDefault: null,
  workspaceCreatedAt: snapshot.workspaceCreatedAt,
  workspaceChatReferenceId: snapshot.workspaceChatReferenceId,
  sources: snapshot.sources.map((source) => ({ ...source })),
  selectedSourceIds: [...snapshot.selectedSourceIds],
  sourceFolders: snapshot.sourceFolders.map((folder) => ({ ...folder })),
  sourceFolderMemberships: snapshot.sourceFolderMemberships.map((membership) => ({
    ...membership
  })),
  selectedSourceFolderIds: [...snapshot.selectedSourceFolderIds],
  activeFolderId: snapshot.activeFolderId,
  generatedArtifacts: snapshot.generatedArtifacts.map((artifact) => ({
    ...artifact
  })),
  notes: snapshot.notes,
  currentNote: { ...snapshot.currentNote },
  workspaceBanner: cloneWorkspaceValue(snapshot.workspaceBanner),
  leftPaneCollapsed: snapshot.leftPaneCollapsed,
  rightPaneCollapsed: snapshot.rightPaneCollapsed,
  audioSettings: { ...snapshot.audioSettings }
})

export const buildWorkspaceSnapshot = (state: WorkspaceState): WorkspaceSnapshot => ({
  workspaceId: state.workspaceId,
  workspaceName: state.workspaceName || "Untitled Workspace",
  workspaceTag: state.workspaceTag,
  studyMaterialsPolicy: state.studyMaterialsPolicy ?? null,
  assistantDefaults: state.assistantDefaults ?? null,
  workspaceCreatedAt: state.workspaceCreatedAt,
  workspaceChatReferenceId: state.workspaceChatReferenceId || state.workspaceId,
  sources: state.sources.map((source) => ({ ...source })),
  selectedSourceIds: [...state.selectedSourceIds],
  sourceFolders: state.sourceFolders.map((folder) => ({ ...folder })),
  sourceFolderMemberships: state.sourceFolderMemberships.map((membership) => ({
    ...membership
  })),
  selectedSourceFolderIds: [...state.selectedSourceFolderIds],
  activeFolderId: state.activeFolderId,
  generatedArtifacts: state.generatedArtifacts.map((artifact) => ({
    ...artifact
  })),
  notes: state.notes,
  currentNote: { ...state.currentNote },
  workspaceBanner: cloneWorkspaceValue(state.workspaceBanner),
  leftPaneCollapsed: state.leftPaneCollapsed,
  rightPaneCollapsed: state.rightPaneCollapsed,
  audioSettings: { ...state.audioSettings }
})

export const buildWorkspaceBundleSnapshot = (
  snapshot: WorkspaceSnapshot
): WorkspaceBundleSnapshot => ({
  workspaceName: snapshot.workspaceName,
  workspaceTag: snapshot.workspaceTag,
  studyMaterialsPolicy: snapshot.studyMaterialsPolicy ?? null,
  workspaceCreatedAt: snapshot.workspaceCreatedAt,
  sources: snapshot.sources.map((source) => ({ ...source })),
  selectedSourceIds: [...snapshot.selectedSourceIds],
  sourceFolders: snapshot.sourceFolders.map((folder) => ({ ...folder })),
  sourceFolderMemberships: snapshot.sourceFolderMemberships.map((membership) => ({
    ...membership
  })),
  selectedSourceFolderIds: [...snapshot.selectedSourceFolderIds],
  activeFolderId: snapshot.activeFolderId,
  generatedArtifacts: snapshot.generatedArtifacts.map((artifact) => ({
    ...artifact
  })),
  notes: snapshot.notes,
  currentNote: { ...snapshot.currentNote },
  workspaceBanner: cloneWorkspaceValue(snapshot.workspaceBanner),
  leftPaneCollapsed: snapshot.leftPaneCollapsed,
  rightPaneCollapsed: snapshot.rightPaneCollapsed,
  audioSettings: { ...snapshot.audioSettings }
})

export const cloneWorkspaceBundleChatSession = (
  session: WorkspaceBundleChatSession
): WorkspaceBundleChatSession => ({
  messages: Array.isArray(session.messages)
    ? session.messages.map((message) => ({ ...message }))
    : [],
  history: Array.isArray(session.history)
    ? session.history.map((entry) => ({ ...entry }))
    : [],
  historyId: typeof session.historyId === "string" ? session.historyId : null,
  serverChatId:
    typeof session.serverChatId === "string" ? session.serverChatId : null
})

export const hydrateWorkspaceBundleSnapshot = (
  snapshot: WorkspaceBundleSnapshot,
  workspaceId: string,
  workspaceName: string,
  workspaceTag: string
): WorkspaceSnapshot => {
  const revivedSources = reviveSources(snapshot.sources || [])
  const sourceIdSet = new Set(revivedSources.map((source) => source.id))
  const selectedSourceIds = (snapshot.selectedSourceIds || []).filter((id) =>
    sourceIdSet.has(id)
  )
  const revivedSourceFolders = reviveSourceFolders(
    snapshot.sourceFolders || [],
    workspaceId
  ).map((folder) => ({
    ...folder,
    workspaceId
  }))
  const folderIdSet = new Set(revivedSourceFolders.map((folder) => folder.id))
  const sourceFolderMemberships = reviveSourceFolderMemberships(
    snapshot.sourceFolderMemberships || [],
    sourceIdSet,
    folderIdSet
  )

  return {
    workspaceId,
    workspaceName,
    workspaceTag,
    studyMaterialsPolicy: snapshot.studyMaterialsPolicy ?? null,
    assistantDefaults: null,
    workspaceCreatedAt: new Date(),
    workspaceChatReferenceId: workspaceId,
    sources: revivedSources,
    selectedSourceIds,
    sourceFolders: revivedSourceFolders,
    sourceFolderMemberships,
    selectedSourceFolderIds: (snapshot.selectedSourceFolderIds || []).filter((id) =>
      folderIdSet.has(id)
    ),
    activeFolderId:
      snapshot.activeFolderId && folderIdSet.has(snapshot.activeFolderId)
        ? snapshot.activeFolderId
        : null,
    generatedArtifacts: reviveArtifacts(snapshot.generatedArtifacts || []),
    notes: typeof snapshot.notes === "string" ? snapshot.notes : "",
    currentNote: snapshot.currentNote || { ...DEFAULT_WORKSPACE_NOTE },
    workspaceBanner: coerceWorkspaceBannerForRehydrate(
      snapshot.workspaceBanner
    ),
    leftPaneCollapsed: Boolean(snapshot.leftPaneCollapsed),
    rightPaneCollapsed: Boolean(snapshot.rightPaneCollapsed),
    audioSettings: snapshot.audioSettings || { ...DEFAULT_AUDIO_SETTINGS }
  }
}

export const buildWorkspaceUndoSnapshot = (
  state: WorkspaceState
): WorkspaceUndoSnapshot => {
  const nextSnapshots = { ...state.workspaceSnapshots }
  if (state.workspaceId) {
    nextSnapshots[state.workspaceId] = buildWorkspaceSnapshot(state)
  }

  const snapshot: WorkspaceUndoSnapshot = {
    workspaceId: state.workspaceId,
    workspaceName: state.workspaceName,
    workspaceTag: state.workspaceTag,
    studyMaterialsPolicy: state.studyMaterialsPolicy ?? null,
    assistantDefaults: state.assistantDefaults ?? null,
    effectiveAssistantDefault: state.effectiveAssistantDefault ?? null,
    workspaceCreatedAt: state.workspaceCreatedAt,
    workspaceChatReferenceId:
      state.workspaceChatReferenceId || state.workspaceId,
    sources: state.sources,
    selectedSourceIds: state.selectedSourceIds,
    sourceFolders: state.sourceFolders,
    sourceFolderMemberships: state.sourceFolderMemberships,
    selectedSourceFolderIds: state.selectedSourceFolderIds,
    activeFolderId: state.activeFolderId,
    generatedArtifacts: state.generatedArtifacts,
    notes: state.notes,
    currentNote: state.currentNote,
    workspaceBanner: state.workspaceBanner,
    leftPaneCollapsed: state.leftPaneCollapsed,
    rightPaneCollapsed: state.rightPaneCollapsed,
    audioSettings: state.audioSettings,
    savedWorkspaces: state.savedWorkspaces,
    archivedWorkspaces: state.archivedWorkspaces,
    workspaceCollections: state.workspaceCollections,
    workspaceSnapshots: nextSnapshots,
    workspaceChatSessions: state.workspaceChatSessions
  }

  return cloneWorkspaceValue(snapshot)
}

export const createSavedWorkspaceEntry = (
  snapshot: WorkspaceSnapshot,
  lastAccessedAt: Date = new Date(),
  collectionId: string | null = null
): SavedWorkspace => ({
  id: snapshot.workspaceId,
  name: snapshot.workspaceName || "Untitled Workspace",
  tag: snapshot.workspaceTag,
  collectionId,
  createdAt: snapshot.workspaceCreatedAt || new Date(),
  lastAccessedAt,
  sourceCount: snapshot.sources.length
})

const findSavedWorkspaceById = (
  savedWorkspaces: SavedWorkspace[],
  archivedWorkspaces: SavedWorkspace[],
  workspaceId: string
): SavedWorkspace | null =>
  savedWorkspaces.find((workspace) => workspace.id === workspaceId) ||
  archivedWorkspaces.find((workspace) => workspace.id === workspaceId) ||
  null

export const getSavedWorkspaceCollectionId = (
  savedWorkspaces: SavedWorkspace[],
  archivedWorkspaces: SavedWorkspace[],
  workspaceId: string
): string | null =>
  findSavedWorkspaceById(savedWorkspaces, archivedWorkspaces, workspaceId)
    ?.collectionId || null

export const upsertSavedWorkspace = (
  workspaces: SavedWorkspace[],
  workspace: SavedWorkspace
): SavedWorkspace[] => [workspace, ...workspaces.filter((w) => w.id !== workspace.id)]

export const upsertArchivedWorkspace = (
  workspaces: SavedWorkspace[],
  workspace: SavedWorkspace
): SavedWorkspace[] =>
  [workspace, ...workspaces.filter((w) => w.id !== workspace.id)].slice(
    0,
    MAX_ARCHIVED_WORKSPACES
  )

export const sortByLastAccessedDesc = (workspaces: SavedWorkspace[]): SavedWorkspace[] =>
  [...workspaces].sort(
    (a, b) =>
      new Date(b.lastAccessedAt).getTime() - new Date(a.lastAccessedAt).getTime()
  )

export const cloneWorkspaceChatSession = (
  session: WorkspaceChatSession
): WorkspaceChatSession => ({
  messages: session.messages.map((message) => ({ ...message })),
  history: session.history.map((entry) => ({ ...entry })),
  historyId: session.historyId,
  serverChatId: session.serverChatId
})

const buildPersistedWorkspaceChatSession = (
  session: WorkspaceChatSession
): PersistedWorkspaceChatSession => {
  const boundedMessages =
    session.messages.length > WORKSPACE_PERSIST_MAX_CHAT_MESSAGES_PER_SESSION
      ? session.messages.slice(-WORKSPACE_PERSIST_MAX_CHAT_MESSAGES_PER_SESSION)
      : session.messages

  return {
    messages: boundedMessages.map((message) => ({ ...message })),
    historyId: session.historyId,
    serverChatId: session.serverChatId
  }
}

const buildPersistedWorkspaceChatSessions = (
  sessions: Record<string, WorkspaceChatSession>
): Record<string, PersistedWorkspaceChatSession> => {
  const persisted: Record<string, PersistedWorkspaceChatSession> = {}
  for (const [workspaceId, session] of Object.entries(sessions)) {
    persisted[workspaceId] = buildPersistedWorkspaceChatSession(session)
  }
  return persisted
}

export const createFallbackWorkspaceSnapshot = (): WorkspaceSnapshot => {
  const replacementId = generateWorkspaceId()
  const replacementName = "New Research"
  const replacementTag = `workspace:${createSlug(replacementName) || replacementId.slice(0, 8)}`
  return createEmptyWorkspaceSnapshot({
    id: replacementId,
    name: replacementName,
    tag: replacementTag,
    createdAt: new Date()
  })
}

export const duplicateWorkspaceSnapshot = (
  snapshot: WorkspaceSnapshot
): WorkspaceSnapshot => {
  const duplicateId = generateWorkspaceId()
  const duplicateName = `${snapshot.workspaceName} (Copy)`
  const duplicateTag = `workspace:${createSlug(duplicateName) || duplicateId.slice(0, 8)}`
  const sourceIdMap = new Map<string, string>()
  const folderIdMap = new Map<string, string>()

  const duplicatedSources = snapshot.sources.map((source) => {
    const nextSourceId = generateWorkspaceId()
    sourceIdMap.set(source.id, nextSourceId)
    return {
      ...source,
      id: nextSourceId,
      addedAt: reviveDateOrNull(source.addedAt) || new Date()
    }
  })

  const duplicatedSelectedSourceIds = snapshot.selectedSourceIds
    .map((sourceId) => sourceIdMap.get(sourceId))
    .filter((sourceId): sourceId is string => Boolean(sourceId))

  for (const folder of snapshot.sourceFolders) {
    folderIdMap.set(folder.id, generateWorkspaceId())
  }

  const duplicatedSourceFolders = snapshot.sourceFolders.map((folder) => ({
    ...folder,
    id: folderIdMap.get(folder.id) || generateWorkspaceId(),
    workspaceId: duplicateId,
    parentFolderId: folder.parentFolderId
      ? folderIdMap.get(folder.parentFolderId) || null
      : null,
    createdAt: reviveDateOrNull(folder.createdAt) || new Date(),
    updatedAt: reviveDateOrNull(folder.updatedAt) || new Date()
  }))

  const duplicatedSourceFolderMemberships = snapshot.sourceFolderMemberships
    .map((membership) => {
      const nextFolderId = folderIdMap.get(membership.folderId)
      const nextSourceId = sourceIdMap.get(membership.sourceId)
      if (!nextFolderId || !nextSourceId) {
        return null
      }
      return {
        folderId: nextFolderId,
        sourceId: nextSourceId
      }
    })
    .filter(
      (
        membership
      ): membership is WorkspaceSourceFolderMembership => Boolean(membership)
    )

  const duplicatedSelectedSourceFolderIds = snapshot.selectedSourceFolderIds
    .map((folderId) => folderIdMap.get(folderId))
    .filter((folderId): folderId is string => Boolean(folderId))

  const duplicatedArtifacts = snapshot.generatedArtifacts.map((artifact) => ({
    ...artifact,
    id: generateWorkspaceId(),
    createdAt: reviveDateOrNull(artifact.createdAt) || new Date(),
    completedAt: reviveDateOrUndefined(artifact.completedAt)
  }))

  return {
    workspaceId: duplicateId,
    workspaceName: duplicateName,
    workspaceTag: duplicateTag,
    studyMaterialsPolicy: snapshot.studyMaterialsPolicy ?? null,
    assistantDefaults: coerceWorkspaceAssistantDefaultsForRehydrate(
      snapshot.assistantDefaults
    ),
    workspaceCreatedAt: new Date(),
    workspaceChatReferenceId: duplicateId,
    sources: duplicatedSources,
    selectedSourceIds: duplicatedSelectedSourceIds,
    sourceFolders: duplicatedSourceFolders,
    sourceFolderMemberships: duplicatedSourceFolderMemberships,
    selectedSourceFolderIds: duplicatedSelectedSourceFolderIds,
    activeFolderId: snapshot.activeFolderId
      ? folderIdMap.get(snapshot.activeFolderId) || null
      : null,
    generatedArtifacts: duplicatedArtifacts,
    notes: snapshot.notes,
    currentNote: {
      ...snapshot.currentNote,
      id: undefined,
      version: undefined,
      isDirty: false
    },
    workspaceBanner: cloneWorkspaceValue(snapshot.workspaceBanner),
    leftPaneCollapsed: snapshot.leftPaneCollapsed,
    rightPaneCollapsed: snapshot.rightPaneCollapsed,
    audioSettings: { ...snapshot.audioSettings }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

// Captured store setter so `onRehydrateStorage` can publish the post-processed
// hydrated state THROUGH the store (notifying subscribers) instead of mutating
// the passed-in state object in place. The creator runs before hydration, so
// this is always assigned by the time the rehydrate callback fires (this also
// avoids a temporal-dead-zone reference to `useWorkspaceStore` when the backing
// storage is synchronous and hydration happens during store construction).
let publishWorkspaceHydration: ((next: WorkspaceState) => void) | null = null

export const useWorkspaceStore = createWithEqualityFn<WorkspaceState>()(
  persist<WorkspaceState, [], [], PersistedWorkspaceState>(
    (set, get) => {
      publishWorkspaceHydration = (next) => set(next, true)
      return {
        ...initialState,
        ...createSourcesSlice(set, get),
        ...createStudioSlice(set, get),
        ...createUISlice(set, get),
        ...createWorkspaceListSlice(set, get),
      }
    },
    {
      name: WORKSPACE_STORAGE_KEY,
      storage: createJSONStorage(() => createWorkspaceStorage()),
      version: 1,
      migrate: (persistedState) => migratePersistedWorkspaceState(persistedState),
      // NOTE: The rest of persist config follows below (was already here)

      // Only persist essential state, not transient UI state
      partialize: (state): PersistedWorkspaceState => {
        const nextSnapshots = { ...state.workspaceSnapshots }
        if (state.workspaceId) {
          nextSnapshots[state.workspaceId] = buildWorkspaceSnapshot(state)
        }

        // WP-16: unchanged workspaces keep their raw snapshot reference
        // between passes, so reuse the previous sanitized persist output
        // instead of re-running the sanitize pass for every workspace on
        // every write. Only the active workspace (rebuilt from top-level
        // state each pass) re-sanitizes.
        const persistedSnapshots: Record<string, WorkspaceSnapshot> = {}
        for (const [workspaceId, snapshot] of Object.entries(nextSnapshots)) {
          const existingRecord =
            workspacePersistPartializeRecords.get(workspaceId)?.snapshot
          if (existingRecord && existingRecord.rawRef === snapshot) {
            persistedSnapshots[workspaceId] = existingRecord.persisted
            continue
          }
          const persisted: WorkspaceSnapshot = {
            ...snapshot,
            generatedArtifacts: sanitizeArtifactsForPersistence(
              snapshot.generatedArtifacts
            )
          }
          setWorkspacePersistPartializeRecord(workspaceId, {
            snapshot: { rawRef: snapshot, persisted }
          })
          persistedSnapshots[workspaceId] = persisted
        }

        // Chat sessions: same reference-based reuse — an unchanged raw
        // session skips the per-message bounding rebuild.
        const persistedChatSessions: Record<string, PersistedWorkspaceChatSession> =
          {}
        for (const [workspaceId, session] of Object.entries(
          state.workspaceChatSessions
        )) {
          const existingRecord =
            workspacePersistPartializeRecords.get(workspaceId)?.chat
          if (existingRecord && existingRecord.rawRef === session) {
            persistedChatSessions[workspaceId] = existingRecord.persisted
            continue
          }
          const persisted = buildPersistedWorkspaceChatSession(session)
          setWorkspacePersistPartializeRecord(workspaceId, {
            chat: { rawRef: session, persisted }
          })
          persistedChatSessions[workspaceId] = persisted
        }

        const persistedState: PersistedWorkspaceState = {
          // Active workspace identity
          workspaceId: state.workspaceId,

          // Workspace lists
          savedWorkspaces: state.savedWorkspaces,
          archivedWorkspaces: state.archivedWorkspaces,
          workspaceCollections: state.workspaceCollections,

          // Workspace snapshots
          workspaceSnapshots: persistedSnapshots,

          // Workspace chat sessions (messages canonical; history derived on rehydrate)
          workspaceChatSessions: persistedChatSessions
        }

        // Flag the records as describing the value zustand persist is about
        // to stringify and hand to storage.setItem (consumed there).
        workspacePersistPartializeRecordsFresh = true

        recordWorkspacePersistenceDiagnostics(
          WORKSPACE_STORAGE_KEY,
          persistedState
        )

        return persistedState
      },
      // Rehydrate dates properly and handle migration
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Ensure dates are Date objects after rehydration
          state.workspaceCreatedAt = reviveDateOrNull(state.workspaceCreatedAt)
          state.sources = reviveSources(
            Array.isArray(state.sources) ? state.sources : []
          )
          const selectableSourceIds = new Set(
            state.sources
              .filter((source) => isWorkspaceSourceSelectable(source))
              .map((source) => source.id)
          )
          state.selectedSourceIds = (
            Array.isArray(state.selectedSourceIds) ? state.selectedSourceIds : []
          ).filter((id) => selectableSourceIds.has(id))
          const persistedArtifacts = Array.isArray(state.generatedArtifacts)
            ? state.generatedArtifacts
            : []
          const interruptedArtifactCount = persistedArtifacts.filter(
            (artifact) =>
              (artifact?.status || "").toString().toLowerCase() === "generating"
          ).length
          state.generatedArtifacts = reviveArtifacts(persistedArtifacts)
          if (interruptedArtifactCount > 0) {
            void trackResearchWorkspaceTelemetry({
              type: "artifact_rehydrated_failed",
              workspace_id: state.workspaceId || null,
              interrupted_count: interruptedArtifactCount
            })
          }

          // Migration: ensure optional fields exist
          if (!state.audioSettings) {
            state.audioSettings = { ...DEFAULT_AUDIO_SETTINGS }
          }
          if (!state.currentNote) {
            state.currentNote = { ...DEFAULT_WORKSPACE_NOTE }
          }
          if (!state.workspaceChatReferenceId) {
            state.workspaceChatReferenceId = state.workspaceId || ""
          }

          // Migration: ensure savedWorkspaces exists and dates are properly converted
          state.savedWorkspaces = (
            Array.isArray(state.savedWorkspaces) ? state.savedWorkspaces : []
          ).map(reviveSavedWorkspace)
          state.archivedWorkspaces = (
            Array.isArray(state.archivedWorkspaces) ? state.archivedWorkspaces : []
          ).map(reviveSavedWorkspace)
          state.workspaceCollections = reviveWorkspaceCollections(
            Array.isArray(state.workspaceCollections)
              ? state.workspaceCollections
              : []
          )

          // Migration: ensure workspace snapshots exist and are hydrated
          state.workspaceSnapshots = normalizeWorkspaceSnapshotsForRehydrate(
            state.workspaceSnapshots
          )
          state.workspaceChatSessions =
            normalizeWorkspaceChatSessionsForRehydrate(
              state.workspaceChatSessions
            )

          // Ensure active workspace snapshot exists and use it as canonical source
          if (state.workspaceId) {
            const activeSnapshot =
              state.workspaceSnapshots[state.workspaceId] ||
              createEmptyWorkspaceSnapshot({
                id: state.workspaceId,
                name: state.workspaceName || "Untitled Workspace",
                tag:
                  state.workspaceTag ||
                  `workspace:${state.workspaceId.slice(0, 8)}`,
                createdAt: state.workspaceCreatedAt || new Date()
              })

            state.workspaceSnapshots[state.workspaceId] = activeSnapshot
            Object.assign(state, applyWorkspaceSnapshot(activeSnapshot))

            state.savedWorkspaces = upsertSavedWorkspace(
              state.savedWorkspaces,
              createSavedWorkspaceEntry(
                activeSnapshot,
                new Date(),
                getSavedWorkspaceCollectionId(
                  state.savedWorkspaces,
                  state.archivedWorkspaces,
                  activeSnapshot.workspaceId
                )
              )
            )
          }

          state.storeHydrated = true

          // Publish the post-processed hydrated state THROUGH the store so
          // subscribers (already-mounted components, loading gates keyed on
          // `storeHydrated`) are notified. Persist applies the raw persisted
          // values via its own `set()` BEFORE this callback, but the mutations
          // above (date revival, snapshot application, `storeHydrated`) happen
          // afterwards and would otherwise never be broadcast. Spreading into a
          // fresh object gives `set` a new reference so the update is not
          // dropped as a no-op; `replace: true` matches the shape we mutated.
          publishWorkspaceHydration?.({ ...state })
        }
      }
    }
  )
)

// Expose for debugging
if (typeof window !== "undefined") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useWorkspaceStore = useWorkspaceStore
}
