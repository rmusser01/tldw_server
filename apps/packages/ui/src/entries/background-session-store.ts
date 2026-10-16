import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

// Durable, worker-survivable storage for MV3 background session state.
//
// Chrome suspends idle MV3 service workers (~30s), which wipes the in-memory
// Maps/Set that `entries/background.ts` uses to track in-flight ingest
// sessions, pending 401 auth-replays, and quick-ingest modal sessions. This
// module serialises that state to `chrome.storage.session` (falling back to
// `chrome.storage.local` when session storage is unavailable) so it can be
// rehydrated when the worker restarts.
//
// The serialise/deserialise helpers are pure (no browser dependency) so they
// are directly unit-testable; the async read/write wrappers accept an injected
// storage area for the same reason.

export type PersistedIngestSession = Record<string, unknown>

export type PersistedQuickIngestSession = {
  requestScope?: ServicePromptRequestScope
  sessionId: string
  cancelled: boolean
}

export type PersistedQuickIngestRemoteJob = {
  jobId: number
  batchId: string
  meta: Record<string, unknown>
}

export type PersistedQuickIngestPlannedItem = {
  key: string
  collectionId: number
  itemId: number
  idempotencyKey?: string | null
}

// A quick-ingest batch that has finished submitting its remote ingest jobs and
// is (or was) polling them to completion. Persisted so the remote-job polling
// phase can be resumed and finalized after an MV3 worker restart, even though
// the in-flight multipart UPLOAD phase that produced these job ids cannot be
// resumed (there is no live fetch to abort/resume).
export type PersistedQuickIngestBatch = {
  requestScope?: ServicePromptRequestScope
  sessionId: string
  totalCount: number
  processedCount: number
  ingestTimeoutMs: number
  remoteJobs: PersistedQuickIngestRemoteJob[]
  collectedResults: Array<Record<string, unknown>>
  plannedConferenceItems: PersistedQuickIngestPlannedItem[]
}

export type PersistedSessionState = {
  ingestSessions: Record<string, PersistedIngestSession>
  pendingAuthReplay: string[]
  quickIngestSessions: PersistedQuickIngestSession[]
  quickIngestBatches: PersistedQuickIngestBatch[]
}

export const SESSION_STATE_STORAGE_KEY = "tldw:backgroundSessionStateV1"

// WP-19: the monolithic key above is the LEGACY shape. New writes shard the
// state into per-funnel keys (`tldw:ingest:<funnelId>`, plus `qi-session:` /
// `qi-batch:` / `auth-replay` sub-keys for the quick-ingest namespaces) with
// one small index key listing every live shard. A status tick for one funnel
// then rewrites only that funnel's shard instead of the whole world. Funnel ids
// are generated as `ingest-*` and quick-ingest session ids as `qi-*`, so the
// namespaces cannot collide.
export const INGEST_STATE_KEY_PREFIX = "tldw:ingest:"
export const SESSION_STATE_INDEX_KEY = `${INGEST_STATE_KEY_PREFIX}index`
const AUTH_REPLAY_SHARD_KEY = `${INGEST_STATE_KEY_PREFIX}auth-replay`
const QUICK_INGEST_SESSION_SHARD_PREFIX = `${INGEST_STATE_KEY_PREFIX}qi-session:`
const QUICK_INGEST_BATCH_SHARD_PREFIX = `${INGEST_STATE_KEY_PREFIX}qi-batch:`

export type SessionStorageArea = {
  get: (keys: string | string[]) => Promise<Record<string, unknown>>
  set: (items: Record<string, unknown>) => Promise<void>
  remove?: (keys: string | string[]) => Promise<void>
}

export const emptySessionState = (): PersistedSessionState => ({
  ingestSessions: {},
  pendingAuthReplay: [],
  quickIngestSessions: [],
  quickIngestBatches: []
})

/** Only an explicit credential-free owner can authorize worker resume. */
export const readQuickIngestRequestScope = (value: unknown): ServicePromptRequestScope | null => {
  if (!value || typeof value !== "object") return null
  const raw = value as Record<string, unknown>
  const config = raw.config as Record<string, unknown> | null
  if (!config || typeof config !== "object" || typeof config.serverUrl !== "string" || !config.serverUrl.trim()) return null
  if (config.authMode !== "single-user" && config.authMode !== "multi-user") return null
  if (config.authMode === "single-user" && (typeof config.expectedSingleUserApiKeyScope !== "string" || !config.expectedSingleUserApiKeyScope)) return null
  if (config.authMode === "multi-user" && ((typeof raw.userId !== "string" && typeof raw.userId !== "number") || !String(raw.userId).trim())) return null
  if (config.authSource != null && config.authSource !== "manual" && config.authSource !== "cookie-session") return null
  return { config: {
    serverUrl: config.serverUrl, authMode: config.authMode,
    ...(config.authSource ? { authSource: config.authSource } : {}),
    ...(typeof config.orgId === "number" ? { orgId: config.orgId } : {}),
    ...(config.authMode === "single-user" ? { expectedSingleUserApiKeyScope: config.expectedSingleUserApiKeyScope as string } : {})
  }, userId: config.authMode === "single-user" ? null : raw.userId as string | number }
}

const toFiniteInt = (value: unknown): number | null => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  return Math.trunc(parsed)
}

// Validate + normalize a single quick-ingest batch record. Shared by the
// serialize (Map -> array) and deserialize (storage -> array) paths so both drop
// the same malformed data instead of persisting/rehydrating junk.
const normalizeQuickIngestBatch = (
  value: unknown
): PersistedQuickIngestBatch | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const sessionId = String(record.sessionId || "").trim()
  if (!sessionId) return null
  const requestScope = readQuickIngestRequestScope(record.requestScope)
  if (!requestScope) return null

  const remoteJobs: PersistedQuickIngestRemoteJob[] = []
  if (Array.isArray(record.remoteJobs)) {
    for (const raw of record.remoteJobs) {
      if (!raw || typeof raw !== "object") continue
      const job = raw as Record<string, unknown>
      const jobId = toFiniteInt(job.jobId)
      const batchId = String(job.batchId || "").trim()
      if (jobId == null || jobId <= 0 || !batchId) continue
      remoteJobs.push({
        jobId,
        batchId,
        meta:
          job.meta && typeof job.meta === "object"
            ? (job.meta as Record<string, unknown>)
            : {}
      })
    }
  }

  const plannedConferenceItems: PersistedQuickIngestPlannedItem[] = []
  if (Array.isArray(record.plannedConferenceItems)) {
    for (const raw of record.plannedConferenceItems) {
      if (!raw || typeof raw !== "object") continue
      const item = raw as Record<string, unknown>
      const collectionId = toFiniteInt(item.collectionId)
      const itemId = toFiniteInt(item.itemId)
      if (collectionId == null || itemId == null) continue
      plannedConferenceItems.push({
        key: String(item.key || "").trim(),
        collectionId,
        itemId,
        idempotencyKey:
          typeof item.idempotencyKey === "string" ? item.idempotencyKey : null
      })
    }
  }

  const collectedResults = Array.isArray(record.collectedResults)
    ? (record.collectedResults.filter(
        (entry) => entry && typeof entry === "object"
      ) as Array<Record<string, unknown>>)
    : []

  return {
    sessionId, requestScope,
    totalCount: Math.max(0, toFiniteInt(record.totalCount) ?? 0),
    processedCount: Math.max(0, toFiniteInt(record.processedCount) ?? 0),
    ingestTimeoutMs: Math.max(0, toFiniteInt(record.ingestTimeoutMs) ?? 0),
    remoteJobs,
    collectedResults,
    plannedConferenceItems
  }
}

export const serializeIngestSessions = (
  sessions: Map<string, PersistedIngestSession>
): Record<string, PersistedIngestSession> => {
  const out: Record<string, PersistedIngestSession> = {}
  for (const [key, value] of sessions) {
    const funnelId = String(key || "").trim()
    if (funnelId && value && typeof value === "object") {
      out[funnelId] = value
    }
  }
  return out
}

export const serializePendingReplay = (ids: Set<string>): string[] =>
  Array.from(ids, (id) => String(id || "").trim()).filter(
    (id) => id.length > 0
  )

export const serializeQuickIngestSessions = (
  sessions: Map<string, { sessionId?: string; cancelled?: boolean; requestScope?: ServicePromptRequestScope }>
): PersistedQuickIngestSession[] => {
  const out: PersistedQuickIngestSession[] = []
  for (const [key, value] of sessions) {
    const sessionId = String(value?.sessionId || key || "").trim()
    const requestScope = readQuickIngestRequestScope(value?.requestScope)
    if (sessionId && requestScope) {
      out.push({ sessionId, requestScope, cancelled: Boolean(value?.cancelled) })
    }
  }
  return out
}

export const serializeQuickIngestBatches = (
  batches:
    | Map<string, PersistedQuickIngestBatch>
    | Iterable<PersistedQuickIngestBatch>
): PersistedQuickIngestBatch[] => {
  const source =
    batches instanceof Map ? Array.from(batches.values()) : Array.from(batches)
  const out: PersistedQuickIngestBatch[] = []
  for (const batch of source) {
    const normalized = normalizeQuickIngestBatch(batch)
    if (normalized) out.push(normalized)
  }
  return out
}

// WP-19: collected quick-ingest results keep fat server response bodies in
// memory for the live UI, but the resume path only needs id/status/error (plus
// the small scalar descriptors). Strip `result.data` at PERSIST time only —
// runtime objects are never mutated.
const stripCollectedResultForPersist = (
  result: Record<string, unknown>
): Record<string, unknown> => {
  if (!("data" in result)) return result
  const { data: _data, ...rest } = result
  return rest
}

export const stripQuickIngestBatchForPersist = (
  batch: PersistedQuickIngestBatch
): Record<string, unknown> => ({
  ...batch,
  collectedResults: batch.collectedResults.map(stripCollectedResultForPersist)
})

// Split a serialized session state into per-funnel storage shards plus the
// index key. Only the shards that CHANGED since the last write are written by
// the coalescing writer, so a tick for one ingest costs O(that funnel) bytes.
export const shardSessionState = (
  state: PersistedSessionState
): Record<string, unknown> => {
  const shards: Record<string, unknown> = {}
  for (const [funnelId, session] of Object.entries(
    state.ingestSessions ?? {}
  )) {
    if (funnelId) shards[INGEST_STATE_KEY_PREFIX + funnelId] = session
  }
  shards[AUTH_REPLAY_SHARD_KEY] = state.pendingAuthReplay ?? []
  for (const session of state.quickIngestSessions ?? []) {
    shards[QUICK_INGEST_SESSION_SHARD_PREFIX + session.sessionId] = session
  }
  for (const batch of state.quickIngestBatches ?? []) {
    shards[QUICK_INGEST_BATCH_SHARD_PREFIX + batch.sessionId] =
      stripQuickIngestBatchForPersist(batch)
  }
  shards[SESSION_STATE_INDEX_KEY] = Object.keys(shards).sort()
  return shards
}

// Rebuild the monolithic state shape from storage shards, then reuse the
// existing validation/normalization in `deserializeSessionState`.
const assembleFromShards = (
  values: Record<string, unknown>
): PersistedSessionState => {
  const assembled: Record<string, unknown> = {
    ingestSessions: {},
    pendingAuthReplay: [],
    quickIngestSessions: [],
    quickIngestBatches: []
  }
  for (const [key, value] of Object.entries(values ?? {})) {
    if (value == null || typeof value !== "object") continue
    if (key === SESSION_STATE_INDEX_KEY) continue
    if (key === AUTH_REPLAY_SHARD_KEY) {
      if (Array.isArray(value)) {
        (assembled.pendingAuthReplay as unknown[]).push(...value)
      }
    } else if (key.startsWith(QUICK_INGEST_SESSION_SHARD_PREFIX)) {
      (assembled.quickIngestSessions as unknown[]).push(value)
    } else if (key.startsWith(QUICK_INGEST_BATCH_SHARD_PREFIX)) {
      (assembled.quickIngestBatches as unknown[]).push(value)
    } else if (key.startsWith(INGEST_STATE_KEY_PREFIX)) {
      (assembled.ingestSessions as Record<string, unknown>)[
        key.slice(INGEST_STATE_KEY_PREFIX.length)
      ] = value
    }
  }
  return deserializeSessionState(assembled)
}

export const deserializeSessionState = (raw: unknown): PersistedSessionState => {
  const state = emptySessionState()
  if (!raw || typeof raw !== "object") return state
  const record = raw as Record<string, unknown>

  const ingest = record.ingestSessions
  if (ingest && typeof ingest === "object") {
    for (const [key, value] of Object.entries(
      ingest as Record<string, unknown>
    )) {
      const funnelId = String(key || "").trim()
      if (funnelId && value && typeof value === "object") {
        state.ingestSessions[funnelId] = value as PersistedIngestSession
      }
    }
  }

  if (Array.isArray(record.pendingAuthReplay)) {
    const seen = new Set<string>()
    for (const entry of record.pendingAuthReplay) {
      const id = String(entry || "").trim()
      if (id && !seen.has(id)) {
        seen.add(id)
        state.pendingAuthReplay.push(id)
      }
    }
  }

  if (Array.isArray(record.quickIngestSessions)) {
    for (const entry of record.quickIngestSessions) {
      if (entry && typeof entry === "object") {
        const sessionId = String(
          (entry as Record<string, unknown>).sessionId || ""
        ).trim()
        const requestScope = readQuickIngestRequestScope((entry as Record<string, unknown>).requestScope)
        if (sessionId && requestScope) {
          state.quickIngestSessions.push({
            sessionId, requestScope,
            cancelled: Boolean((entry as Record<string, unknown>).cancelled)
          })
        }
      }
    }
  }

  if (Array.isArray(record.quickIngestBatches)) {
    for (const entry of record.quickIngestBatches) {
      const normalized = normalizeQuickIngestBatch(entry)
      if (normalized) state.quickIngestBatches.push(normalized)
    }
  }

  return state
}

// Resolve a promise-based storage area, preferring session storage (cleared on
// browser restart, but survives service-worker suspension) and falling back to
// local storage. Returns null when neither is available (e.g. non-extension
// contexts) so callers can no-op gracefully.
export const getSessionStorageArea = (): SessionStorageArea | null => {
  try {
    const chromeApi = (globalThis as { chrome?: any }).chrome
    const area = chromeApi?.storage?.session || chromeApi?.storage?.local
    if (area && typeof area.get === "function" && typeof area.set === "function") {
      return area as SessionStorageArea
    }
  } catch {
    // fall through
  }
  return null
}

// Set when a read observed the legacy monolithic key (either migrating it or
// finding a stale leftover). The coalescing writer removes the legacy key on
// its next successful drain, after the sharded copies are durable.
let legacySessionStateKeyCleanupPending = false

export const readPersistedSessionState = async (
  area: SessionStorageArea | null = getSessionStorageArea()
): Promise<PersistedSessionState> => {
  if (!area) return emptySessionState()
  try {
    const head = await area.get([
      SESSION_STATE_INDEX_KEY,
      SESSION_STATE_STORAGE_KEY
    ])
    const legacy = head?.[SESSION_STATE_STORAGE_KEY]
    const index = head?.[SESSION_STATE_INDEX_KEY]
    if (Array.isArray(index)) {
      // New sharded shape: index present. A lingering legacy key is removed on
      // the next write drain (it is no longer authoritative).
      legacySessionStateKeyCleanupPending = legacy != null
      const shardKeys = index.filter(
        (key): key is string =>
          typeof key === "string" &&
          key.startsWith(INGEST_STATE_KEY_PREFIX) &&
          key !== SESSION_STATE_INDEX_KEY
      )
      if (shardKeys.length === 0) return emptySessionState()
      const values = await area.get(shardKeys)
      return assembleFromShards(values ?? {})
    }
    if (legacy && typeof legacy === "object") {
      // Legacy monolithic shape (pre-WP-19 worker, e.g. across an extension
      // update with an in-flight session): consume it unchanged, then
      // one-time migrate by splitting it into per-funnel shards + index. The
      // legacy key itself is kept until the writer's next drain so an
      // interrupted migration can never lose the only copy.
      const state = deserializeSessionState(legacy)
      try {
        await area.set(shardSessionState(state))
        legacySessionStateKeyCleanupPending = true
      } catch {
        // Best-effort migration; the legacy key remains for the next read.
      }
      return state
    }
    return emptySessionState()
  } catch {
    return emptySessionState()
  }
}

export const writePersistedSessionState = async (
  state: PersistedSessionState,
  area: SessionStorageArea | null = getSessionStorageArea()
): Promise<void> => {
  if (!area) return
  await area.set(shardSessionState(state))
}

// Remove keys from the area, degrading to null-valued overwrites (read as
// absent) when the area has no `remove` (the minimum SessionStorageArea).
const removeFromArea = async (
  area: SessionStorageArea,
  keys: string[]
): Promise<void> => {
  if (keys.length === 0) return
  if (typeof area.remove === "function") {
    await area.remove(keys)
    return
  }
  const nulled: Record<string, unknown> = {}
  for (const key of keys) nulled[key] = null
  await area.set(nulled)
}

// Coalesce fire-and-forget persist requests into diffed storage writes.
// WP-19: the caller passes a SNAPSHOT BUILDER, so serialization of the live
// Maps only happens when a write actually drains — a burst of status ticks
// within one drain window serializes exactly once. Each drain additionally
// diffs the sharded state against what this writer last wrote, so unchanged
// funnels (and an unchanged index) cost zero bytes. Callers are never blocked
// — the returned function is synchronous.
export const createSerializedSessionStateWriter = (
  area: SessionStorageArea | null = getSessionStorageArea(),
  buildState: () => PersistedSessionState,
  onError?: (error: unknown) => void
): (() => void) => {
  let dirty = false
  let inFlight: Promise<void> | null = null
  const lastWritten = new Map<string, string>()

  const drain = async (): Promise<void> => {
    // Yield one microtask before the first serialization so a synchronous
    // burst of persist requests (e.g. 20 status ticks in one window) is fully
    // absorbed before the snapshot is built: one drain window, one serialize.
    await Promise.resolve()
    // Keep writing while newer snapshots have arrived. There is no `await`
    // between the loop's `dirty` check and clearing `inFlight`, so a caller
    // can only ever run while we are suspended on a write below (inFlight
    // set) or once we are fully idle (inFlight null) — never in between.
    while (dirty) {
      dirty = false
      try {
        const shards = shardSessionState(buildState())
        const changed: Record<string, unknown> = {}
        for (const [key, value] of Object.entries(shards)) {
          const signature = JSON.stringify(value) ?? "null"
          if (lastWritten.get(key) === signature) continue
          lastWritten.set(key, signature)
          changed[key] = value
        }
        const stale: string[] = []
        for (const key of Array.from(lastWritten.keys())) {
          if (!(key in shards)) {
            stale.push(key)
            lastWritten.delete(key)
          }
        }
        if (Object.keys(changed).length > 0) {
          await area?.set(changed)
        }
        if (!area) continue
        await removeFromArea(area, stale)
        if (legacySessionStateKeyCleanupPending) {
          legacySessionStateKeyCleanupPending = false
          await removeFromArea(area, [SESSION_STATE_STORAGE_KEY])
        }
      } catch (error) {
        onError?.(error)
      }
    }
    inFlight = null
  }

  return () => {
    dirty = true
    if (!inFlight) {
      inFlight = drain()
    }
  }
}

// True when a persisted ingest session is one a restarted worker cannot resume:
// it is still in a pre-submission lifecycle state (queued/running) but has no
// jobIds to poll and is not awaiting an auth replay. Such a session was
// interrupted (e.g. mid "Checking for existing media...") before its
// non-resumable job submission completed, so it must be reported as failed —
// otherwise the sidepanel is left waiting forever.
export const isInterruptedPreSubmissionIngestSession = (
  session: PersistedIngestSession | null | undefined
): boolean => {
  if (!session || typeof session !== "object") return false
  const record = session as Record<string, unknown>
  const status = record.status
  if (status !== "queued" && status !== "running") return false
  if (record.awaitingAuth === true) return false
  const jobIds = record.jobIds
  if (Array.isArray(jobIds) && jobIds.length > 0) return false
  return true
}

// Given the persisted ingest sessions and the pending-auth-replay ids, return
// the funnelIds that were interrupted mid pre-submission and therefore cannot be
// resumed by the jobId-driven poll. Auth-replay ids are excluded because those
// sessions are re-driven by the auth replay path once credentials return.
export const selectInterruptedIngestFunnelIds = (
  ingestSessions: Record<string, PersistedIngestSession>,
  pendingAuthReplay: Iterable<string> = []
): string[] => {
  const replay = new Set<string>()
  for (const id of pendingAuthReplay) {
    const trimmed = String(id || "").trim()
    if (trimmed) replay.add(trimmed)
  }
  const out: string[] = []
  if (ingestSessions && typeof ingestSessions === "object") {
    for (const [funnelId, session] of Object.entries(ingestSessions)) {
      if (replay.has(funnelId)) continue
      if (isInterruptedPreSubmissionIngestSession(session)) out.push(funnelId)
    }
  }
  return out
}
