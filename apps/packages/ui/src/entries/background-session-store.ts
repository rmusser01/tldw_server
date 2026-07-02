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

export type SessionStorageArea = {
  get: (key: string) => Promise<Record<string, unknown>>
  set: (items: Record<string, unknown>) => Promise<void>
}

export const emptySessionState = (): PersistedSessionState => ({
  ingestSessions: {},
  pendingAuthReplay: [],
  quickIngestSessions: [],
  quickIngestBatches: []
})

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
    sessionId,
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
  sessions: Map<string, { sessionId?: string; cancelled?: boolean }>
): PersistedQuickIngestSession[] => {
  const out: PersistedQuickIngestSession[] = []
  for (const [key, value] of sessions) {
    const sessionId = String(value?.sessionId || key || "").trim()
    if (sessionId) {
      out.push({ sessionId, cancelled: Boolean(value?.cancelled) })
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
        if (sessionId) {
          state.quickIngestSessions.push({
            sessionId,
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

export const readPersistedSessionState = async (
  area: SessionStorageArea | null = getSessionStorageArea()
): Promise<PersistedSessionState> => {
  if (!area) return emptySessionState()
  try {
    const result = await area.get(SESSION_STATE_STORAGE_KEY)
    return deserializeSessionState(result?.[SESSION_STATE_STORAGE_KEY])
  } catch {
    return emptySessionState()
  }
}

export const writePersistedSessionState = async (
  state: PersistedSessionState,
  area: SessionStorageArea | null = getSessionStorageArea()
): Promise<void> => {
  if (!area) return
  await area.set({ [SESSION_STATE_STORAGE_KEY]: state })
}
