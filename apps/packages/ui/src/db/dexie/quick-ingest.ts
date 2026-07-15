import type { StateStorage } from "zustand/middleware"

import { db } from "./schema"
import type { QuickIngestSessionDbRecord } from "./types"

export type QuickIngestPersistenceStatus =
  | "ready"
  | "migrating"
  | "unavailable"
  | "quota_error"

export const QUICK_INGEST_STORAGE_KEY = "tldw-quick-ingest-session"
export const QUICK_INGEST_DRAFT_RETENTION_MS = 7 * 24 * 60 * 60 * 1_000
export const QUICK_INGEST_TERMINAL_RETENTION_MS = 14 * 24 * 60 * 60 * 1_000
export const QUICK_INGEST_INTERRUPTED_RETENTION_MS = 30 * 24 * 60 * 60 * 1_000
export const QUICK_INGEST_PROCESSING_RETENTION_MS = 90 * 24 * 60 * 60 * 1_000
export const QUICK_INGEST_SUBMISSION_LEASE_MS = 60_000

type QuickIngestSessionTable = {
  get: (id: string) => Promise<QuickIngestSessionDbRecord | undefined>
  put: (row: QuickIngestSessionDbRecord) => Promise<unknown>
  delete: (id: string) => Promise<unknown>
  toArray: () => Promise<QuickIngestSessionDbRecord[]>
  bulkDelete: (ids: string[]) => Promise<unknown>
}

type QuickIngestDatabase = {
  quickIngestSessions: QuickIngestSessionTable
  transaction: <T>(
    mode: "rw",
    table: QuickIngestSessionTable,
    operation: () => Promise<T>
  ) => Promise<T>
}

type LegacyStorage = Pick<Storage, "getItem" | "removeItem">

export type QuickIngestIndexedDbStorage = {
  storage: StateStorage
  /** Wait for queued storage work and reject when its latest operation failed. */
  flush: () => Promise<void>
  initialize: () => Promise<void>
  cleanupExpired: () => Promise<void>
  commitReviewHandoff: (
    expectedValue: string,
    nextValue: string
  ) => Promise<boolean>
  commitProcessingHandoff: (
    expectedValue: string,
    nextValue: string
  ) => Promise<boolean>
  clearAuthoritativeSession: (
    expected: Pick<QuickIngestSessionDbRecord, "id" | "lifecycle" | "updatedAt">
  ) => Promise<boolean>
  getStatus: () => QuickIngestPersistenceStatus
  subscribeStatus: (
    listener: (status: QuickIngestPersistenceStatus) => void
  ) => () => void
  acquireSubmissionLease: (
    sessionId: string,
    durationMs?: number
  ) => Promise<boolean>
  renewSubmissionLease: (
    sessionId: string,
    durationMs?: number
  ) => Promise<boolean>
  releaseSubmissionLease: (sessionId: string) => Promise<void>
}

type CreateQuickIngestStorageOptions = {
  database?: QuickIngestDatabase
  legacyStorage?: LegacyStorage | null
  ownerId?: string
  now?: () => number
  normalizeValue?: (value: string) => string | null
}

const LIFECYCLES = new Set<QuickIngestSessionDbRecord["lifecycle"]>([
  "draft",
  "processing",
  "completed",
  "partial_failure",
  "cancelled",
  "interrupted",
])

const validateEnvelope = (value: string): string | null => {
  try {
    const envelope = JSON.parse(value) as {
      state?: { session?: Record<string, unknown> | null }
    }
    const state = envelope?.state
    if (
      !state ||
      typeof state !== "object" ||
      !("session" in state)
    ) {
      return null
    }
    if (state.session === null) return value
    const session = state.session
    if (
      !session ||
      typeof session.id !== "string" ||
      !session.id.trim() ||
      !LIFECYCLES.has(
        session.lifecycle as QuickIngestSessionDbRecord["lifecycle"]
      ) ||
      typeof session.updatedAt !== "number" ||
      !Number.isFinite(session.updatedAt)
    ) {
      return null
    }
    return value
  } catch {
    return null
  }
}

const getDefaultLegacyStorage = (): LegacyStorage | null => {
  if (typeof window === "undefined") return null
  return window.sessionStorage
}

const generateOwnerId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `quick-ingest-owner-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const retentionForLifecycle = (
  lifecycle: QuickIngestSessionDbRecord["lifecycle"]
): number => {
  if (lifecycle === "processing") return QUICK_INGEST_PROCESSING_RETENTION_MS
  if (lifecycle === "interrupted") return QUICK_INGEST_INTERRUPTED_RETENTION_MS
  if (lifecycle === "draft") return QUICK_INGEST_DRAFT_RETENTION_MS
  return QUICK_INGEST_TERMINAL_RETENTION_MS
}

const parseRecord = (
  value: string,
  now: () => number
): QuickIngestSessionDbRecord | null => {
  try {
    const envelope = JSON.parse(value) as {
      state?: { session?: Record<string, unknown> | null }
    }
    const session = envelope?.state?.session
    if (!session) return null
    const id = typeof session.id === "string" ? session.id.trim() : ""
    const lifecycle = session.lifecycle
    if (!id || !LIFECYCLES.has(lifecycle as QuickIngestSessionDbRecord["lifecycle"])) {
      return null
    }
    const updatedAt =
      typeof session.updatedAt === "number" && Number.isFinite(session.updatedAt)
        ? session.updatedAt
        : now()
    const normalizedLifecycle = lifecycle as QuickIngestSessionDbRecord["lifecycle"]
    return {
      id,
      lifecycle: normalizedLifecycle,
      updatedAt,
      expiresAt: updatedAt + retentionForLifecycle(normalizedLifecycle),
      value,
    }
  } catch {
    return null
  }
}

export const classifyQuickIngestPersistenceError = (
  error: unknown
): Exclude<QuickIngestPersistenceStatus, "ready" | "migrating"> => {
  const name =
    error && typeof error === "object" && "name" in error
      ? String((error as { name?: unknown }).name || "")
      : ""
  return name === "QuotaExceededError" ? "quota_error" : "unavailable"
}

export const createQuickIngestIndexedDbStorage = (
  options: CreateQuickIngestStorageOptions = {}
): QuickIngestIndexedDbStorage => {
  const database = (options.database ?? db) as unknown as QuickIngestDatabase
  const table = database.quickIngestSessions
  const ownerId = options.ownerId || generateOwnerId()
  const now = options.now || Date.now
  const normalizeValue = options.normalizeValue
  const listeners = new Set<(status: QuickIngestPersistenceStatus) => void>()
  let status: QuickIngestPersistenceStatus = "migrating"
  let initialization: Promise<void> | null = null
  let currentRecordId: string | null = null
  let storageTail: Promise<void> = Promise.resolve()
  let latestStorageResult: Promise<void> = Promise.resolve()

  const publishStatus = (next: QuickIngestPersistenceStatus) => {
    status = next
    for (const listener of listeners) listener(next)
  }

  const normalizeEnvelope = (value: string): string => {
    let normalized: string | null
    try {
      normalized = normalizeValue ? normalizeValue(value) : value
    } catch (error) {
      throw error instanceof Error
        ? error
        : new TypeError("Invalid quick ingest persistence envelope")
    }
    const validated = normalized === null ? null : validateEnvelope(normalized)
    if (validated === null) {
      throw new TypeError("Invalid quick ingest persistence envelope")
    }
    return validated
  }

  const runWrite = <T>(operation: () => Promise<T>): Promise<T> =>
    database.transaction("rw", table, operation)

  const enqueueStorage = <T>(operation: () => Promise<T>): Promise<T> => {
    const result = storageTail.then(operation, operation)
    latestStorageResult = result.then(() => undefined)
    storageTail = latestStorageResult.catch(() => undefined)
    return result
  }

  const flushStorage = (): Promise<void> => latestStorageResult

  const deleteExpired = async (includeProcessing: boolean): Promise<void> => {
    await runWrite(async () => {
      const currentTime = now()
      const expiredIds = (await table.toArray())
        .filter(
          (record) =>
            record.expiresAt <= currentTime &&
            (includeProcessing || record.lifecycle !== "processing")
        )
        .map((record) => record.id)
      if (expiredIds.length > 0) await table.bulkDelete(expiredIds)
    })
  }

  const putRecordPreservingLease = async (
    record: QuickIngestSessionDbRecord
  ): Promise<boolean> => {
    const existing = await table.get(record.id)
    if (
      record.lifecycle === "draft" &&
      existing?.lifecycle !== undefined &&
      existing.lifecycle !== "draft"
    ) {
      return false
    }
    const lease =
      existing?.submissionLeaseOwnerId &&
      typeof existing.submissionLeaseExpiresAt === "number"
        ? {
            submissionLeaseOwnerId: existing.submissionLeaseOwnerId,
            submissionLeaseExpiresAt: existing.submissionLeaseExpiresAt,
          }
        : {}
    await table.put({ ...record, ...lease })
    return true
  }

  const writeRecord = async (
    record: QuickIngestSessionDbRecord
  ): Promise<void> => {
    let wrote = false
    try {
      await runWrite(async () => {
        wrote = await putRecordPreservingLease(record)
      })
    } catch (error) {
      if (classifyQuickIngestPersistenceError(error) !== "quota_error") {
        publishStatus("unavailable")
        throw error
      }
      try {
        await deleteExpired(false)
        await runWrite(async () => {
          wrote = await putRecordPreservingLease(record)
        })
      } catch (retryError) {
        publishStatus(classifyQuickIngestPersistenceError(retryError))
        throw retryError
      }
    }
    if (wrote) currentRecordId = record.id
    publishStatus("ready")
  }

  const commitAuthoritativeHandoff = (
    expectedValue: string,
    nextValue: string,
    kind: "review" | "processing"
  ): Promise<boolean> =>
    enqueueStorage(async () => {
      await initialize()
      let normalizedExpected: string
      let normalizedNext: string
      try {
        normalizedExpected = normalizeEnvelope(expectedValue)
        normalizedNext = normalizeEnvelope(nextValue)
      } catch (error) {
        publishStatus("unavailable")
        throw error
      }
      const expected = parseRecord(normalizedExpected, now)
      const next = parseRecord(normalizedNext, now)
      if (
        !expected ||
        !next ||
        expected.id !== next.id ||
        (kind === "review" && next.lifecycle !== "draft") ||
        (kind === "processing" &&
          (expected.lifecycle !== "draft" || next.lifecycle !== "processing"))
      ) {
        const error = new TypeError(
          "Invalid quick ingest authoritative persistence handoff"
        )
        publishStatus("unavailable")
        throw error
      }

      const attempt = () =>
        runWrite(async () => {
          const existing = await table.get(expected.id)
          if (
            !existing ||
            existing.id !== expected.id ||
            existing.lifecycle !== expected.lifecycle ||
            existing.updatedAt !== expected.updatedAt
          ) {
            return false
          }
          let normalizedExisting: string
          try {
            normalizedExisting = normalizeEnvelope(existing.value)
          } catch {
            return false
          }
          if (normalizedExisting !== normalizedExpected) return false
          if (
            kind === "processing" &&
            (existing.lifecycle !== "draft" ||
              existing.submissionLeaseOwnerId !== ownerId ||
              (existing.submissionLeaseExpiresAt || 0) <= now())
          ) {
            return false
          }
          const lease =
            existing.submissionLeaseOwnerId &&
            typeof existing.submissionLeaseExpiresAt === "number"
              ? {
                  submissionLeaseOwnerId: existing.submissionLeaseOwnerId,
                  submissionLeaseExpiresAt: existing.submissionLeaseExpiresAt,
                }
              : {}
          await table.put({ ...next, ...lease })
          return true
        })

      let committed: boolean
      try {
        committed = await attempt()
      } catch (error) {
        if (classifyQuickIngestPersistenceError(error) !== "quota_error") {
          publishStatus("unavailable")
          throw error
        }
        try {
          await deleteExpired(false)
          committed = await attempt()
        } catch (retryError) {
          publishStatus(classifyQuickIngestPersistenceError(retryError))
          throw retryError
        }
      }
      if (committed) currentRecordId = next.id
      publishStatus("ready")
      return committed
    })

  const deleteCurrentDraft = async (): Promise<void> => {
    if (!currentRecordId) return
    const id = currentRecordId
    await runWrite(async () => {
      const existing = await table.get(id)
      if (existing?.lifecycle === "draft") await table.delete(id)
    })
    currentRecordId = null
  }

  const initialize = (): Promise<void> => {
    if (initialization) return initialization
    publishStatus("migrating")
    initialization = (async () => {
      try {
        const legacyStorage =
          options.legacyStorage === undefined
            ? getDefaultLegacyStorage()
            : options.legacyStorage
        const legacyValue = legacyStorage?.getItem(QUICK_INGEST_STORAGE_KEY)
        if (legacyValue !== null && legacyValue !== undefined) {
          const normalizedLegacyValue = normalizeEnvelope(legacyValue)
          const imported = parseRecord(normalizedLegacyValue, now)
          if (imported) {
            await runWrite(async () => {
              const existing = await table.get(imported.id)
              if (!existing || existing.updatedAt <= imported.updatedAt) {
                await putRecordPreservingLease(imported)
              }
            })
          }
          try {
            legacyStorage?.removeItem(QUICK_INGEST_STORAGE_KEY)
          } catch {
            // The durable row exists; leaving the source permits a safe retry.
          }
        }
        await deleteExpired(true)
        publishStatus("ready")
      } catch (error) {
        publishStatus(classifyQuickIngestPersistenceError(error))
        throw error
      }
    })()
    return initialization
  }

  const storage: StateStorage = {
    getItem: () => enqueueStorage(async () => {
      try {
        await initialize()
        const latest = (await table.toArray()).sort(
          (left, right) =>
            right.updatedAt - left.updatedAt || right.id.localeCompare(left.id)
        )[0]
        currentRecordId = latest?.id || null
        return latest?.value || null
      } catch (error) {
        publishStatus(classifyQuickIngestPersistenceError(error))
        throw error
      }
    }),
    setItem: (_name, value) => enqueueStorage(async () => {
      await initialize()
      let normalizedValue: string
      try {
        normalizedValue = normalizeEnvelope(value)
      } catch (error) {
        publishStatus("unavailable")
        throw error
      }
      const record = parseRecord(normalizedValue, now)
      if (!record) {
        await deleteCurrentDraft()
        return
      }
      await writeRecord(record)
    }),
    removeItem: () => enqueueStorage(async () => {
      try {
        await initialize()
        await deleteCurrentDraft()
      } catch (error) {
        publishStatus(classifyQuickIngestPersistenceError(error))
        throw error
      }
    }),
  }

  const updateLease = async (
    sessionId: string,
    operation: (record: QuickIngestSessionDbRecord) => boolean
  ): Promise<boolean> => {
    await initialize()
    try {
      return await runWrite(async () => {
        const record = await table.get(sessionId)
        if (!record || !operation(record)) return false
        await table.put(record)
        return true
      })
    } catch (error) {
      publishStatus(classifyQuickIngestPersistenceError(error))
      throw error
    }
  }

  return {
    storage,
    flush: flushStorage,
    initialize,
    cleanupExpired: async () => {
      try {
        await initialize()
        await deleteExpired(true)
      } catch (error) {
        publishStatus(classifyQuickIngestPersistenceError(error))
        throw error
      }
    },
    commitReviewHandoff: (expectedValue, nextValue) =>
      commitAuthoritativeHandoff(expectedValue, nextValue, "review"),
    commitProcessingHandoff: (expectedValue, nextValue) =>
      commitAuthoritativeHandoff(expectedValue, nextValue, "processing"),
    clearAuthoritativeSession: (expected) =>
      enqueueStorage(async () => {
        await initialize()
        try {
          const cleared = await runWrite(async () => {
            const existing = await table.get(expected.id)
            if (
              !existing ||
              existing.lifecycle !== expected.lifecycle ||
              existing.updatedAt !== expected.updatedAt
            ) {
              return false
            }
            await table.delete(expected.id)
            return true
          })
          if (cleared && currentRecordId === expected.id) currentRecordId = null
          return cleared
        } catch (error) {
          publishStatus(classifyQuickIngestPersistenceError(error))
          throw error
        }
      }),
    getStatus: () => status,
    subscribeStatus: (listener) => {
      listeners.add(listener)
      listener(status)
      return () => listeners.delete(listener)
    },
    acquireSubmissionLease: (sessionId, durationMs = QUICK_INGEST_SUBMISSION_LEASE_MS) =>
      updateLease(sessionId, (record) => {
        if (record.lifecycle !== "draft") return false
        const expiresAt = record.submissionLeaseExpiresAt || 0
        if (
          record.submissionLeaseOwnerId &&
          record.submissionLeaseOwnerId !== ownerId &&
          expiresAt > now()
        ) {
          return false
        }
        record.submissionLeaseOwnerId = ownerId
        record.submissionLeaseExpiresAt = now() + durationMs
        return true
      }),
    renewSubmissionLease: (sessionId, durationMs = QUICK_INGEST_SUBMISSION_LEASE_MS) =>
      updateLease(sessionId, (record) => {
        if (
          record.lifecycle !== "draft" ||
          record.submissionLeaseOwnerId !== ownerId ||
          (record.submissionLeaseExpiresAt || 0) <= now()
        ) {
          return false
        }
        record.submissionLeaseExpiresAt = now() + durationMs
        return true
      }),
    releaseSubmissionLease: async (sessionId) => {
      await updateLease(sessionId, (record) => {
        if (record.submissionLeaseOwnerId !== ownerId) return false
        delete record.submissionLeaseOwnerId
        delete record.submissionLeaseExpiresAt
        return true
      })
    },
  }
}
