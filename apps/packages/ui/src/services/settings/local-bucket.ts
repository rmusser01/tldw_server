import { createSafeStorage } from "@/utils/safe-storage"

export type LocalRegistryRecord<T> = {
  value: T
  updatedAt: number
}

type LocalRegistryBucketOptions = {
  prefix: string
  ttlMs?: number
  /** Pin each tab's record, retaining local storage as last-record recovery. */
  tabScoped?: boolean
}

export type LocalRegistryBucket<T> = {
  get: (key: string) => Promise<LocalRegistryRecord<T> | null>
  set: (key: string, value: T, updatedAt?: number) => Promise<void>
  remove: (key: string) => Promise<void>
  cleanup: () => Promise<number>
  buildKey: (key: string) => string
}

const storage = createSafeStorage({ area: "local" })

const withRecordLock = async <T>(key: string, operation: () => Promise<T>): Promise<T> => {
  if (typeof navigator === "undefined" || !navigator.locks?.request) return operation()
  return navigator.locks.request(`tldw:local-registry:${key}`, operation)
}

const parseRecord = <T>(raw: unknown): LocalRegistryRecord<T> | null => {
  if (!raw || typeof raw !== "object") return null
  if (!("value" in raw) || !("updatedAt" in raw)) return null
  const record = raw as LocalRegistryRecord<T>
  if (!Number.isFinite(record.updatedAt)) return null
  return record
}

const isStale = (updatedAt: number, now: number, ttlMs?: number) => {
  if (!ttlMs) return false
  return now - updatedAt > ttlMs
}

export const createLocalRegistryBucket = <T>({
  prefix,
  ttlMs,
  tabScoped = false
}: LocalRegistryBucketOptions): LocalRegistryBucket<T> => {
  const buildKey = (key: string) => `${prefix}${key}`
  let tabStorage: Storage | undefined
  try {
    if (tabScoped && typeof window !== "undefined") tabStorage = window.sessionStorage
  } catch {
    // Retain durable recovery when the browser disables session storage.
  }
  const tabRecords = new Map<string, string>()
  const inTab = <R>(key: string, operation: (area: Storage) => R): R | undefined => {
    try {
      return tabStorage ? operation(tabStorage) : undefined
    } catch {
      // Quota/privacy failures must not prevent the existing durable operation.
      // Remove an older pin so the next reload can recover the newer durable save.
      try { tabStorage?.removeItem(key) } catch { /* Storage may be entirely blocked. */ }
      tabStorage = undefined
      return undefined
    }
  }
  const readTab = (key: string) => tabScoped
    ? inTab(key, area => area.getItem(key)) ?? tabRecords.get(key)
    : undefined
  const pinTab = (key: string, value: string) => {
    if (!tabScoped) return
    // Keep ownership through a session-storage failure in this active tab.
    tabRecords.set(key, value)
    inTab(key, area => area.setItem(key, value))
  }

  const remove = async (key: string, expected?: string): Promise<void> => {
    const storageKey = buildKey(key)
    try {
      const previous = expected ?? readTab(storageKey)
      // An explicit empty record must not fall back to another tab's later save.
      if (expected === undefined || readTab(storageKey) === expected) pinTab(storageKey, "null")
      await withRecordLock(storageKey, async () => {
        if (tabScoped || expected !== undefined) {
          if (previous == null || previous === "null") return
          const durable = await storage.get(storageKey)
          if (JSON.stringify(durable ?? null) !== previous) return
        }
        await storage.remove(storageKey)
      })
    } catch {
      // ignore storage errors
    }
  }

  const get = async (key: string): Promise<LocalRegistryRecord<T> | null> => {
    const storageKey = buildKey(key)
    try {
      const pinned = readTab(storageKey)
      let raw: unknown
      if (pinned != null) {
        raw = JSON.parse(pinned)
      } else {
        raw = await storage.get(storageKey)
        // A same-tab write may have completed while the durable read was pending.
        const current = readTab(storageKey)
        if (current != null) raw = JSON.parse(current)
        else pinTab(storageKey, JSON.stringify(raw ?? null))
      }
      if (raw == null) return null
      const record = parseRecord<T>(raw)
      if (!record) {
        await remove(key, JSON.stringify(raw))
        return null
      }
      if (isStale(record.updatedAt, Date.now(), ttlMs)) {
        await remove(key, JSON.stringify(raw))
        return null
      }
      return record
    } catch {
      return null
    }
  }

  const set = async (key: string, value: T, updatedAt = Date.now()): Promise<void> => {
    const storageKey = buildKey(key)
    try {
      const record = { value, updatedAt }
      pinTab(storageKey, JSON.stringify(record))
      await withRecordLock(storageKey, () => storage.set(storageKey, record))
    } catch {
      // ignore storage errors
    }
  }

  const cleanup = async (): Promise<number> => {
    try {
      const entries = await storage.getAll()
      const now = Date.now()
      const keysToRemove = Object.entries(entries)
        .filter(([key]) => key.startsWith(prefix))
        .filter(([, value]) => {
          const record = parseRecord<T>(value)
          return !record || isStale(record.updatedAt, now, ttlMs)
        })
        .map(([key]) => key)

      let removed = 0
      for (const key of keysToRemove) {
        await withRecordLock(key, async () => {
          // Another tab may have refreshed this record after getAll().
          const current = await storage.get(key)
          if (current == null) return
          const record = parseRecord<T>(current)
          if (!record || isStale(record.updatedAt, Date.now(), ttlMs)) {
            await storage.remove(key)
            removed++
          }
        })
      }
      return removed
    } catch {
      return 0
    }
  }

  return {
    get,
    set,
    remove,
    cleanup,
    buildKey
  }
}
