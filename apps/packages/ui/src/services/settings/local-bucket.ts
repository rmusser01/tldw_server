import { createSafeStorage, safeStorageSerde } from "@/utils/safe-storage"
import { requirePersistentStorage } from "@/utils/persistent-storage"

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

export type LocalRegistryListing<T> = { records: Record<string, LocalRegistryRecord<T>>; unavailable: boolean }

export type LocalRegistryBucket<T> = {
  isDurable: () => boolean
  /** Read candidates without pinning unseen records or deleting stale ones. */
  list: () => Promise<LocalRegistryListing<T>>
  /** Fail-closed reads require an unpinned durable bucket and never delete unreadable records. */
  get: (key: string, options?: { failClosed?: boolean }) => Promise<LocalRegistryRecord<T> | null>
  /** Expected null claims absence; conditional writes require the existing Web Lock. */
  set: (key: string, value: T, updatedAt?: number, expected?: LocalRegistryRecord<T> | null, validate_lease?: () => boolean) => Promise<boolean>
  remove: (key: string) => Promise<void>
  cleanup: () => Promise<number>
  buildKey: (key: string) => string
}

const storage = createSafeStorage({ area: "local" })
const hasDurableBackend = () => {
  try {
    requirePersistentStorage(storage)
    return true
  } catch {
    return false
  }
}


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

  const get = async (key: string, options?: { failClosed?: boolean }): Promise<LocalRegistryRecord<T> | null> => {
    const storageKey = buildKey(key)
    const failClosed = options?.failClosed === true
    try {
      if (failClosed && (tabScoped || !hasDurableBackend())) throw new Error("registry_durable_read_unavailable")
      const pinned = readTab(storageKey)
      let raw: unknown
      if (pinned != null) {
        raw = JSON.parse(pinned)
      } else {
        raw = await storage.get(storageKey)
        if (failClosed && !hasDurableBackend())
          throw new Error("registry_durable_read_unavailable")

        // A same-tab write may have completed while the durable read was pending.
        const current = readTab(storageKey)
        if (current != null) raw = JSON.parse(current)
        else pinTab(storageKey, JSON.stringify(raw ?? null))
      }
      if (raw === undefined || (!failClosed && raw === null)) return null
      const record = parseRecord<T>(raw)
      if (!record || (failClosed && !Number.isFinite(new Date(record.updatedAt).getTime()))) {
        if (failClosed) throw new Error("registry_record_invalid")
        await remove(key, JSON.stringify(raw))
        return null
      }
      if (isStale(record.updatedAt, Date.now(), ttlMs)) {
        if (failClosed) throw new Error("registry_record_expired")
        await remove(key, JSON.stringify(raw))
        return null
      }
      return record
    } catch (error) {
      if (failClosed) throw error
      return null
    }
  }

  const set = async (key: string, value: T, updatedAt = Date.now(), expected?: LocalRegistryRecord<T> | null, validate_lease?: () => boolean): Promise<boolean> => {
    const storageKey = buildKey(key)
    try {
      const conditional = expected !== undefined
      const previous = conditional ? JSON.stringify(expected) : undefined
      if (validate_lease?.() === false || (conditional && !hasDurableBackend()) || (expected === null && tabScoped)) return false
      if (conditional && tabScoped && readTab(storageKey) !== previous) return false
      const record = { value, updatedAt }
      pinTab(storageKey, JSON.stringify(record))
      // A conditional read/write must be serialized across physical contexts.
      // Without Web Locks, retire only this exact pin; keep the durable source.
      if (conditional && (typeof navigator === "undefined" || !navigator.locks?.request)) return false
      return await withRecordLock(storageKey, async () => {
        if (validate_lease?.() === false || (conditional && !hasDurableBackend())) return false
        if (conditional) {
          const current = await storage.get(storageKey)
          if (validate_lease?.() === false || !hasDurableBackend()) return false
          if (expected === null ? current !== undefined : JSON.stringify(current ?? null) !== previous) return false
        }
        await storage.set(storageKey, record)
        return hasDurableBackend() && validate_lease?.() !== false
      })
    } catch {
      return false
    }
  }

  const list = async (): Promise<LocalRegistryListing<T>> => {
    let entries: Record<string, unknown> = {}
    let unavailable = !hasDurableBackend()
    try { entries = { ...await storage.getAll() } } catch { unavailable = true }
    if (tabScoped) {
      const pinned = new Map(tabRecords)
      try {
        for (let i = 0; tabStorage && i < tabStorage.length; i++) {
          const key = tabStorage.key(i)
          if (key?.startsWith(prefix)) {
            const value = tabStorage.getItem(key)
            if (value !== null) pinned.set(key, value)
          }
        }
      } catch { unavailable = true }
      // Overlay after the asynchronous read, including empty tombstones and
      // quota-fallback pins from this exact mounted bucket.
      for (const [key, value] of pinned) {
        try { entries[key] = JSON.parse(value) } catch { entries[key] = null }
      }
    }
    const records: LocalRegistryListing<T>["records"] = {}
    const now = Date.now()
    for (const [key, value] of Object.entries(entries)) {
      if (!key.startsWith(prefix)) continue
      const record = parseRecord<T>(safeStorageSerde.deserializer(value))
      if (record && Number.isFinite(new Date(record.updatedAt).getTime()) && !isStale(record.updatedAt, now, ttlMs)) records[key.slice(prefix.length)] = record
    }
    return { records, unavailable }
  }

  const cleanup = async (): Promise<number> => {
    try {
      const entries = await storage.getAll()
      const now = Date.now()
      const keysToRemove = Object.entries(entries)
        .filter(([key]) => key.startsWith(prefix))
        .filter(([, value]) => {
          const record = parseRecord<T>(safeStorageSerde.deserializer(value))
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
    isDurable: hasDurableBackend,
    list,
    get,
    set,
    remove,
    cleanup,
    buildKey
  }
}
