export type SerdeOptions = {
  serializer?: (value: unknown) => string
  deserializer?: (value: unknown) => unknown
}

export type StorageOptions = {
  area?: "local" | "sync" | "session"
  key?: string
  serde?: SerdeOptions
}

type StorageChange = {
  oldValue?: unknown
  newValue?: unknown
}

type WatchCallback = (change: StorageChange) => void

type StorageBackend = {
  getItem: (key: string) => string | null
  setItem: (key: string, value: string) => void
  removeItem: (key: string) => void
  clear: () => void
  key: (index: number) => string | null
  length: number
}

const createMemoryStorage = (): StorageBackend => {
  const map = new Map<string, string>()
  return {
    getItem: (key) => (map.has(key) ? map.get(key)! : null),
    setItem: (key, value) => {
      map.set(key, value)
    },
    removeItem: (key) => {
      map.delete(key)
    },
    clear: () => {
      map.clear()
    },
    key: (index) => Array.from(map.keys())[index] ?? null,
    get length() {
      return map.size
    }
  }
}

const getBackend = (): StorageBackend => {
  if (typeof window !== "undefined" && window.localStorage) {
    return window.localStorage
  }
  return createMemoryStorage()
}

const defaultSerde: Required<SerdeOptions> = {
  serializer: (value) => JSON.stringify(value),
  deserializer: (value) => {
    if (typeof value !== "string") return value
    try {
      return JSON.parse(value)
    } catch {
      return value
    }
  }
}

export class Storage {
  private backend: StorageBackend
  private serde: Required<SerdeOptions>
  private area: StorageOptions["area"]
  private watchers = new Map<string, Set<WatchCallback>>()

  constructor(options: StorageOptions = {}) {
    this.backend = getBackend()
    this.area = options.area || "local"
    this.serde = {
      ...defaultSerde,
      ...(options.serde || {})
    }
  }

  private storageKey(key: string): string {
    if (this.area === "local") return key
    return `plasmo-${this.area}:${key}`
  }

  private unscopedKey(key: string): string | null {
    if (this.area === "local") {
      return key.startsWith("plasmo-sync:") || key.startsWith("plasmo-session:")
        ? null
        : key
    }

    const prefix = `plasmo-${this.area}:`
    return key.startsWith(prefix) ? key.slice(prefix.length) : null
  }

  async get<T = unknown>(key: string): Promise<T | undefined> {
    const raw = this.backend.getItem(this.storageKey(key))
    if (raw == null) return undefined
    return this.serde.deserializer(raw) as T
  }

  async getAll(): Promise<Record<string, unknown>> {
    const entries: Record<string, unknown> = {}
    for (let i = 0; i < this.backend.length; i += 1) {
      const storedKey = this.backend.key(i)
      if (!storedKey) continue
      const key = this.unscopedKey(storedKey)
      if (!key) continue
      entries[key] = this.serde.deserializer(this.backend.getItem(storedKey))
    }
    return entries
  }

  async set<T = unknown>(key: string, value: T): Promise<void> {
    const prev = await this.get(key)
    this.backend.setItem(this.storageKey(key), this.serde.serializer(value))
    this.emitWatch(key, { oldValue: prev, newValue: value })
  }

  async remove(key: string): Promise<void> {
    const prev = await this.get(key)
    this.backend.removeItem(this.storageKey(key))
    this.emitWatch(key, { oldValue: prev, newValue: undefined })
  }

  async removeMany(keys: string[]): Promise<void> {
    await Promise.all(keys.map((key) => this.remove(key)))
  }

  async clear(): Promise<void> {
    const keys: string[] = []
    for (let i = 0; i < this.backend.length; i += 1) {
      const storedKey = this.backend.key(i)
      if (storedKey && this.unscopedKey(storedKey)) {
        keys.push(storedKey)
      }
    }
    keys.forEach((key) => this.backend.removeItem(key))
  }

  watch(map: Record<string, WatchCallback>): () => void {
    const entries = Object.entries(map)
    entries.forEach(([key, cb]) => {
      if (!this.watchers.has(key)) {
        this.watchers.set(key, new Set())
      }
      this.watchers.get(key)!.add(cb)
    })

    return () => {
      entries.forEach(([key, cb]) => {
        const set = this.watchers.get(key)
        if (!set) return
        set.delete(cb)
        if (set.size === 0) {
          this.watchers.delete(key)
        }
      })
    }
  }

  unwatch(map: Record<string, WatchCallback>): void {
    Object.entries(map).forEach(([key, cb]) => {
      const set = this.watchers.get(key)
      if (!set) return
      set.delete(cb)
      if (set.size === 0) {
        this.watchers.delete(key)
      }
    })
  }

  private emitWatch(key: string, change: StorageChange) {
    const callbacks = this.watchers.get(key)
    if (!callbacks) return
    callbacks.forEach((cb) => {
      try {
        cb(change)
      } catch {
        // ignore watcher errors
      }
    })
  }
}
