type BrowserListener = (...args: unknown[]) => void
type BrowserTab = {
  id?: number
  title?: string
  url?: string
  favIconUrl?: string
  active?: boolean
  status?: string
}

const createEventTarget = () => {
  const listeners = new Set<BrowserListener>()
  return {
    addListener: (listener: BrowserListener) => listeners.add(listener),
    removeListener: (listener: BrowserListener) => listeners.delete(listener),
    hasListener: (listener: BrowserListener) => listeners.has(listener),
    trigger: (...args: unknown[]) =>
      listeners.forEach((listener) => listener(...args))
  }
}

const noopAsync = async (..._args: unknown[]) => undefined

const runtime = {
  id: undefined,
  getURL: (path: string) => {
    if (typeof window === "undefined") return path
    try {
      return new URL(path, window.location.origin).toString()
    } catch {
      return path
    }
  },
  getManifest: () => ({
    name: "tldw",
    version: "0.0.0"
  }),
  lastError: undefined,
  sendMessage: async (..._args: unknown[]) => undefined,
  sendNativeMessage: async (
    _host?: string,
    _message?: unknown
  ) => {
    throw new Error("Native messaging is not available in web mode.")
  },
  connect: (_info?: Record<string, unknown>) => ({
    postMessage: (_message?: unknown) => {},
    onMessage: createEventTarget(),
    onDisconnect: createEventTarget(),
    disconnect: () => {}
  }),
  openOptionsPage: () => {
    if (typeof window !== "undefined") {
      window.location.href = "/settings"
    }
  },
  onMessage: createEventTarget(),
  onConnect: createEventTarget()
}

const tabs = {
  query: async (
    _query?: Record<string, unknown>,
    callback?: (tabs: BrowserTab[]) => void
  ): Promise<BrowserTab[]> => {
    const result: BrowserTab[] = []
    callback?.(result)
    return result
  },
  create: async ({ url }: { url: string }) => {
    if (typeof window !== "undefined") {
      window.open(url, "_blank", "noopener,noreferrer")
    }
  },
  captureVisibleTab: async (
    _windowId?: number | null,
    _options?: Record<string, unknown>,
    callback?: (dataUrl: string | null) => void
  ) => {
    const result = null
    callback?.(result)
    return result
  }
}

const notifications = {
  create: async (_options?: Record<string, unknown>) => undefined
}

type StorageAreaName = "local" | "sync" | "session"

type StorageBackendLike = {
  getItem: (key: string) => string | null
  setItem: (key: string, value: string) => void
  removeItem: (key: string) => void
  key: (index: number) => string | null
  readonly length: number
}

// Per-area key prefixes. `local` stays UNPREFIXED so existing data
// (tldwConfig, tldw-api-host, ...) and the plasmo shim (which also writes
// `local` unprefixed and uses `plasmo-sync:` / `plasmo-session:` for the
// other areas) remain cross-compatible.
const SYNC_PREFIX = "plasmo-sync:"
const SESSION_PREFIX = "plasmo-session:"

const scopedKey = (areaName: StorageAreaName, key: string): string => {
  if (areaName === "sync") return `${SYNC_PREFIX}${key}`
  if (areaName === "session") return `${SESSION_PREFIX}${key}`
  return key
}

// Returns the logical (unprefixed) key if `storedKey` belongs to `areaName`,
// otherwise null. `local` explicitly excludes keys owned by the other areas.
const unscopedKey = (
  areaName: StorageAreaName,
  storedKey: string
): string | null => {
  if (areaName === "sync") {
    return storedKey.startsWith(SYNC_PREFIX)
      ? storedKey.slice(SYNC_PREFIX.length)
      : null
  }
  if (areaName === "session") {
    return storedKey.startsWith(SESSION_PREFIX)
      ? storedKey.slice(SESSION_PREFIX.length)
      : null
  }
  return storedKey.startsWith(SYNC_PREFIX) ||
    storedKey.startsWith(SESSION_PREFIX)
    ? null
    : storedKey
}

// `session` is memory-only (matches chrome.storage.session semantics): a
// module-level Map that is never persisted to localStorage/disk.
const sessionMemory = new Map<string, string>()
const sessionBackend: StorageBackendLike = {
  getItem: (key) => (sessionMemory.has(key) ? sessionMemory.get(key)! : null),
  setItem: (key, value) => {
    sessionMemory.set(key, value)
  },
  removeItem: (key) => {
    sessionMemory.delete(key)
  },
  key: (index) => Array.from(sessionMemory.keys())[index] ?? null,
  get length() {
    return sessionMemory.size
  }
}

const getLocalStorageBackend = (): StorageBackendLike | null => {
  if (typeof window !== "undefined" && window.localStorage) {
    return window.localStorage
  }
  return null
}

const getAreaBackend = (areaName: StorageAreaName): StorageBackendLike | null => {
  if (areaName === "session") return sessionBackend
  return getLocalStorageBackend()
}

const storageOnChanged = createEventTarget()

const parseStoredValue = (raw: string | null): unknown => {
  if (raw == null) return raw
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

const createStorageArea = (areaName: StorageAreaName) => ({
  get: (
    keys?: string | string[] | null,
    callback?: (items: Record<string, unknown>) => void
  ) => {
    const backend = getAreaBackend(areaName)
    const result: Record<string, unknown> = {}
    if (!backend) {
      callback?.(result)
      return Promise.resolve(result)
    }
    if (!keys) {
      // Enumerate only the keys belonging to this area.
      for (let i = 0; i < backend.length; i += 1) {
        const storedKey = backend.key(i)
        if (!storedKey) continue
        const logicalKey = unscopedKey(areaName, storedKey)
        if (logicalKey == null) continue
        result[logicalKey] = parseStoredValue(backend.getItem(storedKey))
      }
    } else {
      const keyList = Array.isArray(keys) ? keys : [keys]
      keyList.forEach((key) => {
        result[key] = parseStoredValue(
          backend.getItem(scopedKey(areaName, key))
        )
      })
    }
    callback?.(result)
    return Promise.resolve(result)
  },
  set: (items: Record<string, unknown>, callback?: () => void) => {
    const backend = getAreaBackend(areaName)
    const changes: Record<string, { oldValue?: unknown; newValue?: unknown }> =
      {}
    let writeError: Error | null = null
    if (backend) {
      for (const [key, value] of Object.entries(items)) {
        const storedKey = scopedKey(areaName, key)
        const oldRaw = backend.getItem(storedKey)
        let newRaw: string
        try {
          newRaw = JSON.stringify(value)
          backend.setItem(storedKey, newRaw)
        } catch (err) {
          // Quota exceeded, circular refs, etc. Do NOT record a change for
          // this key so no phantom onChanged fires, and surface the error.
          writeError = err instanceof Error ? err : new Error(String(err))
          break
        }
        // Only record the change after the write actually succeeded.
        if (oldRaw !== newRaw) {
          changes[key] = {
            oldValue: parseStoredValue(oldRaw),
            newValue: parseStoredValue(newRaw)
          }
        }
      }
    }
    if (Object.keys(changes).length > 0) {
      storageOnChanged.trigger(changes, areaName)
    }
    if (writeError) {
      // Propagate the failure instead of resolving as a success.
      return Promise.reject(writeError)
    }
    callback?.()
    return Promise.resolve()
  },
  remove: (keys: string | string[], callback?: () => void) => {
    const backend = getAreaBackend(areaName)
    const changes: Record<string, { oldValue?: unknown; newValue?: unknown }> =
      {}
    if (backend) {
      const keyList = Array.isArray(keys) ? keys : [keys]
      keyList.forEach((key) => {
        const storedKey = scopedKey(areaName, key)
        try {
          const oldRaw = backend.getItem(storedKey)
          if (oldRaw !== null) {
            changes[key] = {
              oldValue: parseStoredValue(oldRaw),
              newValue: undefined
            }
          }
          backend.removeItem(storedKey)
        } catch {
          // Silently ignore storage failures
        }
      })
    }
    if (Object.keys(changes).length > 0) {
      storageOnChanged.trigger(changes, areaName)
    }
    callback?.()
    return Promise.resolve()
  },
  clear: (callback?: () => void) => {
    const backend = getAreaBackend(areaName)
    const changes: Record<string, { oldValue?: unknown; newValue?: unknown }> =
      {}
    if (backend) {
      try {
        // Collect only THIS area's keys first, then remove them so we never
        // wipe the whole origin (H9) and don't mutate while indexing.
        const keysToRemove: string[] = []
        for (let i = 0; i < backend.length; i += 1) {
          const storedKey = backend.key(i)
          if (!storedKey) continue
          const logicalKey = unscopedKey(areaName, storedKey)
          if (logicalKey == null) continue
          changes[logicalKey] = {
            oldValue: parseStoredValue(backend.getItem(storedKey)),
            newValue: undefined
          }
          keysToRemove.push(storedKey)
        }
        keysToRemove.forEach((key) => backend.removeItem(key))
      } catch {
        // Silently ignore storage failures
      }
    }
    if (Object.keys(changes).length > 0) {
      storageOnChanged.trigger(changes, areaName)
    }
    callback?.()
    return Promise.resolve()
  }
})

const storage = {
  local: createStorageArea("local"),
  sync: createStorageArea("sync"),
  session: createStorageArea("session"),
  onChanged: storageOnChanged
}

const permissions = {
  request: async () => false
}

const i18n = {
  getMessage: () => ""
}

const action = {
  setTitle: noopAsync,
  setBadgeText: noopAsync,
  setBadgeBackgroundColor: noopAsync
}

const browserAction = action

const contextMenus = {
  create: noopAsync,
  remove: noopAsync,
  removeAll: noopAsync,
  onClicked: createEventTarget()
}

const commands = {
  onCommand: createEventTarget()
}

const alarms = {
  create: noopAsync,
  clear: noopAsync,
  onAlarm: createEventTarget()
}

type ScriptResult = { result?: unknown }

const scripting = {
  executeScript: async (
    _options?: Record<string, unknown>
  ): Promise<ScriptResult[]> => []
}

const tts = {
  speak: (_utterance: string, _options?: Record<string, unknown>) => {},
  stop: () => {},
  getVoices: async () => []
}

const extension = {
  inIncognitoContext: false
}

const sidePanel = {
  open: noopAsync,
  setOptions: noopAsync
}

export const browser = {
  runtime,
  tabs,
  notifications,
  storage,
  permissions,
  i18n,
  action,
  browserAction,
  contextMenus,
  commands,
  alarms,
  scripting,
  tts,
  extension,
  sidePanel
}

export type Browser = typeof browser
