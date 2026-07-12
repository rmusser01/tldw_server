/** Background notification subscription for the browser extension. */

import { notify } from "@/services/background-helpers"
import {
  buildNotificationScopeKey,
  classifyNotificationError,
  reduceNotificationLifecycle,
  type NotificationLifecycleAction,
  type NotificationLifecycleState,
  type NotificationScopeInput
} from "@/services/notification-lifecycle"
import { getUnreadCount, subscribeNotificationsStream } from "@/services/notifications"
import type { NotificationStreamEvent } from "@/services/notifications"
import { createSafeStorage } from "@/utils/safe-storage"
import { toUnreadCount } from "@/utils/notifications"

const CONFIG_KEY = "tldwConfig"
const ACTIVE_SCOPE_KEY = "tldw:notifications:activeScope"

type NotificationConfig = NotificationScopeInput
type ExposedNotificationState = Exclude<NotificationLifecycleState, "idle">
type NotificationRecord = {
  state: ExposedNotificationState
  unreadCount: number
  updatedAt: number
}
type StorageChange = { oldValue?: unknown; newValue?: unknown }
type SafeStorage = ReturnType<typeof createSafeStorage>

let storageInstance: SafeStorage | null = null
let watchedStorage: SafeStorage | null = null
let unsubscribe: (() => void) | null = null
let currentConfig: NotificationConfig | null = null
let currentScopeKey: string | null = null
let generation = 0
let activeScopeWrite: Promise<void> | null = null

const getStorage = (): SafeStorage => {
  storageInstance ??= createSafeStorage({ area: "local" })
  return storageInstance
}

const asConfig = (value: unknown): NotificationConfig | null => {
  if (!value || typeof value !== "object") return null
  const config = value as Record<string, unknown>
  const serverUrl = String(config.serverUrl || "").trim()
  const authMode = String(config.authMode || "").trim().toLowerCase()
  if (!serverUrl || (authMode !== "single-user" && authMode !== "multi-user")) return null

  const accessToken = String(config.accessToken || "").trim()
  const apiKey = String(config.apiKey || "").trim()
  if (authMode === "multi-user" ? !accessToken : !apiKey) return null

  return {
    serverUrl,
    authMode,
    orgId:
      typeof config.orgId === "string" || typeof config.orgId === "number"
        ? config.orgId
        : null,
    userId:
      typeof config.userId === "string" || typeof config.userId === "number"
        ? config.userId
        : null,
    accessToken,
    apiKey
  }
}

const configIdentity = (config: NotificationConfig | null): string =>
  JSON.stringify([
    config?.serverUrl ?? null,
    config?.authMode ?? null,
    config?.orgId ?? null,
    config?.userId ?? null,
    config?.accessToken ?? null,
    config?.apiKey ?? null
  ])

const recordKeyFor = (config: NotificationConfig): string =>
  `tldw:${buildNotificationScopeKey(config)}`

const readRecord = (value: unknown): NotificationRecord | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Partial<NotificationRecord>
  if (
    record.state !== "connecting" &&
    record.state !== "active" &&
    record.state !== "degraded" &&
    record.state !== "auth-required" &&
    record.state !== "unavailable"
  ) {
    return null
  }
  return {
    state: record.state,
    unreadCount: toUnreadCount(record.unreadCount),
    updatedAt: typeof record.updatedAt === "number" ? record.updatedAt : 0
  }
}

const stopStream = (): void => {
  const stop = unsubscribe
  unsubscribe = null
  stop?.()
}

const writeActiveScope = (storage: SafeStorage, scopeKey: string | null): Promise<void> => {
  const previousWrite = activeScopeWrite
  const write = previousWrite
    ? previousWrite.catch(() => undefined).then(() => storage.set(ACTIVE_SCOPE_KEY, scopeKey))
    : storage.set(ACTIVE_SCOPE_KEY, scopeKey)
  activeScopeWrite = write
  void write.then(
    () => {
      if (activeScopeWrite === write) activeScopeWrite = null
    },
    () => {
      if (activeScopeWrite === write) activeScopeWrite = null
    }
  )
  return write
}

const clearActiveScope = (storage: SafeStorage): Promise<void> => {
  currentScopeKey = null
  return writeActiveScope(storage, null)
}

const transitionToConfig = async (
  value: unknown,
  options: { force?: boolean } = {}
): Promise<void> => {
  const storage = getStorage()
  const nextConfig = asConfig(value)
  if (
    !options.force &&
    nextConfig &&
    currentConfig &&
    configIdentity(nextConfig) === configIdentity(currentConfig)
  ) {
    return
  }

  const activeGeneration = ++generation
  stopStream()
  currentConfig = nextConfig
  const cleared = clearActiveScope(storage)
  if (!nextConfig) {
    await cleared
    return
  }

  await cleared
  if (activeGeneration !== generation) return

  const scopeKey = recordKeyFor(nextConfig)
  currentScopeKey = scopeKey
  let lifecycleState: NotificationLifecycleState = "idle"
  let record: NotificationRecord = {
    state: "connecting",
    unreadCount: toUnreadCount(
      readRecord(await storage.get<NotificationRecord>(scopeKey))?.unreadCount
    ),
    updatedAt: Date.now()
  }
  if (activeGeneration !== generation) return

  const isCurrent = (): boolean =>
    activeGeneration === generation && currentScopeKey === scopeKey

  const writeRecord = async (
    action: NotificationLifecycleAction | null,
    unreadCount = record.unreadCount
  ): Promise<void> => {
    if (!isCurrent()) return
    if (action) lifecycleState = reduceNotificationLifecycle(lifecycleState, action)
    if (lifecycleState === "idle") return
    record = {
      state: lifecycleState,
      unreadCount: toUnreadCount(unreadCount),
      updatedAt: Date.now()
    }
    await storage.set(scopeKey, record)
  }

  await writeActiveScope(storage, scopeKey)
  await writeRecord({ type: "start" })
  if (!isCurrent()) return

  try {
    const { unread_count } = await getUnreadCount()
    await writeRecord({ type: "start" }, unread_count)
  } catch (error) {
    if (!isCurrent()) return
    const classification = classifyNotificationError(error)
    if (classification.kind === "auth-required") {
      await writeRecord({ type: "auth-required" })
      return
    }
    if (classification.kind === "unavailable") {
      await writeRecord({ type: "unavailable" })
      return
    }
    if (classification.kind === "idle") return
    await writeRecord({ type: "retry" })
  }
  if (!isCurrent()) return

  let terminal = false
  let unreadCountWrite = Promise.resolve()
  const handleTerminalError = async (error: unknown): Promise<void> => {
    if (!isCurrent()) return
    const classification = classifyNotificationError(error)
    if (classification.kind === "retry") {
      await writeRecord({ type: "retry" })
      return
    }
    if (classification.kind === "auth-required") {
      terminal = true
      stopStream()
      await writeRecord({ type: "auth-required" })
      return
    }
    if (classification.kind === "unavailable") {
      terminal = true
      stopStream()
      await writeRecord({ type: "unavailable" })
    }
  }

  try {
    const stop = subscribeNotificationsStream({
      onOpen: () => {
        if (!terminal) void writeRecord({ type: "open" })
      },
      onError: (error) => {
        void handleTerminalError(error)
      },
      onEvent: async (event: NotificationStreamEvent) => {
        if (!isCurrent() || terminal) return
        if (event.event === "notification") {
          const payload = event.payload as {
            title?: string
            message?: string
          } | null
          if (payload?.title) notify(payload.title, payload.message || "")

          unreadCountWrite = unreadCountWrite
            .catch(() => undefined)
            .then(() => writeRecord(null, record.unreadCount + 1))
            .catch((error) => {
              console.debug(
                "[background] Failed to update unread count from notification event:",
                error
              )
            })
          await unreadCountWrite
        }

        if (event.event === "notifications_coalesced") {
          try {
            const { unread_count } = await getUnreadCount()
            await writeRecord(null, unread_count)
          } catch (error) {
            await handleTerminalError(error)
          }
        }
      }
    })
    if (terminal || !isCurrent()) {
      stop()
      return
    }
    unsubscribe = stop
  } catch (error) {
    await handleTerminalError(error)
  }
}

const handleConfigChange = (change: StorageChange): void => {
  void transitionToConfig(change?.newValue)
}

const ensureConfigWatcher = (storage: SafeStorage): void => {
  if (watchedStorage) return
  watchedStorage = storage
  storage.watch({ tldwConfig: handleConfigChange })
}

/** Start listening for notifications for the current authenticated scope. */
export async function startNotificationSubscription(config?: unknown): Promise<void> {
  const storage = getStorage()
  ensureConfigWatcher(storage)
  if (config === undefined) {
    const readGeneration = generation
    const resolvedConfig = await storage.get(CONFIG_KEY)
    if (readGeneration !== generation) return
    await transitionToConfig(resolvedConfig)
    return
  }
  const resolvedConfig = config
  await transitionToConfig(resolvedConfig)
}

/** Explicitly retry a stopped terminal subscription. */
export async function retryNotificationSubscription(): Promise<void> {
  if (!currentConfig) return
  await transitionToConfig(currentConfig, { force: true })
}

/** Stop listening and synchronously clear the rendered scope selector. */
export function stopNotificationSubscription(): void {
  generation += 1
  stopStream()
  currentConfig = null
  const storage = storageInstance ?? createSafeStorage({ area: "local" })
  void clearActiveScope(storage)
  if (watchedStorage) {
    watchedStorage.unwatch({ tldwConfig: handleConfigChange })
  }
  watchedStorage = null
  storageInstance = null
}

/** Read the unread count for the currently rendered scope. */
export async function getStoredUnreadCount(): Promise<number> {
  const storage = getStorage()
  const scopeKey = await storage.get<string | null>(ACTIVE_SCOPE_KEY)
  if (!scopeKey) return 0
  return toUnreadCount(readRecord(await storage.get(scopeKey))?.unreadCount)
}

/** Reset the unread count for the currently rendered scope. */
export async function resetStoredUnreadCount(): Promise<void> {
  const storage = getStorage()
  const scopeKey = await storage.get<string | null>(ACTIVE_SCOPE_KEY)
  if (!scopeKey) return
  const record = readRecord(await storage.get(scopeKey))
  if (!record) return
  await storage.set(scopeKey, { ...record, unreadCount: 0, updatedAt: Date.now() })
}
