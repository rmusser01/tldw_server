import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { buildNotificationScopeKey } from "@/services/notification-lifecycle"
import type { NotificationStreamEvent } from "@/services/notifications"

const subscribeNotificationsStreamMock = vi.fn()
const getUnreadCountMock = vi.fn()
const notifyMock = vi.fn()

type StorageChange = { oldValue?: unknown; newValue?: unknown }
type StorageWatcher = (change: StorageChange) => void
type MockStorage = {
  get: ReturnType<typeof vi.fn>
  set: ReturnType<typeof vi.fn>
  watch: ReturnType<typeof vi.fn>
  unwatch: ReturnType<typeof vi.fn>
}

const ACTIVE_SCOPE_KEY = "tldw:notifications:activeScope"

let storageState = new Map<string, unknown>()
let storageMock: MockStorage
let watchers = new Map<string, Set<StorageWatcher>>()
let operationOrder: string[] = []

vi.mock("@/services/notifications", () => ({
  subscribeNotificationsStream: (...args: unknown[]) => subscribeNotificationsStreamMock(...args),
  getUnreadCount: (...args: unknown[]) => getUnreadCountMock(...args)
}))

vi.mock("@/services/background-helpers", () => ({
  notify: (...args: unknown[]) => notifyMock(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => storageMock
}))

import * as notificationSubscription from "@/entries/shared/notification-subscription"

const multiUserConfig = (userId: string, token = `token-${userId}`) => ({
  serverUrl: "https://api.example.test",
  authMode: "multi-user",
  orgId: "org-1",
  userId,
  accessToken: token
})

const recordKeyFor = (config: ReturnType<typeof multiUserConfig>) =>
  `tldw:${buildNotificationScopeKey(config)}`

const flushAsync = async () => {
  for (let index = 0; index < 12; index += 1) {
    await Promise.resolve()
  }
}

describe("notification subscription", () => {
  beforeEach(() => {
    storageState = new Map()
    watchers = new Map()
    operationOrder = []
    storageMock = {
      get: vi.fn(async (key: string) => storageState.get(key)),
      set: vi.fn(async (key: string, value: unknown) => {
        operationOrder.push(`set:${key}:${String(value)}`)
        const oldValue = storageState.get(key)
        storageState.set(key, value)
        for (const watcher of watchers.get(key) ?? []) {
          watcher({ oldValue, newValue: value })
        }
      }),
      watch: vi.fn((entries: Record<string, StorageWatcher>) => {
        for (const [key, watcher] of Object.entries(entries)) {
          const callbacks = watchers.get(key) ?? new Set<StorageWatcher>()
          callbacks.add(watcher)
          watchers.set(key, callbacks)
        }
        return () => {
          for (const [key, watcher] of Object.entries(entries)) {
            watchers.get(key)?.delete(watcher)
          }
        }
      }),
      unwatch: vi.fn((entries: Record<string, StorageWatcher>) => {
        for (const [key, watcher] of Object.entries(entries)) {
          watchers.get(key)?.delete(watcher)
        }
      })
    }

    subscribeNotificationsStreamMock.mockReset()
    getUnreadCountMock.mockReset()
    notifyMock.mockReset()
    getUnreadCountMock.mockResolvedValue({ unread_count: 4 })
  })

  afterEach(() => {
    notificationSubscription.stopNotificationSubscription()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("persists lifecycle and unread count under a server/principal-scoped key", async () => {
    let onOpen: (() => void) | undefined
    subscribeNotificationsStreamMock.mockImplementation((options: { onOpen?: () => void }) => {
      onOpen = options.onOpen
      return () => operationOrder.push("unsubscribe")
    })
    const config = multiUserConfig("user-a")
    const recordKey = recordKeyFor(config)

    await notificationSubscription.startNotificationSubscription(config)

    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBe(recordKey)
    expect(recordKey).not.toContain(config.accessToken)
    expect(storageState.get(recordKey)).toEqual({
      state: "connecting",
      unreadCount: 4,
      updatedAt: expect.any(Number)
    })
    expect(storageState.has("tldw:notifications:unreadCount")).toBe(false)

    onOpen?.()
    await flushAsync()

    expect(storageState.get(recordKey)).toEqual({
      state: "active",
      unreadCount: 4,
      updatedAt: expect.any(Number)
    })
  })

  it("prevents a stale initial fetch from writing or subscribing after a scope switch", async () => {
    let resolveFirstFetch: ((value: { unread_count: number }) => void) | undefined
    getUnreadCountMock
      .mockImplementationOnce(
        () =>
          new Promise<{ unread_count: number }>((resolve) => {
            resolveFirstFetch = resolve
          })
      )
      .mockResolvedValueOnce({ unread_count: 9 })
    subscribeNotificationsStreamMock.mockReturnValue(vi.fn())
    const firstConfig = multiUserConfig("user-a")
    const secondConfig = multiUserConfig("user-b")

    const firstStart = notificationSubscription.startNotificationSubscription(firstConfig)
    await flushAsync()

    for (const watcher of watchers.get("tldwConfig") ?? []) {
      watcher({ oldValue: firstConfig, newValue: secondConfig })
    }
    await flushAsync()

    resolveFirstFetch?.({ unread_count: 77 })
    await firstStart
    await flushAsync()

    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBe(recordKeyFor(secondConfig))
    expect(storageState.get(recordKeyFor(secondConfig))).toEqual({
      state: "connecting",
      unreadCount: 9,
      updatedAt: expect.any(Number)
    })
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)
  })

  it("aborts and clears the rendered selector before starting a switched principal scope", async () => {
    subscribeNotificationsStreamMock.mockImplementation(() => {
      operationOrder.push("subscribe")
      return () => operationOrder.push("unsubscribe")
    })
    const firstConfig = multiUserConfig("user-a")
    const secondConfig = multiUserConfig("user-b")

    await notificationSubscription.startNotificationSubscription(firstConfig)
    operationOrder = []

    for (const watcher of watchers.get("tldwConfig") ?? []) {
      watcher({ oldValue: firstConfig, newValue: secondConfig })
    }
    await flushAsync()

    const unsubscribeIndex = operationOrder.indexOf("unsubscribe")
    const clearIndex = operationOrder.findIndex((entry) =>
      entry.startsWith(`set:${ACTIVE_SCOPE_KEY}:`)
    )
    const subscribeIndex = operationOrder.indexOf("subscribe")

    expect(unsubscribeIndex).toBeGreaterThanOrEqual(0)
    expect(clearIndex).toBeGreaterThan(unsubscribeIndex)
    expect(subscribeIndex).toBeGreaterThan(clearIndex)
    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBe(recordKeyFor(secondConfig))
  })

  it("does not mark the lifecycle active before the stream onOpen callback", async () => {
    let onOpen: (() => void) | undefined
    subscribeNotificationsStreamMock.mockImplementation((options: { onOpen?: () => void }) => {
      onOpen = options.onOpen
      return vi.fn()
    })
    const config = multiUserConfig("user-a")
    const recordKey = recordKeyFor(config)

    await notificationSubscription.startNotificationSubscription(config)

    expect((storageState.get(recordKey) as { state: string }).state).toBe("connecting")
    onOpen?.()
    await flushAsync()
    expect((storageState.get(recordKey) as { state: string }).state).toBe("active")
  })

  it("stops on 401 and restarts only after a successful auth config change", async () => {
    vi.useFakeTimers()
    let onError: ((error: unknown) => void) | undefined
    subscribeNotificationsStreamMock.mockImplementation((options: { onError?: (error: unknown) => void }) => {
      onError = options.onError
      return () => operationOrder.push("unsubscribe")
    })
    const firstConfig = multiUserConfig("user-a", "expired-token")
    const refreshedConfig = multiUserConfig("user-a", "fresh-token")

    await notificationSubscription.startNotificationSubscription(firstConfig)
    onError?.({ status: 401 })
    await flushAsync()

    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)
    expect((storageState.get(recordKeyFor(firstConfig)) as { state: string }).state).toBe(
      "auth-required"
    )

    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)

    for (const watcher of watchers.get("tldwConfig") ?? []) {
      watcher({ oldValue: firstConfig, newValue: refreshedConfig })
    }
    await flushAsync()

    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(2)
  })

  it("aborts and synchronously clears the active scope on logout without restarting", async () => {
    subscribeNotificationsStreamMock.mockImplementation(() => {
      operationOrder.push("subscribe")
      return () => operationOrder.push("unsubscribe")
    })
    const config = multiUserConfig("user-a")

    await notificationSubscription.startNotificationSubscription(config)
    operationOrder = []

    for (const watcher of watchers.get("tldwConfig") ?? []) {
      watcher({ oldValue: config, newValue: { ...config, accessToken: "" } })
    }

    expect(operationOrder[0]).toBe("unsubscribe")
    expect(operationOrder[1]).toBe(`set:${ACTIVE_SCOPE_KEY}:null`)
    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBeNull()
    await flushAsync()
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)
  })

  it("clears a persisted selector when startup finds no authenticated config", async () => {
    storageState.set(ACTIVE_SCOPE_KEY, "tldw:notifications:stale-scope")

    await notificationSubscription.startNotificationSubscription(null)

    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBeNull()
    expect(subscribeNotificationsStreamMock).not.toHaveBeenCalled()
  })

  it("stops on 403 until an explicit retry and never passively polls", async () => {
    vi.useFakeTimers()
    let onError: ((error: unknown) => void) | undefined
    subscribeNotificationsStreamMock.mockImplementation((options: { onError?: (error: unknown) => void }) => {
      onError = options.onError
      return vi.fn()
    })
    const config = multiUserConfig("restricted-user")

    await notificationSubscription.startNotificationSubscription(config)
    onError?.({ status: 403 })
    await flushAsync()

    expect((storageState.get(recordKeyFor(config)) as { state: string }).state).toBe(
      "unavailable"
    )

    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(getUnreadCountMock).toHaveBeenCalledTimes(1)
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)

    expect(notificationSubscription.retryNotificationSubscription).toBeTypeOf("function")
    await notificationSubscription.retryNotificationSubscription()

    expect(getUnreadCountMock).toHaveBeenCalledTimes(2)
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(2)
  })

  it("classifies a terminal initial count failure without starting or polling the stream", async () => {
    vi.useFakeTimers()
    getUnreadCountMock.mockRejectedValue({ status: 401 })
    const config = multiUserConfig("expired-user")

    await notificationSubscription.startNotificationSubscription(config)

    expect((storageState.get(recordKeyFor(config)) as { state: string }).state).toBe(
      "auth-required"
    )
    expect(subscribeNotificationsStreamMock).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(getUnreadCountMock).toHaveBeenCalledTimes(1)
    expect(subscribeNotificationsStreamMock).not.toHaveBeenCalled()
  })

  it("aborts a stream that reports a terminal error during subscription startup", async () => {
    const stop = vi.fn()
    subscribeNotificationsStreamMock.mockImplementation(
      (options: { onError?: (error: unknown) => void }) => {
        options.onError?.({ status: 403 })
        return stop
      }
    )
    const config = multiUserConfig("restricted-user")

    await notificationSubscription.startNotificationSubscription(config)

    expect(stop).toHaveBeenCalledTimes(1)
    expect((storageState.get(recordKeyFor(config)) as { state: string }).state).toBe(
      "unavailable"
    )
  })

  it("shows transient stream failures as degraded and recovers only on onOpen", async () => {
    let onOpen: (() => void) | undefined
    let onError: ((error: unknown) => void) | undefined
    subscribeNotificationsStreamMock.mockImplementation(
      (options: { onOpen?: () => void; onError?: (error: unknown) => void }) => {
        onOpen = options.onOpen
        onError = options.onError
        return vi.fn()
      }
    )
    const config = multiUserConfig("user-a")
    const recordKey = recordKeyFor(config)

    await notificationSubscription.startNotificationSubscription(config)
    onOpen?.()
    await flushAsync()
    onError?.({ status: 503 })
    await flushAsync()

    expect((storageState.get(recordKey) as { state: string }).state).toBe("degraded")
    expect(subscribeNotificationsStreamMock).toHaveBeenCalledTimes(1)

    onOpen?.()
    await flushAsync()
    expect((storageState.get(recordKey) as { state: string }).state).toBe("active")
  })

  it("increments the active scoped unread count without losing concurrent events", async () => {
    let onEvent: ((event: NotificationStreamEvent) => Promise<void> | void) | null = null
    subscribeNotificationsStreamMock.mockImplementation((options: { onEvent: typeof onEvent }) => {
      onEvent = options.onEvent
      return vi.fn()
    })
    getUnreadCountMock.mockResolvedValue({ unread_count: 0 })
    const config = multiUserConfig("user-a")

    await notificationSubscription.startNotificationSubscription(config)
    await Promise.all([
      onEvent?.({ event: "notification", payload: { title: "First", message: "One" } }),
      onEvent?.({ event: "notification", payload: { title: "Second", message: "Two" } })
    ])

    expect((storageState.get(recordKeyFor(config)) as { unreadCount: number }).unreadCount).toBe(2)
    expect(notifyMock).toHaveBeenCalledTimes(2)
  })

  it("keeps idle internal-only when the active subscription stops", async () => {
    subscribeNotificationsStreamMock.mockReturnValue(vi.fn())
    const config = multiUserConfig("user-a")
    const recordKey = recordKeyFor(config)

    await notificationSubscription.startNotificationSubscription(config)
    notificationSubscription.stopNotificationSubscription()
    await flushAsync()

    expect(storageState.get(ACTIVE_SCOPE_KEY)).toBeNull()
    expect((storageState.get(recordKey) as { state: string }).state).not.toBe("idle")
  })
})
