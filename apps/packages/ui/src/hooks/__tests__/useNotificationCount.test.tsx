import { StrictMode } from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { notificationRecordKeyForConfig } from "@/services/notification-runtime-scope"

const mocks = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  watchers: new Map<string, Set<(change: { newValue: unknown }) => void>>(),
  get: vi.fn<(key: string) => Promise<unknown>>()
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {},
  createSafeStorage: () => ({
    get: mocks.get,
    watch(map: Record<string, (change: { newValue: unknown }) => void>) {
      for (const [key, callback] of Object.entries(map)) {
        const callbacks = mocks.watchers.get(key) ?? new Set()
        callbacks.add(callback)
        mocks.watchers.set(key, callbacks)
      }
    },
    unwatch(map: Record<string, (change: { newValue: unknown }) => void>) {
      for (const [key, callback] of Object.entries(map)) {
        mocks.watchers.get(key)?.delete(callback)
      }
    }
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (options: { key: string }) => [mocks.values.get(options.key)]
}))
import { useNotificationCount } from "../useNotificationCount"

const firstConfig = { serverUrl: "https://first.test", authMode: "multi-user", userId: "user-a", accessToken: "a" }
const secondConfig = { serverUrl: "https://second.test", authMode: "multi-user", userId: "user-b", accessToken: "b" }
const firstScope = notificationRecordKeyForConfig(firstConfig)!
const secondScope = notificationRecordKeyForConfig(secondConfig)!
const record = (unreadCount: number) => ({ state: "active", unreadCount, updatedAt: 123 })
const select = (config: typeof firstConfig) => {
  mocks.values.set("tldwConfig", config)
  mocks.values.set("tldw:notifications:activeScope", notificationRecordKeyForConfig(config))
}
const publish = (key: string, value: unknown) => {
  for (const callback of mocks.watchers.get(key) ?? []) callback({ newValue: value })
}
const deferred = () => {
  let resolve!: (value: unknown) => void
  const promise = new Promise<unknown>((done) => { resolve = done })
  return { promise, resolve }
}

describe("useNotificationCount", () => {
  beforeEach(() => {
    mocks.values.clear()
    mocks.watchers.clear()
    mocks.get.mockReset().mockImplementation(async (key) => mocks.values.get(key))
    select(firstConfig)
    mocks.values.set(firstScope, record(7))
    mocks.values.set(secondScope, record(9))
  })

  it("reads unread count from the active scoped lifecycle record", async () => {
    const { result } = renderHook(() => useNotificationCount())
    await waitFor(() => expect(result.current).toBe(7))
  })

  it.each([false, true])("recovers after a scope change without an unrelated rerender (StrictMode=%s)", async (strict) => {
    const observed: number[] = []
    const { result, rerender } = renderHook(() => {
      const count = useNotificationCount()
      observed.push(count)
      return count
    }, strict ? { wrapper: StrictMode } : undefined)
    await waitFor(() => expect(result.current).toBe(7))
    const switchIndex = observed.length
    select(secondConfig)
    rerender()
    expect(result.current).toBe(0)
    await waitFor(() => expect(result.current).toBe(9))
    expect(observed.slice(switchIndex)).not.toContain(7)
  })

  it("never displays the previous account while the new record read is pending", async () => {
    const next = deferred()
    const { result, rerender } = renderHook(() => useNotificationCount())
    await waitFor(() => expect(result.current).toBe(7))
    mocks.get.mockImplementation(() => next.promise)
    select(secondConfig)
    rerender()
    rerender()
    expect(result.current).toBe(0)
    await act(async () => next.resolve(record(9)))
    expect(result.current).toBe(9)
  })

  it("suppresses the old count while the active selector lags an account switch", async () => {
    const { result, rerender } = renderHook(() => useNotificationCount())
    await waitFor(() => expect(result.current).toBe(7))
    mocks.values.set("tldwConfig", secondConfig)
    rerender()
    expect(result.current).toBe(0)
  })

  it.each([null, {}, { ...firstConfig, accessToken: "" }])("returns zero for unresolved auth config %j", (config) => {
    mocks.values.set("tldwConfig", config)
    const { result } = renderHook(() => useNotificationCount())
    expect(result.current).toBe(0)
  })

  it("returns zero without an active scope", () => {
    mocks.values.delete("tldw:notifications:activeScope")
    const { result } = renderHook(() => useNotificationCount())
    expect(result.current).toBe(0)
  })

  it.each([undefined, null, [], { unreadCount: "bad" }])("returns zero for a missing or invalid record %j", async (value) => {
    const { result } = renderHook(() => useNotificationCount())
    await waitFor(() => expect(result.current).toBe(7))
    act(() => publish(firstScope, value))
    expect(result.current).toBe(0)
  })

  it("keeps the latest watch update when an older initial read resolves", async () => {
    const initial = deferred()
    mocks.get.mockReturnValue(initial.promise)
    const { result } = renderHook(() => useNotificationCount())
    act(() => publish(firstScope, record(11)))
    expect(result.current).toBe(11)
    await act(async () => initial.resolve(record(7)))
    expect(result.current).toBe(11)
  })

  it("ignores superseded reads during a rapid A to B to A switch", async () => {
    const oldA = deferred()
    const oldB = deferred()
    const newA = deferred()
    mocks.get.mockReturnValueOnce(oldA.promise).mockReturnValueOnce(oldB.promise).mockReturnValueOnce(newA.promise)
    const { result, rerender, unmount } = renderHook(() => useNotificationCount())
    select(secondConfig)
    rerender()
    select(firstConfig)
    rerender()
    await act(async () => newA.resolve(record(12)))
    await act(async () => { oldA.resolve(record(7)); oldB.resolve(record(9)) })
    expect(result.current).toBe(12)
    expect(mocks.watchers.get(secondScope)?.size).toBe(0)
    unmount()
    expect(mocks.watchers.get(firstScope)?.size).toBe(0)
  })

  it("handles a failed record read and can recover from a later watch update", async () => {
    mocks.get.mockRejectedValue(new Error("storage unavailable"))
    const { result } = renderHook(() => useNotificationCount())
    await act(async () => {})
    expect(result.current).toBe(0)
    act(() => publish(firstScope, record(4)))
    expect(result.current).toBe(4)
  })
})
