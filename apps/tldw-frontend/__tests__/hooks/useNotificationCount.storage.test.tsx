import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { createSafeStorage } from "@/utils/safe-storage"
import { notificationRecordKeyForConfig } from "@/services/notification-runtime-scope"
import { useNotificationCount } from "@/hooks/useNotificationCount"

const storage = createSafeStorage({ area: "local" })
const firstConfig = { serverUrl: "https://first.test", authMode: "multi-user", userId: "user-a", accessToken: "a" }
const secondConfig = { serverUrl: "https://second.test", authMode: "multi-user", userId: "user-b", accessToken: "b" }
const firstScope = notificationRecordKeyForConfig(firstConfig)!
const secondScope = notificationRecordKeyForConfig(secondConfig)!

beforeEach(async () => {
  await storage.set("tldwConfig", firstConfig)
  await storage.set("tldw:notifications:activeScope", firstScope)
  await storage.set(firstScope, { unreadCount: 7 })
  await storage.set(secondScope, { unreadCount: 9 })
})

it("follows real storage subscriptions through account switches and record removal", async () => {
  const observed: number[] = []
  const { result } = renderHook(() => {
    const count = useNotificationCount()
    observed.push(count)
    return count
  })
  await waitFor(() => expect(result.current).toBe(7))
  const switchIndex = observed.length
  await act(async () => storage.set("tldwConfig", secondConfig))
  expect(result.current).toBe(0)
  await act(async () => storage.set("tldw:notifications:activeScope", secondScope))
  await waitFor(() => expect(result.current).toBe(9))
  expect(observed.slice(switchIndex)).not.toContain(7)
  await act(async () => storage.remove(secondScope))
  expect(result.current).toBe(0)
  await act(async () => storage.set(secondScope, { unreadCount: 4 }))
  expect(result.current).toBe(4)
})

it("only reads storage during mount and leaves shared values untouched", async () => {
  const set = vi.spyOn(storage.constructor.prototype, "set")
  const { result } = renderHook(() => useNotificationCount())
  await waitFor(() => expect(result.current).toBe(7))
  expect(set).not.toHaveBeenCalled()
  set.mockRestore()
})
