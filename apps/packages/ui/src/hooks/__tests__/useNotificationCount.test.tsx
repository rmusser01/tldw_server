import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { safeStorageSerde } from "@/utils/safe-storage"
import { notificationRecordKeyForConfig } from "@/services/notification-runtime-scope"

const useStorageMock = vi.fn()

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => useStorageMock(...args)
}))

import { useNotificationCount } from "@/hooks/useNotificationCount"

describe("useNotificationCount", () => {
  beforeEach(() => {
    useStorageMock.mockReset()
  })

  it("reads unread count from the active scoped lifecycle record", () => {
    const config = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      userId: "user-a",
      accessToken: "token-a"
    }
    const recordKey = notificationRecordKeyForConfig(config)
    useStorageMock
      .mockReturnValueOnce([config])
      .mockReturnValueOnce([recordKey])
      .mockReturnValueOnce([{ state: "active", unreadCount: 7, updatedAt: 123 }])

    const { result } = renderHook(() => useNotificationCount())

    expect(useStorageMock).toHaveBeenNthCalledWith(
      1,
      {
        key: "tldwConfig",
        area: "local",
        serde: safeStorageSerde
      },
      expect.any(Function)
    )
    expect(useStorageMock).toHaveBeenNthCalledWith(
      2,
      {
        key: "tldw:notifications:activeScope",
        area: "local",
        serde: safeStorageSerde
      },
      expect.any(Function)
    )
    expect(useStorageMock).toHaveBeenNthCalledWith(
      3,
      {
        key: recordKey,
        area: "local",
        serde: safeStorageSerde
      },
      expect.any(Function)
    )
    expect(result.current).toBe(7)
  })

  it("clears the rendered count when no scoped selector is active", () => {
    const config = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      userId: "user-a",
      accessToken: "token-a"
    }
    useStorageMock
      .mockReturnValueOnce([config])
      .mockReturnValueOnce([null])
      .mockReturnValueOnce([undefined])

    const { result } = renderHook(() => useNotificationCount())

    expect(result.current).toBe(0)
    expect(useStorageMock.mock.calls[2]?.[0]).not.toEqual(
      expect.objectContaining({ key: "tldw:notifications:unreadCount" })
    )
  })

  it("clears synchronously for one render when the active scope is replaced", () => {
    const firstConfig = { serverUrl: "https://first.test", authMode: "multi-user", userId: "user-a", accessToken: "a" }
    const secondConfig = { serverUrl: "https://second.test", authMode: "multi-user", userId: "user-b", accessToken: "b" }
    const firstScope = notificationRecordKeyForConfig(firstConfig)
    const secondScope = notificationRecordKeyForConfig(secondConfig)
    let config = firstConfig
    let activeScope = firstScope
    const records = new Map([
      [firstScope, { state: "active", unreadCount: 7, updatedAt: 123 }],
      [secondScope, { state: "active", unreadCount: 9, updatedAt: 456 }]
    ])
    useStorageMock.mockImplementation((options: { key: string }) => [
      options.key === "tldwConfig"
        ? config
        : options.key === "tldw:notifications:activeScope"
        ? activeScope
        : records.get(options.key)
    ])
    const { result, rerender } = renderHook(() => useNotificationCount())

    expect(result.current).toBe(7)

    config = secondConfig
    activeScope = secondScope
    rerender()
    expect(result.current).toBe(0)

    rerender()
    expect(result.current).toBe(9)
  })

  it("suppresses the old count while the active selector lags a config account switch", () => {
    const firstConfig = { serverUrl: "https://api.test", authMode: "multi-user", userId: "user-a", accessToken: "a" }
    const secondConfig = { serverUrl: "https://api.test", authMode: "multi-user", userId: "user-b", accessToken: "b" }
    const firstScope = notificationRecordKeyForConfig(firstConfig)
    useStorageMock
      .mockReturnValueOnce([secondConfig])
      .mockReturnValueOnce([firstScope])
      .mockReturnValueOnce([{ state: "active", unreadCount: 7, updatedAt: 123 }])

    const { result } = renderHook(() => useNotificationCount())

    expect(result.current).toBe(0)
  })
})
