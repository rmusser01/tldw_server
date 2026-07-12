import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { safeStorageSerde } from "@/utils/safe-storage"

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
    const recordKey = "tldw:notifications:server:api.example.test:user:user-a"
    useStorageMock
      .mockReturnValueOnce([recordKey])
      .mockReturnValueOnce([{ state: "active", unreadCount: 7, updatedAt: 123 }])

    const { result } = renderHook(() => useNotificationCount())

    expect(useStorageMock).toHaveBeenNthCalledWith(
      1,
      {
        key: "tldw:notifications:activeScope",
        area: "local",
        serde: safeStorageSerde
      },
      expect.any(Function)
    )
    expect(useStorageMock).toHaveBeenNthCalledWith(
      2,
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
    useStorageMock
      .mockReturnValueOnce([null])
      .mockReturnValueOnce([undefined])

    const { result } = renderHook(() => useNotificationCount())

    expect(result.current).toBe(0)
    expect(useStorageMock.mock.calls[1]?.[0]).not.toEqual(
      expect.objectContaining({ key: "tldw:notifications:unreadCount" })
    )
  })

  it("clears synchronously for one render when the active scope is replaced", () => {
    const firstScope = "tldw:notifications:server:first:user:user-a"
    const secondScope = "tldw:notifications:server:second:user:user-b"
    let activeScope = firstScope
    const records = new Map([
      [firstScope, { state: "active", unreadCount: 7, updatedAt: 123 }],
      [secondScope, { state: "active", unreadCount: 9, updatedAt: 456 }]
    ])
    useStorageMock.mockImplementation((options: { key: string }) => [
      options.key === "tldw:notifications:activeScope"
        ? activeScope
        : records.get(options.key)
    ])
    const { result, rerender } = renderHook(() => useNotificationCount())

    expect(result.current).toBe(7)

    activeScope = secondScope
    rerender()
    expect(result.current).toBe(0)

    rerender()
    expect(result.current).toBe(9)
  })
})
