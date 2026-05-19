import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { safeStorageSerde } from "@/utils/safe-storage"

const useStorageMock = vi.fn()

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => useStorageMock(...args)
}))

import { useNotificationCount } from "@/hooks/useNotificationCount"

describe("useNotificationCount", () => {
  it("subscribes to unread count changes from extension local storage", () => {
    useStorageMock.mockReturnValue([7])

    const { result } = renderHook(() => useNotificationCount())

    expect(useStorageMock).toHaveBeenCalledWith(
      {
        key: "tldw:notifications:unreadCount",
        area: "local",
        serde: safeStorageSerde
      },
      expect.any(Function)
    )

    const transform = useStorageMock.mock.calls[0]?.[1] as ((value: unknown) => number) | undefined
    expect(transform).toBeTypeOf("function")
    expect(transform?.(undefined)).toBe(0)
    expect(transform?.("12")).toBe(12)
    expect(transform?.("abc")).toBe(0)
    expect(result.current).toBe(7)
  })
})
