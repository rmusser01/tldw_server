import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { openModelSelector } from "@/components/Option/Playground/playground-cockpit-actions"
import { useModelPickerOpenRequests } from "../useModelPickerOpenRequests"

describe("useModelPickerOpenRequests (CC-02)", () => {
  it("opens the side panel's model picker when the page asks for it", () => {
    const picker = { openAndFocus: vi.fn() }
    renderHook(() => useModelPickerOpenRequests({ current: picker }))

    openModelSelector()

    expect(picker.openAndFocus).toHaveBeenCalledTimes(1)
  })

  it("stops listening once the composer is gone", () => {
    const picker = { openAndFocus: vi.fn() }
    const { unmount } = renderHook(() => useModelPickerOpenRequests({ current: picker }))
    unmount()

    openModelSelector()

    expect(picker.openAndFocus).not.toHaveBeenCalled()
  })

  it("does nothing while the picker is not mounted", () => {
    renderHook(() => useModelPickerOpenRequests({ current: null }))
    expect(() => openModelSelector()).not.toThrow()
  })
})
