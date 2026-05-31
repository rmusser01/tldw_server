import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useRouteTransitionStore } from "@/store/route-transition"
import { useModeNavigationShortcuts } from "../useKeyboardShortcuts"

describe("useModeNavigationShortcuts", () => {
  beforeEach(() => {
    document.body.innerHTML = ""
    useRouteTransitionStore.getState().stop()
  })

  it("navigates to Sources on Alt+2", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))

    act(() => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", { key: "2", altKey: true })
      )
    })

    expect(navigate).toHaveBeenCalledWith("/sources")
    expect(useRouteTransitionStore.getState().pendingPath).toBe("/sources")
  })

  it("navigates to Notes on Alt+5", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))

    act(() => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", { key: "5", altKey: true })
      )
    })

    expect(navigate).toHaveBeenCalledWith("/notes")
    expect(useRouteTransitionStore.getState().pendingPath).toBe("/notes")
  })

  it("does not navigate while disabled", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate, false))

    act(() => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", { key: "2", altKey: true })
      )
    })

    expect(navigate).not.toHaveBeenCalled()
    expect(useRouteTransitionStore.getState().pendingPath).toBeNull()
  })

  it("does not navigate while an editable element is focused", () => {
    const navigate = vi.fn()
    const input = document.createElement("input")
    document.body.appendChild(input)
    input.focus()

    renderHook(() => useModeNavigationShortcuts(navigate))

    act(() => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", { key: "2", altKey: true })
      )
    })

    expect(navigate).not.toHaveBeenCalled()
    expect(useRouteTransitionStore.getState().pendingPath).toBeNull()
  })
})
