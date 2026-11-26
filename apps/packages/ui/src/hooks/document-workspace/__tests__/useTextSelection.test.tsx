import { act, renderHook } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { useTextSelection } from "../useTextSelection"

afterEach(() => {
  vi.restoreAllMocks()
  document.getSelection()?.removeAllRanges()
})

it("captures a keyboard selection on its PDF page and keeps it when moving focus to an action", () => {
  const viewer = document.createElement("div")
  viewer.innerHTML = '<div data-page-number="4">Selected passage</div>'
  const node = viewer.firstChild!.firstChild!
  const range = document.createRange()
  range.selectNodeContents(node)
  range.getBoundingClientRect = () => new DOMRect(10, 30, 80, 12)
  const removeAllRanges = vi.fn()
  const nativeSelection = {
    rangeCount: 1, anchorNode: node, focusNode: node, isCollapsed: false,
    toString: () => "Selected passage", getRangeAt: () => range, removeAllRanges
  }
  vi.spyOn(window, "getSelection").mockReturnValue(nativeSelection as unknown as Selection)
  const { result } = renderHook(() => useTextSelection({ current: viewer }))
  act(() => { document.dispatchEvent(new Event("selectionchange")) })
  expect(result.current.selection).toMatchObject({ text: "Selected passage", page: 4 })
  nativeSelection.isCollapsed = true
  act(() => { document.dispatchEvent(new Event("selectionchange")) })
  expect(result.current.selection?.text).toBe("Selected passage")
  act(() => result.current.clearSelection())
  expect(result.current.selection).toBeNull()
  expect(removeAllRanges).toHaveBeenCalledOnce()
})
