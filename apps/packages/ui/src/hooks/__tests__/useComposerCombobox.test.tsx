// @vitest-environment jsdom
import { renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { comboboxOptionId, useComposerCombobox } from "../useComposerCombobox"

// AX-12: one combobox contract for the slash, mention and wikilink menus.
describe("useComposerCombobox", () => {
  it("is a collapsed combobox while no suggestions show", () => {
    const { result } = renderHook(() =>
      useComposerCombobox({ open: false, count: 3, activeIndex: 0 })
    )
    expect(result.current.inputProps).toEqual({
      role: "combobox",
      "aria-autocomplete": "list",
      "aria-expanded": false,
      "aria-controls": undefined,
      "aria-activedescendant": undefined
    })
  })

  it("controls its listbox and points at the active option while open", () => {
    const { result, rerender } = renderHook(
      ({ activeIndex }) => useComposerCombobox({ open: true, count: 3, activeIndex }),
      { initialProps: { activeIndex: 0 } }
    )
    const { listboxId } = result.current
    expect(listboxId).toMatch(/^[A-Za-z][\w-]*$/)
    expect(result.current.inputProps).toMatchObject({
      "aria-expanded": true,
      "aria-controls": listboxId,
      "aria-activedescendant": comboboxOptionId(listboxId, 0)
    })
    rerender({ activeIndex: 2 })
    expect(result.current.inputProps["aria-activedescendant"]).toBe(
      comboboxOptionId(listboxId, 2)
    )
  })

  it("stays collapsed when an open menu has no suggestions", () => {
    const { result } = renderHook(() =>
      useComposerCombobox({ open: true, count: 0, activeIndex: 0 })
    )
    expect(result.current.inputProps).toMatchObject({
      "aria-expanded": false,
      "aria-controls": undefined,
      "aria-activedescendant": undefined
    })
  })

  it("gives each textarea its own listbox ids", () => {
    const first = renderHook(() => useComposerCombobox({ open: true, count: 1, activeIndex: 0 }))
    const second = renderHook(() => useComposerCombobox({ open: true, count: 1, activeIndex: 0 }))
    expect(first.result.current.listboxId).not.toBe(second.result.current.listboxId)
  })
})
