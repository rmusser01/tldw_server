import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { useFilterPresets } from "../useFilterPresets"

const STORAGE_KEY = "tldw-prompt-filter-presets-v1"

describe("useFilterPresets", () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    window.localStorage.clear()
  })

  it("ignores malformed preset entries without dropping valid presets", () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        null,
        {
          id: "preset-1",
          name: "Saved preset",
          typeFilter: "all",
          syncFilter: "all",
          usageFilter: "used",
          tagFilter: [],
          tagMatchMode: "any",
          savedView: "grid"
        }
      ])
    )

    const { result } = renderHook(() => useFilterPresets())

    expect(result.current.presets).toHaveLength(1)
    expect(result.current.presets[0]).toMatchObject({
      id: "preset-1",
      name: "Saved preset",
      usageFilter: "used"
    })
  })

  it("normalizes unknown usage filters back to all", () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        {
          id: "preset-1",
          name: "Saved preset",
          typeFilter: "all",
          syncFilter: "all",
          usageFilter: "unexpected",
          tagFilter: [],
          tagMatchMode: "all",
          savedView: "list"
        }
      ])
    )

    const { result } = renderHook(() => useFilterPresets())

    expect(result.current.presets[0]?.usageFilter).toBe("all")
  })

  it("persists new presets after recovering malformed storage", () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify([null]))

    const { result } = renderHook(() => useFilterPresets())

    act(() => {
      result.current.savePreset("New preset", {
        typeFilter: "all",
        syncFilter: "all",
        usageFilter: "unused",
        tagFilter: [],
        tagMatchMode: "any",
        savedView: "grid"
      })
    })

    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]")
    expect(stored).toHaveLength(1)
    expect(stored[0]).toMatchObject({
      name: "New preset",
      usageFilter: "unused"
    })
  })
})
