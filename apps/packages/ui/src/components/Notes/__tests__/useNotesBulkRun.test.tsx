import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { useNotesBulkRun } from "../hooks/useNotesBulkRun"

describe("useNotesBulkRun (NL-05)", () => {
  it("refuses a second run while one is active, and allows one after it finishes", () => {
    const { result } = renderHook(() => useNotesBulkRun("scope-a"))
    let first: ReturnType<typeof result.current.begin> = null
    act(() => {
      first = result.current.begin("delete")
    })
    expect(first).not.toBeNull()
    expect(result.current.begin("addTags")).toBeNull()

    act(() => {
      first!.start(2)
      first!.step()
    })
    expect(result.current.progress).toEqual({ action: "delete", done: 1, total: 2 })

    act(() => first!.finish([{ id: "n2", title: "Beta" }]))
    expect(result.current.isRunning).toBe(false)
    expect(result.current.failures).toEqual({ action: "delete", notes: [{ id: "n2", title: "Beta" }] })

    let second: ReturnType<typeof result.current.begin> = null
    act(() => {
      second = result.current.begin("addTags")
    })
    expect(second).not.toBeNull()
    // A new run clears the last run's failures.
    expect(result.current.failures).toBeNull()
  })

  it("drops failures when the notes account changes", () => {
    const { result, rerender } = renderHook(({ scope }) => useNotesBulkRun(scope), {
      initialProps: { scope: "scope-a" }
    })
    act(() => {
      result.current.begin("delete")!.finish([{ id: "n1", title: "Alpha" }])
    })
    expect(result.current.failures).not.toBeNull()
    rerender({ scope: "scope-b" })
    expect(result.current.failures).toBeNull()
  })
})
