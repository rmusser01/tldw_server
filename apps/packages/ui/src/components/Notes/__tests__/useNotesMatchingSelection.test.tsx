import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useNotesMatchingSelection } from "../hooks/useNotesMatchingSelection"

const rows = (prefix: string, count: number) =>
  Array.from({ length: count }, (_, index) => ({ id: `${prefix}${index}`, title: `${prefix}${index}` }))

const renderMatching = (initialKey: string) => {
  const fetchPage = vi.fn(async (page: number) => ({
    items: page === 1 ? rows("n", 100) : rows("m", 50),
    total: 150
  }))
  const hook = renderHook(
    ({ filterKey }) => useNotesMatchingSelection({ filterKey, fetchPage, onError: () => undefined }),
    { initialProps: { filterKey: initialKey } }
  )
  return { ...hook, fetchPage }
}

describe("useNotesMatchingSelection (NL-05)", () => {
  it.each([
    ["Notes -> Trash -> Notes", ["notes", "trash", "notes"]],
    ["tag on -> off -> on", ["tag:research", "no-tags", "tag:research"]]
  ])("does not bring a selection back when the list returns to its filters (%s)", async (_name, keys) => {
    const { result, rerender } = renderMatching(keys[0])
    await act(async () => {
      await result.current.selectAllMatching()
    })
    expect(result.current.matching?.notes).toHaveLength(150)

    rerender({ filterKey: keys[1] })
    expect(result.current.matching).toBeNull()
    rerender({ filterKey: keys[2] })
    expect(result.current.matching).toBeNull()
  })
})
