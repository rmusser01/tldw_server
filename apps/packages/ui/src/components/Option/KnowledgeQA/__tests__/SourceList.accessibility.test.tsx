import { render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourceList } from "../SourceList"

const state = {
  results: [
    {
      id: "r1",
      content: "Alpha content",
      metadata: { title: "Alpha Source", source_type: "media_db" },
      score: 0.9,
    },
    {
      id: "r2",
      content: "Beta content",
      metadata: { title: "Beta Source", source_type: "notes" },
      score: 0.8,
    },
  ] as Array<Record<string, any>>,
  citations: [{ index: 1 }] as Array<{ index: number }>,
  focusedSourceIndex: null as number | null,
  focusSource: vi.fn(),
  setQuery: vi.fn(),
  search: vi.fn(),
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    storageScopeKey: "test-owner",
    isAuthorityCurrent: () => true,
    results: state.results,
    citations: state.citations,
    focusedSourceIndex: state.focusedSourceIndex,
    focusSource: state.focusSource,
    setQuery: state.setQuery,
    search: state.search,
  })
}))

describe("SourceList accessibility semantics", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.focusedSourceIndex = null
  })

  it("renders sources using list and listitem semantics", () => {
    render(<SourceList />)

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const items = within(list).getAllByRole("listitem")
    expect(items).toHaveLength(2)
    expect(items[0]).toHaveAttribute("tabindex", "0")
    expect(items[1]).not.toHaveAttribute("tabindex")
  })
})
