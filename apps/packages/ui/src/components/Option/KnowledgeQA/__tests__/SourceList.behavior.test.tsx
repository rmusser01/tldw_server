import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SourceList } from "../SourceList"

const state = {
  results: [] as Array<Record<string, any>>,
  citations: [] as Array<{ index: number }>,
  focusedSourceIndex: null as number | null,
  focusSource: vi.fn(),
  setQuery: vi.fn(),
  query: "source test query",
  answer: "First claim cites [3]. Second claim cites [8]. Third claim revisits [3].",
  searchDetails: {
    expandedQueries: ["evidence source"],
  } as { expandedQueries: string[] } | null,
  currentThreadId: "thread-source-test" as string | null,
  scrollToCitation: vi.fn(),
  setPinnedSourceFilters: vi.fn(),
  messages: [{ id: "assistant-1", role: "assistant" }] as Array<{
    id: string
    role: string
  }>,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    results: state.results,
    citations: state.citations,
    focusedSourceIndex: state.focusedSourceIndex,
    focusSource: state.focusSource,
    setQuery: state.setQuery,
    query: state.query,
    answer: state.answer,
    searchDetails: state.searchDetails,
    currentThreadId: state.currentThreadId,
    scrollToCitation: state.scrollToCitation,
    setPinnedSourceFilters: state.setPinnedSourceFilters,
    messages: state.messages,
  }),
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session-source-test",
  submitExplicitFeedback: vi.fn().mockResolvedValue({ ok: true }),
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: vi.fn(),
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: vi.fn().mockResolvedValue(undefined),
}))

function makeResult(index: number, overrides: Partial<Record<string, unknown>> = {}) {
  const sourceType = index % 2 === 0 ? "media_db" : "notes"
  return {
    id: `source-${index}`,
    content: `Body for source ${index}`,
    metadata: {
      title: `Source ${index}`,
      source_type: sourceType,
      media_id: sourceType === "media_db" ? index : undefined,
      note_id: sourceType === "notes" ? `note-${index}` : undefined,
      created_at: `2026-02-${String(10 + index).padStart(2, "0")}T12:00:00.000Z`,
      page_number: index,
      url: `https://example.com/source-${index}`,
      chunk_id: `chunk_${index}_of_20`,
    },
    score: 1 - index * 0.02,
    ...overrides,
  }
}

describe("SourceList researcher workflows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    state.focusedSourceIndex = null
    state.results = Array.from({ length: 12 }, (_, idx) => makeResult(idx + 1))
    state.citations = [{ index: 3 }, { index: 8 }]
    state.answer = "First claim cites [3]. Second claim cites [8]. Third claim revisits [3]."
    state.currentThreadId = "thread-source-test"
    state.scrollToCitation = vi.fn()
    state.setPinnedSourceFilters = vi.fn()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("supports source-type filter chips with counts", () => {
    render(<SourceList />)

    expect(screen.getByRole("button", { name: /All\s*12/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Notes\s*6/i }))

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const headings = within(list).getAllByRole("heading", { level: 4 })
    expect(headings.every((heading) => /Source (1|3|5|7|9|11)/.test(heading.textContent || ""))).toBe(
      true
    )
  })

  it("supports content-type facet chips", () => {
    state.results = [
      makeResult(1, {
        metadata: {
          title: "Source 1",
          source_type: "media_db",
          file_type: "pdf",
          media_id: 1,
          created_at: "2026-02-11T12:00:00.000Z",
          page_number: 1,
          url: "https://example.com/source-1.pdf",
          chunk_id: "chunk_1_of_20",
        },
      }),
      makeResult(2, {
        metadata: {
          title: "Source 2",
          source_type: "media_db",
          file_type: "transcript",
          media_id: 2,
          created_at: "2026-02-12T12:00:00.000Z",
          page_number: 2,
          url: "https://example.com/source-2.txt",
          chunk_id: "chunk_2_of_20",
        },
      }),
      makeResult(3, {
        metadata: {
          title: "Source 3",
          source_type: "media_db",
          mime_type: "video/mp4",
          media_id: 3,
          created_at: "2026-02-13T12:00:00.000Z",
          page_number: 3,
          url: "https://example.com/source-3.mp4",
          chunk_id: "chunk_3_of_20",
        },
      }),
    ]

    render(<SourceList />)

    // 3 results = compact density; reveal extended filters first
    fireEvent.click(screen.getByRole("button", { name: "Show filters" }))

    fireEvent.click(screen.getByRole("button", { name: /PDF\s*1/i }))

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const headings = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)
    expect(headings).toEqual(["Source 1"])
  })

  it("uses compact source-card defaults for mobile density", () => {
    render(<SourceList />)

    const sourceCard = document.getElementById("source-card-0")
    expect(sourceCard).not.toBeNull()

    const headerRow = sourceCard!.querySelector("div.flex.items-start")
    expect(headerRow).not.toBeNull()
    expect(headerRow!.className).toContain("p-3")
    expect(headerRow!.className).toContain("sm:p-4")

    const excerpt = screen.getByText((_, element) => {
      if (!element) return false
      if (element.tagName.toLowerCase() !== "p") return false
      return (element.textContent || "").trim() === "Body for source 1"
    })
    expect(excerpt.className).toContain("text-xs")
    expect(excerpt.className).toContain("sm:text-sm")
  })

  it("highlights query-aligned terms in source excerpts", () => {
    render(<SourceList />)

    const highlightedTerms = document.querySelectorAll("mark")
    expect(highlightedTerms.length).toBeGreaterThan(0)
    expect(
      Array.from(highlightedTerms).some((element) =>
        /source/i.test(element.textContent || "")
      )
    ).toBe(true)
  })

  it("shows freshness badges for stale sources", () => {
    state.results = [
      makeResult(1, {
        metadata: {
          title: "Source 1",
          source_type: "media_db",
          media_id: 1,
          created_at: "2018-02-01T12:00:00.000Z",
          page_number: 1,
          url: "https://example.com/source-1",
          chunk_id: "chunk_1_of_20",
        },
      }),
    ]
    state.citations = [{ index: 1 }]

    render(<SourceList />)

    const freshnessBadge = screen.getByText("From 2018")
    expect(freshnessBadge).toBeInTheDocument()
    expect(freshnessBadge.className).toContain("text-danger")
  })

  it("does not hijack Tab navigation from interactive controls inside a source card", () => {
    render(<SourceList />)

    const viewFullButton = screen.getByRole("button", { name: "View source 1" })
    viewFullButton.focus()
    vi.clearAllMocks()
    fireEvent.keyDown(viewFullButton, { key: "Tab" })

    expect(state.focusSource).not.toHaveBeenCalled()
  })

  it("supports keyword and date-range filtering within sources", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-01T00:00:00.000Z"))

    state.results = [
      makeResult(1, {
        content: "Legacy archive analysis and migration notes",
        metadata: {
          title: "Source 1",
          source_type: "media_db",
          media_id: 1,
          created_at: "2018-02-01T12:00:00.000Z",
          page_number: 1,
          url: "https://example.com/source-1",
          chunk_id: "chunk_1_of_20",
        },
      }),
      makeResult(2, {
        content: "Recent release update and roadmap",
        metadata: {
          title: "Source 2",
          source_type: "media_db",
          media_id: 2,
          created_at: "2026-02-25T12:00:00.000Z",
          page_number: 2,
          url: "https://example.com/source-2",
          chunk_id: "chunk_2_of_20",
        },
      }),
    ]
    state.citations = [{ index: 1 }]

    render(<SourceList />)

    // 2 results = compact density; reveal filters first
    fireEvent.click(screen.getByRole("button", { name: "Show filters" }))

    fireEvent.change(screen.getByLabelText("Filter sources by keyword"), {
      target: { value: "archive migration" },
    })
    fireEvent.change(screen.getByLabelText("Filter sources by date range"), {
      target: { value: "older_365d" },
    })

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const headings = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)
    expect(headings).toEqual(["Source 1"])

    fireEvent.change(screen.getByLabelText("Filter sources by keyword"), {
      target: { value: "release roadmap" },
    })
    expect(screen.getByText("No sources match the selected filters.")).toBeInTheDocument()
  })

  it("resets all source filters to defaults", () => {
    render(<SourceList />)

    fireEvent.change(screen.getByLabelText("Sort sources"), {
      target: { value: "date" },
    })
    fireEvent.change(screen.getByLabelText("Filter sources by date range"), {
      target: { value: "older_365d" },
    })
    fireEvent.change(screen.getByLabelText("Filter sources by keyword"), {
      target: { value: "source 1" },
    })
    fireEvent.click(screen.getByRole("button", { name: /Notes\s*6/i }))
    fireEvent.click(screen.getByRole("button", { name: /Note\s*6/i }))

    expect(screen.getByText("5 filters active")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Reset source filters" }))

    expect((screen.getByLabelText("Sort sources") as HTMLSelectElement).value).toBe(
      "relevance"
    )
    expect(
      (screen.getByLabelText("Filter sources by date range") as HTMLSelectElement).value
    ).toBe("all")
    expect(
      (screen.getByLabelText("Filter sources by keyword") as HTMLInputElement).value
    ).toBe("")
    expect(
      screen.getByRole("button", { name: /All\s*12/i })
    ).toHaveAttribute("aria-pressed", "true")
    expect(
      screen.getByRole("button", { name: /Any type\s*12/i })
    ).toHaveAttribute("aria-pressed", "true")
    expect(screen.queryByText("5 filters active")).not.toBeInTheDocument()
  })

  it("persists filter controls per thread and restores when switching back", async () => {
    const { rerender } = render(<SourceList />)

    const sourceTypeGroup = screen.getByLabelText("Source type filters")
    const contentTypeGroup = screen.getByLabelText("Content type filters")
    const sortSelect = screen.getByLabelText("Sort sources") as HTMLSelectElement
    const dateSelect = screen.getByLabelText(
      "Filter sources by date range"
    ) as HTMLSelectElement
    const keywordInput = screen.getByLabelText(
      "Filter sources by keyword"
    ) as HTMLInputElement

    fireEvent.change(sortSelect, { target: { value: "date" } })
    fireEvent.change(dateSelect, { target: { value: "older_365d" } })
    fireEvent.change(keywordInput, { target: { value: "source 1" } })
    fireEvent.click(
      within(sourceTypeGroup).getByRole("button", { name: /Notes\s*6/i })
    )
    fireEvent.click(
      within(contentTypeGroup).getByRole("button", { name: /Note\s*6/i })
    )

    state.currentThreadId = "thread-other"
    rerender(<SourceList />)

    await waitFor(() => {
      expect((screen.getByLabelText("Sort sources") as HTMLSelectElement).value).toBe(
        "relevance"
      )
      expect(
        (
          screen.getByLabelText(
            "Filter sources by date range"
          ) as HTMLSelectElement
        ).value
      ).toBe("all")
      expect(
        (screen.getByLabelText("Filter sources by keyword") as HTMLInputElement).value
      ).toBe("")
    })

    fireEvent.change(screen.getByLabelText("Sort sources"), {
      target: { value: "title" },
    })
    fireEvent.change(screen.getByLabelText("Filter sources by keyword"), {
      target: { value: "source 2" },
    })

    state.currentThreadId = "thread-source-test"
    rerender(<SourceList />)

    await waitFor(() => {
      expect((screen.getByLabelText("Sort sources") as HTMLSelectElement).value).toBe(
        "date"
      )
      expect(
        (
          screen.getByLabelText(
            "Filter sources by date range"
          ) as HTMLSelectElement
        ).value
      ).toBe("older_365d")
      expect(
        (screen.getByLabelText("Filter sources by keyword") as HTMLInputElement).value
      ).toBe("source 1")
      expect(
        within(screen.getByLabelText("Source type filters")).getByRole("button", {
          name: /Notes\s*6/i,
        })
      ).toHaveAttribute("aria-pressed", "true")
      expect(
        within(screen.getByLabelText("Content type filters")).getByRole("button", {
          name: /Note\s*6/i,
        })
      ).toHaveAttribute("aria-pressed", "true")
    })
  })

  it("falls back to default filters when localStorage reads are blocked", () => {
    const getItemSpy = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })

    expect(() => render(<SourceList />)).not.toThrow()
    expect((screen.getByLabelText("Sort sources") as HTMLSelectElement).value).toBe(
      "relevance"
    )
    expect(
      (screen.getByLabelText("Filter sources by keyword") as HTMLInputElement).value
    ).toBe("")

    getItemSpy.mockRestore()
  })

  it("keeps filter controls interactive when localStorage writes are blocked", async () => {
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })

    render(<SourceList />)

    expect(() => {
      fireEvent.change(screen.getByLabelText("Sort sources"), {
        target: { value: "date" },
      })
      fireEvent.change(screen.getByLabelText("Filter sources by keyword"), {
        target: { value: "source 1" },
      })
    }).not.toThrow()

    await waitFor(() => {
      expect((screen.getByLabelText("Sort sources") as HTMLSelectElement).value).toBe("date")
      expect(
        (screen.getByLabelText("Filter sources by keyword") as HTMLInputElement).value
      ).toBe("source 1")
    })

    setItemSpy.mockRestore()
  })

  it("supports reverse citation jump from cited source badges", () => {
    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "Jump to citation 3 in answer" }))

    expect(state.scrollToCitation).toHaveBeenCalledWith(3, 1)
    expect(state.focusSource).toHaveBeenCalledWith(2)
  })

  it("supports reverse citation jump from cited source card activation", () => {
    render(<SourceList />)

    fireEvent.click(screen.getByRole("heading", { name: "Source 3" }))

    expect(state.scrollToCitation).toHaveBeenCalledWith(3, 1)
    expect(state.focusSource).toHaveBeenCalledWith(2)

    const citedCard = document.getElementById("source-card-2")
    expect(citedCard).not.toBeNull()
    expect(citedCard).toHaveAttribute("tabindex", "0")

    fireEvent.keyDown(citedCard!, { key: "Enter" })

    expect(state.scrollToCitation).toHaveBeenLastCalledWith(3, 1)
  })

  it("shows sentence-level usage anchors for cited sources", () => {
    render(<SourceList />)

    const sentenceChip = screen.getByRole("button", {
      name: /Jump to citation 3 in answer sentence 3:/i,
    })

    expect(sentenceChip).toHaveAttribute(
      "title",
      "Sentence 3: Third claim revisits [3]."
    )

    fireEvent.click(sentenceChip)

    expect(state.scrollToCitation).toHaveBeenCalledWith(3, 2)
    expect(state.focusSource).toHaveBeenCalledWith(2)
  })

  it("supports pinning sources and prioritizes pinned cards in ordering", () => {
    render(<SourceList />)

    fireEvent.click(screen.getByLabelText("More actions for source 10"))
    fireEvent.click(screen.getByRole("menuitem", { name: "Pin" }))

    expect(screen.getByText("1 pinned")).toBeInTheDocument()

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const titlesAfterPin = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)
    expect(titlesAfterPin[0]).toBe("Source 10")
    expect(state.setPinnedSourceFilters).toHaveBeenLastCalledWith(
      expect.objectContaining({
        mediaIds: expect.arrayContaining([10]),
      })
    )

    fireEvent.click(screen.getByLabelText("More actions for source 10"))
    fireEvent.click(screen.getByRole("menuitem", { name: "Unpin" }))

    const titlesAfterUnpin = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)
    expect(titlesAfterUnpin[0]).toBe("Source 1")
  })

  it("syncs focused source on hover for citation back-link highlighting", () => {
    render(<SourceList />)

    const firstCard = document.getElementById("source-card-0")
    expect(firstCard).not.toBeNull()

    fireEvent.mouseEnter(firstCard!)
    expect(state.focusSource).toHaveBeenCalledWith(0)

    fireEvent.mouseLeave(firstCard!)
    expect(state.focusSource).toHaveBeenCalledWith(null)
  })

  it("adds sort modes including date and cited-first", () => {
    render(<SourceList />)

    const sortSelect = screen.getByLabelText("Sort sources")
    fireEvent.change(sortSelect, { target: { value: "cited" } })

    const list = screen.getByRole("list", { name: "Retrieved sources" })
    const titles = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)

    expect(titles[0]).toBe("Source 3")
    expect(titles[1]).toBe("Source 8")

    fireEvent.change(sortSelect, { target: { value: "date" } })
    const resortedTitles = within(list)
      .getAllByRole("heading", { level: 4 })
      .map((heading) => heading.textContent)
    expect(resortedTitles[0]).toBe("Source 12")
  })

  it("paginates large result sets with show-more progression", async () => {
    render(<SourceList />)

    expect(screen.getByText("Showing 10 of 12 sources")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Show more \(2 remaining\)/i }))

    await waitFor(() =>
      expect(screen.getByText("Showing 12 of 12 sources")).toBeInTheDocument()
    )
  })

  it("supports ask-template variants and populates the query", () => {
    render(<SourceList />)

    fireEvent.click(screen.getByLabelText("More actions for source 1"))
    fireEvent.click(screen.getByRole("menuitem", { name: "Ask: Summarize" }))

    expect(state.setQuery).toHaveBeenCalledWith("Summarize Source 1")
  })

  it("copies citation-formatted output for individual sources", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })

    render(<SourceList />)
    fireEvent.click(screen.getAllByRole("button", { name: /Copy citation/i })[0])

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(
        "Source 1 (Page 1) - https://example.com/source-1"
      )
    )
  })

  it("opens keyboard shortcut legend via ? and closes on escape", async () => {
    render(<SourceList />)

    fireEvent.keyDown(window, { key: "?" })
    expect(
      screen.getByRole("dialog", { name: "Source keyboard shortcuts" })
    ).toBeInTheDocument()

    fireEvent.keyDown(window, { key: "Escape" })
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Source keyboard shortcuts" })
      ).not.toBeInTheDocument()
    )
  })

  it("hides the date filter in compact density (<=3 results)", () => {
    state.results = [makeResult(1), makeResult(2)]
    state.citations = [{ index: 1 }]

    render(<SourceList />)

    expect(screen.queryByLabelText("Filter sources by date range")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Filter sources by keyword")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Sort sources")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Source type filters")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Content type filters")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show filters" })).toBeInTheDocument()
  })

  it("reveals all filters in compact density after clicking Show filters", () => {
    state.results = [makeResult(1), makeResult(2)]
    state.citations = [{ index: 1 }]

    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "Show filters" }))

    expect(screen.getByLabelText("Filter sources by date range")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter sources by keyword")).toBeInTheDocument()
    expect(screen.getByLabelText("Sort sources")).toBeInTheDocument()
    expect(screen.getByLabelText("Source type filters")).toBeInTheDocument()
    expect(screen.getByLabelText("Content type filters")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Hide filters" })).toBeInTheDocument()
  })

  it("shows all filters by default in full density (>9 results)", () => {
    state.results = Array.from({ length: 12 }, (_, idx) => makeResult(idx + 1))
    state.citations = [{ index: 3 }, { index: 8 }]

    render(<SourceList />)

    expect(screen.getByLabelText("Filter sources by date range")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter sources by keyword")).toBeInTheDocument()
    expect(screen.getByLabelText("Sort sources")).toBeInTheDocument()
    expect(screen.getByLabelText("Source type filters")).toBeInTheDocument()
    expect(screen.getByLabelText("Content type filters")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Show filters" })).not.toBeInTheDocument()
  })

  it("shows keyword and sort but hides date and facets in default density (4-9 results)", () => {
    state.results = Array.from({ length: 6 }, (_, idx) => makeResult(idx + 1))
    state.citations = [{ index: 1 }]

    render(<SourceList />)

    expect(screen.getByLabelText("Filter sources by keyword")).toBeInTheDocument()
    expect(screen.getByLabelText("Sort sources")).toBeInTheDocument()
    expect(screen.queryByLabelText("Filter sources by date range")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Source type filters")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Content type filters")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "More filters" })).toBeInTheDocument()
  })

  it("uses a single-column source stack and shorter keyboard hint in rail layout", () => {
    state.results = Array.from({ length: 6 }, (_, idx) => makeResult(idx + 1))
    state.citations = [{ index: 1 }]

    render(<SourceList layout="rail" />)

    const grid = screen.getByTestId("knowledge-source-grid")
    expect(grid.className).toContain("grid-cols-1")
    expect(grid.className).not.toContain("md:grid-cols-2")
    expect(
      screen.getByText((_, element) =>
        element?.tagName === "SPAN" && element.textContent === "Jump: 1-9 • Tab cycles"
      )
    ).toBeInTheDocument()
  })
})
