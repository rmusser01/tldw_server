import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SearchBar } from "../SearchBar"

const trackMetricMock = vi.fn()

const state = {
  query: "",
  setQuery: vi.fn(),
  search: vi.fn(),
  cancelSearch: vi.fn(),
  isSearching: false,
  clearResults: vi.fn(),
  results: [] as Array<{ id: string }>,
  answer: null as string | null,
  queryWarning: null as string | null,
  searchHistory: [] as Array<{ query: string }>,
  isLocalOnlyThread: false,
  settings: {
    sources: ["documents", "notes"] as string[],
    enable_web_fallback: true,
  },
  updateSetting: vi.fn(),
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    query: state.query,
    setQuery: state.setQuery,
    search: state.search,
    cancelSearch: state.cancelSearch,
    isSearching: state.isSearching,
    clearResults: state.clearResults,
    results: state.results,
    answer: state.answer,
    queryWarning: state.queryWarning,
    searchHistory: state.searchHistory,
    isLocalOnlyThread: state.isLocalOnlyThread,
    settings: state.settings,
    updateSetting: state.updateSetting,
  })
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: (...args: unknown[]) => trackMetricMock(...args),
}))

describe("SearchBar behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    trackMetricMock.mockResolvedValue(undefined)
    state.query = ""
    state.isSearching = false
    state.results = []
    state.answer = null
    state.queryWarning = null
    state.searchHistory = []
    state.isLocalOnlyThread = false
    state.settings.sources = ["documents", "notes"]
    state.settings.enable_web_fallback = true
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("provides accessible input description and input constraints", () => {
    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })
    expect(input).toHaveAttribute("id", "knowledge-search-input")
    expect(input).toHaveAttribute("maxlength", "20000")
    expect(input).toHaveAttribute("aria-describedby", "knowledge-qa-search-description")
    expect(screen.getByText(/Ask questions about your documents/i)).toHaveAttribute(
      "id",
      "knowledge-qa-search-description"
    )
  })

  it("keeps the compact width cap by default", () => {
    render(<SearchBar autoFocus={false} />)

    expect(screen.getByRole("textbox", { name: "Search your knowledge base" }).closest("form"))
      .toHaveClass("max-w-3xl")
  })

  it("removes the compact width cap when wide mode is enabled", () => {
    render(<SearchBar autoFocus={false} widthMode="wide" />)

    expect(screen.getByRole("textbox", { name: "Search your knowledge base" }).closest("form"))
      .not.toHaveClass("max-w-3xl")
  })

  it("uses explicit searching label and loading indicator", () => {
    state.query = "active query"
    state.isSearching = true

    render(<SearchBar autoFocus={false} />)

    expect(screen.getByRole("button", { name: "Searching..." })).toBeInTheDocument()
    expect(screen.getByTestId("knowledge-search-loading-indicator")).toBeInTheDocument()
  })

  it("shows character count near the max length", () => {
    state.query = "a".repeat(17000)

    render(<SearchBar autoFocus={false} />)

    expect(screen.getByText("17000/20000")).toBeInTheDocument()
  })

  it("shows inline truncation warning when query payload is shortened", () => {
    state.queryWarning = "Query exceeded 20,000 characters and was shortened before search."

    render(<SearchBar autoFocus={false} />)

    expect(
      screen.getByText(
        "Query exceeded 20,000 characters and was shortened before search."
      )
    ).toBeInTheDocument()
  })

  it("uses descriptive web fallback tooltip text", () => {
    render(<SearchBar autoFocus={false} />)

    const toggle = screen.getByRole("button", { name: /Web fallback is currently enabled/i })
    expect(toggle).toHaveAttribute(
      "title",
      "Falls back to web search when local source relevance is below threshold (configurable in settings)."
    )
  })

  it("truncates typed query updates to max length", () => {
    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })
    const longQuery = "x".repeat(25000)
    fireEvent.change(input, { target: { value: longQuery } })

    expect(state.setQuery).toHaveBeenCalledWith(longQuery.slice(0, 20000))
  })

  it("clears only the query when clear icon is clicked", () => {
    state.query = "existing query"

    render(<SearchBar autoFocus={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }))

    expect(state.setQuery).toHaveBeenCalledWith("")
    expect(state.clearResults).not.toHaveBeenCalled()
  })

  it("shows explicit new-search action and clears full state", () => {
    state.query = "existing query"
    state.results = [{ id: "r1" }]

    render(<SearchBar autoFocus={false} />)
    fireEvent.click(screen.getByRole("button", { name: "New search" }))

    expect(state.clearResults).toHaveBeenCalledTimes(1)
    expect(state.setQuery).toHaveBeenCalledWith("")
  })

  it("focuses the search input with slash shortcut", () => {
    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })
    expect(document.activeElement).not.toBe(input)

    fireEvent.keyDown(window, { key: "/" })
    expect(document.activeElement).toBe(input)
  })

  it("runs clear-full shortcut with Cmd/Ctrl+K", () => {
    state.query = "existing query"
    state.results = [{ id: "r1" }]

    render(<SearchBar autoFocus={false} />)

    fireEvent.keyDown(window, { key: "k", metaKey: true })
    expect(state.clearResults).toHaveBeenCalledTimes(1)
    expect(state.setQuery).toHaveBeenCalledWith("")

    vi.clearAllMocks()
    fireEvent.keyDown(window, { key: "k", ctrlKey: true })
    expect(state.clearResults).toHaveBeenCalledTimes(1)
    expect(state.setQuery).toHaveBeenCalledWith("")
  })

  it("runs clear-full shortcut when the knowledge search input itself has focus", () => {
    state.query = "existing query"
    state.results = [{ id: "r1" }]

    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })
    input.focus()
    fireEvent.keyDown(input, { key: "k", metaKey: true })

    expect(state.clearResults).toHaveBeenCalledTimes(1)
    expect(state.setQuery).toHaveBeenCalledWith("")
  })

  it("does not clear the active session when Cmd/Ctrl+K originates from another editable control", () => {
    state.query = "existing query"
    state.results = [{ id: "r1" }]

    render(
      <>
        <input aria-label="History filter" />
        <SearchBar autoFocus={false} />
      </>
    )

    const secondaryInput = screen.getByRole("textbox", { name: "History filter" })
    secondaryInput.focus()
    fireEvent.keyDown(secondaryInput, { key: "k", metaKey: true })

    expect(state.clearResults).not.toHaveBeenCalled()
    expect(state.setQuery).not.toHaveBeenCalledWith("")
  })

  it("does not clear the active session when Cmd/Ctrl+K originates from another interactive control", () => {
    state.query = "existing query"
    state.results = [{ id: "r1" }]

    render(
      <>
        <button type="button" aria-label="Export answer">
          Export
        </button>
        <SearchBar autoFocus={false} />
      </>
    )

    const exportButton = screen.getByRole("button", { name: "Export answer" })
    exportButton.focus()
    fireEvent.keyDown(exportButton, { key: "k", metaKey: true })

    expect(state.clearResults).not.toHaveBeenCalled()
    expect(state.setQuery).not.toHaveBeenCalledWith("")
  })

  it("shows stop action during search and triggers cancellation", () => {
    state.query = "active query"
    state.isSearching = true

    render(<SearchBar autoFocus={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Cancel search" }))

    expect(state.cancelSearch).toHaveBeenCalledTimes(1)
  })

  it("shows local-only persistence indicator for fallback threads", () => {
    state.isLocalOnlyThread = true

    render(<SearchBar autoFocus={false} />)

    expect(screen.getByText("Not synced")).toBeInTheDocument()
    expect(screen.getByText("Not synced")).toHaveAttribute(
      "title",
      "Working offline - conversation is stored locally and not synced to server."
    )
  })

  it("rotates placeholders through expanded example set", () => {
    vi.useFakeTimers()
    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })

    act(() => {
      vi.advanceTimersByTime(32000)
    })

    expect(input).toHaveAttribute("placeholder", "Compare the conclusions across my PDFs")
  })

  it("shows query suggestions and applies selection", () => {
    state.query = "comp"
    state.searchHistory = [{ query: "Compare findings across Q1 and Q2 reports" }]

    render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })

    fireEvent.focus(input)
    expect(
      screen.getByRole("option", {
        name: /Compare findings across Q1 and Q2 reports/i,
      })
    ).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("option", {
        name: /Compare findings across Q1 and Q2 reports/i,
      })
    )
    expect(state.setQuery).toHaveBeenCalledWith(
      "Compare findings across Q1 and Q2 reports"
    )
    expect(trackMetricMock).toHaveBeenCalledWith({
      type: "suggestion_accept",
      source: "history",
    })
  })

  it("disables submit when no sources selected and web fallback is off", async () => {
    state.query = "test query"
    state.settings.sources = []
    state.settings.enable_web_fallback = false

    render(<SearchBar autoFocus={false} />)

    const submit = screen.getByRole("button", { name: "Ask" })
    expect(submit).toBeDisabled()
    expect(submit).not.toHaveAttribute("title")

    fireEvent.mouseEnter(submit.parentElement!)
    expect(
      await screen.findByText("Select source categories or enable web fallback to search")
    ).toBeInTheDocument()
  })

  it("allows submit when no sources selected but web fallback is on", () => {
    state.query = "test query"
    state.settings.sources = []
    state.settings.enable_web_fallback = true

    render(<SearchBar autoFocus={false} />)

    const submit = screen.getByRole("button", { name: "Ask" })
    expect(submit).not.toBeDisabled()
  })

  it("prevents search submission when no sources and no web fallback", () => {
    state.query = "test query"
    state.settings.sources = []
    state.settings.enable_web_fallback = false

    render(<SearchBar autoFocus={false} />)

    fireEvent.submit(screen.getByRole("textbox", { name: "Search your knowledge base" }).closest("form")!)

    expect(state.search).not.toHaveBeenCalled()
  })

  it("clamps the active suggestion index when the suggestion list shrinks", () => {
    state.query = "co"
    state.searchHistory = [
      { query: "Compare findings across Q1 and Q2 reports" },
      { query: "Compare confidence intervals across studies" },
    ]

    const { rerender } = render(<SearchBar autoFocus={false} />)

    const input = screen.getByRole("textbox", {
      name: "Search your knowledge base",
    })

    fireEvent.focus(input)
    fireEvent.keyDown(input, { key: "ArrowDown" })
    fireEvent.keyDown(input, { key: "ArrowDown" })

    state.query = "q1"
    rerender(<SearchBar autoFocus={false} />)

    const remainingSuggestion = screen.getByRole("option", {
      name: /Compare findings across Q1 and Q2 reports/i,
    })
    expect(remainingSuggestion).toHaveAttribute("aria-selected", "true")

    fireEvent.keyDown(input, { key: "Enter" })

    expect(state.setQuery).toHaveBeenCalledWith(
      "Compare findings across Q1 and Q2 reports"
    )
    expect(state.search).not.toHaveBeenCalled()
  })
})
