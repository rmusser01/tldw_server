// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, opts?: { defaultValue?: string } | string) => {
      if (typeof opts === "string") return opts
      return opts?.defaultValue ?? _key
    }
  })
}))

import { ResultsList } from "../ResultsList"

const emptyProps = {
  results: [] as any[],
  selectedId: null,
  onSelect: vi.fn(),
  totalCount: 0,
  loadedCount: 0,
  isLoading: false,
  hasActiveFilters: false,
  searchQuery: "",
  onOpenQuickIngest: vi.fn()
}

describe("ResultsList FTUX", () => {
  beforeEach(() => {
    localStorage.clear()
    emptyProps.onOpenQuickIngest = vi.fn()
  })

  it("shows first-ingest tutorial when library is empty", () => {
    render(<ResultsList {...emptyProps} />)
    expect(screen.getByTestId("first-ingest-tutorial")).toBeInTheDocument()
    expect(screen.getByText(/Get started/i)).toBeInTheDocument()
  })

  it("has inline URL input and Ingest button", () => {
    render(<ResultsList {...emptyProps} />)
    expect(screen.getByPlaceholderText(/youtube/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /ingest/i })).toBeInTheDocument()
  })

  it("passes typed URL to onOpenQuickIngest when clicking Ingest", () => {
    render(<ResultsList {...emptyProps} />)
    const input = screen.getByPlaceholderText(/youtube/i)
    fireEvent.change(input, { target: { value: "https://youtube.com/watch?v=test" } })
    fireEvent.click(screen.getByRole("button", { name: /ingest/i }))
    expect(emptyProps.onOpenQuickIngest).toHaveBeenCalledWith(
      expect.objectContaining({ source: "https://youtube.com/watch?v=test" })
    )
  })

  it("calls onOpenQuickIngest without source when URL is empty", () => {
    render(<ResultsList {...emptyProps} />)
    fireEvent.click(screen.getByRole("button", { name: /ingest/i }))
    expect(emptyProps.onOpenQuickIngest).toHaveBeenCalledWith()
  })

  it("hides tutorial after dismiss and shows fallback with Quick Ingest button", () => {
    render(<ResultsList {...emptyProps} />)
    fireEvent.click(screen.getByText(/skip/i))
    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
    // Should now show the regular empty state with Quick Ingest button
    expect(screen.getByRole("button", { name: /quick ingest/i })).toBeInTheDocument()
  })

  it("does not show tutorial when filters are active", () => {
    render(<ResultsList {...emptyProps} hasActiveFilters />)
    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
  })

  it("does not show tutorial when search query is present", () => {
    render(<ResultsList {...emptyProps} searchQuery="something" />)
    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
  })

  it("persists dismissal to localStorage", () => {
    render(<ResultsList {...emptyProps} />)
    fireEvent.click(screen.getByText(/skip/i))
    expect(localStorage.getItem("tldw:media:first-ingest-dismissed")).toBe("true")
  })

  it("migrates the legacy tutorial dismissal key on read", () => {
    localStorage.setItem("tldw_first_ingest_tutorial_dismissed", "true")

    render(<ResultsList {...emptyProps} />)

    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
    expect(localStorage.getItem("tldw:media:first-ingest-dismissed")).toBe("true")
    expect(localStorage.getItem("tldw_first_ingest_tutorial_dismissed")).toBeNull()
  })
})
