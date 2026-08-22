import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listPresentations: vi.fn(),
  getPresentation: vi.fn(),
  navigate: vi.fn(),
  online: true
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return { ...actual, useNavigate: () => mocks.navigate }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listPresentations: (...args: unknown[]) => mocks.listPresentations(...args),
    getPresentation: (...args: unknown[]) => mocks.getPresentation(...args)
  }
}))

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => mocks.online }))

const structured = {
  id: "structured-1",
  title: "Quarterly review",
  description: null,
  theme: "white",
  content_kind: "structured_slides",
  slide_count: 6,
  created_at: "2026-08-01T00:00:00Z",
  last_modified: "2026-08-02T00:00:00Z",
  deleted: false,
  version: 2,
  provenance: { source_kind: "prompt", provider: null, model: null }
}

const standalone = {
  id: "html-1",
  title: "Architecture briefing",
  description: null,
  theme: "dark",
  content_kind: "standalone_html",
  html_slide_count: 9,
  html_bytes: 12_345,
  created_at: "2026-08-03T00:00:00Z",
  last_modified: "2026-08-04T00:00:00Z",
  deleted: false,
  version: 1,
  provenance: { source_kind: "prompt", provider: "openai", model: "gpt-5-mini" }
}

const unsupported = {
  id: "future-1",
  title: "Future deck",
  description: null,
  theme: "unknown",
  content_kind: "unsupported",
  unsupported_content_kind: "immersive_canvas",
  read_only: true,
  created_at: "2026-08-05T00:00:00Z",
  last_modified: "2026-08-05T00:00:00Z",
  deleted: false,
  version: 1,
  provenance: { source_kind: null, provider: null, model: null }
}

const page = (presentations: unknown[], offset: number, hasMore: boolean, nextOffset: number | null) => ({
  presentations,
  total: 4,
  limit: 2,
  offset,
  pagination: { mode: "offset", limit: 2, offset, total: 4, has_more: hasMore, next_offset: nextOffset },
  has_more: hasMore,
  next_offset: nextOffset
})

const loadSubject = () =>
  vi.importActual<typeof import("../PresentationStudioIndex")>(
    ["..", "PresentationStudioIndex"].join("/")
  )

describe("PresentationStudioIndex", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.online = true
  })

  it("loads source-free summaries with kind-specific metadata and a prominent New action", async () => {
    mocks.listPresentations.mockResolvedValue(page([structured, standalone, unsupported], 0, false, null))
    const { PresentationStudioIndex } = await loadSubject()
    render(<PresentationStudioIndex />)

    expect(screen.getByRole("status", { name: "Loading presentations" })).toBeInTheDocument()
    expect(await screen.findByRole("heading", { name: "Presentation Studio" })).toBeVisible()
    expect(screen.getByRole("button", { name: "New presentation" })).toBeVisible()
    expect(screen.getByText("Structured slides")).toBeVisible()
    expect(screen.getByText("6 slides")).toBeVisible()
    expect(screen.getByText("Standalone HTML + JavaScript")).toBeVisible()
    expect(screen.getByText("9 HTML slides")).toBeVisible()
    expect(screen.getByText("12.1 KB")).toBeVisible()
    expect(screen.getByText("Unsupported kind")).toBeVisible()
    expect(screen.getByText("Read only")).toBeVisible()
    expect(mocks.getPresentation).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "New presentation" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/presentation-studio/new")
  })

  it("accumulates offset pages with ID deduplication and keeps every project reachable", async () => {
    mocks.listPresentations
      .mockResolvedValueOnce(page([structured, standalone], 0, true, 2))
      .mockResolvedValueOnce(page([{ ...standalone, title: "Architecture briefing updated" }, unsupported], 2, false, null))
    const { PresentationStudioIndex } = await loadSubject()
    render(<PresentationStudioIndex />)

    await screen.findByText("Quarterly review")
    fireEvent.click(screen.getByRole("button", { name: "Load more" }))

    await screen.findByText("Future deck")
    expect(screen.getAllByText(/Architecture briefing/)).toHaveLength(1)
    expect(mocks.listPresentations).toHaveBeenNthCalledWith(1, { limit: 25, offset: 0 })
    expect(mocks.listPresentations).toHaveBeenNthCalledWith(2, { limit: 25, offset: 2 })
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Future deck" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/presentation-studio/future-1")
  })

  it("teaches from the empty state and retries a failed request", async () => {
    mocks.listPresentations
      .mockRejectedValueOnce(new Error("server unavailable"))
      .mockResolvedValueOnce(page([], 0, false, null))
    const { PresentationStudioIndex } = await loadSubject()
    render(<PresentationStudioIndex />)

    expect(await screen.findByText("Presentations could not load")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(mocks.listPresentations).toHaveBeenCalledTimes(2))
    expect(await screen.findByText("No presentations yet")).toBeVisible()
    expect(screen.getByText(/Start with direct material or create a structured deck/)).toBeVisible()
  })

  it("shows an offline recovery state without requesting the index", async () => {
    mocks.online = false
    const { PresentationStudioIndex } = await loadSubject()
    render(<PresentationStudioIndex />)

    expect(screen.getByText("Presentation Studio is offline")).toBeVisible()
    expect(screen.getByText(/Reconnect to load your presentations/)).toBeVisible()
    expect(mocks.listPresentations).not.toHaveBeenCalled()
  })
})
