/**
 * NL-07: a text search keeps the server's relevance order ("Best match"), and
 * each row carries the server's match snippet.
 *
 * The list used to re-sort each search page by date, so the best title match
 * for "sourdough" rendered last, and body-only hits showed the first line.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useNotesListManagement } from "../useNotesListManagement"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  confirmDanger: vi.fn(),
  setKeywordTokens: vi.fn(),
  message: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting
  }
})

const SNIPPET = [
  { text: "…A friend mentioned ", match: false },
  { text: "sourdough", match: true },
  { text: " once.", match: false }
]

// Server relevance order: the oldest note is the best match.
const searchRows = [
  { id: "loaf", title: "Sourdough country loaf", content: "Feed the starter", last_modified: "2026-01-01T00:00:00Z", snippet: null },
  { id: "apple", title: "Apple pie", content: "Notes", last_modified: "2026-03-01T00:00:00Z", snippet: SNIPPET },
  { id: "bad", title: "Bad snippet", content: "Notes", last_modified: "2026-02-01T00:00:00Z", snippet: [{ text: 3 }] }
]

const searchPaths = () =>
  mocks.bgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter((path) => path.startsWith("/api/v1/notes/search/?"))

const lastSearchParams = () => {
  const paths = searchPaths()
  return new URLSearchParams(paths[paths.length - 1].split("?")[1])
}

const renderList = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return renderHook(
    () =>
      useNotesListManagement({
        isOnline: true,
        message: mocks.message as any,
        confirmDanger: mocks.confirmDanger,
        queryClient,
        t: (key: string) => key,
        keywordTokens: [],
        setKeywordTokens: mocks.setKeywordTokens,
        notebookKeywordTokens: []
      }),
    { wrapper }
  )
}

describe("useNotesListManagement Best match (NL-07)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/search/?")) {
        return { items: searchRows, count: 3, total: 3, pagination: { total: 3 } }
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], total: 0, pagination: { total: 0 } }
      }
      return {}
    })
  })

  it("sorts a text search by best match and keeps the server's order", async () => {
    const { result } = renderList()
    expect(result.current.sortOption).toBe("modified_desc")

    act(() => result.current.setQuery("sourdough"))

    expect(result.current.sortOption).toBe("relevance")
    await waitFor(() => expect(result.current.rawNotes.map((note) => note.id)).toEqual(["loaf", "apple", "bad"]))
    expect(lastSearchParams().get("sort_by")).toBe("relevance")
  })

  it("carries the match snippet and drops a malformed one", async () => {
    const { result } = renderList()
    act(() => result.current.setQuery("sourdough"))

    await waitFor(() => expect(result.current.rawNotes).toHaveLength(3))
    const byId = new Map(result.current.rawNotes.map((note) => [note.id, note]))
    expect(byId.get("apple")?.snippet).toEqual(SNIPPET)
    expect(byId.get("loaf")?.snippet).toBeNull()
    expect(byId.get("bad")?.snippet).toBeNull()
  })

  it("lets a search pick another sort, then returns to the list's sort", async () => {
    const { result } = renderList()
    act(() => result.current.setSortOption("title_asc"))
    act(() => result.current.setQuery("sourdough"))
    expect(result.current.sortOption).toBe("relevance")

    act(() => result.current.setSortOption("modified_desc"))
    expect(result.current.sortOption).toBe("modified_desc")
    await waitFor(() => expect(lastSearchParams().get("sort_by")).toBe("last_modified"))
    // A date-sorted search is still sorted by date on the client.
    await waitFor(() => expect(result.current.rawNotes.map((note) => note.id)).toEqual(["apple", "bad", "loaf"]))

    act(() => result.current.setQuery(""))
    expect(result.current.sortOption).toBe("title_asc")

    // The next search starts on Best match again.
    act(() => result.current.setQuery("loaf"))
    expect(result.current.sortOption).toBe("relevance")
  })

  it("gathers every match without snippets; the list still asks for them", async () => {
    const { result } = renderList()
    act(() => result.current.setQuery("sourdough"))
    await waitFor(() => expect(result.current.rawNotes).toHaveLength(3))
    expect(lastSearchParams().has("include_snippets")).toBe(false)
    const listRequests = searchPaths().length

    await act(async () => {
      result.current.bulkSelectionControls.onSelectAllMatching()
    })

    await waitFor(() => expect(result.current.bulkSelectionControls.matching).not.toBeNull())
    const gathered = searchPaths()
      .slice(listRequests)
      .map((path) => new URLSearchParams(path.split("?")[1]))
    expect(gathered.length).toBeGreaterThan(0)
    expect(gathered.every((params) => params.get("include_snippets") === "false")).toBe(true)
  })

  it("starts a new 'all matching' selection when the search's sort changes", async () => {
    const { result } = renderList()
    act(() => result.current.setQuery("sourdough"))
    await waitFor(() => expect(result.current.rawNotes).toHaveLength(3))
    await act(async () => {
      result.current.bulkSelectionControls.onSelectAllMatching()
    })
    await waitFor(() => expect(result.current.bulkSelectionControls.matching).not.toBeNull())

    // Best match to Title is another order of another gather.
    act(() => result.current.setSortOption("title_asc"))

    expect(result.current.bulkSelectionControls.matching).toBeNull()
  })
})
