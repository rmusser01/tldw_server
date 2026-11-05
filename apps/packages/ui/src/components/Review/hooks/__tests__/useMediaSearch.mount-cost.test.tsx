import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import { useMediaSearch } from "../useMediaSearch"
import { useConnectionStore } from "@/store/connection"

// W4 Stage 3 (WP-43) pins: one media + one notes request per search, facets
// from the search response itself, and concurrent media/notes dispatch.
vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))
vi.mock("@plasmohq/storage", () => ({
  Storage: class {
    get = async () => undefined
    set = async () => undefined
  }
}))

const MEDIA_LISTING_PATH = "/api/v1/media/?page=1&results_per_page=20&include_keywords=true"
const NOTES_LISTING_PATH = "/api/v1/notes/?page=1&results_per_page=20"
// The retired type-discovery fan-out fetched /api/v1/media/ pages 1-3 at
// results_per_page=50 before/alongside the first search.
const DISCOVERY_PATH = "/api/v1/media/?page=1&results_per_page=50"

const t = (key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? key)

const mountSearch = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const feedback: string[] = []
  const message = {
    error: (text: string) => feedback.push(text),
    warning: (text: string) => feedback.push(text)
  }
  const hook = renderHook(() => useMediaSearch({ t, message }), {
    wrapper: ({ children }: React.PropsWithChildren) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  })
  return { ...hook, feedback, client }
}

const callPaths = () =>
  vi.mocked(bgRequest).mock.calls.map((call) => String(call?.[0]?.path || ""))

beforeEach(() => {
  useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: "http://fixture.invalid" } }))
  vi.clearAllMocks()
  vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0, total_pages: 1 } })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("Media search mount cost (WP-43)", () => {
  it("mounts with exactly one media request (no duplicate search, no type-discovery pages)", async () => {
    vi.mocked(bgRequest).mockResolvedValue({
      items: [{ id: 1, title: "Rowan", type: "pdf" }],
      pagination: { total_items: 1, total_pages: 1 }
    })
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.results).toHaveLength(1))
    await act(async () => {})
    expect(callPaths()).toEqual([MEDIA_LISTING_PATH])
    expect(callPaths().some((path) => path === DISCOVERY_PATH)).toBe(false)
  })

  it("derives media-type facets from the first search response instead of discovery pages", async () => {
    vi.mocked(bgRequest).mockImplementation(async (request) => {
      const path = String((request as { path?: string })?.path || "")
      if (path.includes("results_per_page=50")) {
        return {
          items: [
            { id: 50, title: "Discovery only", type: "video" },
            { id: 51, title: "Discovery only", type: "audio" }
          ],
          pagination: { total_items: 2, total_pages: 3 }
        }
      }
      return {
        items: [
          { id: 1, title: "Rowan", type: "pdf" },
          { id: 2, title: "Cedar", type: "pdf" }
        ],
        pagination: { total_items: 2, total_pages: 1 }
      }
    })
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.results).toHaveLength(2))
    await act(async () => {})
    expect(result.current.availableMediaTypes).toEqual(["pdf"])
  })

  it("keeps the facet list empty when the first search response has no items", async () => {
    vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0, total_pages: 1 } })
    const { result } = mountSearch()
    await waitFor(() => expect(bgRequest).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(result.current.availableMediaTypes).toEqual([])
    expect(bgRequest).toHaveBeenCalledTimes(1)
  })

  it("issues exactly one media and one notes request per search and runs them concurrently", async () => {
    vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0, total_pages: 1 } })
    const { result } = mountSearch()
    await waitFor(() => expect(bgRequest).toHaveBeenCalledTimes(1))
    // Enabling notes is a criteria change: exactly one media + one notes
    // request must follow (the mount search already counted above).
    await act(async () => { result.current.setKinds({ media: true, notes: true }) })
    await waitFor(() => expect(bgRequest).toHaveBeenCalledTimes(3))
    expect(callPaths().slice(1).sort()).toEqual([MEDIA_LISTING_PATH, NOTES_LISTING_PATH].sort())

    // Deferred-promise proof: the notes request must be in flight before the
    // media request resolves (and vice versa) within a single search.
    vi.mocked(bgRequest).mockClear()
    const pending: Array<{ path: string; resolve: (value: unknown) => void }> = []
    vi.mocked(bgRequest).mockImplementation(
      (request) =>
        new Promise((resolve) => {
          pending.push({ path: String((request as { path?: string })?.path || ""), resolve })
        })
    )
    await act(async () => { void result.current.refetch() })
    await waitFor(() => expect(pending.length).toBe(2))
    expect(pending.map((entry) => entry.path).sort()).toEqual([MEDIA_LISTING_PATH, NOTES_LISTING_PATH].sort())
    await act(async () => {
      for (const entry of pending) {
        entry.resolve(
          entry.path.startsWith("/api/v1/notes")
            ? { items: [{ id: 9, title: "Note 9", content: "note body" }], pagination: { total_items: 1 } }
            : { items: [{ id: 1, title: "Rowan", type: "pdf" }], pagination: { total_items: 1 } }
        )
      }
    })
    await waitFor(() =>
      expect(result.current.results.map((item) => item.title).sort()).toEqual(["Note 9", "Rowan"])
    )
    expect(bgRequest).toHaveBeenCalledTimes(2)
  })

  it("issues exactly one page-2 listing request on an explicit page change (arrow-key pagination path)", async () => {
    // Arrow keys page through results via setPage when page < totalPages;
    // ViewMediaPage-level harnesses mock the search query away, so the
    // request behavior is pinned here against the real query pipeline.
    vi.mocked(bgRequest).mockResolvedValue({
      items: [{ id: 1, title: "Rowan", type: "pdf" }],
      pagination: { total_items: 40, total_pages: 2 }
    })
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.results).toHaveLength(1))
    await waitFor(() => expect(result.current.totalPages).toBe(2))
    vi.mocked(bgRequest).mockClear()
    await act(async () => { result.current.setPage(2) })
    await waitFor(() =>
      expect(callPaths().some((path) => path.startsWith("/api/v1/media/?page=2&results_per_page=20"))).toBe(true)
    )
    await act(async () => {})
    expect(callPaths().some((path) => path.startsWith("/api/v1/media/?page=0"))).toBe(false)
  })
})
