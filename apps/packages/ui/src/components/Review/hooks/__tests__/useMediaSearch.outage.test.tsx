import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import { useMediaSearch } from "../useMediaSearch"
import { useConnectionStore } from "@/store/connection"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))
vi.mock("@plasmohq/storage", () => ({
  Storage: class {
    get = async () => undefined
    set = async () => undefined
  }
}))

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

beforeEach(() => {
  useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: "http://fixture.invalid" } }))
  vi.clearAllMocks()
  vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0 } })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("Media search outage handling", () => {
  it.each(["resolve", "reject"])("discards late search/type work after unmount (%s)", async outcome => {
    const pending: Array<{ resolve: (value: unknown) => void; reject: (error: Error) => void }> = []
    vi.mocked(bgRequest).mockImplementation(() => new Promise((resolve, reject) => { pending.push({ resolve, reject }) }))
    const { unmount, feedback, client } = mountSearch()
    await waitFor(() => expect(pending.length).toBeGreaterThanOrEqual(2))
    const before = vi.mocked(bgRequest).mock.calls.length
    unmount()
    await act(async () => {
      for (const request of pending.slice()) {
        if (outcome === "reject") request.reject(new Error("Retired source request"))
        else request.resolve({ items: [{ id: 1, title: "Old owner source" }], pagination: { total_items: 1 } })
      }
    })
    expect(feedback).toEqual([])
    expect(bgRequest).toHaveBeenCalledTimes(before)
    expect(JSON.stringify(client.getQueriesData({ queryKey: ["media-search"] }))).not.toContain("Old owner source")
  })

  it("cancels callbacks on a synchronous connection authority loss before React unmount", async () => {
    const rejects: Array<(error: Error) => void> = []
    vi.mocked(bgRequest).mockImplementation(() => new Promise((_resolve, reject) => { rejects.push(reject) }))
    const { feedback } = mountSearch()
    await waitFor(() => expect(bgRequest).toHaveBeenCalled())
    await act(async () => {
      useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: false } }))
      useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true } }))
      for (const reject of rejects.slice()) reject(new Error("Old authority failure"))
    })
    expect(feedback).toEqual([])
  })

  it("refreshes current catalogue criteria and page, but rejects an obsolete owned completion", async () => {
    vi.mocked(bgRequest).mockResolvedValue({ items: [{ id: 7, title: "Cedar" }], pagination: { total_items: 60, total_pages: 3 } })
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.results).toHaveLength(1))
    act(() => result.current.setQuery("cedar"))
    await waitFor(() => expect(result.current.debouncedQuery).toBe("cedar"))
    act(() => result.current.setPage(2))
    await waitFor(() => expect(result.current.isFetching).toBe(false))
    const before = vi.mocked(bgRequest).mock.calls.length
    await act(async () => window.dispatchEvent(new CustomEvent('tldw:quick-ingest-complete', { detail: { isCurrent: () => false } })))
    expect(bgRequest).toHaveBeenCalledTimes(before)
    await act(async () => window.dispatchEvent(new CustomEvent('tldw:quick-ingest-complete', { detail: { isCurrent: () => true } })))
    expect(vi.mocked(bgRequest).mock.calls.length).toBeGreaterThan(before)
    expect(result.current.query).toBe("cedar")
    expect(result.current.page).toBe(2)
    expect(vi.mocked(bgRequest).mock.calls.at(-1)?.[0]).toMatchObject({ path: '/api/v1/media/search?page=2&results_per_page=20&include_keywords=true', body: expect.objectContaining({ query: 'cedar' }) })
  })

  it("loads a replacement owner without cached content from the previous mounted lifetime", async () => {
    const previous = mountSearch()
    vi.mocked(bgRequest).mockResolvedValue({ items: [{ id: 1, title: 'Old private source' }], pagination: { total_items: 1 } })
    await act(async () => { await previous.result.current.refetch() })
    expect(previous.result.current.results[0].title).toBe('Old private source')
    previous.unmount()
    vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0 } })
    const next = renderHook(() => useMediaSearch({ t, message: { error: vi.fn(), warning: vi.fn() } }), {
      wrapper: ({ children }: React.PropsWithChildren) => <QueryClientProvider client={previous.client}>{children}</QueryClientProvider>
    })
    expect(next.result.current.results).toEqual([])
    await waitFor(() => expect(next.result.current.isFetching).toBe(false))
    expect(next.result.current.results).toEqual([])
  })

  it("keeps current requests through a connected background check", async () => {
    const resolvers: Array<(value: unknown) => void> = []
    vi.mocked(bgRequest).mockImplementation(() => new Promise(resolve => { resolvers.push(resolve) }))
    const { result, feedback } = mountSearch()
    await waitFor(() => expect(resolvers.length).toBeGreaterThanOrEqual(2))
    const data = { items: [{ id: 7, title: 'Current source' }], pagination: { total_items: 1 } }
    vi.mocked(bgRequest).mockResolvedValue(data)
    await act(async () => {
      useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, isChecking: true } }))
      for (const resolve of resolvers) resolve(data)
    })
    await waitFor(() => expect(result.current.results[0]?.title).toBe('Current source'))
    expect(feedback).toEqual([])
  })

  it("keeps a transport outage recoverable without reporting a runtime exception", async () => {
    const runtimeErrors = vi.spyOn(console, "error").mockImplementation(() => {})
    vi.spyOn(console, "warn").mockImplementation(() => {})
    const outage = Object.assign(new Error("Failed to fetch (GET /api/v1/media)"), { status: 0 })
    vi.mocked(bgRequest).mockRejectedValue(outage)
    const { result, feedback } = mountSearch()

    await waitFor(() => expect(feedback).toContain("Failed to search media"))
    expect(result.current.mediaApiUnavailable).toBe(false)
    expect(runtimeErrors).not.toHaveBeenCalled()

    vi.mocked(bgRequest).mockResolvedValue({
      items: [{ id: 7, title: "Preserved Rowan source" }],
      pagination: { total_items: 1 }
    })
    await act(async () => { await result.current.refetch() })
    expect(result.current.results.map(item => item.title)).toEqual(["Preserved Rowan source"])
  })

  it("keeps missing-endpoint guidance distinct from a transport outage", async () => {
    vi.mocked(bgRequest).mockRejectedValue(Object.assign(new Error("Not found (GET /api/v1/media/)"), { status: 404 }))
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.mediaApiUnavailable).toBe(true))
  })

  it("retains diagnostics for unexpected failures", async () => {
    const runtimeErrors = vi.spyOn(console, "error").mockImplementation(() => {})
    const unexpected = new Error("Unexpected media mapping failure")
    vi.mocked(bgRequest).mockRejectedValue(unexpected)
    const { feedback } = mountSearch()
    await waitFor(() => expect(feedback).toContain("Failed to search media"))
    expect(runtimeErrors).toHaveBeenCalledWith("Media search error:", unexpected)
  })
})
