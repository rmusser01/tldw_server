import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { usePdfOutline } from "../usePdfOutline"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import type { MediaNavigationNode } from "@/hooks/useMediaNavigation"

const state = vi.hoisted(() => ({ connected: true, operation: null as QuickIngestOperation | null }))
const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: transport.request }))
vi.mock("@/services/tldw", () => ({ tldwClient: {
  getDocumentOutline: (id: number) => transport.request({ path: `/api/v1/media/${id}/outline` })
} }))
vi.mock("../useDocumentOperation", () => ({ useDocumentOperation: () => state.operation }))
vi.mock("@/store/connection", () => ({ useConnectionStore: (select: (value: unknown) => unknown) =>
  select({ state: { isConnected: state.connected, mode: "normal" } }) }))

const embedded = (mediaId = 1, entries: Array<{ title: string; page: number; level: number }> = []) =>
  ({ media_id: mediaId, has_outline: entries.length > 0, entries, total_pages: 922 })
const node = (id: string, page = 352, extra: Partial<MediaNavigationNode> = {}): MediaNavigationNode => ({
  id, title: id, parent_id: null, level: 1, order: 0, path_label: null,
  target_type: "page", target_start: page, target_end: null, target_href: null,
  source: "generated_toc", confidence: 0.85, ...extra
})
const navigation = (nodes: MediaNavigationNode[], mediaId = 1) => ({
  media_id: mediaId, available: nodes.length > 0, nodes, navigation_version: "nav-v1",
  source_order_used: ["generated_toc"], stats: { node_count: nodes.length, returned_node_count: nodes.length, max_depth: 2, truncated: false }
})
const resolved = (extra = {}) => ({
  media_id: 1, node_id: "Education", navigation_version: "nav-v1", coverage_status: "exact",
  resolved_start_char: 100, resolved_end_char: 200,
  page_spans: [{ page_number: 352, printed_page_label: "319", start_char: 100, end_char: 200 }],
  ...extra
})
const owner = (revision: number): QuickIngestOperation => {
  const controller = new AbortController()
  const operation: QuickIngestOperation = {
    authorityKey: "server-a:user-1", authorityRevision: revision,
    requestScope: { userId: "1", config: { serverUrl: "https://owned.test", authMode: "multi-user" } },
    signal: controller.signal,
    isCurrent: () => state.operation === operation && !controller.signal.aborted,
    assertCurrent: () => { if (!operation.isCurrent()) throw new Error("Retired owner") }
  }
  return operation
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
let client: QueryClient
const wrapper = ({ children }: { children: React.ReactNode }) =>
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
beforeEach(() => {
  vi.clearAllMocks()
  state.connected = true
  state.operation = owner(1)
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline")
    ? embedded() : navigation([node("Education", 352)]))
})
afterEach(() => { cleanup(); client.clear() })

describe("owned PDF navigation fallback", () => {
  it("prefers embedded bookmarks and does not request generated navigation", async () => {
    transport.request.mockResolvedValue(embedded(1, [{ title: "Embedded", page: 5, level: 1 }]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.data?.items[0].title).toBe("Embedded"))
    expect(transport.request).toHaveBeenCalledTimes(1)
    expect(hook.result.current.data).toMatchObject({ documentId: 1, source: "embedded", items: [{ page: 5 }] })
  })

  it("resolves printed targets lazily to exact physical pages, retaining hierarchy and captured request scope", async () => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? embedded() : path.includes("/content?") ? resolved() : navigation([
      node("child", 320, { parent_id: "Education", level: 2 }), node("Education", 319, { title: "Education (printed 319)" })
    ]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.data?.items).toHaveLength(1))
    expect(hook.result.current.data).toMatchObject({ source: "generated", partial: false,
      items: [{ title: "Education (printed 319)", page: 0, navigationNodeId: "Education", children: [{ title: "child", page: 0 }] }] })
    expect(transport.request.mock.calls.map(([request]) => request.path)).toEqual([
      "/api/v1/media/1/outline", "/api/v1/media/1/navigation?include_generated_fallback=false"
    ])
    expect(await hook.result.current.resolvePage("Education", new AbortController().signal)).toBe(352)
    expect(transport.request.mock.calls[2][0].path).toBe("/api/v1/media/1/navigation/Education/content?format=auto&include_alternates=false")
    for (const [request] of transport.request.mock.calls) {
      expect(request.servicePromptConfig).toMatchObject({ serverUrl: "https://owned.test", expectedUserId: "1" })
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
    }
  })

  it("accepts exact page maps with source separators and blank physical pages", async () => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? embedded()
      : path.includes("/content?") ? resolved({ page_spans: [
        { page_number: 352, start_char: 100, end_char: 150 },
        { page_number: 354, start_char: 152, end_char: 198 }
      ] }) : navigation([node("Education", 319)]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.isSuccess).toBe(true))
    expect(await hook.result.current.resolvePage("Education", new AbortController().signal)).toBe(352)
  })

  it("keeps invalid targets unavailable and excludes ambiguous hierarchy", async () => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? embedded() : navigation([
      node("valid"), node("zero", 0), node("fraction", 1.5), node("past-end", 923),
      node("characters", 40, { target_type: "char_range", target_end: 50 }),
      node("duplicate", 4), node("duplicate", 8), node("duplicate-child", 9, { parent_id: "duplicate" }),
      node("cycle-a", 3, { parent_id: "cycle-b" }), node("cycle-b", 4, { parent_id: "cycle-a" }),
      node("orphan", 3, { parent_id: "missing" }), node("bad-range", 6, { target_end: 5 })
    ]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.data?.items).toHaveLength(6))
    const items = hook.result.current.data!.items
    expect(items.map(item => item.title)).toEqual(["bad-range", "characters", "fraction", "past-end", "valid", "zero"])
    expect(items.every(item => item.page === 0)).toBe(true)
    expect(items.filter(item => item.unavailable).map(item => item.title)).toEqual(["bad-range", "fraction", "zero"])
    expect(hook.result.current.data?.partial).toBe(true)
  })

  it("preserves page children of a valid non-page grouping node without guessing its own page", async () => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? embedded() : navigation([
      node("group", 0, { target_type: "char_range", target_end: 20 }), node("child", 9, { parent_id: "group", level: 2 })
    ]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.data?.items).toHaveLength(1))
    expect(hook.result.current.data).toMatchObject({ partial: false, items: [{ title: "group", page: 0, children: [{ title: "child", page: 0 }] }] })
  })

  it.each(["unavailable", "empty"])("returns honest empty data for %s", async mode => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline")
      ? embedded()
      : { ...navigation(mode === "empty" ? [] : [node("chapter")]), available: mode !== "unavailable" })
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.isSuccess).toBe(true))
    expect(hook.result.current.data?.items).toEqual([])
  })

  it.each([
    { media_id: 2 }, { node_id: "other" }, { navigation_version: "changed" }, { coverage_status: "unavailable" },
    { page_spans: [] }, { page_spans: [{ page_number: 923, start_char: 100, end_char: 200 }] },
    { page_spans: [{ page_number: 1.5, start_char: 100, end_char: 200 }] },
    { page_spans: [{ page_number: 352, start_char: 150, end_char: 200 }] },
    { resolved_end_char: 100 }
  ])("refuses unresolved or inconsistent exact page mapping %j", async invalid => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? embedded()
      : path.includes("/content?") ? resolved(invalid) : navigation([node("Education", 319)]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.isSuccess).toBe(true))
    await expect(hook.result.current.resolvePage("Education", new AbortController().signal)).rejects.toThrow()
  })

  it("refuses a page jump when the PDF's physical page count is unavailable", async () => {
    transport.request.mockImplementation(async ({ path }) => path.endsWith("/outline") ? { ...embedded(), total_pages: 0 }
      : path.includes("/content?") ? resolved() : navigation([node("Education", 319)]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.isSuccess).toBe(true))
    await expect(hook.result.current.resolvePage("Education", new AbortController().signal)).rejects.toThrow()
  })

  it.each(["outline-error", "navigation-error", "wrong-outline-owner", "wrong-navigation-owner"])("fails closed for %s", async mode => {
    transport.request.mockImplementation(async ({ path }) => {
      const outline = path.endsWith("/outline")
      if ((outline && mode === "outline-error") || (!outline && mode === "navigation-error"))
        throw Object.assign(new Error("Unavailable"), { status: 403 })
      return outline ? embedded(mode === "wrong-outline-owner" ? 2 : 1)
        : navigation([node("foreign")], mode === "wrong-navigation-owner" ? 2 : 1)
    })
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.isError).toBe(true))
    expect(hook.result.current.data).toBeUndefined()
    if (mode.startsWith("outline") || mode === "wrong-outline-owner") expect(transport.request).toHaveBeenCalledTimes(1)
  })

  it("drops a held document response when the reader switches documents", async () => {
    const pending = deferred<ReturnType<typeof embedded>>()
    transport.request.mockImplementation(async ({ path }) => path.includes("/1/") ? pending.promise
      : embedded(2, [{ title: "Second document", page: 2, level: 1 }]))
    const hook = renderHook(({ id }) => usePdfOutline(id), { wrapper, initialProps: { id: 1 } })
    await waitFor(() => expect(transport.request).toHaveBeenCalledTimes(1))
    hook.rerender({ id: 2 })
    await waitFor(() => expect(hook.result.current.data?.documentId).toBe(2))
    await act(async () => { pending.resolve(embedded(1, [{ title: "Old private", page: 1, level: 1 }])) })
    expect(hook.result.current.data?.items[0].title).toBe("Second document")
    expect(transport.request.mock.calls[0][0].abortSignal.aborted).toBe(true)
  })

  it("isolates the same Media ID across account revisions and masks unavailable owners", async () => {
    transport.request.mockResolvedValue(embedded(1, [{ title: "Old owner", page: 2, level: 1 }]))
    const hook = renderHook(() => usePdfOutline(1), { wrapper })
    await waitFor(() => expect(hook.result.current.data?.items[0].title).toBe("Old owner"))
    state.operation = null
    hook.rerender()
    expect(hook.result.current.data).toBeUndefined()
    const held = deferred<ReturnType<typeof embedded>>()
    transport.request.mockReturnValue(held.promise)
    state.operation = owner(2)
    hook.rerender()
    expect(hook.result.current.data).toBeUndefined()
    await act(async () => { held.resolve(embedded(1, [{ title: "New owner", page: 7, level: 1 }])) })
    await waitFor(() => expect(hook.result.current.data?.items[0].title).toBe("New owner"))
    state.connected = false
    hook.rerender()
    expect(hook.result.current.data).toBeUndefined()
  })
})
