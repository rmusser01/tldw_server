import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import { useMediaSearch } from "../useMediaSearch"
import { useConnectionStore } from "@/store/connection"

// #3149: Review's notes results must use the notes API's real contract, as the
// /notes page does since NL-01. The mock below serves notes the way
// GET /api/v1/notes/ and GET /api/v1/notes/search/ do: limit/offset paging
// (they ignore page/results_per_page), exact case-insensitive tokens, and the
// total in pagination.total. There is no POST /api/v1/notes/search/.

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))
vi.mock("@plasmohq/storage", () => ({
  Storage: class {
    get = async () => undefined
    set = async () => undefined
  }
}))

type ServedNote = { id: string; title: string; content: string; keywords: Array<{ keyword: string }> }

const NOTES: ServedNote[] = Array.from({ length: 105 }, (_, index) => ({
  id: `note-${index + 1}`,
  title: `Note ${index + 1}`,
  content: `Body ${index + 1}`,
  keywords: [{ keyword: index % 3 === 0 ? "Research" : "Researcher" }]
}))

const notesPage = (rows: ServedNote[], limit: number, offset: number) => ({
  items: rows.slice(offset, offset + limit),
  count: Math.min(limit, Math.max(0, rows.length - offset)),
  limit,
  offset,
  total: rows.length,
  pagination: { limit, offset, total: rows.length, has_more: offset + limit < rows.length }
})

const serveNotes = (path: string, method: string) => {
  const url = new URL(path, "http://fixture.invalid")
  if (method !== "GET") {
    throw Object.assign(new Error(`Method Not Allowed (${method} ${url.pathname})`), { status: 405 })
  }
  const search = url.pathname === "/api/v1/notes/search/"
  const limit = Number(url.searchParams.get("limit") ?? (search ? 10 : 100))
  const offset = Number(url.searchParams.get("offset") ?? 0)
  if (!search) return notesPage(NOTES, limit, offset)
  const query = (url.searchParams.get("query") ?? "").toLowerCase()
  const tokens = url.searchParams.getAll("tokens").map(token => token.toLowerCase())
  const rows = NOTES.filter(note =>
    (!query || `${note.title} ${note.content}`.toLowerCase().includes(query)) &&
    tokens.every(token => note.keywords.some(({ keyword }) => keyword.toLowerCase() === token))
  )
  return notesPage(rows, limit, offset)
}

const t = (key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? key)

const mountNotesSearch = async () => {
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
  await waitFor(() => expect(hook.result.current.isFetching).toBe(false))
  act(() => hook.result.current.setKinds({ media: false, notes: true }))
  await waitFor(() => expect(hook.result.current.results.length).toBeGreaterThan(0))
  await waitFor(() => expect(hook.result.current.isFetching).toBe(false))
  return { ...hook, feedback }
}

const notesRequests = () =>
  vi.mocked(bgRequest).mock.calls
    .map(([request]) => request as { path?: string; method?: string })
    .filter(request => String(request.path ?? "").startsWith("/api/v1/notes"))

const lastNotesRequest = () => {
  const request = notesRequests().at(-1)
  return new URL(String(request?.path), "http://fixture.invalid")
}

const titles = (results: Array<{ title?: string }>) => results.map(item => item.title)

beforeEach(() => {
  useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: "http://fixture.invalid" } }))
  vi.clearAllMocks()
  vi.mocked(bgRequest).mockImplementation(async (request: any) => {
    const path = String(request?.path ?? "")
    if (path.startsWith("/api/v1/notes")) return serveNotes(path, String(request?.method ?? "GET").toUpperCase())
    return { items: [], pagination: { total_items: 0 } }
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("Review notes results follow the notes API contract (#3149)", () => {
  it("pages the notes list with limit/offset and reads pagination.total", async () => {
    const { result } = await mountNotesSearch()

    expect(result.current.notesTotal).toBe(105)
    expect(titles(result.current.results)).toEqual(NOTES.slice(0, 20).map(note => note.title))
    const first = lastNotesRequest()
    expect(first.pathname).toBe("/api/v1/notes/")
    expect(first.searchParams.get("limit")).toBe("20")
    expect(first.searchParams.get("offset")).toBe("0")
    expect(first.searchParams.has("page")).toBe(false)
    expect(first.searchParams.has("results_per_page")).toBe(false)

    act(() => result.current.setPage(2))
    await waitFor(() => expect(titles(result.current.results)).toEqual(NOTES.slice(20, 40).map(note => note.title)))
    expect(lastNotesRequest().searchParams.get("offset")).toBe("20")
    expect(result.current.notesTotal).toBe(105)
  })

  it("searches with one GET, sends tag tokens, and keeps the server's exact tag matches", async () => {
    const { result, feedback } = await mountNotesSearch()
    vi.mocked(bgRequest).mockClear()

    act(() => {
      result.current.setQuery("body")
      result.current.setKeywordTokens(["research"])
    })
    await waitFor(() => expect(result.current.debouncedQuery).toBe("body"))
    await waitFor(() => expect(result.current.isFetching).toBe(false))
    await waitFor(() => expect(result.current.notesTotal).toBe(35))

    expect(notesRequests().every(request => String(request.method ?? "GET").toUpperCase() === "GET")).toBe(true)
    const search = lastNotesRequest()
    expect(search.pathname).toBe("/api/v1/notes/search/")
    expect(search.searchParams.get("query")).toBe("body")
    expect(search.searchParams.getAll("tokens")).toEqual(["research"])
    expect(search.searchParams.get("tokens_match")).toBe("all")
    expect(search.searchParams.get("limit")).toBe("20")
    expect(search.searchParams.get("offset")).toBe("0")
    // "Researcher" contains "research" but is another tag.
    expect(result.current.results.every(item => item.keywords?.includes("Research"))).toBe(true)
    expect(result.current.results).toHaveLength(20)
    expect(feedback).toEqual([])
  })

  it("filters by tags alone through the search endpoint", async () => {
    const { result } = await mountNotesSearch()

    act(() => result.current.setKeywordTokens(["Research"]))
    await waitFor(() => expect(result.current.notesTotal).toBe(35))

    const search = lastNotesRequest()
    expect(search.pathname).toBe("/api/v1/notes/search/")
    expect(search.searchParams.has("query")).toBe(false)
    expect(search.searchParams.getAll("tokens")).toEqual(["Research"])
  })
})
