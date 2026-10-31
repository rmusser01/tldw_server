import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { MessageInstance } from "antd/es/message/interface"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesListManagement } from "../useNotesListManagement"

// NS-04 follow-up: while the next list loads, the previous rows may stand in
// only for the same kind of list. Active notes must never stand in for Trash
// (or the reverse), and one collection's notes never stand in for another's.

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
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting
  }
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

const notesResponse = (id: string) => ({
  items: [{ id, title: id, content: `${id} content` }],
  pagination: { total_items: 1, total_pages: 1 },
  total: 1
})

const moodboardNotesResponse = (id: string) => ({
  notes: [{ id, title: id, content: `${id} content` }],
  total: 1
})

const renderList = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return renderHook(
    () =>
      useNotesListManagement({
        authorityScope: "scope-a",
        isOnline: true,
        message: mocks.message as unknown as MessageInstance,
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

const rowIds = (rows: Array<{ id: string | number }>) => rows.map((row) => String(row.id))

const callCount = (prefix: string) =>
  mocks.bgRequest.mock.calls.filter(([request]) =>
    String(request?.path || "").startsWith(prefix)
  ).length

describe("useNotesListManagement NS-04 placeholder rows stay within one list", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
  })

  it("does not show active notes while Trash loads", async () => {
    const trash = deferred<ReturnType<typeof notesResponse>>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/trash?")) return trash.promise
      if (path.startsWith("/api/v1/notes/?")) return notesResponse("active-note")
      return {}
    })
    const { result } = renderList()
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))

    act(() => result.current.setListMode("trash"))

    await waitFor(() => expect(callCount("/api/v1/notes/trash?")).toBe(1))
    expect(result.current.isFetching).toBe(true)
    expect(rowIds(result.current.rawNotes)).toEqual([])

    trash.resolve(notesResponse("trashed-note"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["trashed-note"]))
  })

  it("does not show trashed notes while the active list loads", async () => {
    let activeCalls = 0
    const secondActive = deferred<ReturnType<typeof notesResponse>>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/trash?")) return notesResponse("trashed-note")
      if (path.startsWith("/api/v1/notes/?")) {
        activeCalls += 1
        return activeCalls === 1 ? notesResponse("active-note") : secondActive.promise
      }
      return {}
    })
    const { result, unmount } = renderList()
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))
    act(() => result.current.setListMode("trash"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["trashed-note"]))

    // Make the cached active list stale so returning to it refetches; the
    // cached active rows are the right kind of list and may be shown.
    act(() => result.current.setSortOption("title_asc"))
    await waitFor(() => expect(callCount("/api/v1/notes/trash?")).toBe(2))
    await waitFor(() => expect(result.current.isFetching).toBe(false))
    act(() => result.current.setListMode("active"))

    await waitFor(() => expect(activeCalls).toBe(2))
    expect(rowIds(result.current.rawNotes)).not.toContain("trashed-note")

    secondActive.resolve(notesResponse("active-note"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))
    unmount()
  })

  it("does not show one collection's notes while another collection loads", async () => {
    const secondBoard = deferred<ReturnType<typeof moodboardNotesResponse>>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/moodboards?")) {
        return {
          moodboards: [
            { id: 1, name: "Board one", version: 1 },
            { id: 2, name: "Board two", version: 1 }
          ],
          total: 2
        }
      }
      if (path.startsWith("/api/v1/notes/moodboards/1/notes?")) {
        return moodboardNotesResponse("board-one-note")
      }
      if (path.startsWith("/api/v1/notes/moodboards/2/notes?")) return secondBoard.promise
      if (path.startsWith("/api/v1/notes/?")) return notesResponse("active-note")
      return {}
    })
    const { result } = renderList()
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))

    act(() => result.current.setListViewMode("moodboard"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["board-one-note"]))

    act(() => result.current.setSelectedMoodboardId(2))

    await waitFor(() => expect(callCount("/api/v1/notes/moodboards/2/notes?")).toBe(1))
    expect(result.current.isFetching).toBe(true)
    expect(rowIds(result.current.rawNotes)).toEqual([])

    secondBoard.resolve(moodboardNotesResponse("board-two-note"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["board-two-note"]))
  })

  it("does not show the active list while a collection loads", async () => {
    const board = deferred<ReturnType<typeof moodboardNotesResponse>>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/moodboards?")) {
        return { moodboards: [{ id: 1, name: "Board one", version: 1 }], total: 1 }
      }
      if (path.startsWith("/api/v1/notes/moodboards/1/notes?")) return board.promise
      if (path.startsWith("/api/v1/notes/?")) return notesResponse("active-note")
      return {}
    })
    const { result } = renderList()
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))

    act(() => result.current.setListViewMode("moodboard"))

    await waitFor(() => expect(callCount("/api/v1/notes/moodboards/1/notes?")).toBe(1))
    expect(rowIds(result.current.rawNotes)).not.toContain("active-note")

    board.resolve(moodboardNotesResponse("board-one-note"))
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["board-one-note"]))
  })

  it("still keeps the current rows while a search in the same list loads", async () => {
    let listCalls = 0
    const searched = deferred<unknown>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/search/")) {
        listCalls += 1
        return searched.promise
      }
      if (path.startsWith("/api/v1/notes/?")) return notesResponse("active-note")
      return {}
    })
    const { result } = renderList()
    await waitFor(() => expect(rowIds(result.current.rawNotes)).toEqual(["active-note"]))

    act(() => result.current.setQuery("plans"))

    await waitFor(() => expect(listCalls).toBe(1))
    expect(result.current.isPlaceholderData).toBe(true)
    expect(rowIds(result.current.rawNotes)).toEqual(["active-note"])
    searched.resolve({ items: [], pagination: { total_items: 0 } })
  })
})
