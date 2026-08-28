import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { MessageInstance } from "antd/es/message/interface"
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

const notesResponse = (id: string, total = 1) => ({
  items: [{ id, title: id, content: `${id} content` }],
  pagination: { total_items: total, total_pages: 1 }
})

const createHarness = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return { queryClient, wrapper }
}

const renderList = (initialAuthorityScope: string | null | undefined) => {
  const { queryClient, wrapper } = createHarness()
  return renderHook(
    ({ authorityScope }: { authorityScope: string | null | undefined }) =>
      useNotesListManagement({
        authorityScope,
        isOnline: true,
        message: mocks.message as unknown as MessageInstance,
        confirmDanger: mocks.confirmDanger,
        queryClient,
        t: (key: string) => key,
        keywordTokens: [],
        setKeywordTokens: mocks.setKeywordTokens,
        notebookKeywordTokens: []
      }),
    {
      initialProps: { authorityScope: initialAuthorityScope },
      wrapper
    }
  )
}

describe("useNotesListManagement authority boundaries", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
  })

  it("does not fetch or expose list evidence while supplied authority is guarded", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [] })
    const { result } = renderList(null)

    await act(async () => Promise.resolve())

    expect(result.current.data).toBeUndefined()
    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)
    expect(
      mocks.bgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").startsWith("/api/v1/notes/?")
      )
    ).toBe(false)
  })

  it("drops account A placeholders synchronously while account B loads", async () => {
    const accountB = deferred<ReturnType<typeof notesResponse>>()
    let noteListCalls = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      noteListCalls += 1
      return noteListCalls === 1
        ? notesResponse("account-a-note", 7)
        : accountB.promise
    })
    const { result, rerender } = renderList("scope-a")
    await waitFor(() => {
      expect(result.current.rawNotes.map((note) => note.id)).toEqual([
        "account-a-note"
      ])
    })
    expect(result.current.total).toBe(7)

    rerender({ authorityScope: "scope-b" })

    expect(result.current.data).toBeUndefined()
    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)
    accountB.resolve({
      items: [],
      pagination: { total_items: 0, total_pages: 0 }
    })
    await waitFor(() => expect(noteListCalls).toBe(2))
    await waitFor(() => expect(result.current.rawNotes).toEqual([]))
    expect(result.current.total).toBe(0)
  })

  it("ignores a stale account A completion after account B becomes current", async () => {
    const accountA = deferred<ReturnType<typeof notesResponse>>()
    let noteListCalls = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      noteListCalls += 1
      if (noteListCalls === 1) return accountA.promise
      return { items: [], pagination: { total_items: 0 } }
    })
    const { result, rerender } = renderList("scope-a")
    await waitFor(() => expect(noteListCalls).toBe(1))

    rerender({ authorityScope: "scope-b" })
    await waitFor(() => expect(noteListCalls).toBe(2))
    await waitFor(() => expect(result.current.total).toBe(0))

    await act(async () => {
      accountA.resolve(notesResponse("stale-account-a-note", 99))
      await Promise.resolve()
    })

    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)
  })

  it("retains pagination placeholders only within the same authority", async () => {
    const secondPage = deferred<ReturnType<typeof notesResponse>>()
    let noteListCalls = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      noteListCalls += 1
      return noteListCalls === 1
        ? notesResponse("page-one-note", 2)
        : secondPage.promise
    })
    const { result } = renderList("scope-a")
    await waitFor(() => expect(result.current.rawNotes).toHaveLength(1))

    act(() => result.current.setPage(2))

    await waitFor(() => expect(noteListCalls).toBe(2))
    expect(result.current.isPlaceholderData).toBe(true)
    expect(result.current.rawNotes.map((note) => note.id)).toEqual([
      "page-one-note"
    ])
    secondPage.resolve(notesResponse("page-two-note", 2))
    await waitFor(() => {
      expect(result.current.rawNotes.map((note) => note.id)).toEqual([
        "page-two-note"
      ])
    })
  })
})
