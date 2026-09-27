import { NOTES_NOTEBOOKS_SETTING } from "@/services/settings/ui-settings"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { MessageInstance } from "antd/es/message/interface"
import React from "react"
import { createRoot } from "react-dom/client"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesListManagement } from "../useNotesListManagement"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  confirmDanger: vi.fn(),
  promptModal: vi.fn(),
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

vi.mock("../../notes-manager-utils", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../../notes-manager-utils")>()
  return {
    ...actual,
    promptModal: mocks.promptModal
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

const notebookResponse = (id: number, name: string) => ({
  collections: [
    { id, name, keywords: [name.toLowerCase().replaceAll(" ", "-")] }
  ],
  pagination: { total_items: 1 }
})

const moodboardResponse = (id: number, name: string) => ({
  moodboards: [{ id, name, version: 1 }],
  total: 1
})

const notesRequestPaths = () =>
  mocks.bgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter((path) => path.startsWith("/api/v1/notes"))

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

const renderList = (
  initialAuthorityScope: string | null | undefined,
  initialNotebookKeywordTokens: string[] = []
) => {
  const { queryClient, wrapper } = createHarness()
  return renderHook(
    ({
      authorityScope,
      notebookKeywordTokens
    }: {
      authorityScope: string | null | undefined
      notebookKeywordTokens?: string[]
    }) =>
      useNotesListManagement({
        authorityScope,
        isOnline: true,
        message: mocks.message as unknown as MessageInstance,
        confirmDanger: mocks.confirmDanger,
        queryClient,
        t: (key: string) => key,
        keywordTokens: [],
        setKeywordTokens: mocks.setKeywordTokens,
        notebookKeywordTokens: notebookKeywordTokens ?? []
      }),
    {
      initialProps: {
        authorityScope: initialAuthorityScope,
        notebookKeywordTokens: initialNotebookKeywordTokens
      },
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
    mocks.promptModal.mockResolvedValue("Renamed collection")
  })

  it("blocks every Notes discovery path and imperative refetch while supplied authority is guarded", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [] })
    const { result } = renderList(null)

    await act(async () => {
      await Promise.resolve()
      await result.current.refetch()
      await result.current.fetchFilteredNotesRaw("guarded", [], 1, 20)
    })
    act(() => {
      result.current.setListMode("trash")
    })
    await act(async () => {
      await Promise.resolve()
      await result.current.refetch()
    })
    act(() => {
      result.current.setListMode("active")
      result.current.setListViewMode("moodboard")
      result.current.setSelectedMoodboardId(17)
    })
    await act(async () => {
      await Promise.resolve()
      await result.current.refetch()
    })

    expect(result.current.data).toBeUndefined()
    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)
    expect(result.current.moodboards).toEqual([])
    expect(result.current.notebookOptions).toEqual([])
    expect(notesRequestPaths()).toEqual([])
  })

  it("hides all account A discovery evidence synchronously through null and empty account B", async () => {
    let account: "a" | "b" = "a"
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (account === "b") {
        if (path.startsWith("/api/v1/notes/collections")) {
          return { collections: [], pagination: { total_items: 0 } }
        }
        if (path.startsWith("/api/v1/notes/moodboards?")) {
          return { moodboards: [], total: 0 }
        }
        return { items: [], pagination: { total_items: 0 } }
      }
      if (path.startsWith("/api/v1/notes/collections")) {
        return notebookResponse(10, "Account A notebook")
      }
      if (path.startsWith("/api/v1/notes/moodboards?")) {
        return moodboardResponse(20, "Account A moodboard")
      }
      if (path.includes("/moodboards/20/notes?")) {
        return { items: notesResponse("account-a-note").items, total: 1 }
      }
      return notesResponse("account-a-note", 7)
    })
    const { result, rerender } = renderList("scope-a")
    await waitFor(() => expect(result.current.notebookOptions).toHaveLength(1))
    act(() => {
      result.current.setSelectedNotebookId(10)
      result.current.setListViewMode("moodboard")
    })
    await waitFor(() => expect(result.current.moodboards).toHaveLength(1))
    await waitFor(() => expect(result.current.rawNotes).toHaveLength(1))

    rerender({ authorityScope: null })

    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)
    expect(result.current.moodboards).toEqual([])
    expect(result.current.selectedMoodboardId).toBeNull()
    expect(result.current.selectedMoodboard).toBeNull()
    expect(result.current.notebookOptions).toEqual([])
    expect(result.current.selectedNotebookId).toBeNull()
    expect(result.current.selectedNotebook).toBeNull()

    account = "b"
    rerender({ authorityScope: "scope-b" })

    expect(result.current.rawNotes).toEqual([])
    expect(result.current.moodboards).toEqual([])
    expect(result.current.notebookOptions).toEqual([])
    expect(result.current.selectedMoodboardId).toBeNull()
    expect(result.current.selectedNotebookId).toBeNull()
    await waitFor(() => {
      expect(result.current.rawNotes).toEqual([])
      expect(result.current.moodboards).toEqual([])
      expect(result.current.notebookOptions).toEqual([])
    })
  })

  it("does not carry account A notebook-derived filters into account B discovery", async () => {
    mocks.bgRequest.mockResolvedValue({
      items: [],
      pagination: { total_items: 0 }
    })
    const { result, rerender } = renderList("scope-a", ["account-a"])
    act(() => {
      result.current.setNotebookOptions([
        { id: 7, name: "Account A notebook", keywords: ["account-a"] }
      ])
      result.current.setSelectedNotebookId(7)
    })
    await waitFor(() => {
      expect(
        notesRequestPaths().some((path) => path.includes("account-a"))
      ).toBe(true)
    })
    mocks.bgRequest.mockClear()

    rerender({
      authorityScope: "scope-b",
      notebookKeywordTokens: ["account-a"]
    })

    await waitFor(() => expect(notesRequestPaths().length).toBeGreaterThan(0))
    expect(notesRequestPaths().some((path) => path.includes("account-a"))).toBe(
      false
    )
  })

  it("ignores deferred A discovery and never migrates global local notebooks into scoped B", async () => {
    const accountACollections = deferred<{
      collections: never[]
      pagination: { total_items: number }
    }>()
    const accountAMoodboards = deferred<ReturnType<typeof moodboardResponse>>()
    let collectionGetCalls = 0
    let moodboardGetCalls = 0
    mocks.getSetting.mockImplementation(async (setting) => {
      if (setting !== NOTES_NOTEBOOKS_SETTING) return null
      return [{ id: 5, name: "Account A local", keywords: ["account-a"] }]
    })
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        const path = String(request.path || "")
        const method = String(request.method || "GET").toUpperCase()
        if (path.startsWith("/api/v1/notes/collections") && method === "GET") {
          collectionGetCalls += 1
          return collectionGetCalls === 1
            ? accountACollections.promise
            : { collections: [], pagination: { total_items: 0 } }
        }
        if (path.startsWith("/api/v1/notes/collections")) {
          return { id: 5, name: "Account A local", keywords: ["account-a"] }
        }
        if (path.startsWith("/api/v1/notes/moodboards?")) {
          moodboardGetCalls += 1
          return moodboardGetCalls === 1
            ? accountAMoodboards.promise
            : { moodboards: [], total: 0 }
        }
        return { items: [], pagination: { total_items: 0 } }
      }
    )
    const { result, rerender } = renderList("scope-a")
    act(() => result.current.setListViewMode("moodboard"))
    await waitFor(() => {
      expect(collectionGetCalls).toBe(1)
      expect(moodboardGetCalls).toBe(1)
    })

    rerender({ authorityScope: null })
    rerender({ authorityScope: "scope-b" })

    await act(async () => {
      accountACollections.resolve({
        collections: [],
        pagination: { total_items: 0 }
      })
      accountAMoodboards.resolve(moodboardResponse(20, "Account A moodboard"))
      await Promise.resolve()
    })
    await waitFor(() => expect(collectionGetCalls).toBeGreaterThanOrEqual(2))

    expect(result.current.moodboards).toEqual([])
    expect(result.current.notebookOptions).toEqual([])
    expect(result.current.selectedMoodboardId).toBeNull()
    expect(result.current.selectedNotebookId).toBeNull()
    expect(
      mocks.bgRequest.mock.calls.filter(([request]) => {
        const path = String(request?.path || "")
        const method = String(request?.method || "GET").toUpperCase()
        return path.startsWith("/api/v1/notes/collections") && method !== "GET"
      })
    ).toEqual([])
    expect(
      mocks.setSetting.mock.calls.some(([, value]) =>
        JSON.stringify(value).includes("Account A local")
      )
    ).toBe(false)
    expect(
      mocks.getSetting.mock.calls.filter(
        ([setting]) => setting === NOTES_NOTEBOOKS_SETTING
      )
    ).toEqual([])
  })

  it("rejects retained account A collection mutation callbacks after authority is guarded", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) {
        return notebookResponse(10, "Account A notebook")
      }
      if (path.startsWith("/api/v1/notes/moodboards?")) {
        return moodboardResponse(20, "Account A moodboard")
      }
      return notesResponse("account-a-note")
    })
    const { result, rerender } = renderList("scope-a")
    await waitFor(() => expect(result.current.notebookOptions).toHaveLength(1))
    act(() => {
      result.current.setSelectedNotebookId(10)
      result.current.setListViewMode("moodboard")
    })
    await waitFor(() => expect(result.current.selectedMoodboardId).toBe(20))
    const renameAccountAMoodboard = result.current.renameMoodboard
    const removeAccountANotebook = result.current.removeSelectedNotebook

    rerender({ authorityScope: null })
    mocks.bgRequest.mockClear()
    await act(async () => {
      await renameAccountAMoodboard()
      await removeAccountANotebook()
    })

    expect(notesRequestPaths()).toEqual([])
    expect(mocks.promptModal).not.toHaveBeenCalled()
    expect(mocks.confirmDanger).not.toHaveBeenCalled()
  })

  it("preserves legacy local notebook migration when authority is omitted", async () => {
    let collectionGetCalls = 0
    mocks.getSetting.mockImplementation(async (setting) =>
      setting === NOTES_NOTEBOOKS_SETTING
        ? [{ id: 4, name: "Legacy notebook", keywords: ["legacy"] }]
        : null
    )
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        const path = String(request.path || "")
        const method = String(request.method || "GET").toUpperCase()
        if (path.startsWith("/api/v1/notes/collections") && method === "GET") {
          collectionGetCalls += 1
          return collectionGetCalls === 1
            ? { collections: [], pagination: { total_items: 0 } }
            : notebookResponse(44, "Legacy notebook")
        }
        if (path.startsWith("/api/v1/notes/collections")) {
          return { id: 44, name: "Legacy notebook", keywords: ["legacy"] }
        }
        return { items: [], pagination: { total_items: 0 } }
      }
    )

    const { result } = renderList(undefined)

    await waitFor(() => {
      expect(result.current.notebookOptions).toEqual([
        { id: 44, name: "Legacy notebook", keywords: ["legacy-notebook"] }
      ])
    })
    expect(notesRequestPaths()).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/^\/api\/v1\/notes\/\?/),
        expect.stringMatching(/^\/api\/v1\/notes\/collections/)
      ])
    )
    expect(
      mocks.bgRequest.mock.calls.some(([request]) =>
        ["PATCH", "POST"].includes(
          String(request?.method || "GET").toUpperCase()
        )
      )
    ).toBe(true)
    await waitFor(() => expect(mocks.setSetting).toHaveBeenCalled())
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

  it("keeps committed authority stable when a speculative transition is abandoned", async () => {
    const pendingSearch = deferred<ReturnType<typeof notesResponse>>()
    const suspendedTransition = new Promise<never>(() => {})
    const { queryClient } = createHarness()
    let browseCalls = 0
    let searchCalls = 0
    let suspendedBRenderCount = 0
    let shouldSuspendB = true
    let committedList: ReturnType<typeof useNotesListManagement> | null = null

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (path.startsWith("/api/v1/notes/search/?")) {
        searchCalls += 1
        return pendingSearch.promise
      }
      if (path.startsWith("/api/v1/notes/?")) {
        browseCalls += 1
        return browseCalls === 1
          ? notesResponse("account-a-note", 1)
          : { items: [], pagination: { total_items: 0 } }
      }
      return { items: [] }
    })

    const Suspender = ({ authorityScope }: { authorityScope: string }) => {
      if (authorityScope === "scope-b" && shouldSuspendB) {
        suspendedBRenderCount += 1
        throw suspendedTransition
      }
      return null
    }
    const Harness = ({ authorityScope }: { authorityScope: string }) => {
      const list = useNotesListManagement({
        authorityScope,
        isOnline: true,
        message: mocks.message as unknown as MessageInstance,
        confirmDanger: mocks.confirmDanger,
        queryClient,
        t: (key: string) => key,
        keywordTokens: [],
        setKeywordTokens: mocks.setKeywordTokens,
        notebookKeywordTokens: []
      })
      React.useLayoutEffect(() => {
        committedList = list
      })
      return (
        <>
          <output data-testid="concurrent-note-ids">
            {list.rawNotes.map((note) => note.id).join(",")}
          </output>
          <output data-testid="concurrent-total">{list.total}</output>
          <output data-testid="concurrent-bulk-selection">
            {list.bulkSelectedIds.join(",")}
          </output>
          <Suspender authorityScope={authorityScope} />
        </>
      )
    }
    const renderScope = (authorityScope: string) => (
      <React.StrictMode>
        <QueryClientProvider client={queryClient}>
          <React.Suspense fallback={<span>Suspended</span>}>
            <Harness authorityScope={authorityScope} />
          </React.Suspense>
        </QueryClientProvider>
      </React.StrictMode>
    )
    const authorityGenerations = (scope: string) =>
      Array.from(
        new Set(
          queryClient
            .getQueryCache()
            .getAll()
            .filter((query) => query.queryKey[0] === "notes")
            .map((query) => {
              const authorityIndex = query.queryKey.indexOf("authority")
              const generationIndex = query.queryKey.indexOf("generation")
              if (
                authorityIndex < 0 ||
                generationIndex < 0 ||
                query.queryKey[authorityIndex + 1] !== scope
              ) {
                return null
              }
              return Number(query.queryKey[generationIndex + 1])
            })
            .filter((generation): generation is number => generation != null)
        )
      ).sort((left, right) => left - right)

    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    try {
      await act(async () => {
        root.render(renderScope("scope-a"))
      })
      await waitFor(() => {
        expect(
          container.querySelector('[data-testid="concurrent-note-ids"]')
            ?.textContent
        ).toBe("account-a-note")
      })
      expect(authorityGenerations("scope-a")).toEqual([0])
      expect(browseCalls).toBe(1)

      const retainedToggleBulkSelection =
        committedList!.handleToggleBulkSelection
      let pendingCompletion!: ReturnType<
        typeof committedList.fetchFilteredNotesRaw
      >
      act(() => {
        pendingCompletion = committedList!.fetchFilteredNotesRaw(
          "pending",
          [],
          1,
          20
        )
      })
      await waitFor(() => expect(searchCalls).toBe(1))

      act(() => {
        React.startTransition(() => {
          root.render(renderScope("scope-b"))
        })
      })
      await waitFor(() => expect(suspendedBRenderCount).toBeGreaterThan(0))

      await act(async () => {
        root.render(renderScope("scope-a"))
      })

      expect(
        container.querySelector('[data-testid="concurrent-note-ids"]')
          ?.textContent
      ).toBe("account-a-note")
      expect(
        container.querySelector('[data-testid="concurrent-total"]')?.textContent
      ).toBe("1")
      expect(authorityGenerations("scope-a")).toEqual([0])
      expect(browseCalls).toBe(1)

      act(() => {
        retainedToggleBulkSelection("account-a-note", true, false)
      })
      expect(
        container.querySelector('[data-testid="concurrent-bulk-selection"]')
          ?.textContent
      ).toBe("account-a-note")

      let completionResult!: Awaited<typeof pendingCompletion>
      await act(async () => {
        pendingSearch.resolve(notesResponse("accepted-account-a-note", 5))
        completionResult = await pendingCompletion
      })
      expect(completionResult.items.map((item) => item.id)).toEqual([
        "accepted-account-a-note"
      ])
      expect(completionResult.total).toBe(5)

      shouldSuspendB = false
      await act(async () => {
        React.startTransition(() => {
          root.render(renderScope("scope-b"))
        })
      })
      await waitFor(() => expect(authorityGenerations("scope-b")).toEqual([1]))
      expect(authorityGenerations("scope-a")).toEqual([0])
      expect(
        container.querySelector('[data-testid="concurrent-note-ids"]')
          ?.textContent
      ).toBe("")
      expect(
        container.querySelector('[data-testid="concurrent-bulk-selection"]')
          ?.textContent
      ).toBe("")
    } finally {
      act(() => root.unmount())
      container.remove()
    }
  })

  it("starts a fresh browse request when the same authority returns after a guarded transition", async () => {
    const staleAccountA = deferred<ReturnType<typeof notesResponse>>()
    const freshAccountA = deferred<ReturnType<typeof notesResponse>>()
    let noteListCalls = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      noteListCalls += 1
      return noteListCalls === 1 ? staleAccountA.promise : freshAccountA.promise
    })
    const { result, rerender } = renderList("scope-a")
    await waitFor(() => expect(noteListCalls).toBe(1))

    rerender({ authorityScope: null })
    rerender({ authorityScope: "scope-a" })

    await waitFor(() => expect(noteListCalls).toBe(2))
    await act(async () => {
      staleAccountA.resolve(notesResponse("stale-account-a-note", 99))
      await Promise.resolve()
    })

    expect(result.current.isFetching).toBe(true)
    expect(result.current.rawNotes).toEqual([])
    expect(result.current.total).toBe(0)

    await act(async () => {
      freshAccountA.resolve(notesResponse("fresh-account-a-note", 1))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(result.current.rawNotes.map((note) => note.id)).toEqual([
        "fresh-account-a-note"
      ])
    })
    expect(result.current.total).toBe(1)
  })

  it("starts fresh moodboard enumeration when the same authority returns after a guarded transition", async () => {
    const staleAccountA = deferred<ReturnType<typeof moodboardResponse>>()
    const freshAccountA = deferred<ReturnType<typeof moodboardResponse>>()
    let moodboardGetCalls = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/collections")) return { items: [] }
      if (path.startsWith("/api/v1/notes/moodboards?")) {
        moodboardGetCalls += 1
        return moodboardGetCalls === 1
          ? staleAccountA.promise
          : freshAccountA.promise
      }
      return { items: [], pagination: { total_items: 0 } }
    })
    const { result, rerender } = renderList("scope-a")
    act(() => result.current.setListViewMode("moodboard"))
    await waitFor(() => expect(moodboardGetCalls).toBe(1))

    rerender({ authorityScope: null })
    rerender({ authorityScope: "scope-a" })

    await waitFor(() => expect(moodboardGetCalls).toBe(2))
    await act(async () => {
      staleAccountA.resolve(moodboardResponse(20, "Stale account A moodboard"))
      await Promise.resolve()
    })

    expect(result.current.isMoodboardsFetching).toBe(true)
    expect(result.current.moodboards).toEqual([])
    expect(result.current.selectedMoodboardId).toBeNull()

    await act(async () => {
      freshAccountA.resolve(moodboardResponse(21, "Fresh account A moodboard"))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(result.current.moodboards).toEqual([
        expect.objectContaining({
          id: 21,
          name: "Fresh account A moodboard"
        })
      ])
    })
    expect(result.current.selectedMoodboardId).toBe(21)
    expect(result.current.selectedMoodboard?.id).toBe(21)
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
