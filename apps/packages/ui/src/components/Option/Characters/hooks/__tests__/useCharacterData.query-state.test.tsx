import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useCharacterData, type UseCharacterDataDeps } from "../useCharacterData"

const client = vi.hoisted(() => ({
  initialize: vi.fn(), listCharactersPage: vi.fn(), listCharacters: vi.fn(),
  listAllCharacters: vi.fn(), listChats: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: client }))
vi.mock("@/services/folder-api", () => ({ fetchFolders: vi.fn().mockResolvedValue([]) }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [true, vi.fn(), { isLoading: false }] }))

const clients: QueryClient[] = []
const empty = { items: [], total: 0, page: 1, page_size: 10, has_more: false }
beforeEach(() => {
  vi.clearAllMocks()
  client.initialize.mockResolvedValue(undefined)
  client.listCharactersPage.mockResolvedValue(empty)
  client.listCharacters.mockResolvedValue([])
  client.listAllCharacters.mockResolvedValue([])
  client.listChats.mockResolvedValue([])
})
afterEach(() => { clients.splice(0).forEach(query => query.clear()) })

const mount = () => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  clients.push(qc)
  const notification = { error: vi.fn(), warning: vi.fn(), success: vi.fn(), info: vi.fn() }
  const deps: UseCharacterDataDeps = {
    t: (_key, options) => options?.defaultValue || _key, notification, qc,
    searchTerm: "", debouncedSearchTerm: "", filterTags: [], folderFilterId: undefined,
    matchAllTags: false, creatorFilter: undefined, createdFromDate: "", createdToDate: "",
    updatedFromDate: "", updatedToDate: "", hasConversationsOnly: false, favoritesOnly: false,
    characterListScope: "active", sortColumn: null, sortOrder: null, currentPage: 1, pageSize: 10,
    setCurrentPage: vi.fn(), previewCharacter: null, setPreviewCharacter: vi.fn(),
    editCharacterNumericId: null, editWorldBooksInitializedRef: { current: false },
    open: false, openEdit: false, editForm: {}, defaultCharacterSelection: null,
    setDefaultCharacterSelection: vi.fn(), defaultCharacterId: undefined
  }
  const hook = renderHook(() => useCharacterData(deps), {
    wrapper: ({ children }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  })
  return { ...hook, notification }
}

it("publishes a failed Library read as an error instead of successful empty data", async () => {
  const failure = new Error("Library read unavailable")
  client.listCharactersPage.mockRejectedValue(failure)
  const mounted = mount()
  await waitFor(() => expect(mounted.result.current.status).toBe("error"))
  expect(mounted.result.current.error).toBe(failure)
  expect(mounted.notification.error).toHaveBeenCalled()
})

it("keeps a genuinely empty successful Library distinct from failure", async () => {
  const mounted = mount()
  await waitFor(() => expect(mounted.result.current.status).toBe("success"))
  expect(mounted.result.current.data).toEqual([])
  expect(mounted.result.current.totalCharacters).toBe(0)
  expect(mounted.notification.error).not.toHaveBeenCalled()
})

it("returns ready Library results after an explicit Retry succeeds", async () => {
  client.listCharactersPage.mockRejectedValueOnce(new Error("Library read unavailable"))
  const mounted = mount()
  await waitFor(() => expect(mounted.result.current.status).toBe("error"))
  client.listCharactersPage.mockResolvedValue({ ...empty, items: [{ id: 1, name: "Guide" }], total: 1 })
  await act(async () => { await mounted.result.current.refetch() })
  await waitFor(() => expect(mounted.result.current.status).toBe("success"))
  expect(mounted.result.current.data).toMatchObject([{ id: 1, name: "Guide" }])
  expect(mounted.result.current.error).toBeNull()
})

it("retains the supported legacy route fallback as a successful read", async () => {
  client.listCharactersPage.mockRejectedValueOnce(Object.assign(new Error("route unavailable"), { status: 404 }))
  client.listCharacters.mockResolvedValue([{ id: 2, name: "Legacy Guide" }])
  const mounted = mount()
  await waitFor(() => expect(mounted.result.current.status).toBe("success"))
  expect(mounted.result.current.data).toMatchObject([{ id: 2, name: "Legacy Guide" }])
  expect(mounted.notification.error).not.toHaveBeenCalled()
})
