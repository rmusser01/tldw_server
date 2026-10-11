import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider, QueryObserver, useQuery } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useCharacterCrud, type UseCharacterCrudDeps } from "../hooks/useCharacterCrud"

const { createCharacter } = vi.hoisted(() => ({ createCharacter: vi.fn() }))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { createCharacter }
}))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [null, vi.fn()]
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((complete) => { resolve = complete })
  return { promise, resolve }
}

const makeDeps = (qc: QueryClient): UseCharacterCrudDeps => ({
  qc,
  t: (key, options) => String(options?.defaultValue ?? key),
  notification: { error: vi.fn(), warning: vi.fn(), success: vi.fn(), info: vi.fn() },
  createForm: { resetFields: vi.fn() },
  editForm: { resetFields: vi.fn() },
  editId: null,
  editVersion: null,
  editCharacterNumericId: null,
  setEditId: vi.fn(),
  setEditVersion: vi.fn(),
  setOpen: vi.fn(),
  setOpenEdit: vi.fn(),
  setConversationsOpen: vi.fn(),
  setConversationCharacter: vi.fn(),
  setPreviewCharacter: vi.fn(),
  setShowTemplates: vi.fn(),
  setShowCreateSystemPromptExample: vi.fn(),
  setShowEditSystemPromptExample: vi.fn(),
  setShowCreateAdvanced: vi.fn(),
  setShowEditAdvanced: vi.fn(),
  setCreateFormDirty: vi.fn(),
  setEditFormDirty: vi.fn(),
  setExporting: vi.fn(),
  newButtonRef: { current: null },
  lastEditTriggerRef: { current: null },
  editWorldBooksInitializedRef: { current: false },
  clearCreateDraft: vi.fn(),
  clearEditDraft: vi.fn(),
  data: [],
  effectiveDefaultCharacterId: undefined,
  defaultCharacterSelection: null,
  setDefaultCharacterSelection: vi.fn(),
  writeDefaultCharacterPreference: vi.fn(),
  activeChatModel: null,
  availableChatModels: [],
  setChatIntentBlocker: vi.fn()
})

const clients: QueryClient[] = []
const setup = (queryKey: readonly unknown[], queryFn: () => Promise<unknown>) => {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  clients.push(qc)
  const deps = makeDeps(qc)
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
  const hook = renderHook(() => {
    const { data, isSuccess } = useQuery({ queryKey, queryFn, staleTime: 5 * 60 * 1000 })
    return { list: { data, isSuccess }, crud: useCharacterCrud(deps) }
  }, { wrapper })
  return { ...hook, qc, deps }
}

describe("character creation list refresh", () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => {
    cleanup()
    clients.splice(0).forEach(client => client.clear())
  })

  it.each(["server", "legacy"])(
    "refreshes a cold coalesced %s list after its pre-create read settles",
    async (mode) => {
      const firstRead = deferred<unknown>()
      const fresh = { items: [{ id: 7, name: "New character" }], total: 1 }
      const read = vi.fn().mockReturnValueOnce(firstRead.promise).mockResolvedValue(fresh)
      let inFlight: Promise<unknown> | undefined
      const coalescedRead = () => inFlight ??= read().finally(() => { inFlight = undefined })
      createCharacter.mockResolvedValueOnce(fresh.items[0])
      const { result, qc, deps } = setup(["tldw:listCharacters", { page: 1 }, mode], coalescedRead)
      qc.setQueryData(["unrelated"], { id: "preserved" })
      await waitFor(() => expect(read).toHaveBeenCalledTimes(1))

      act(() => { result.current.crud.createCharacter({ name: "New character" }) })
      await waitFor(() => expect(deps.clearCreateDraft).toHaveBeenCalledTimes(1))
      await act(async () => { firstRead.resolve({ items: [], total: 0 }) })

      await waitFor(() => expect(result.current.list.data).toEqual(fresh))
      expect(read).toHaveBeenCalledTimes(2)
      expect(createCharacter).toHaveBeenCalledTimes(1)
      expect(qc.getQueryData(["unrelated"])).toEqual({ id: "preserved" })
      expect(deps.notification.error).not.toHaveBeenCalled()
    }
  )

  it("still refreshes a settled list after create", async () => {
    const fresh = { items: [{ id: 7, name: "New character" }], total: 1 }
    const read = vi.fn().mockResolvedValueOnce({ items: [], total: 0 }).mockResolvedValue(fresh)
    createCharacter.mockResolvedValueOnce(fresh.items[0])
    const { result } = setup(["tldw:listCharacters", { page: 1 }, "server"], read)
    await waitFor(() => expect(result.current.list.isSuccess).toBe(true))

    act(() => { result.current.crud.createCharacter({ name: "New character" }) })

    await waitFor(() => expect(result.current.list.data).toEqual(fresh))
    expect(createCharacter).toHaveBeenCalledTimes(1)
  })

  it("invalidates a cold list whose observer was removed before create", async () => {
    const fresh = { items: [{ id: 7, name: "New character" }], total: 1 }
    const activeRead = vi.fn().mockResolvedValueOnce({ items: [], total: 0 }).mockResolvedValue(fresh)
    const { result, qc, deps } = setup(["tldw:listCharacters", { query: "B" }, "server"], activeRead)
    await waitFor(() => expect(result.current.list.isSuccess).toBe(true))
    const firstRead = deferred<unknown>()
    const inactiveRead = vi.fn().mockReturnValueOnce(firstRead.promise).mockResolvedValue(fresh)
    const inactiveKey = ["tldw:listCharacters", { query: "A" }, "server"]
    const observer = new QueryObserver(qc, { queryKey: inactiveKey, queryFn: inactiveRead, staleTime: 5 * 60 * 1000 })
    const stop = observer.subscribe(() => undefined)
    await waitFor(() => expect(inactiveRead).toHaveBeenCalledTimes(1))
    stop()
    createCharacter.mockResolvedValueOnce(fresh.items[0])

    act(() => { result.current.crud.createCharacter({ name: "New character" }) })
    await waitFor(() => expect(deps.clearCreateDraft).toHaveBeenCalledTimes(1))
    await act(async () => { firstRead.resolve({ items: [], total: 0 }) })
    await waitFor(() => expect(result.current.list.data).toEqual(fresh))

    expect(qc.getQueryState(inactiveKey)?.isInvalidated).toBe(true)
    expect(await qc.fetchQuery({ queryKey: inactiveKey, queryFn: inactiveRead, staleTime: 5 * 60 * 1000 })).toEqual(fresh)
    expect(inactiveRead).toHaveBeenCalledTimes(2)
    expect(createCharacter).toHaveBeenCalledTimes(1)
  })

  it("retains the draft and ongoing list read when create fails", async () => {
    const firstRead = deferred<unknown>()
    const read = vi.fn().mockReturnValue(firstRead.promise)
    createCharacter.mockRejectedValueOnce(new Error("Creation rejected"))
    const { result, deps } = setup(["tldw:listCharacters", { page: 1 }, "server"], read)
    await waitFor(() => expect(read).toHaveBeenCalledTimes(1))

    act(() => { result.current.crud.createCharacter({ name: "New character" }) })
    await waitFor(() => expect(deps.notification.error).toHaveBeenCalledTimes(1))
    await act(async () => { firstRead.resolve({ items: [], total: 0 }) })

    await waitFor(() => expect(result.current.list.isSuccess).toBe(true))
    expect(read).toHaveBeenCalledTimes(1)
    expect(deps.clearCreateDraft).not.toHaveBeenCalled()
    expect(deps.setOpen).not.toHaveBeenCalled()
  })
})
