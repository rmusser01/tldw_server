// @vitest-environment jsdom
import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  act,
  render,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { DictionaryQuickAssignModal } from "../components/DictionaryQuickAssignModal"
import { useDictionaryListData } from "../components/useDictionaryListData"
import { useDictionaryManagerDataFlows } from "../components/useDictionaryManagerDataFlows"
import { useDictionaryQuickAssign } from "../components/useDictionaryQuickAssign"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  listDictionaries: vi.fn(),
  listChats: vi.fn(),
  listChatsWithMeta: vi.fn(),
  getChatSettings: vi.fn(),
  updateChatSettings: vi.fn(),
  release: vi.fn(),
  snapshot: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: mocks }))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: mocks.snapshot
}))
const dictionary = {
  id: 42,
  name: "Terms",
  used_by_chat_refs: [{ chat_id: "ordinary", title: "Ordinary" }]
}
const chats = [
  {
    id: "ordinary",
    title: "Ordinary",
    scope_type: "global",
    character_id: null,
    assistant_kind: null,
    parent_conversation_id: null,
    version: 1
  },
  {
    id: "character",
    title: "Character",
    scope_type: "global",
    character_id: 4,
    assistant_kind: "character",
    version: 1
  },
  {
    id: "persona",
    title: "Persona",
    scope_type: "global",
    assistant_kind: "persona",
    version: 1
  },
  {
    id: "fork",
    title: "Fork",
    scope_type: "global",
    parent_conversation_id: "ordinary",
    version: 1
  },
  { id: "unknown", title: "Unknown", version: 1 },
  {
    id: "workspace",
    title: "Workspace",
    scope_type: "workspace",
    workspace_id: "w",
    version: 1
  }
]
const notice = {
  info: vi.fn(),
  success: vi.fn(),
  warning: vi.fn(),
  error: vi.fn()
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { resolve, promise }
}
const makeOwner = (key: string) => {
  let current = true
  const owner = {
    ownerKey: key,
    serverScopeKey: key,
    requestScope: {
      config: {
        serverUrl: `https://${key}.test`,
        authMode: "multi-user" as const
      },
      userId: key
    },
    isCurrent: () => current,
    verificationState: "ready" as const,
    hasEverOwned: () => true,
    retryVerification: () => undefined
  }
  return {
    owner,
    retire: () => {
      current = false
    }
  }
}
const setup = (
  owner: ReturnType<typeof useChatDraftOwner>,
  onOpenChat = vi.fn(async () => true)
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  const wrapper = ({ children }: React.PropsWithChildren) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  const hook = renderHook(
    ({ owner }) => {
      const listInput = {
        isOnline: true,
        dictionarySearch: "",
        dictionaryCategoryFilter: "",
        dictionaryTagFilters: [],
        openEntries: null,
        notification: notice,
        queryClient,
        owner
      }
      const list = useDictionaryListData(listInput)
      const assignInput = {
        isOnline: true,
        notification: notice,
        queryClient,
        dictionaries: list.data ?? [],
        owner,
        onOpenChat,
        isDictionaryCurrent: list.isDictionaryCurrent
      }
      const assign = useDictionaryQuickAssign(assignInput)
      return { list, assign }
    },
    { wrapper, initialProps: { owner } }
  )
  return { ...hook, queryClient }
}
const open = async (hook: ReturnType<typeof setup>) => {
  await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
  act(() =>
    hook.result.current.assign.openQuickAssignModal(
      hook.result.current.list.data![0]
    )
  )
  await waitFor(() =>
    expect(hook.result.current.assign.assignableChatsStatus).toBe("success")
  )
}
beforeEach(() => {
  vi.resetAllMocks()
  mocks.initialize.mockResolvedValue(undefined)
  mocks.listDictionaries.mockResolvedValue({ dictionaries: [dictionary] })
  mocks.listChats.mockResolvedValue(chats)
  mocks.listChatsWithMeta.mockResolvedValue({ chats, total: chats.length })
  mocks.getChatSettings.mockImplementation(async (id: string) => ({
    conversation_id: id,
    settings: {}
  }))
  mocks.updateChatSettings.mockImplementation(
    async (id: string, patch: object) => ({
      conversation_id: id,
      settings: patch
    })
  )
  mocks.snapshot.mockImplementation(
    async (_ids: unknown, options: { signal?: AbortSignal }) => ({
      scopeSignal: options.signal ?? new AbortController().signal,
      release: mocks.release
    })
  )
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => ({
      matches: false,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {}
    })
  })
})

describe("dictionary assignment actual qualified query and controls", () => {
  it("carries one original owner and global scope into both source and chat reads", async () => {
    const { owner } = makeOwner("A")
    const hook = setup(owner)
    await open(hook)
    expect(mocks.listDictionaries).toHaveBeenCalledWith(
      true,
      true,
      expect.objectContaining({
        requestScope: owner.requestScope,
        signal: expect.any(AbortSignal)
      })
    )
    expect(mocks.listChatsWithMeta).toHaveBeenCalledWith(
      { limit: 100, offset: 0, include_deleted: false },
      expect.objectContaining({
        requestScope: owner.requestScope,
        scope: { type: "global" },
        signal: expect.any(AbortSignal)
      })
    )
    expect(mocks.listChats).not.toHaveBeenCalled()
  })

  it("does not authorize usage hints or any current saved row as assignable", async () => {
    const hook = setup(makeOwner("A").owner)
    await open(hook)
    expect(hook.result.current.assign.assignChatIds).toEqual([])
    expect(hook.result.current.assign.quickAssignChatOptions).toHaveLength(
      chats.length
    )
    for (const option of hook.result.current.assign.quickAssignChatOptions)
      expect(option).toMatchObject({
        disabledReason: expect.stringMatching(/saved.*chat|server/i)
      })
    act(() => hook.result.current.assign.toggleAssignChatSelection("ordinary"))
    await act(() => hook.result.current.assign.handleConfirmQuickAssign())
    expect(mocks.getChatSettings).not.toHaveBeenCalled()
    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
    expect(notice.success).not.toHaveBeenCalled()
  })

  it("does not publish an A source read after B becomes current, including an A key ABA", async () => {
    const A = makeOwner("A"),
      B = makeOwner("B"),
      A2 = makeOwner("A")
    const entered = deferred<void>(),
      held = deferred<{ dictionaries: (typeof dictionary)[] }>()
    mocks.listDictionaries.mockImplementationOnce(async () => {
      entered.resolve()
      return held.promise
    })
    const hook = setup(A.owner)
    await entered.promise
    A.retire()
    hook.rerender({ owner: B.owner })
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    B.retire()
    hook.rerender({ owner: A2.owner })
    await act(async () => {
      held.resolve({ dictionaries: [{ ...dictionary, name: "Retired A" }] })
    })
    await waitFor(() =>
      expect(hook.result.current.list.data?.[0]?.name).toBe("Terms")
    )
    expect(
      mocks.listDictionaries.mock.calls.some(
        (call) => call[2]?.requestScope === A2.owner.requestScope
      )
    ).toBe(true)
  })

  it("reloads a completed catalogue for a replacement owner with the same server key", async () => {
    const A = makeOwner("A"),
      A2 = makeOwner("A")
    const hook = setup(A.owner)
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    A.retire()
    hook.rerender({ owner: A2.owner })
    await waitFor(() => expect(hook.result.current.list.status).toBe("success"))
    expect(mocks.listDictionaries).toHaveBeenCalledTimes(2)
  })

  it("shows a finite owner error and retries verification without an uncertified read", async () => {
    const retryVerification = vi.fn(),
      owner = {
        ...makeOwner("A").owner,
        isCurrent: () => false,
        verificationState: "error" as const,
        retryVerification
      }
    const hook = setup(owner)
    expect(hook.result.current.list.status).toBe("error")
    await act(async () => {
      await hook.result.current.list.refetch()
    })
    expect(retryVerification).toHaveBeenCalledOnce()
    expect(mocks.listDictionaries).not.toHaveBeenCalled()
  })

  it("does not let a retired source Retry invoke replacement account verification", async () => {
    const retryVerification = vi.fn(),
      A = makeOwner("A"),
      B = makeOwner("B")
    const hook = setup({ ...A.owner, retryVerification })
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    const oldRetry = hook.result.current.list.refetch
    A.retire()
    hook.rerender({ owner: { ...B.owner, retryVerification } })
    await act(async () => {
      await oldRetry()
    })
    expect(retryVerification).not.toHaveBeenCalled()
  })

  it.each(["fetching", "paused", "error", "replaced"])(
    "refuses a retained Open immediately after the catalogue becomes %s",
    async (transition) => {
      const onOpenChat = vi.fn(async () => true),
        hook = setup(makeOwner("A").owner, onOpenChat)
      await open(hook)
      const old = hook.result.current.assign,
        chat = old.quickAssignChatOptions[0].chat
      const query = hook.queryClient
        .getQueryCache()
        .find({
          queryKey: ["tldw:listChatsForDictionaryAssign"],
          exact: false
        })!
      await act(async () => {
        if (transition === "replaced")
          query.setState({
            data: { ...(query.state.data as object), chats: [{ ...chat }] }
          })
        else if (transition === "error")
          query.setState({
            status: "error",
            error: new Error("Catalogue failed")
          })
        else query.setState({ fetchStatus: transition })
        expect(await old.openQuickAssignChat(chat)).toBe(false)
      })
      expect(onOpenChat).not.toHaveBeenCalled()
    }
  )

  it("cannot open a retained source row immediately after its library begins refreshing", async () => {
    const hook = setup(makeOwner("A").owner)
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    const oldOpen = hook.result.current.assign.openQuickAssignModal,
      row = hook.result.current.list.data![0]
    const query = hook.queryClient
      .getQueryCache()
      .find({ queryKey: ["tldw:listDictionaries"], exact: false })!
    act(() => {
      query.setState({ fetchStatus: "fetching" })
      oldOpen(row)
    })
    expect(hook.result.current.assign.assignFor).toBeNull()
  })

  it("a retired search callback cannot alter a replacement modal", async () => {
    const A = makeOwner("A"),
      B = makeOwner("B")
    const hook = setup(A.owner)
    await open(hook)
    const retired = hook.result.current.assign.setAssignSearch
    A.retire()
    hook.rerender({ owner: B.owner })
    await open(hook)
    act(() => {
      hook.result.current.assign.setAssignSearch("Current")
      retired("Retired")
    })
    expect(hook.result.current.assign.assignSearch).toBe("Current")
  })

  it("a same-owner retired row Open cannot replace a newer modal", async () => {
    const hook = setup(makeOwner("A").owner)
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    const oldOpen = hook.result.current.assign.openQuickAssignModal
    await open(hook)
    act(() => hook.result.current.assign.setAssignSearch("Current"))
    act(() => oldOpen(hook.result.current.list.data![0]))
    expect(hook.result.current.assign.assignSearch).toBe("Current")
  })

  it("retained row callbacks cannot open, close or refetch a replacement modal", async () => {
    const A = makeOwner("A"),
      B = makeOwner("B")
    const hook = setup(A.owner)
    await open(hook)
    const retired = hook.result.current.assign
    const oldDictionary = hook.result.current.list.data![0]
    A.retire()
    hook.rerender({ owner: B.owner })
    await open(hook)
    vi.clearAllMocks()
    act(() => {
      retired.closeQuickAssignModal()
      retired.openQuickAssignModal(oldDictionary)
    })
    await act(() => retired.refetchAssignableChats())
    expect(hook.result.current.assign.assignFor?.name).toBe("Terms")
    expect(mocks.listChatsWithMeta).not.toHaveBeenCalled()
  })

  it("retains only the current modal catalogue after close and reopen", async () => {
    const hook = setup(makeOwner("A").owner)
    const entered = deferred<void>(),
      held = deferred<{ chats: typeof chats; total: number }>()
    mocks.listChats.mockImplementationOnce(async () => {
      entered.resolve()
      return (await held.promise).chats
    })
    mocks.listChatsWithMeta.mockImplementationOnce(async () => {
      entered.resolve()
      return held.promise
    })
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    act(() =>
      hook.result.current.assign.openQuickAssignModal(
        hook.result.current.list.data![0]
      )
    )
    await entered.promise
    const oldClose = hook.result.current.assign.closeQuickAssignModal
    act(() => oldClose())
    await open(hook)
    await act(async () =>
      held.resolve({
        chats: [{ ...chats[0], title: "Retired modal" }],
        total: 1
      })
    )
    expect(
      hook.result.current.assign.quickAssignChatOptions.some(
        (option) => option.title === "Retired modal"
      )
    ).toBe(false)
    act(() => oldClose())
    expect(hook.result.current.assign.assignFor).not.toBeNull()
  })

  it("loads later catalogue pages rather than silently searching only the first100 rows", async () => {
    mocks.listChatsWithMeta.mockImplementation(
      async ({ offset }: { offset: number }) => ({
        chats:
          offset === 0
            ? Array.from({ length: 100 }, (_, i) => ({
                ...chats[0],
                id: `row-${i}`,
                title: `Row ${i}`
              }))
            : [{ ...chats[0], id: "page-2", title: "Later row" }],
        total: 101
      })
    )
    const hook = setup(makeOwner("A").owner)
    await open(hook)
    act(() => hook.result.current.assign.setAssignSearch("Later row"))
    expect(
      hook.result.current.assign.quickAssignChatOptions.map(
        (option) => option.title
      )
    ).toEqual(["Later row"])
  })

  it("fails closed at a full repeated final page before deduplicating it", async () => {
    mocks.listChatsWithMeta.mockResolvedValue({
      chats: Array.from({ length: 100 }, () => chats[0]),
      total: 100
    })
    const hook = setup(makeOwner("A").owner)
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    act(() =>
      hook.result.current.assign.openQuickAssignModal(
        hook.result.current.list.data![0]
      )
    )
    await waitFor(() =>
      expect(hook.result.current.assign.assignableChatsStatus).toBe("error")
    )
    expect(hook.result.current.assign.quickAssignChatOptions).toEqual([])
  })

  it("offers finite recovery after a warm catalogue read pauses", async () => {
    const hook = setup(makeOwner("A").owner)
    await open(hook)
    const query = hook.queryClient
      .getQueryCache()
      .find({ queryKey: ["tldw:listChatsForDictionaryAssign"], exact: false })!
    act(() => query.setState({ fetchStatus: "paused" }))
    await waitFor(() =>
      expect(hook.result.current.assign.assignableChatsStatus).toBe("error")
    )
    expect(hook.result.current.assign.quickAssignChatOptions).toEqual([])
  })

  it("invalidates the original source qualification immediately when dictionary usage is replaced", async () => {
    const hook = setup(makeOwner("A").owner)
    await waitFor(() => expect(hook.result.current.list.data).toHaveLength(1))
    const row = hook.result.current.list.data![0],
      guard = hook.result.current.list.isDictionaryCurrent
    const query = hook.queryClient
      .getQueryCache()
      .find({ queryKey: ["tldw:listDictionaries"], exact: false })!
    act(() => {
      query.setState({
        data: { ...(query.state.data as object), dictionaries: [{ ...row }] }
      })
      expect(guard(row)).toBe(false)
    })
  })

  it("never treats a malformed dictionary source response as empty authoritative absence", async () => {
    mocks.listDictionaries.mockResolvedValue({ dictionaries: {} })
    const hook = setup(makeOwner("A").owner)
    await waitFor(() => expect(hook.result.current.list.status).toBe("error"))
    expect(hook.result.current.list.data).toBeUndefined()
  })

  it("binds Usage Open to its original dictionary and reference through a source replacement", async () => {
    const owner = makeOwner("A").owner,
      onOpenChat = vi.fn(async (_chat: unknown, current: () => boolean) =>
        current()
      )
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const wrapper = ({ children }: React.PropsWithChildren) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
    const hook = renderHook(
      () =>
        useDictionaryManagerDataFlows({
          owner,
          onOpenChat,
          queryClient,
          isOnline: true,
          dictionarySearch: "",
          dictionaryCategoryFilter: "",
          dictionaryTagFilters: [],
          notification: notice,
          confirmDanger: async () => false,
          t: (_key, fallback) =>
            typeof fallback === "string" ? fallback : _key,
          entriesDrawerFocusReturnRef: React.useRef(null),
          quickAssignFocusReturnRef: React.useRef(null),
          importDialogFocusReturnRef: React.useRef(null)
        }),
      { wrapper }
    )
    await waitFor(() => expect(hook.result.current.data).toHaveLength(1))
    const row = hook.result.current.data![0],
      reference = row.used_by_chat_refs[0],
      oldOpen = hook.result.current.openLinkedDictionaryChat
    await expect(oldOpen(reference, row)).resolves.toBe(true)
    onOpenChat.mockClear()
    const query = queryClient
      .getQueryCache()
      .find({ queryKey: ["tldw:listDictionaries"], exact: false })!
    await act(async () => {
      query.setState({
        data: {
          ...(query.state.data as object),
          dictionaries: [{ ...row, used_by_chat_refs: [] }]
        }
      })
      expect(await oldOpen(reference, row)).toBe(false)
    })
    expect(onOpenChat).not.toHaveBeenCalled()
  })

  it("renders saved rows as unavailable with separate inspection controls", async () => {
    const hook = setup(makeOwner("A").owner)
    await open(hook)
    render(
      <DictionaryQuickAssignModal
        dictionaryName="Terms"
        open
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        selectedChatIds={hook.result.current.assign.assignChatIds}
        assignSaving={false}
        searchValue=""
        onSearchChange={vi.fn()}
        chatsStatus="success"
        chatsError={null}
        chatOptions={hook.result.current.assign.quickAssignChatOptions}
        onRetry={vi.fn()}
        onToggleChatSelection={vi.fn()}
        onOpenChat={vi.fn()}
      />
    )
    for (const chat of chats)
      expect(
        screen.getByRole("checkbox", { name: `Select chat ${chat.title}` })
      ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Assignment unavailable" })
    ).toBeDisabled()
    expect(
      screen
        .getByRole("button", { name: "Open chat Ordinary" })
        .closest("label")
    ).toBeNull()
  })
})
