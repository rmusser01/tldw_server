// @vitest-environment jsdom
import type { HistoryInfo } from "@/db/dexie/types"
import { onlineManager, QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { SidepanelChatSidebar } from "../Sidebar"

const fullTextSearchChatHistoriesMock = vi.hoisted(() =>
  vi.fn<(...args: unknown[]) => Promise<HistoryInfo[]>>(async () => [])
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : defaultValueOrOptions?.defaultValue || _key
  })
}))

vi.mock("antd", () => ({
  message: {
    success: vi.fn(),
    error: vi.fn()
  },
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Modal: ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null)
}))

vi.mock("@/store/sidepanel-chat-tabs", () => ({
  useSidepanelChatTabsStore: (
    selector?: (state: {
      togglePinned: ReturnType<typeof vi.fn>
      renameTab: ReturnType<typeof vi.fn>
      setStatus: ReturnType<typeof vi.fn>
    }) => unknown
  ) =>
    typeof selector === "function"
      ? selector({
          togglePinned: vi.fn(),
          renameTab: vi.fn(),
          setStatus: vi.fn()
        })
      : {}
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector?: (state: { mode: string }) => unknown) =>
    typeof selector === "function" ? selector({ mode: "pro" }) : { mode: "pro" }
}))

vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: vi.fn(function MockPageAssistDatabase(this: {
    fullTextSearchChatHistories: typeof fullTextSearchChatHistoriesMock
  }) {
    this.fullTextSearchChatHistories = fullTextSearchChatHistoriesMock
  })
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: (
    selector?: (state: {
      getFoldersForConversation: (conversationId: string) => string[]
      uiPrefs: { showFolders: boolean }
      folderApiAvailable: boolean
    }) => unknown
  ) =>
    typeof selector === "function"
      ? selector({
          getFoldersForConversation: () => [],
          uiPrefs: { showFolders: true },
          folderApiAvailable: true
        })
      : {
          getFoldersForConversation: () => [],
          uiPrefs: { showFolders: true },
          folderApiAvailable: true
        }
}))

vi.mock("@/hooks/useBulkChatOperations", () => ({
  useBulkChatOperations: () => ({
    openBulkFolderPicker: vi.fn(),
    openBulkTagPicker: vi.fn(),
    applyBulkDelete: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [288, vi.fn()]
}))

vi.mock("../ModeToggle", () => ({
  ModeToggle: () => null
}))

vi.mock("../ConversationContextMenu", () => ({
  ConversationContextMenu: ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  )
}))

vi.mock("../FolderPickerModal", () => ({
  FolderPickerModal: () => null
}))

const search = vi.hoisted(() => vi.fn())
const connection = vi.hoisted(() => ({ isConnected: true }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connection
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: (select: (state: unknown) => unknown) =>
    select({ checkOnce: async () => {} })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    searchConversationsWithMeta: (...args: unknown[]) => search(...args)
  }
}))
const props = {
  open: true,
  variant: "docked" as const,
  tabs: [],
  activeTabId: null,
  onSelectTab: () => {},
  onCloseTab: () => {},
  onNewTab: () => {},
  searchQuery: "private",
  onSearchQueryChange: () => {}
}
const owner = (userId: number) => ({
  ownerKey: String(userId),
  isCurrent: () => true,
  snapshot: {
    requestScope: {
      config: {
        serverUrl: "http://chat.test",
        authMode: "multi-user" as const
      },
      userId
    },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: () => {}
  }
})
const response = (title: string) => ({
  chats: [{ id: title, title, created_at: "2026-09-20T00:00:00Z" }],
  total: 1
})
const withClient = (
  client: QueryClient,
  currentOwner: ReturnType<typeof owner> | undefined,
  overrides: Partial<React.ComponentProps<typeof SidepanelChatSidebar>> = {}
) => (
  <QueryClientProvider client={client}>
    <SidepanelChatSidebar {...props} {...overrides} owner={currentOwner} />
  </QueryClientProvider>
)
beforeEach(() => {
  search.mockReset()
  connection.isConnected = true
  fullTextSearchChatHistoriesMock.mockReset().mockResolvedValue([])
})

const clients: QueryClient[] = []
const createClient = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } })
  clients.push(client)
  return client
}
afterEach(() => {
  clients.splice(0).forEach(client => client.clear())
  onlineManager.setOnline(true)
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(finish => { resolve = finish })
  return { promise, resolve }
}
const localResult = (title: string): HistoryInfo => ({
  id: title, title, server_scope_key: "1", createdAt: 1
} as HistoryInfo)
const empty = { chats: [], total: 0 }
const serverFailure = () => Object.assign(new Error("HTTP 500"), { status: 500 })

it("shows server loading until a successful empty search settles", async () => {
  const held = deferred<typeof empty>()
  search.mockReturnValue(held.promise)
  render(withClient(createClient(), owner(1)))
  await waitFor(() => expect(search).toHaveBeenCalledOnce())
  expect(screen.getByText("Searching chat history…")).toBeInTheDocument()
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  await act(async () => { held.resolve(empty) })
  await screen.findByText("No matches found")
})

it("waits for local search before declaring successful empty results", async () => {
  const held = deferred<HistoryInfo[]>()
  search.mockResolvedValue(empty)
  fullTextSearchChatHistoriesMock.mockReturnValue(held.promise)
  render(withClient(createClient(), owner(1)))
  await waitFor(() => expect(search).toHaveBeenCalledOnce())
  expect(screen.getByText("Searching chat history…")).toBeInTheDocument()
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  await act(async () => { held.resolve([]) })
  await screen.findByText("No matches found")
})

it("reports HTTP failure and retries the unchanged query with its owner scope", async () => {
  search.mockRejectedValue(serverFailure())
  const alice = owner(1)
  render(withClient(createClient(), alice))
  await screen.findByText("Server history search failed. Try again.")
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  search.mockResolvedValue(response("Recovered private chat"))
  fireEvent.click(screen.getByRole("button", { name: "Retry server history search" }))
  await screen.findByText("Recovered private chat")
  expect(search).toHaveBeenLastCalledWith(
    expect.objectContaining({ query: "private" }),
    expect.objectContaining({ requestScope: alice.snapshot.requestScope })
  )
  expect(screen.queryByText("Server history search failed. Try again.")).not.toBeInTheDocument()
})

it("keeps valid local matches visible alongside a server failure", async () => {
  search.mockRejectedValue(serverFailure())
  fullTextSearchChatHistoriesMock.mockResolvedValue([localResult("Local private chat")])
  render(withClient(createClient(), owner(1)))
  await screen.findByText("Server history search failed. Try again.")
  expect(screen.getByText("Local private chat")).toBeInTheDocument()
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
})

it("reports a local read failure and retries while preserving server matches", async () => {
  search.mockResolvedValue(response("Server private chat"))
  fullTextSearchChatHistoriesMock.mockRejectedValue(new Error("disk read failed"))
  render(withClient(createClient(), owner(1)))
  await screen.findByText("Local history search failed. Try again.")
  await screen.findByText("Server private chat")
  fullTextSearchChatHistoriesMock.mockResolvedValue([localResult("Recovered local chat")])
  fireEvent.click(screen.getByRole("button", { name: "Retry local history search" }))
  await screen.findByText("Recovered local chat")
  expect(search).toHaveBeenCalledOnce()
  expect(fullTextSearchChatHistoriesMock).toHaveBeenLastCalledWith("private")
})

it("hides previous-query results throughout debounce and held replacement reads", async () => {
  const nextServer = deferred<typeof empty>()
  const nextLocal = deferred<HistoryInfo[]>()
  search.mockResolvedValueOnce(response("Old server private")).mockReturnValue(nextServer.promise)
  fullTextSearchChatHistoriesMock.mockResolvedValueOnce([localResult("Old local private")]).mockReturnValue(nextLocal.promise)
  const client = createClient()
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await screen.findByText("Old server private")
  await screen.findByText("Old local private")
  view.rerender(withClient(client, alice, { searchQuery: "new query" }))
  expect(screen.queryByText("Old server private")).not.toBeInTheDocument()
  expect(screen.queryByText("Old local private")).not.toBeInTheDocument()
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  await waitFor(() => expect(search).toHaveBeenCalledTimes(2))
  expect(screen.queryByText("Old local private")).not.toBeInTheDocument()
  await act(async () => { nextServer.resolve(empty); nextLocal.resolve([]) })
  await screen.findByText("No matches found")
})

it("does not offer the previous query retry during debounce", async () => {
  search.mockRejectedValue(serverFailure())
  const client = createClient()
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await screen.findByText("Server history search failed. Try again.")
  search.mockResolvedValue(empty)
  const before = search.mock.calls.length
  view.rerender(withClient(client, alice, { searchQuery: "replacement" }))
  expect(screen.queryByRole("button", { name: "Retry server history search" })).not.toBeInTheDocument()
  expect(search).toHaveBeenCalledTimes(before)
  await screen.findByText("No matches found")
  expect(search).toHaveBeenLastCalledWith(expect.objectContaining({ query: "replacement" }), expect.anything())
})

it("hides stale server cache after a failed refresh but retains current local matches", async () => {
  search.mockResolvedValue(response("Cached private server chat"))
  fullTextSearchChatHistoriesMock.mockResolvedValue([localResult("Current local chat")])
  const client = createClient()
  render(withClient(client, owner(1)))
  await screen.findByText("Cached private server chat")
  search.mockRejectedValue(serverFailure())
  await act(async () => { await client.invalidateQueries({ queryKey: ["serverChatHistory"] }) })
  await screen.findByText("Server history search failed. Try again.")
  expect(screen.queryByText("Cached private server chat")).not.toBeInTheDocument()
  expect(screen.getByText("Current local chat")).toBeInTheDocument()
})

it("does not dispatch retry after the captured owner is revoked", async () => {
  search.mockRejectedValue(serverFailure())
  const alice = owner(1)
  render(withClient(createClient(), alice))
  const retry = await screen.findByRole("button", { name: "Retry server history search" })
  const before = search.mock.calls.length
  alice.isCurrent = () => false
  fireEvent.click(retry)
  expect(search).toHaveBeenCalledTimes(before)
})

it("rejects late retry results after switching accounts", async () => {
  search.mockRejectedValue(serverFailure())
  const alice = owner(1)
  const client = createClient()
  const view = render(withClient(client, alice))
  const retry = await screen.findByRole("button", { name: "Retry server history search" })
  const held = deferred<ReturnType<typeof response>>()
  search.mockReturnValueOnce(held.promise).mockResolvedValue(response("Bob private chat"))
  fireEvent.click(retry)
  await waitFor(() => expect(search).toHaveBeenCalledTimes(3))
  alice.isCurrent = () => false
  view.rerender(withClient(client, owner(2)))
  await screen.findByText("Bob private chat")
  await act(async () => { held.resolve(response("Late Alice private chat")) })
  expect(screen.queryByText("Late Alice private chat")).not.toBeInTheDocument()
})

it("shows offline server search as unavailable and blocks its retry", async () => {
  connection.isConnected = false
  search.mockResolvedValue(empty)
  const client = createClient()
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await screen.findByText("Server history search is unavailable while disconnected.")
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  const retry = screen.queryByRole("button", { name: "Retry server history search" })
  if (retry) {
    expect(retry).toBeDisabled()
    fireEvent.click(retry)
  }
  expect(search).not.toHaveBeenCalled()
  connection.isConnected = true
  view.rerender(withClient(client, alice))
  await screen.findByText("No matches found")
})

it("does not claim empty search or dispatch without a verified owner", async () => {
  render(withClient(createClient(), undefined))
  await screen.findByText("History search is unavailable until your account is verified.")
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Retry .* history search/ })).not.toBeInTheDocument()
  expect(search).not.toHaveBeenCalled()
  expect(fullTextSearchChatHistoriesMock).not.toHaveBeenCalled()
})

it("does not present server cache as current search results after disconnection", async () => {
  search.mockResolvedValue(response("Cached server private chat"))
  fullTextSearchChatHistoriesMock.mockResolvedValue([localResult("Owned local private chat")])
  const client = createClient()
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await screen.findByText("Cached server private chat")
  connection.isConnected = false
  view.rerender(withClient(client, alice))
  expect(screen.queryByText("Cached server private chat")).not.toBeInTheDocument()
  expect(screen.getByText("Owned local private chat")).toBeInTheDocument()
  expect(screen.getByText("Server history search is unavailable while disconnected.")).toBeInTheDocument()
})

it("shows a paused network query as unavailable rather than endless loading", async () => {
  onlineManager.setOnline(false)
  search.mockResolvedValue(empty)
  render(withClient(createClient(), owner(1)))
  await screen.findByText("Server history search is unavailable while disconnected.")
  await waitFor(() => expect(screen.queryByText("Searching chat history…")).not.toBeInTheDocument())
  expect(screen.queryByText("No matches found")).not.toBeInTheDocument()
  expect(search).not.toHaveBeenCalled()
})

it("does not retry a local search after its owner is revoked", async () => {
  search.mockResolvedValue(empty)
  fullTextSearchChatHistoriesMock.mockRejectedValue(new Error("disk read failed"))
  const alice = owner(1)
  render(withClient(createClient(), alice))
  const retry = await screen.findByRole("button", { name: "Retry local history search" })
  const before = fullTextSearchChatHistoriesMock.mock.calls.length
  alice.isCurrent = () => false
  fireEvent.click(retry)
  expect(fullTextSearchChatHistoriesMock).toHaveBeenCalledTimes(before)
})

it("does not publish a previous-query local completion after the next query settles", async () => {
  search.mockResolvedValue(empty)
  const held = deferred<HistoryInfo[]>()
  fullTextSearchChatHistoriesMock.mockReturnValueOnce(held.promise).mockResolvedValue([localResult("Current result")])
  const client = createClient()
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await waitFor(() => expect(fullTextSearchChatHistoriesMock).toHaveBeenCalledOnce())
  view.rerender(withClient(client, alice, { searchQuery: "current" }))
  await screen.findByText("Current result")
  await act(async () => { held.resolve([localResult("Obsolete private result")]) })
  expect(screen.queryByText("Obsolete private result")).not.toBeInTheDocument()
  expect(screen.getByText("Current result")).toBeInTheDocument()
})

it("does not display Alice's cached server titles for Bob's identical search", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const alice = owner(1)
  search
    .mockResolvedValueOnce(response("Alice private title"))
    .mockResolvedValueOnce(response("Bob private title"))
  const view = render(withClient(client, alice))
  await screen.findByText("Alice private title")
  alice.isCurrent = () => false
  view.rerender(withClient(client, owner(2)))
  expect(screen.queryByText("Alice private title")).not.toBeInTheDocument()
  await screen.findByText("Bob private title")
  view.unmount()
  client.clear()
})

it("rejects a late Alice server title response after Bob starts the same search", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  let finish!: (value: ReturnType<typeof response>) => void
  search
    .mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve
      })
    )
    .mockResolvedValueOnce(response("Bob private title"))
  const alice = owner(1)
  const view = render(withClient(client, alice))
  await waitFor(() => expect(search).toHaveBeenCalledOnce())
  alice.isCurrent = () => false
  view.rerender(withClient(client, owner(2)))
  await act(async () => {
    finish(response("Alice delayed title"))
  })
  expect(screen.queryByText("Alice delayed title")).not.toBeInTheDocument()
  await screen.findByText("Bob private title")
  view.unmount()
  client.clear()
})

it("preserves same-owner cached title search across remounts", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  search.mockResolvedValue(response("Alice private title"))
  const view = render(withClient(client, owner(1)))
  await screen.findByText("Alice private title")
  view.unmount()
  const remount = render(withClient(client, owner(1)))
  await screen.findByText("Alice private title")
  expect(search).toHaveBeenCalledOnce()
  remount.unmount()
  client.clear()
})
