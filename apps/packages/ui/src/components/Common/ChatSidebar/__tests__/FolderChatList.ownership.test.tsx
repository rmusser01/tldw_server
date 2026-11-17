import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider, message } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import i18next from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import common from "@/assets/locale/en/common.json"
import sidepanel from "@/assets/locale/en/sidepanel.json"

// Actual portal/dialog behavior with CSS transitions disabled for jsdom.
// ConfigProvider's token alone does not remove Modal's transitionName.
vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const react = await vi.importActual<typeof import("react")>("react")
  return { ...actual, Modal: (props: React.ComponentProps<typeof actual.Modal>) => {
    io.modalOpen = props.open === true
    return react.createElement(actual.Modal, { ...props, transitionName: "", maskTransitionName: "" })
  } }
})
const io = vi.hoisted(() => ({ getChat: vi.fn(), initialize: vi.fn(), select: vi.fn(),
  connected: true, modalOpen: false, cachedFolders: [] as unknown[] }))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => ({ isConnected: io.connected }) }))
vi.mock("@/store/connection", () => ({ useConnectionStore: Object.assign(
  (selector: (state: unknown) => unknown) => selector({ checkOnce: vi.fn() }),
  { getState: () => ({ state: { isConnected: io.connected } }) }) }))
vi.mock("@/services/chat-account-boundary", () => ({ watchChatAccountChanges: () => () => undefined }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getChat: io.getChat, initialize: io.initialize } }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => io.select }))
vi.mock("@/hooks/useServerChatHistory", () => ({ useServerChatHistory: () => ({ data: [], isLoading: false }) }))
vi.mock("@/services/folder-api", () => ({ fetchFolders: vi.fn(), fetchKeywords: vi.fn(),
  fetchFolderKeywordLinks: vi.fn(), fetchConversationKeywordLinks: vi.fn(), createFolder: vi.fn(),
  updateFolder: vi.fn(), deleteFolder: vi.fn(), createKeyword: vi.fn(), deleteKeyword: vi.fn(),
  linkKeywordToFolder: vi.fn(), unlinkKeywordFromFolder: vi.fn(), linkKeywordToConversation: vi.fn(), unlinkKeywordFromConversation: vi.fn() }))
vi.mock("@/db/dexie/schema", () => {
  const table = () => ({ clear: vi.fn(), bulkPut: vi.fn(), toArray: vi.fn(async () => []), put: vi.fn(), update: vi.fn() })
  return { db: { transaction: vi.fn(async (...args: unknown[]) => (args.at(-1) as () => unknown)()),
    folders: { ...table(), toArray: vi.fn(async () => io.cachedFolders) }, keywords: table(), folderKeywordLinks: table(), conversationKeywordLinks: table() } }
})

import * as api from "@/services/folder-api"
import { db } from "@/db/dexie/schema"
import { useFolderStore } from "@/store/folder"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { FolderTree } from "@/components/Folders/FolderTree"
import { FolderChatList } from "../FolderChatList"

const translations = i18next.createInstance()
await translations.init({ lng: "en", fallbackLng: "en", resources: { en: { common, sidepanel } }, interpolation: { escapeValue: false } })
const ok = <T,>(data: T) => ({ ok: true, status: 200, data })
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r }); return { promise, resolve } }
const folder = { id: 1, name: "Adventure", parent_id: null, deleted: false }
const keyword = { id: 9, keyword: "Adventure", deleted: false }
const chat = (id = "owned-chat", title = "Owned chat") => ({ id, title, scope_type: "global", workspace_id: null })
const owner = (user = 7) => ({ key: `account-${user}`, requestScope: { config: {
  serverUrl: "https://folders.test", authMode: "multi-user" as const }, userId: user }, isCurrent: vi.fn(() => true) })
const seedReads = (ids = ["owned-chat"]) => {
  vi.mocked(api.fetchFolders).mockResolvedValue(ok([folder]) as never)
  vi.mocked(api.fetchKeywords).mockResolvedValue(ok([keyword]) as never)
  vi.mocked(api.fetchFolderKeywordLinks).mockResolvedValue(ok([{ folder_id: 1, keyword_id: 9 }]) as never)
  vi.mocked(api.fetchConversationKeywordLinks).mockResolvedValue(ok(ids.map(id => ({ conversation_id: id, keyword_id: 9 }))) as never)
}
const clients: QueryClient[] = []
const renderView = (element: React.ReactElement) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } }); clients.push(client)
  const wrap = (ui: React.ReactElement) => <I18nextProvider i18n={translations}><ConfigProvider theme={{ token: { motion: false } }}>
    <QueryClientProvider client={client}>{ui}</QueryClientProvider></ConfigProvider></I18nextProvider>
  const view = render(wrap(element))
  return { ...view, rerender: (ui: React.ReactElement) => view.rerender(wrap(ui)) }
}
const list = (account: ReturnType<typeof owner>, extra: Record<string, unknown> = {}) =>
  <FolderChatList {...{ owner: account, verificationState: "ready", onConversationSelected: vi.fn(), ...extra }} />
const openCreate = async (user: ReturnType<typeof userEvent.setup>) => {
  await waitFor(() => expect(screen.getByRole("button", { name: "New Folder" })).toBeEnabled())
  await user.click(screen.getByRole("button", { name: "New Folder" }))
  await waitFor(() => expect(screen.getByPlaceholderText("Folder name")).toBeVisible())
  return screen.getByPlaceholderText("Folder name")
}

describe("actual Folders discovery retains its verified account and view", () => {
  beforeEach(() => { vi.resetAllMocks(); localStorage.clear(); io.connected = true; io.cachedFolders = []; seedReads()
    io.initialize.mockResolvedValue(undefined); io.getChat.mockImplementation(async (id: string) => chat(id)); io.select.mockReset()
    useStoreMessageOption.setState({ temporaryChat: false, chatDraftRevision: 0 })
    usePlaygroundSessionStore.setState({ restoreRevision: 0, sourceSelectionRevision: 0 })
    useFolderStore.setState({ folders: [], keywords: [], folderKeywordLinks: [], conversationKeywordLinks: [],
      isLoading: false, error: null, folderApiAvailable: null, serverDataScope: null, uiPrefs: { 1: { isOpen: true } } })
    vi.spyOn(message, "success"); vi.spyOn(message, "error") })
  afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); message.destroy(); vi.restoreAllMocks() })
  it("loads organization with the parent verified scope and never promotes its unqualified cache", async () => {
    const account = owner(); renderView(list(account))
    await waitFor(() => expect(api.fetchFolders).toHaveBeenCalledTimes(1))
    expect(api.fetchFolders).toHaveBeenCalledWith(expect.objectContaining({ requestScope: account.requestScope }))
    expect(db.folders.bulkPut).not.toHaveBeenCalled()
  })
  it("a failed fresh read shows Retry without another account's cached names", async () => {
    io.cachedFolders = [{ ...folder, id: 42, name: "Cached private folder" }]
    vi.mocked(api.fetchFolders).mockResolvedValue({ ok: false, status: 503, error: "Offline" } as never)
    renderView(list(owner()))
    await waitFor(() => expect(useFolderStore.getState().error).toBe("Offline"))
    expect(screen.queryByText("Cached private folder")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled()
    expect(db.folders.toArray).not.toHaveBeenCalled()
  })
  it("title lookups use the captured account and explicit global namespace", async () => {
    const account = owner(); renderView(list(account))
    await waitFor(() => expect(io.getChat).toHaveBeenCalled())
    expect(io.getChat).toHaveBeenCalledWith("owned-chat", expect.objectContaining({ requestScope: account.requestScope, scope: { type: "global" }, signal: expect.any(AbortSignal) }))
  })
  it("a held selected-chat read cannot select after actual New draft retirement", async () => {
    const held = deferred<ReturnType<typeof chat>>()
    io.getChat.mockResolvedValueOnce(chat()).mockReturnValueOnce(held.promise)
    const accepted = vi.fn(); renderView(list(owner(), { onConversationSelected: accepted }))
    await userEvent.click(await screen.findByText("Owned chat"))
    await waitFor(() => expect(io.getChat).toHaveBeenCalledTimes(2))
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    await act(async () => held.resolve(chat()))
    expect(io.select).not.toHaveBeenCalled(); expect(accepted).not.toHaveBeenCalled(); expect(message.error).not.toHaveBeenCalled()
  })
  it("a replacement account cannot continue the retired title batch", async () => {
    seedReads(Array.from({ length: 6 }, (_, n) => `old-${n}`))
    const held = deferred<void>(); io.getChat.mockImplementation((id: string) => held.promise.then(() => chat(id, "Old private title")))
    const old = owner(); const view = renderView(list(old))
    await waitFor(() => expect(io.getChat).toHaveBeenCalledTimes(5))
    old.isCurrent.mockReturnValue(false); seedReads([])
    view.rerender(list(owner(8)))
    await act(async () => held.resolve())
    expect(io.getChat).toHaveBeenCalledTimes(5)
    expect(screen.queryByText("Old private title")).not.toBeInTheDocument()
    expect(message.error).not.toHaveBeenCalled()
  })
  it("a failed folder creation retains its name and never reports success", async () => {
    const user = userEvent.setup(); vi.mocked(api.createFolder).mockResolvedValue({ ok: false, status: 503, error: "Unavailable" } as never)
    renderView(list(owner())); const input = await openCreate(user)
    await user.type(input, "Keep this name"); await user.click(screen.getByRole("button", { name: "Create" }))
    await waitFor(() => expect(api.createFolder).toHaveBeenCalledTimes(1))
    expect(message.success).not.toHaveBeenCalled()
    await waitFor(() => expect(message.error).toHaveBeenCalledTimes(1))
    expect(input).toBeVisible(); expect(input).toHaveValue("Keep this name")
  })
  it("Temporary disables keyboard management while leaving owned history readable", async () => {
    useStoreMessageOption.setState({ temporaryChat: true }); renderView(list(owner(), { readOnly: true }))
    expect(await screen.findByText("Owned chat")).toBeVisible()
    expect(screen.getByRole("button", { name: "New Folder" })).toBeDisabled()
    expect(api.createFolder).not.toHaveBeenCalled()
  })
  it("a stale handler cannot create after Temporary switches before rendering", async () => {
    const user = userEvent.setup(); renderView(list(owner())); const input = await openCreate(user)
    await user.type(input, "Private draft")
    useStoreMessageOption.getState().setTemporaryChat(true)
    fireEvent.click(screen.getByRole("button", { name: "Create" }))
    expect(api.createFolder).not.toHaveBeenCalled()
  })
  it("Cancel and reopening do not let an old create ACK close or clear the newer form", async () => {
    const user = userEvent.setup(); const held = deferred<ReturnType<typeof ok<typeof folder>>>()
    vi.mocked(api.createFolder).mockReturnValueOnce(held.promise as never)
    renderView(list(owner())); const first = await openCreate(user)
    await user.type(first, "Old request")
    const createButton = screen.getByRole("button", { name: "Create" })
    await user.click(createButton)
    await waitFor(() => expect(api.createFolder).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(createButton).toHaveClass("ant-btn-loading"))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    // Assert the actual dialog's controlled state; jsdom cannot prove its
    // CSS-motion close/focus geometry. Native default-motion is a later gate.
    await waitFor(() => expect(io.modalOpen).toBe(false))
    const next = await openCreate(user); await user.type(next, "New request")
    await act(async () => held.resolve(ok({ ...folder, id: 2, name: "Old request" })))
    expect(io.modalOpen).toBe(true); expect(next).toHaveValue("New request"); expect(message.success).not.toHaveBeenCalled()
  })
  it("the actual tree rejects a visible stale conversation handler imperatively", async () => {
    const account = owner(); const operation = { requestScope: account.requestScope, isCurrent: account.isCurrent }
    await useFolderStore.getState().refreshFromServer(operation)
    const selected = vi.fn()
    renderView(<FolderTree {...{ operation }} conversations={[{ id: "owned-chat", title: "Owned chat" }]} onConversationSelect={selected} />)
    const target = await screen.findByText("Owned chat"); account.isCurrent.mockReturnValue(false)
    fireEvent.click(target)
    expect(selected).not.toHaveBeenCalled()
  })
  it("an unverified account exposes Retry without dispatching or showing cached organization", async () => {
    const retry = vi.fn(); io.cachedFolders = [{ ...folder, name: "Cached private folder" }]
    renderView(<FolderChatList verificationState="error" onRetryVerification={retry} />)
    expect(screen.getByRole("alert")).toHaveTextContent("Unable to verify this account")
    await userEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(retry).toHaveBeenCalledTimes(1); expect(api.fetchFolders).not.toHaveBeenCalled(); expect(io.getChat).not.toHaveBeenCalled()
    expect(screen.queryByText("Cached private folder")).not.toBeInTheDocument()
  })
  it("a current failed organization read recovers through the same captured Retry", async () => {
    const account = owner(); vi.mocked(api.fetchFolders).mockResolvedValueOnce({ ok: false, status: 503, error: "Offline" } as never)
    renderView(list(account))
    await userEvent.click(await screen.findByRole("button", { name: "Retry" }))
    expect(await screen.findByText("Owned chat")).toBeVisible()
    expect(api.fetchFolders).toHaveBeenCalledTimes(2)
    expect(vi.mocked(api.fetchFolders).mock.calls.every(([options]) => options?.requestScope === account.requestScope)).toBe(true)
  })
  it("partial title failure keeps folder links readable and explicit Retry usable", async () => {
    io.getChat.mockRejectedValueOnce(Object.assign(new Error("Unavailable"), { status: 503 }))
    renderView(list(owner()))
    expect(await screen.findByText("Some conversation details are unavailable. Folder links remain visible.")).toBeVisible()
    expect(screen.getByText("Adventure")).toBeVisible(); expect(screen.getByText("Conversation owned-chat")).toBeVisible()
    await userEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("Owned chat")).toBeVisible(); expect(io.getChat).toHaveBeenCalledTimes(2)
  })
  it("foreign workspace metadata is unavailable rather than accepted global selection", async () => {
    io.getChat.mockResolvedValue({ ...chat(), title: "Foreign workspace", scope_type: "workspace", workspace_id: "private-workspace" })
    const accepted = vi.fn(); renderView(list(owner(), { onConversationSelected: accepted }))
    const target = await screen.findByText("Conversation owned-chat")
    await waitFor(() => expect(screen.getByText("Some conversation details are unavailable. Folder links remain visible.")).toBeVisible())
    await userEvent.click(target)
    await waitFor(() => expect(message.error).toHaveBeenCalledTimes(1))
    expect(screen.queryByText("Foreign workspace")).not.toBeInTheDocument(); expect(io.select).not.toHaveBeenCalled(); expect(accepted).not.toHaveBeenCalled()
  })
  it("a cookie-account refusal clears private names and requests the existing owner recheck", async () => {
    io.getChat.mockRejectedValue(Object.assign(new Error("Account changed"), { status: 412, code: "request_config_scope_changed" }))
    const verify = vi.fn(); renderView(list(owner(), { onRetryVerification: verify }))
    await waitFor(() => expect(useFolderStore.getState().folders).toEqual([]))
    await waitFor(() => expect(verify).toHaveBeenCalledTimes(1))
    expect(screen.queryByText("Adventure")).not.toBeInTheDocument(); expect(io.select).not.toHaveBeenCalled()
  })
  it("an organization read cookie refusal requests the existing owner recheck before titles", async () => {
    vi.mocked(api.fetchFolders).mockResolvedValue({ ok: false, status: 412, error: "Account changed" } as never)
    const verify = vi.fn(); renderView(list(owner(), { onRetryVerification: verify }))
    await waitFor(() => expect(verify).toHaveBeenCalledTimes(1))
    expect(screen.queryByText("Adventure")).not.toBeInTheDocument(); expect(io.getChat).not.toHaveBeenCalled()
    expect(useFolderStore.getState().serverDataScope).toBeNull(); expect(db.folders.toArray).not.toHaveBeenCalled()
  })
  it("a fresh current creation uses captured transport and reports its ACK", async () => {
    const user = userEvent.setup(); const account = owner()
    vi.mocked(api.createFolder).mockResolvedValue(ok({ ...folder, id: 2, name: "Quest" }) as never)
    renderView(list(account)); const input = await openCreate(user)
    await user.type(input, "Quest"); await user.click(screen.getByRole("button", { name: "Create" }))
    await waitFor(() => expect(message.success).toHaveBeenCalledTimes(1))
    expect(api.createFolder).toHaveBeenCalledWith("Quest", undefined, expect.objectContaining({ requestScope: account.requestScope }))
    expect(io.modalOpen).toBe(false); expect(useFolderStore.getState().folders.some(folder => folder.name === "Quest")).toBe(true)
  })
  it("Temporary selection validates metadata and hands off through the existing H1 entrypoint", async () => {
    useStoreMessageOption.setState({ temporaryChat: true })
    const accepted = vi.fn(); renderView(list(owner(), { readOnly: true, onConversationSelected: accepted }))
    await userEvent.click(await screen.findByText("Owned chat"))
    await waitFor(() => expect(accepted).toHaveBeenCalledTimes(1))
    expect(io.getChat).toHaveBeenCalledTimes(2); expect(io.select).toHaveBeenCalledWith(chat())
    expect(useStoreMessageOption.getState().temporaryChat).toBe(true); expect(api.createFolder).not.toHaveBeenCalled()
    expect(io.select.mock.invocationCallOrder[0]).toBeLessThan(accepted.mock.invocationCallOrder[0])
  })
  it("identical folder chat IDs on a replacement account never reuse the prior private title", async () => {
    io.getChat.mockResolvedValueOnce(chat("owned-chat", "Private A"))
    const old = owner(); const view = renderView(list(old))
    expect(await screen.findByText("Private A")).toBeVisible()
    const held = deferred<ReturnType<typeof chat>>(); io.getChat.mockReturnValueOnce(held.promise)
    vi.mocked(api.fetchFolders).mockResolvedValue(ok([{ ...folder, name: "B folder" }]) as never)
    old.isCurrent.mockReturnValue(false); view.rerender(list(owner(8)))
    expect(await screen.findByText("B folder")).toBeVisible()
    expect(screen.queryByText("Private A")).not.toBeInTheDocument()
    await act(async () => held.resolve(chat("owned-chat", "Current B")))
    expect(await screen.findByText("Current B")).toBeVisible()
  })
  it("a held create ACK after New cannot publish into the new folder view or show old success", async () => {
    const user = userEvent.setup(); const held = deferred<ReturnType<typeof ok<typeof folder>>>()
    vi.mocked(api.createFolder).mockReturnValueOnce(held.promise as never)
    renderView(list(owner())); const input = await openCreate(user)
    await user.type(input, "Old request"); await user.click(screen.getByRole("button", { name: "Create" }))
    await waitFor(() => expect(api.createFolder).toHaveBeenCalledTimes(1))
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    await act(async () => held.resolve(ok({ ...folder, id: 2, name: "Old request" })))
    expect(message.success).not.toHaveBeenCalled(); expect(io.modalOpen).toBe(false)
    expect(useFolderStore.getState().folders.some(folder => folder.name === "Old request")).toBe(false)
  })

})
