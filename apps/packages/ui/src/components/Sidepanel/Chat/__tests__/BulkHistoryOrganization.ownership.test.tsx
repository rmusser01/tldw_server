import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => typeof fallback === "string" ? fallback : key }) }))
vi.mock("@/services/folder-api", () => ({ fetchFolders: vi.fn(), fetchKeywords: vi.fn(),
  fetchFolderKeywordLinks: vi.fn(), fetchConversationKeywordLinks: vi.fn(), createFolder: vi.fn(),
  updateFolder: vi.fn(), deleteFolder: vi.fn(), createKeyword: vi.fn(), deleteKeyword: vi.fn(),
  linkKeywordToFolder: vi.fn(), unlinkKeywordFromFolder: vi.fn(), linkKeywordToConversation: vi.fn(), unlinkKeywordFromConversation: vi.fn() }))
vi.mock("@/db/dexie/schema", () => {
  const table = () => ({ clear: vi.fn(), bulkPut: vi.fn(), toArray: vi.fn(async () => []), put: vi.fn(), update: vi.fn() })
  return { db: { transaction: vi.fn(async (...args: unknown[]) => (args.at(-1) as () => unknown)()),
    folders: table(), keywords: table(), folderKeywordLinks: table(), conversationKeywordLinks: table() } }
})
import * as api from "@/services/folder-api"
import { useFolderStore } from "@/store/folder"
import { BulkFolderPickerModal } from "../BulkFolderPickerModal"
import { BulkTagPickerModal } from "../BulkTagPickerModal"

const ok = <T,>(data: T) => ({ ok: true, status: 200, data })
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r }); return { promise, resolve } }
const operation = () => ({ requestScope: { config: { serverUrl: "https://folders.test", authMode: "multi-user" as const }, userId: 7 }, isCurrent: vi.fn(() => true) })
const renderModal = (ui: React.ReactNode) => render(<ConfigProvider theme={{ token: { motion: false } }}>{ui}</ConfigProvider>)
const seedReads = () => {
  vi.mocked(api.fetchFolders).mockResolvedValue(ok([{ id: 1, name: "Adventure", parent_id: null, deleted: false }]) as never)
  vi.mocked(api.fetchKeywords).mockResolvedValue(ok([{ id: 9, keyword: "Adventure", deleted: false }]) as never)
  vi.mocked(api.fetchFolderKeywordLinks).mockResolvedValue(ok([{ folder_id: 1, keyword_id: 9 }]) as never)
  vi.mocked(api.fetchConversationKeywordLinks).mockResolvedValue(ok([]) as never)
}
const selectTag = async (user: ReturnType<typeof userEvent.setup>, tag: string) => {
  const input = await screen.findByRole("combobox"); await user.type(input, `${tag}{Enter}`)
}
const selectFolder = async (user: ReturnType<typeof userEvent.setup>) => {
  const title = await screen.findByText("Adventure")
  const checkbox = title.closest(".ant-tree-treenode")?.querySelector(".ant-tree-checkbox")
  expect(checkbox).not.toBeNull(); await user.click(checkbox as Element)
}

describe("actual history organization dialogs retire nested operations", () => {
  beforeEach(() => { vi.resetAllMocks(); localStorage.clear(); seedReads()
    useFolderStore.setState({ folders: [], keywords: [], folderKeywordLinks: [], conversationKeywordLinks: [], isLoading: false,
      error: null, folderApiAvailable: null, serverDataScope: null } as never)
    vi.spyOn(message, "success"); vi.spyOn(message, "error") })
  afterEach(() => { cleanup(); message.destroy(); vi.restoreAllMocks() })
  it("stops tag links and stale feedback when held tag creation finishes after close/New", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    const held = deferred<ReturnType<typeof ok<{ id: number; keyword: string }>>>()
    vi.mocked(api.createKeyword).mockReturnValue(held.promise as never)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    const view = renderModal(<BulkTagPickerModal {...{ operation: op }} open conversationIds={["old-chat"]} onClose={close} onSuccess={success} />)
    await selectTag(user, "New tag"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.createKeyword).toHaveBeenCalledTimes(1))
    op.isCurrent.mockReturnValue(false); view.unmount()
    await act(async () => held.resolve(ok({ id: 10, keyword: "New tag" })))
    expect(api.linkKeywordToConversation).not.toHaveBeenCalled()
    expect(success).not.toHaveBeenCalled(); expect(close).not.toHaveBeenCalled()
    expect(message.success).not.toHaveBeenCalled(); expect(message.error).not.toHaveBeenCalled()
  })
  it("does not start a later folder batch after the first batch retires", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    const held = deferred<ReturnType<typeof ok<undefined>>>()
    vi.mocked(api.linkKeywordToConversation).mockReturnValue(held.promise as never)
    const view = renderModal(<BulkFolderPickerModal {...{ operation: op }} open conversationIds={Array.from({ length: 13 }, (_, i) => `old-${i}`)} onClose={close} onSuccess={success} />)
    await selectFolder(user); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.linkKeywordToConversation).toHaveBeenCalledTimes(12))
    op.isCurrent.mockReturnValue(false); view.unmount()
    await act(async () => held.resolve(ok(undefined)))
    expect(api.linkKeywordToConversation).toHaveBeenCalledTimes(12)
    expect(success).not.toHaveBeenCalled(); expect(close).not.toHaveBeenCalled(); expect(message.success).not.toHaveBeenCalled()
  })
  it("shows a retryable owned read error instead of unqualified cached options", async () => {
    const user = userEvent.setup(); const op = operation()
    useFolderStore.setState({ keywords: [{ id: 42, keyword: "Cached private tag", deleted: false }] } as never)
    vi.mocked(api.fetchFolders).mockResolvedValueOnce({ ok: false, error: "Offline", status: 503 } as never)
    renderModal(<BulkTagPickerModal {...{ operation: op }} open conversationIds={["owned-chat"]} onClose={vi.fn()} />)
    await waitFor(() => expect(screen.getByText("Unable to load folders and tags for this account.")).toBeVisible())
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByRole("combobox")).toBeVisible()
    expect(api.fetchFolders).toHaveBeenCalledTimes(2)
  })
  it("current tag apply retains normal success and captured-account writes", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    renderModal(<BulkTagPickerModal {...{ operation: op }} open conversationIds={["owned-chat"]} onClose={close} onSuccess={success} />)
    await selectTag(user, "Adventure"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(success).toHaveBeenCalledTimes(1))
    expect(close).toHaveBeenCalledTimes(1)
    expect(api.linkKeywordToConversation).toHaveBeenCalledWith("owned-chat", 9, expect.objectContaining({ requestScope: op.requestScope }))
  })
  it("Cancel retires a held tag operation before a parent can unmount the dialog", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    const held = deferred<ReturnType<typeof ok<{ id: number; keyword: string }>>>()
    vi.mocked(api.createKeyword).mockReturnValue(held.promise as never)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    renderModal(<BulkTagPickerModal operation={op} open conversationIds={["old-chat"]} onClose={close} onSuccess={success} />)
    await selectTag(user, "New tag"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.createKeyword).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => held.resolve(ok({ id: 10, keyword: "New tag" })))
    expect(api.linkKeywordToConversation).not.toHaveBeenCalled()
    expect(close).toHaveBeenCalledTimes(1); expect(success).not.toHaveBeenCalled()
    expect(message.success).not.toHaveBeenCalled()
  })
  it("partial tag apply retains selection and retry avoids already acknowledged links", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    vi.mocked(api.linkKeywordToConversation).mockResolvedValueOnce(ok(undefined) as never)
      .mockResolvedValueOnce({ ok: false, status: 503, error: "Try again" } as never)
      .mockResolvedValue(ok(undefined) as never)
    renderModal(<BulkTagPickerModal operation={op} open conversationIds={["good-chat", "retry-chat"]} onClose={close} onSuccess={success} />)
    await selectTag(user, "Adventure"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(message.error).toHaveBeenCalledTimes(1))
    expect(success).not.toHaveBeenCalled(); expect(close).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(success).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.linkKeywordToConversation).mock.calls.map(([id]) => id)).toEqual(["good-chat", "retry-chat", "retry-chat"])
  })
  it("one failed tag creation cannot report that all requested tags were applied", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    vi.mocked(api.createKeyword).mockImplementation(async (name) => name === "Good"
      ? ok({ id: 10, keyword: name }) as never
      : { ok: false, status: 503, error: "Unavailable" } as never)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    renderModal(<BulkTagPickerModal operation={op} open conversationIds={["owned-chat"]} onClose={close} onSuccess={success} />)
    await selectTag(user, "Good"); await selectTag(user, "Unavailable"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.linkKeywordToConversation).toHaveBeenCalledTimes(1))
    expect(message.success).not.toHaveBeenCalled(); expect(success).not.toHaveBeenCalled(); expect(close).not.toHaveBeenCalled()
    expect(message.error).toHaveBeenCalledTimes(1)
  })
  it("current folder creation failure explains the error and retains the name", async () => {
    const user = userEvent.setup(); const op = operation()
    vi.mocked(api.createFolder).mockResolvedValue({ ok: false, status: 503, error: "Unavailable" } as never)
    renderModal(<BulkFolderPickerModal operation={op} open conversationIds={["owned-chat"]} onClose={vi.fn()} />)
    await selectFolder(user)
    const input = screen.getByPlaceholderText("New folder name")
    await user.type(input, "Keep this name"); await user.click(screen.getByRole("button", { name: "Add" }))
    await waitFor(() => expect(api.createFolder).toHaveBeenCalledTimes(1))
    expect(message.error).toHaveBeenCalledWith("Unable to create the folder. Try again.")
    expect(input).toHaveValue("Keep this name")
  })

  it("a new target selection in the same open dialog retires its old tag creation", async () => {
    const user = userEvent.setup(); const op = operation(); const close = vi.fn(); const success = vi.fn()
    const held = deferred<ReturnType<typeof ok<{ id: number; keyword: string }>>>()
    vi.mocked(api.createKeyword).mockReturnValue(held.promise as never)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    const ui = (ids: string[]) => <ConfigProvider theme={{ token: { motion: false } }}><BulkTagPickerModal operation={op} open conversationIds={ids} onClose={close} onSuccess={success} /></ConfigProvider>
    const view = render(ui(["old-chat"])); await selectTag(user, "New tag")
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.createKeyword).toHaveBeenCalledTimes(1))
    view.rerender(ui(["new-chat"]))
    await act(async () => held.resolve(ok({ id: 10, keyword: "New tag" })))
    expect(api.linkKeywordToConversation).not.toHaveBeenCalled()
    expect(success).not.toHaveBeenCalled(); expect(close).not.toHaveBeenCalled()
    expect(await screen.findByRole("combobox")).toHaveValue("")
    await selectTag(user, "Adventure"); await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.linkKeywordToConversation).toHaveBeenCalledWith("new-chat", 9, expect.objectContaining({ requestScope: op.requestScope })))
  })

})
