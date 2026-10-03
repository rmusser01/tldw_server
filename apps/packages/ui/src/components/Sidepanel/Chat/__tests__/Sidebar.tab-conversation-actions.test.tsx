// @vitest-environment jsdom
/**
 * XS-07 (#3105): the side-panel tab menu's Delete and Rename act on the real
 * conversation, and say what they do. Renders the real context menu (antd
 * Dropdown and Modal) and checks the chat services it calls.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SidepanelChatTab } from "@/store/sidepanel-chat-tabs"
import { SidepanelChatSidebar } from "../Sidebar"

const io = vi.hoisted(() => ({
  deleteChat: vi.fn(),
  updateChat: vi.fn(),
  restoreChat: vi.fn(),
  removeServerChatMirror: vi.fn(),
  updateHistory: vi.fn(),
  renameTab: vi.fn(),
  showUndoNotification: vi.fn(),
  messageError: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | ({ defaultValue?: string } & Record<string, unknown>),
      maybeOptions?: Record<string, unknown>
    ) => {
      const options =
        typeof defaultValueOrOptions === "object" ? defaultValueOrOptions : maybeOptions || {}
      const template =
        typeof defaultValueOrOptions === "string"
          ? defaultValueOrOptions
          : defaultValueOrOptions?.defaultValue || key
      return template.replace(/\{\{(\w+)\}\}/g, (_match, name: string) => String(options[name] ?? ""))
    }
  })
}))
vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return { ...actual, message: { ...actual.message, error: io.messageError, success: vi.fn(), info: vi.fn() } }
})
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    searchConversationsWithMeta: async () => ({ chats: [], total: 0 }),
    deleteChat: (...args: unknown[]) => io.deleteChat(...args),
    updateChat: (...args: unknown[]) => io.updateChat(...args),
    restoreChat: (...args: unknown[]) => io.restoreChat(...args)
  }
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  removeServerChatMirror: (...args: unknown[]) => io.removeServerChatMirror(...args)
}))
vi.mock("@/db/dexie/helpers", () => ({
  updateHistory: (...args: unknown[]) => io.updateHistory(...args)
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: vi.fn(function MockPageAssistDatabase(this: Record<string, unknown>) {
    this.fullTextSearchChatHistories = async () => []
  })
}))
vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: io.showUndoNotification })
}))
vi.mock("@/store/sidepanel-chat-tabs", () => ({
  useSidepanelChatTabsStore: (select?: (state: unknown) => unknown) =>
    typeof select === "function"
      ? select({ togglePinned: vi.fn(), renameTab: io.renameTab, setStatus: vi.fn() })
      : {}
}))
vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (select?: (state: { mode: string }) => unknown) =>
    typeof select === "function" ? select({ mode: "pro" }) : { mode: "pro" }
}))
vi.mock("@/store/folder", () => {
  const state = { getFoldersForConversation: () => [], uiPrefs: {}, folderApiAvailable: false }
  return {
    useFolderStore: (select?: (value: typeof state) => unknown) =>
      typeof select === "function" ? select(state) : state
  }
})
vi.mock("@/hooks/useBulkChatOperations", () => ({
  useBulkChatOperations: () => ({
    openBulkFolderPicker: vi.fn(),
    openBulkTagPicker: vi.fn(),
    applyBulkDelete: vi.fn()
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [288, vi.fn()] }))
vi.mock("../ModeToggle", () => ({ ModeToggle: () => null }))
vi.mock("../FolderPickerModal", () => ({ FolderPickerModal: () => null }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ isConnected: true })
}))

const owner = {
  ownerKey: "alice-owner",
  isCurrent: () => true,
  snapshot: {
    requestScope: {
      config: { serverUrl: "http://chat.test", authMode: "multi-user" as const },
      userId: 1
    }
  }
}
const serverTab: SidepanelChatTab = {
  id: "tab-server",
  label: "Quarterly plan",
  labelSource: "auto",
  historyId: "mirror-1",
  serverChatId: "chat-1",
  serverChatTopic: null,
  updatedAt: Date.now()
}
const localTab: SidepanelChatTab = {
  id: "tab-local",
  label: "Scratch notes",
  labelSource: "auto",
  historyId: "local-1",
  serverChatId: null,
  serverChatTopic: null,
  updatedAt: Date.now()
}

const renderSidebar = (overrides: Partial<React.ComponentProps<typeof SidepanelChatSidebar>> = {}) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const props = {
    owner,
    open: true,
    variant: "docked" as const,
    tabs: [serverTab, localTab],
    activeTabId: serverTab.id,
    onSelectTab: vi.fn(),
    onCloseTab: vi.fn(),
    onNewTab: vi.fn(),
    onOpenServerChat: vi.fn(),
    searchQuery: "",
    onSearchQueryChange: vi.fn(),
    ...overrides
  }
  render(
    <QueryClientProvider client={client}>
      <SidepanelChatSidebar {...props} />
    </QueryClientProvider>
  )
  return { ...props, client }
}

/** Right-click a tab row and pick an item from its menu, as a user does. */
const chooseTabMenuItem = async (label: string, item: string) => {
  fireEvent.contextMenu(screen.getByRole("button", { name: label }))
  fireEvent.click(await screen.findByRole("menuitem", { name: item }))
}

describe("side-panel tab menu acts on the real conversation (XS-07)", () => {
  beforeEach(() => {
    io.deleteChat.mockReset().mockResolvedValue(undefined)
    io.updateChat.mockReset().mockImplementation(async (_id: string, data: { title: string }) => ({ id: "chat-1", title: data.title, version: 3 }))
    io.restoreChat.mockReset().mockResolvedValue({ id: "chat-1", title: "Quarterly plan", created_at: "2026-10-01T00:00:00Z" })
    io.removeServerChatMirror.mockReset().mockResolvedValue(["mirror-1"])
    io.updateHistory.mockReset().mockResolvedValue(undefined)
    io.renameTab.mockReset()
    io.showUndoNotification.mockReset()
    io.messageError.mockReset()
  })
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("moves a server chat to Trash, drops its local copy and closes its tab, with Undo", async () => {
    const props = renderSidebar()
    await chooseTabMenuItem("Quarterly plan", "Delete")
    const dialog = await screen.findByRole("dialog", { name: "Delete conversation" })
    expect(dialog).not.toHaveTextContent(/cannot be undone/i)
    expect(dialog).toHaveTextContent(/Trash/)
    fireEvent.click(within(dialog).getByRole("button", { name: "Move to Trash" }))

    await waitFor(() => expect(props.onCloseTab).toHaveBeenCalledWith("tab-server"))
    expect(io.deleteChat).toHaveBeenCalledWith("chat-1", { requestScope: owner.snapshot.requestScope })
    expect(io.deleteChat.mock.calls[0][1]).not.toHaveProperty("hardDelete")
    expect(io.removeServerChatMirror).toHaveBeenCalledWith({ chatId: "chat-1", ownerKey: "alice-owner" })
    expect(io.showUndoNotification).toHaveBeenCalledTimes(1)

    const undo = io.showUndoNotification.mock.calls[0][0] as { title: string; onUndo: () => Promise<void> }
    expect(undo.title).toBe("Moved to Trash")
    await act(async () => {
      await undo.onUndo()
    })
    expect(io.restoreChat).toHaveBeenCalledWith("chat-1")
    expect(props.onOpenServerChat).toHaveBeenCalledWith(expect.objectContaining({ id: "chat-1", title: "Quarterly plan" }))
  })

  it("keeps the tab and the dialog when the server cannot move the chat to Trash", async () => {
    io.deleteChat.mockRejectedValue(new Error("HTTP 500"))
    const props = renderSidebar()
    await chooseTabMenuItem("Quarterly plan", "Delete")
    const dialog = await screen.findByRole("dialog", { name: "Delete conversation" })
    fireEvent.click(within(dialog).getByRole("button", { name: "Move to Trash" }))

    await waitFor(() => expect(io.messageError).toHaveBeenCalled())
    expect(props.onCloseTab).not.toHaveBeenCalled()
    expect(io.removeServerChatMirror).not.toHaveBeenCalled()
    expect(io.showUndoNotification).not.toHaveBeenCalled()
    expect(screen.getByRole("dialog", { name: "Delete conversation" })).toBeInTheDocument()
  })

  it("offers 'Close tab' rather than 'Delete' for a tab with no server chat", async () => {
    const props = renderSidebar()
    fireEvent.contextMenu(screen.getByRole("button", { name: "Scratch notes" }))
    await screen.findByRole("menuitem", { name: "Close tab" })
    expect(screen.queryByRole("menuitem", { name: /Delete/ })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("menuitem", { name: "Close tab" }))

    expect(props.onCloseTab).toHaveBeenCalledWith("tab-local")
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(io.deleteChat).not.toHaveBeenCalled()
  })

  it("renames the server chat, its local copy and the tab", async () => {
    renderSidebar()
    await chooseTabMenuItem("Quarterly plan", "Rename")
    const dialog = await screen.findByRole("dialog", { name: "Rename conversation" })
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "Q4 plan" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.renameTab).toHaveBeenCalledWith("tab-server", "Q4 plan"))
    expect(io.updateChat).toHaveBeenCalledWith("chat-1", { title: "Q4 plan" }, { requestScope: owner.snapshot.requestScope })
    expect(io.updateHistory).toHaveBeenCalledWith("mirror-1", "Q4 plan")
  })

  it("keeps the old name when the server rename fails", async () => {
    io.updateChat.mockRejectedValue(new Error("HTTP 409"))
    renderSidebar()
    await chooseTabMenuItem("Quarterly plan", "Rename")
    const dialog = await screen.findByRole("dialog", { name: "Rename conversation" })
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "Q4 plan" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.messageError).toHaveBeenCalled())
    expect(io.renameTab).not.toHaveBeenCalled()
    expect(io.updateHistory).not.toHaveBeenCalled()
    expect(screen.getByRole("dialog", { name: "Rename conversation" })).toBeInTheDocument()
  })

  it("renames a local-only conversation's saved history and its tab", async () => {
    renderSidebar()
    await chooseTabMenuItem("Scratch notes", "Rename")
    const dialog = await screen.findByRole("dialog", { name: "Rename conversation" })
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "Ideas" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.renameTab).toHaveBeenCalledWith("tab-local", "Ideas"))
    expect(io.updateHistory).toHaveBeenCalledWith("local-1", "Ideas")
    expect(io.updateChat).not.toHaveBeenCalled()
  })
})
