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
  getChat: vi.fn(),
  getTitleById: vi.fn(),
  removeServerChatMirror: vi.fn(),
  updateHistory: vi.fn(),
  renameTab: vi.fn(),
  showUndoNotification: vi.fn(),
  messageError: vi.fn(),
  messageInfo: vi.fn(),
  messageSuccess: vi.fn(),
  messageWarning: vi.fn()
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
  return { ...actual, message: { ...actual.message, error: io.messageError, success: io.messageSuccess, info: io.messageInfo, warning: io.messageWarning } }
})
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    searchConversationsWithMeta: async () => ({ chats: [], total: 0 }),
    listChatsWithMeta: async () => ({ chats: [], total: 0 }),
    deleteChat: (...args: unknown[]) => io.deleteChat(...args),
    updateChat: (...args: unknown[]) => io.updateChat(...args),
    restoreChat: (...args: unknown[]) => io.restoreChat(...args),
    getChat: (...args: unknown[]) => io.getChat(...args)
  }
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  removeServerChatMirror: (...args: unknown[]) => io.removeServerChatMirror(...args)
}))
vi.mock("@/db/dexie/helpers", () => ({
  updateHistory: (...args: unknown[]) => io.updateHistory(...args),
  getTitleById: (...args: unknown[]) => io.getTitleById(...args)
}))
vi.mock("@/db/dexie/chat", () => ({
  watchOnDeviceHistoryChanges: () => () => {},
  PageAssistDatabase: vi.fn(function MockPageAssistDatabase(this: Record<string, unknown>) {
    this.fullTextSearchChatHistories = async () => []
    this.getRecentChatHistories = async () => []
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
/** A chat whose title is longer than the 40 characters a tab label keeps. */
const fullTitle = "Quarterly planning review for the northern region sales team"
const longTab: SidepanelChatTab = {
  id: "tab-long",
  label: `${fullTitle.slice(0, 40)}...`,
  labelSource: "auto",
  historyId: "mirror-long",
  serverChatId: "chat-long",
  serverChatTopic: null,
  updatedAt: Date.now()
}
const serverTitles: Record<string, string> = { "chat-1": "Quarterly plan", "chat-long": fullTitle }
const localTitles: Record<string, string> = { "mirror-1": "Quarterly plan", "mirror-long": fullTitle, "local-1": "Scratch notes" }
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
  const view = render(
    <QueryClientProvider client={client}>
      <SidepanelChatSidebar {...props} />
    </QueryClientProvider>
  )
  return { ...props, client, view }
}

/** Right-click a tab row and pick an item from its menu, as a user does. */
const chooseTabMenuItem = async (label: string, item: string) => {
  fireEvent.contextMenu(screen.getByRole("button", { name: label }))
  fireEvent.click(await screen.findByRole("menuitem", { name: item }))
}

/**
 * Whether antd has started closing a dialog. jsdom never finishes the close
 * animation, so a closed dialog stays in the document in its leave state.
 */
const isClosing = (dialog: HTMLElement) => /-leave\b/.test(dialog.className)

/** Open a tab's Rename dialog and wait until its title field is ready to edit. */
const openRenameDialog = async (label: string) => {
  await chooseTabMenuItem(label, "Rename")
  const dialog = await screen.findByRole("dialog", { name: "Rename conversation" })
  const textbox = within(dialog).getByRole("textbox")
  await waitFor(() => expect(textbox).toBeEnabled())
  return { dialog, textbox }
}

describe("side-panel tab menu acts on the real conversation (XS-07)", () => {
  beforeEach(() => {
    io.deleteChat.mockReset().mockResolvedValue(undefined)
    io.updateChat.mockReset().mockImplementation(async (_id: string, data: { title: string }) => ({ id: "chat-1", title: data.title, version: 3 }))
    io.restoreChat.mockReset().mockResolvedValue({ id: "chat-1", title: "Quarterly plan", created_at: "2026-10-01T00:00:00Z" })
    io.getChat.mockReset().mockImplementation(async (id: string) => ({ id, title: serverTitles[id] }))
    io.getTitleById.mockReset().mockImplementation(async (id: string) => localTitles[id] ?? "")
    io.removeServerChatMirror.mockReset().mockResolvedValue(["mirror-1"])
    io.updateHistory.mockReset().mockResolvedValue(undefined)
    io.renameTab.mockReset()
    io.showUndoNotification.mockReset()
    io.messageError.mockReset()
    io.messageInfo.mockReset()
    io.messageSuccess.mockReset()
    io.messageWarning.mockReset()
  })
  afterEach(() => {
    document.body.innerHTML = ""
  })

  const openBulkDialog = async () => {
    fireEvent.click(screen.getByRole("button", { name: "Select chats" }))
    fireEvent.click(screen.getByRole("button", { name: "Select all" }))
    fireEvent.click(screen.getByRole("button", { name: "Delete" }))
    return screen.findByRole("dialog", { name: "Delete selected chats?" })
  }

  it("names local Temporary discard before accepting one bulk choice, and Cancel closes nothing", async () => {
    const props = renderSidebar({ tabs: [localTab] })
    const dialog = await openBulkDialog()
    expect(dialog).toHaveTextContent(
      "Closing a Temporary tab discards its messages."
    )
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(props.onCloseTab).not.toHaveBeenCalled()
    expect(io.deleteChat).not.toHaveBeenCalled()
  })

  it("awaits each local closure and counts only accepted tabs", async () => {
    let finish!: (accepted: boolean) => void
    const second = { ...localTab, id: "tab-second", label: "Second local" }
    const close = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<boolean>((resolve) => {
            finish = resolve
          })
      )
      .mockResolvedValueOnce(false)
    renderSidebar({ tabs: [localTab, second], onCloseTab: close })
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(close).toHaveBeenCalledTimes(1))
    expect(close).toHaveBeenCalledWith(
      "tab-local",
      expect.objectContaining({
        discardTemporaryConfirmed: true,
        isCurrent: expect.any(Function)
      })
    )
    expect(io.messageInfo).not.toHaveBeenCalled()
    await act(async () => {
      finish(true)
    })
    await waitFor(() => expect(close).toHaveBeenCalledTimes(2))
    expect(io.messageInfo).toHaveBeenCalledWith("1 local chat(s) were closed.")
    expect(screen.getByText("1 selected")).toBeInTheDocument()
  })

  it("keeps declined local tabs selected without reporting closure", async () => {
    renderSidebar({ tabs: [localTab], onCloseTab: vi.fn(async () => false) })
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(isClosing(dialog)).toBe(true))
    expect(io.messageInfo).not.toHaveBeenCalled()
    expect(screen.getByText("1 selected")).toBeInTheDocument()
  })

  it("uses the verified request scope for bulk Trash and retains failed server tabs", async () => {
    io.deleteChat.mockRejectedValue(new Error("Trash unavailable"))
    const props = renderSidebar()
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(isClosing(dialog)).toBe(true))
    expect(io.deleteChat).toHaveBeenCalledWith("chat-1", {
      requestScope: owner.snapshot.requestScope
    })
    expect(props.onCloseTab).toHaveBeenCalledExactlyOnceWith(
      "tab-local",
      expect.objectContaining({
        discardTemporaryConfirmed: true,
        isCurrent: expect.any(Function)
      })
    )
    expect(screen.getByText("1 selected")).toBeInTheDocument()
  })

  it.each(["server", "local"])(
    "retires aggregate closure and feedback after held %s work loses its owner",
    async (boundary) => {
      let current = true
      let finish!: () => void
      const held = new Promise<void>((resolve) => {
        finish = resolve
      })
      const owned = { ...owner, isCurrent: () => current }
      const close = boundary === "local" ? vi.fn(() => held) : vi.fn()
      if (boundary === "server") io.deleteChat.mockReturnValue(held)
      renderSidebar({
        owner: owned,
        tabs:
          boundary === "local"
            ? [localTab, { ...localTab, id: "second", label: "Second local" }]
            : [serverTab, localTab],
        onCloseTab: close
      })
      const dialog = await openBulkDialog()
      fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
      await waitFor(() =>
        expect(
          boundary === "local" ? close : io.deleteChat
        ).toHaveBeenCalledTimes(1)
      )
      current = false
      await act(async () => {
        finish()
      })
      expect(close).toHaveBeenCalledTimes(boundary === "local" ? 1 : 0)
      expect(io.messageSuccess).not.toHaveBeenCalled()
      expect(io.messageInfo).not.toHaveBeenCalled()
      expect(screen.getByText("2 selected")).toBeInTheDocument()
    }
  )

  it("retires held bulk Trash when a selected tab binds a different conversation under the same ID", async () => {
    let finish!: () => void
    io.deleteChat.mockReturnValue(
      new Promise<void>((resolve) => {
        finish = resolve
      })
    )
    const props = renderSidebar()
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(io.deleteChat).toHaveBeenCalledTimes(1))
    props.view.rerender(
      <QueryClientProvider client={props.client}>
        <SidepanelChatSidebar
          {...props}
          tabs={[{ ...serverTab, serverChatId: "replacement-chat" }, localTab]}
        />
      </QueryClientProvider>
    )
    await act(async () => {
      finish()
    })
    expect(props.onCloseTab).not.toHaveBeenCalled()
    expect(io.messageInfo).not.toHaveBeenCalled()
    expect(io.messageSuccess).not.toHaveBeenCalled()
    expect(screen.getByText("2 selected")).toBeInTheDocument()
  })

  it.each(["cancel", "selection", "unmount"])(
    "retires held local closure after %s without later closes or aggregate feedback",
    async (ending) => {
      let finish!: () => void
      const close = vi.fn(
        () =>
          new Promise<void>((resolve) => {
            finish = resolve
          })
      )
      const props = renderSidebar({
        tabs: [localTab, { ...localTab, id: "second", label: "Second local" }],
        onCloseTab: close
      })
      const dialog = await openBulkDialog()
      fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
      await waitFor(() => expect(close).toHaveBeenCalledTimes(1))
      if (ending === "cancel")
        fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
      else if (ending === "selection")
        fireEvent.click(screen.getByRole("button", { name: "Clear" }))
      else props.view.unmount()
      await act(async () => {
        finish()
      })
      expect(close).toHaveBeenCalledTimes(1)
      expect(io.messageInfo).not.toHaveBeenCalled()
      if (ending === "cancel")
        expect(screen.getByText("2 selected")).toBeInTheDocument()
      if (ending === "selection")
        expect(screen.getByText("0 selected")).toBeInTheDocument()
    }
  )

  it("allows a new bulk attempt after Cancel and prevents old completion from clearing its loading state", async () => {
    let finishOld!: () => void
    let finishNew!: () => void
    const close = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishOld = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishNew = resolve
          })
      )
    renderSidebar({ tabs: [localTab], onCloseTab: close })
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: /Delete$/ }))
    await waitFor(() => expect(close).toHaveBeenCalledTimes(1))
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    fireEvent.click(
      within(screen.getByTestId("sidepanel-chat-sidebar")).getByRole("button", {
        name: "Delete"
      })
    )
    const currentDialog = await screen.findByRole("dialog", {
      name: "Delete selected chats?"
    })
    fireEvent.click(
      within(currentDialog).getByRole("button", { name: /Delete$/ })
    )
    await waitFor(() => expect(close).toHaveBeenCalledTimes(2))
    await act(async () => {
      finishOld()
    })
    expect(
      within(currentDialog).getByRole("button", { name: /Delete$/ }).className
    ).toContain("ant-btn-loading")
    expect(io.messageInfo).not.toHaveBeenCalled()
    await act(async () => {
      finishNew()
    })
    expect(io.messageInfo).toHaveBeenCalledExactlyOnceWith(
      "1 local chat(s) were closed."
    )
  })

  it.each(["owner", "unmount"])(
    "silences an old single Trash failure after %s retirement",
    async (ending) => {
      let current = true
      let fail!: (error: Error) => void
      io.deleteChat.mockReturnValue(
        new Promise<void>((_resolve, reject) => {
          fail = reject
        })
      )
      const props = renderSidebar({
        owner: { ...owner, isCurrent: () => current }
      })
      await chooseTabMenuItem("Quarterly plan", "Delete")
      const dialog = await screen.findByRole("dialog", {
        name: "Delete conversation"
      })
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Move to Trash" })
      )
      await waitFor(() => expect(io.deleteChat).toHaveBeenCalledTimes(1))
      if (ending === "owner") current = false
      else props.view.unmount()
      await act(async () => {
        fail(new Error("Old Trash failure"))
      })
      expect(io.messageError).not.toHaveBeenCalled()
      expect(props.onCloseTab).not.toHaveBeenCalled()
      expect(io.removeServerChatMirror).not.toHaveBeenCalled()
    }
  )

  it("removes acknowledged Trash from device discovery even when tab closure declines", async () => {
    let finish!: (accepted: boolean) => void
    const props = renderSidebar({
      onCloseTab: vi.fn(
        () =>
          new Promise<boolean>((resolve) => {
            finish = resolve
          })
      )
    })
    await chooseTabMenuItem("Quarterly plan", "Delete")
    const dialog = await screen.findByRole("dialog", {
      name: "Delete conversation"
    })
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Move to Trash" })
    )
    await waitFor(() => expect(props.onCloseTab).toHaveBeenCalled())
    expect(io.removeServerChatMirror).toHaveBeenCalledWith(
      expect.objectContaining({ chatId: "chat-1", ownerKey: "alice-owner" })
    )
    await act(async () => {
      finish(false)
    })
    expect(io.removeServerChatMirror).toHaveBeenCalledTimes(1)
    expect(io.deleteChat).toHaveBeenCalledExactlyOnceWith("chat-1", {
      requestScope: owner.snapshot.requestScope
    })
    // Trash already succeeded; keep its Undo available even when the tab stays open.
    expect(io.showUndoNotification).toHaveBeenCalledTimes(1)
  })
  it.each(["cancel", "binding", "owner"])(
    "retires held compact cleanup at %s before any close or aggregate feedback",
    async (ending) => {
      let current = true
      let finish!: () => void
      io.removeServerChatMirror.mockReturnValue(
        new Promise<void>((resolve) => {
          finish = resolve
        })
      )
      const props = renderSidebar({
        owner: { ...owner, isCurrent: () => current },
        tabs: [serverTab]
      })
      const dialog = await openBulkDialog()
      fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
      await waitFor(() => expect(io.removeServerChatMirror).toHaveBeenCalled())
      const lease = io.removeServerChatMirror.mock.calls[0][0]
        .validate_lease as () => boolean
      if (ending === "cancel")
        fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
      else if (ending === "owner") current = false
      else
        props.view.rerender(
          <QueryClientProvider client={props.client}>
            <SidepanelChatSidebar
              {...props}
              tabs={[{ ...serverTab, serverChatId: "new-chat" }]}
            />
          </QueryClientProvider>
        )
      expect(lease()).toBe(false)
      await act(async () => {
        finish()
      })
      expect(props.onCloseTab).not.toHaveBeenCalled()
      expect(io.messageWarning).not.toHaveBeenCalled()
      expect(io.messageInfo).not.toHaveBeenCalled()
    }
  )

  it("cleans acknowledged bulk Trash even if the saved tab close is refused", async () => {
    const props = renderSidebar({
      tabs: [serverTab],
      onCloseTab: vi.fn().mockResolvedValue(false)
    })
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(props.onCloseTab).toHaveBeenCalled())
    expect(io.removeServerChatMirror).toHaveBeenCalledWith(
      expect.objectContaining({
        chatId: "chat-1",
        ownerKey: "alice-owner",
        scope: { type: "global" },
        validate_lease: expect.any(Function)
      })
    )
    expect(io.deleteChat).toHaveBeenCalledTimes(1)
    expect(io.messageInfo).not.toHaveBeenCalled()
  })
  it("warns about the retained device cache without retrying acknowledged bulk Trash", async () => {
    io.removeServerChatMirror.mockRejectedValue(
      new Error("Pending local settings")
    )
    const props = renderSidebar({
      tabs: [serverTab],
      onCloseTab: vi.fn().mockResolvedValue(true)
    })
    const dialog = await openBulkDialog()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(io.messageWarning).toHaveBeenCalled())
    expect(props.onCloseTab).toHaveBeenCalledTimes(1)
    expect(io.deleteChat).toHaveBeenCalledTimes(1)
    expect(io.messageError).not.toHaveBeenCalled()
    expect(io.messageSuccess).toHaveBeenCalledWith("Chats moved to trash.")
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
    expect(io.removeServerChatMirror).toHaveBeenCalledWith(expect.objectContaining({ chatId: "chat-1", ownerKey: "alice-owner" }))
    expect(io.showUndoNotification).toHaveBeenCalledTimes(1)

    const undo = io.showUndoNotification.mock.calls[0][0] as { title: string; onUndo: () => Promise<void> }
    expect(undo.title).toBe("Moved to Trash")
    await act(async () => {
      await undo.onUndo()
    })
    expect(io.restoreChat).toHaveBeenCalledWith("chat-1", { requestScope: owner.snapshot.requestScope })
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
    expect(isClosing(screen.getByRole("dialog", { name: "Delete conversation" }))).toBe(false)
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
    const { dialog, textbox } = await openRenameDialog("Quarterly plan")
    fireEvent.change(textbox, { target: { value: "Q4 plan" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.renameTab).toHaveBeenCalledWith("tab-server", "Q4 plan"))
    expect(io.updateChat).toHaveBeenCalledWith("chat-1", { title: "Q4 plan" }, { requestScope: owner.snapshot.requestScope })
    expect(io.updateHistory).toHaveBeenCalledWith("mirror-1", "Q4 plan")
  })

  it("keeps the old name when the server rename fails", async () => {
    io.updateChat.mockRejectedValue(new Error("HTTP 409"))
    renderSidebar()
    const { dialog, textbox } = await openRenameDialog("Quarterly plan")
    fireEvent.change(textbox, { target: { value: "Q4 plan" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.messageError).toHaveBeenCalled())
    expect(io.renameTab).not.toHaveBeenCalled()
    expect(io.updateHistory).not.toHaveBeenCalled()
    expect(isClosing(screen.getByRole("dialog", { name: "Rename conversation" }))).toBe(false)
  })

  it("renames a local-only conversation's saved history and its tab", async () => {
    renderSidebar()
    const { dialog, textbox } = await openRenameDialog("Scratch notes")
    fireEvent.change(textbox, { target: { value: "Ideas" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.renameTab).toHaveBeenCalledWith("tab-local", "Ideas"))
    expect(io.updateHistory).toHaveBeenCalledWith("local-1", "Ideas")
    expect(io.updateChat).not.toHaveBeenCalled()
  })

  it("prefills Rename with the chat's full title, not its truncated tab label", async () => {
    renderSidebar({ tabs: [longTab, localTab], activeTabId: longTab.id })
    const { dialog, textbox } = await openRenameDialog(longTab.label)
    expect(textbox).toHaveValue(fullTitle)
    fireEvent.change(textbox, { target: { value: `${(textbox as HTMLInputElement).value} v2` } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(io.updateChat).toHaveBeenCalled())
    expect(io.updateChat).toHaveBeenCalledWith(
      "chat-long",
      { title: `${fullTitle} v2` },
      { requestScope: owner.snapshot.requestScope }
    )
    expect(io.getChat).toHaveBeenCalledWith("chat-long", { requestScope: owner.snapshot.requestScope })
  })

  it("does nothing when the full title is saved unchanged", async () => {
    renderSidebar({ tabs: [longTab, localTab], activeTabId: longTab.id })
    const { dialog, textbox } = await openRenameDialog(longTab.label)
    expect(textbox).toHaveValue(fullTitle)
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(isClosing(dialog)).toBe(true))
    expect(io.updateChat).not.toHaveBeenCalled()
    expect(io.renameTab).not.toHaveBeenCalled()
    expect(io.updateHistory).not.toHaveBeenCalled()
  })

  it("falls back to the local copy's full title when the server can't be read", async () => {
    io.getChat.mockRejectedValue(new Error("offline"))
    renderSidebar({ tabs: [longTab, localTab], activeTabId: longTab.id })
    const { textbox } = await openRenameDialog(longTab.label)
    expect(textbox).toHaveValue(fullTitle)
    expect(io.getTitleById).toHaveBeenCalledWith("mirror-long")
  })
})
