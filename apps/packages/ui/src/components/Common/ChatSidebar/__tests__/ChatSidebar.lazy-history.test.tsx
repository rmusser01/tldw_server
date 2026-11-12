// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatSidebar } from "../../ChatSidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const useSettingMock = vi.hoisted(() => vi.fn())
const optionState = vi.hoisted(() => ({ temporaryChat: false }))
const verifiedOwner = vi.hoisted(() => ({
  ownerKey: "verified-lazy-owner", verificationState: "ready", isCurrent: () => true,
  requestScope: {config: {serverUrl: "https://chat.test", authMode: "multi-user" as const}, userId: 7},
  retryVerification: vi.fn()
}))
const folderListProps = vi.hoisted(() => vi.fn())
vi.mock("@/hooks/useChatDraftOwner", () => ({ useChatDraftOwner: () => verifiedOwner }))
const serverListProps = vi.hoisted(() => vi.fn())
const activeTab = vi.hoisted(() => ({ value: "server" }))
const useServerChatHistoryMock = vi.hoisted(() =>
  vi.fn((..._args: [string, Record<string, unknown>]) => ({
    data: [],
    isLoading: false
  }))
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (...args: unknown[]) => {
    const result = useSettingMock(...args)
    const [value, setValue] = React.useState(result[0])
    const setting = args[0] as { key?: string }
    return setting?.key === "tldw:sidebar:recentCollapsed"
      ? [value, setValue]
      : setting?.key === "tldw:sidebar:activeTab"
        ? [value, (next: string) => { result[1](next); setValue(next) }, result[2]]
        : result
  }
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: (...args: [string, Record<string, unknown>]) =>
    useServerChatHistoryMock(...args)
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: { temporaryChat: boolean }) => unknown) =>
    typeof selector === "function" ? selector(optionState) : optionState
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: (selector?: (state: { conversationKeywordLinks: never[] }) => unknown) =>
    typeof selector === "function"
      ? selector({ conversationKeywordLinks: [] })
      : { conversationKeywordLinks: [] }
}))

vi.mock("@/store/route-transition", () => ({
  useRouteTransitionStore: (selector?: (state: { start: ReturnType<typeof vi.fn> }) => unknown) =>
    typeof selector === "function"
      ? selector({ start: vi.fn() })
      : { start: vi.fn() }
}))

vi.mock("../ServerChatList", () => ({
  ServerChatList: (props: Record<string, unknown>) => {
    serverListProps(props)
    return <div data-testid="server-chat-list" />
  }
}))

vi.mock("../FolderChatList", () => ({
  FolderChatList: (props: Record<string, unknown>) => {
    folderListProps(props)
    return <div data-testid="folder-chat-list" />
  }
}))

vi.mock("../../QuickChatHelper", () => ({
  QuickChatHelperButton: () => null
}))

vi.mock("../../NotesDock", () => ({
  NotesDockButton: () => null
}))

vi.mock("@/components/Sidepanel/Chat/ModeToggle", () => ({
  ModeToggle: () => null
}))

describe("ChatSidebar lazy history loading", () => {
  beforeEach(() => {
    optionState.temporaryChat = false
    verifiedOwner.verificationState = "ready"
    folderListProps.mockClear()
    activeTab.value = "server"
    serverListProps.mockClear()
    useSettingMock.mockReset()
    useServerChatHistoryMock.mockClear()
    const setCurrentTab = vi.fn()
    const setShortcutsCollapsed = vi.fn()
    const setShortcutSelection = vi.fn()
    useSettingMock.mockImplementation((setting: { key?: string } | string) => {
      const key = typeof setting === "string" ? setting : setting?.key
      if (key === "tldw:sidebar:recentCollapsed") return [true, vi.fn()]
      if (key === "tldw:sidebar:activeTab") {
        return [activeTab.value, setCurrentTab]
      }
      if (key === "tldw:sidebar:shortcutsCollapsed") {
        return [false, setShortcutsCollapsed]
      }
      if (key === "tldw:sidebar:shortcutSelection") {
        return [[], setShortcutSelection]
      }
      return [null, vi.fn()]
    })

    useChatSurfaceCoordinatorStore.setState({
      routeId: null,
      surface: null,
      visiblePanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      },
      engagedPanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      }
    })
  })

  it.each(["server", "folders"])("opens Folders with history collapsed from saved %s view", async (savedView) => {
    activeTab.value = savedView
    render(<MemoryRouter><ChatSidebar collapsed={false} /></MemoryRouter>)
    const history = screen.getByRole("button", { name: /Recent conversations/i })
    expect(history).toHaveAttribute("aria-expanded", "false")
    const folders = screen.getByRole("radio", { name: "Folders", exact: true })
    expect(folders.closest("label")).toBeVisible()
    fireEvent.click(folders)
    await waitFor(() => expect(folders).toBeChecked())
    expect(history).toHaveAttribute("aria-expanded", "true")
    const [, saveTab] = useSettingMock.mock.results.find((result, index) =>
      useSettingMock.mock.calls[index][0]?.key === "tldw:sidebar:activeTab"
    )!.value
    if (savedView !== "folders") expect(saveTab).toHaveBeenCalledWith("folders")
    expect(useServerChatHistoryMock).toHaveBeenLastCalledWith("", expect.objectContaining({ enabled: false }))
  })

  it("keeps server history overview disabled while recent conversations are collapsed", async () => {
    useChatSurfaceCoordinatorStore.setState({
      engagedPanels: {
        "server-history": true,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      }
    })

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
        "",
        expect.objectContaining({
          enabled: false,
          mode: "overview"
        })
      )
    })
  })

  it("keeps saved chats clickable in temporary mode so they open read-only (TASK-13389)", () => {
    optionState.temporaryChat = true

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    const list = screen.getByTestId("server-chat-list")
    expect(list.closest(".pointer-events-none")).toBeNull()
    expect(serverListProps).toHaveBeenLastCalledWith(
      expect.objectContaining({ readOnly: true, selectionMode: false })
    )
    expect(screen.queryByRole("button", { name: "Select chats" })).toBeNull()
  })

  it("keeps folder management disabled in temporary mode", () => {
    optionState.temporaryChat = true
    activeTab.value = "folders"

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    expect(screen.getByTestId("folder-chat-list").closest(".pointer-events-none")).toBeNull()
    expect(folderListProps).toHaveBeenLastCalledWith(expect.objectContaining({readOnly: true}))
  })

  it("enables server history overview after the user expands recent conversations", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    await waitFor(() => {
      expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
        "",
        expect.objectContaining({
          enabled: true,
          mode: "overview"
        })
      )
    })
  })

  it("keeps server history overview enabled when an active search survives reset", async () => {
    const { rerender } = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} openResetKey={1} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
    fireEvent.change(screen.getByTestId("chat-sidebar-search"), {
      target: { value: "alpha" }
    })

    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} openResetKey={2} />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
        "",
        expect.objectContaining({
          enabled: true,
          mode: "overview"
        })
      )
    })
  })

  it("does not authorize server history from expansion before account verification", () => {
    verifiedOwner.verificationState = "pending"
    render(<MemoryRouter><ChatSidebar collapsed={false} /></MemoryRouter>)
    fireEvent.click(screen.getByRole("button", {name: /Recent conversations/i}))
    expect(useServerChatHistoryMock).toHaveBeenLastCalledWith("", expect.objectContaining({enabled: false, owner: undefined}))
  })

  it("does not fetch a server count badge on first expanded mount", () => {
    useChatSurfaceCoordinatorStore.setState({
      engagedPanels: {
        "server-history": true,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      }
    })

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(useServerChatHistoryMock).toHaveBeenCalledTimes(1)
    expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({ enabled: false })
    )
  })
it("lazily mounts On this device while leaving the server coordinator disabled", async () => {
  activeTab.value = "local"
  const view = render(<MemoryRouter><ChatSidebar collapsed={false} /></MemoryRouter>)
  expect(screen.queryByTestId("local-chat-list")).toBeNull()
  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
  expect(screen.getByTestId("local-chat-list")).toBeInTheDocument()
  expect(screen.getByText("On this device")).toBeInTheDocument()
  expect(useServerChatHistoryMock).toHaveBeenLastCalledWith("", expect.objectContaining({ enabled: false }))
  expect(useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]).toBe(false)
  view.rerender(<MemoryRouter><ChatSidebar collapsed /></MemoryRouter>)
  expect(screen.queryByTestId("local-chat-list")).toBeNull()
})

})

vi.mock("../LocalChatList", () => ({ LocalChatList: () => <div data-testid="local-chat-list" /> }))
