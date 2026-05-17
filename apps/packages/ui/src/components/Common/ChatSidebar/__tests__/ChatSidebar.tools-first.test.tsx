// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatSidebar } from "../../ChatSidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const settingState = vi.hoisted(() => ({
  activeTab: "server",
  shortcutsCollapsed: true,
  shortcutSelection: ["quick-ingest", "chat"],
  setActiveTab: vi.fn((next: string) => {
    settingState.activeTab = next
  }),
  setShortcutsCollapsed: vi.fn(async (next: boolean) => {
    settingState.shortcutsCollapsed = next
  }),
  setShortcutSelection: vi.fn()
}))

const debounceState = vi.hoisted(() => ({
  override: undefined as unknown
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key?: string }) => {
    if (setting.key === "tldw:sidebar:activeTab") {
      return [settingState.activeTab, settingState.setActiveTab]
    }
    if (setting.key === "tldw:sidebar:shortcutsCollapsed") {
      return [
        settingState.shortcutsCollapsed,
        settingState.setShortcutsCollapsed
      ]
    }
    if (setting.key === "tldw:sidebar:shortcutSelection") {
      return [settingState.shortcutSelection, settingState.setShortcutSelection]
    }
    return [null, vi.fn()]
  }
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) =>
    debounceState.override === undefined ? value : (debounceState.override as T)
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: () => ({ data: [], total: 0 })
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: { temporaryChat: boolean }) => unknown) =>
    typeof selector === "function"
      ? selector({ temporaryChat: false })
      : { temporaryChat: false }
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: (selector?: (state: { conversationKeywordLinks: never[] }) => unknown) =>
    typeof selector === "function"
      ? selector({ conversationKeywordLinks: [] })
      : { conversationKeywordLinks: [] }
}))

vi.mock("@/store/route-transition", () => ({
  useRouteTransitionStore: (selector?: (state: {
    start: ReturnType<typeof vi.fn>
  }) => unknown) =>
    typeof selector === "function"
      ? selector({ start: vi.fn() })
      : { start: vi.fn() }
}))

vi.mock("../ServerChatList", () => ({
  ServerChatList: () => <div data-testid="server-chat-list" />
}))

vi.mock("../FolderChatList", () => ({
  FolderChatList: () => <div data-testid="folder-chat-list" />
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

const renderSidebar = (
  props?: Partial<React.ComponentProps<typeof ChatSidebar>>
) =>
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} {...props} />
    </MemoryRouter>
  )

describe("ChatSidebar tools-first reset", () => {
  beforeEach(() => {
    settingState.activeTab = "server"
    settingState.shortcutsCollapsed = true
    settingState.shortcutSelection = ["quick-ingest", "chat"]
    settingState.setActiveTab.mockClear()
    settingState.setShortcutsCollapsed.mockClear()
    settingState.setShortcutSelection.mockClear()
    debounceState.override = undefined
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

  it("direct expanded mount opens shortcuts and keeps recent conversations collapsed", async () => {
    const { rerender } = renderSidebar()

    await waitFor(() => {
      expect(settingState.setShortcutsCollapsed).toHaveBeenCalledWith(false)
    })
    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(screen.getByRole("button", { name: /Shortcuts/i })).toHaveAttribute(
      "aria-expanded",
      "true"
    )
    expect(
      screen.getByRole("button", { name: /Recent conversations/i })
    ).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByTestId("chat-sidebar-shortcut-quick-ingest")).toBeInTheDocument()
    expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
  })

  it("resets to tools-first when collapsed sidebar is expanded", async () => {
    const { rerender } = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed />
      </MemoryRouter>
    )

    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(settingState.setShortcutsCollapsed).toHaveBeenCalledWith(false)
    })
    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(screen.getByRole("button", { name: /Shortcuts/i })).toHaveAttribute(
      "aria-expanded",
      "true"
    )
    expect(
      screen.getByRole("button", { name: /Recent conversations/i })
    ).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
  })

  it("shows recent conversation controls when the user expands recent conversations", () => {
    renderSidebar()

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    expect(screen.getByTestId("chat-sidebar-search")).toBeInTheDocument()
    expect(screen.getByLabelText("Chat view")).toBeInTheDocument()
    expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()
  })

  it("hides server selection controls while recent conversations are collapsed", () => {
    renderSidebar()

    expect(
      screen.queryByRole("button", { name: /Select chats/i })
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    expect(screen.getByRole("button", { name: /Select chats/i })).toBeInTheDocument()
  })

  it("exits selection mode when recent conversations collapse", () => {
    renderSidebar()

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
    fireEvent.click(screen.getByRole("button", { name: /Select chats/i }))
    expect(screen.getByRole("button", { name: /Exit selection/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    expect(screen.getByRole("button", { name: /Select chats/i })).toBeInTheDocument()
  })

  it("keeps search controls reachable when a query is active across reset", () => {
    debounceState.override = ""
    const { rerender } = renderSidebar({ openResetKey: 1 })

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
    fireEvent.change(screen.getByTestId("chat-sidebar-search"), {
      target: { value: "alpha" }
    })

    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} openResetKey={2} />
      </MemoryRouter>
    )

    expect(screen.getByTestId("chat-sidebar-search")).toBeInTheDocument()
    expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()
  })

  it("resets to tools-first when openResetKey changes while already expanded", async () => {
    const { rerender } = renderSidebar({ openResetKey: 1 })

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
    expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()

    rerender(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} openResetKey={2} />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: /Shortcuts/i })).toHaveAttribute(
      "aria-expanded",
      "true"
    )
  })
})
