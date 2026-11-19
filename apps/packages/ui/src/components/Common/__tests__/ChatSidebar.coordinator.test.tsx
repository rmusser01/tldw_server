// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { ChatSidebar } from "../ChatSidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const useSettingMock = vi.hoisted(() => vi.fn())

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
      : result
  }
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value
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
    typeof selector === "function" ? selector({ temporaryChat: false }) : { temporaryChat: false }
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

vi.mock("../ChatSidebar/ServerChatList", () => ({
  ServerChatList: () => <div data-testid="server-chat-list" />
}))

vi.mock("../ChatSidebar/FolderChatList", () => ({
  FolderChatList: () => <div data-testid="folder-chat-list" />
}))

vi.mock("../QuickChatHelper", () => ({
  QuickChatHelperButton: () => null
}))

vi.mock("../NotesDock", () => ({
  NotesDockButton: () => null
}))

vi.mock("@/components/Sidepanel/Chat/ModeToggle", () => ({
  ModeToggle: () => null
}))

describe("ChatSidebar coordinator integration", () => {
  beforeEach(() => {
    useSettingMock.mockReset()
    const setCurrentTab = vi.fn()
    const setShortcutsCollapsed = vi.fn()
    const setShortcutSelection = vi.fn()
    useSettingMock.mockImplementation((setting: { key?: string } | string) => {
      const key = typeof setting === "string" ? setting : setting?.key
      if (key === "tldw:sidebar:recentCollapsed") return [true, vi.fn()]
      if (key === "tldw:sidebar:activeTab") {
        return ["server", setCurrentTab]
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

  it("marks server history visible only when recent conversations are expanded", () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(
      useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
    ).toBe(false)

    fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

    expect(
      useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
    ).toBe(true)
  })

  // AX-14: screen-reader users find chat history by landmark.
  it.each([true, false])("is a Chat history navigation landmark when collapsed=%s", (collapsed) => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={collapsed} />
      </MemoryRouter>
    )

    const landmark = screen.getByRole("navigation", { name: "Chat history" })
    expect(landmark).toBe(screen.getByTestId("chat-sidebar"))
  })

  // XP-11 (#3101, decision 6): the chat list belongs to /chat. Other routes
  // keep the sidebar's navigation, Settings and tools, without the chat list,
  // its New Chat button or its search.
  it.each([false, true])("leaves the chat list out of a non-chat route when collapsed=%s", (collapsed) => {
    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <ChatSidebar collapsed={collapsed} />
      </MemoryRouter>
    )

    const landmark = screen.getByRole("navigation", { name: "App navigation" })
    expect(landmark).toBe(screen.getByTestId("chat-sidebar"))
    expect(screen.queryByRole("navigation", { name: "Chat history" })).not.toBeInTheDocument()
    expect(screen.queryByTestId("chat-sidebar-new-chat")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Find conversations" })).not.toBeInTheDocument()
    expect(screen.queryByTestId("chat-sidebar-search")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Recent conversations/i })).not.toBeInTheDocument()
    expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
    expect(screen.getByTestId("chat-sidebar-settings")).toBeInTheDocument()
    expect(screen.getByTestId("chat-sidebar-toggle")).toBeInTheDocument()
  })

  it("keeps the expanded sidebar's navigation on a non-chat route, under a Menu heading", () => {
    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(screen.getByRole("heading", { level: 2, name: "Menu" })).toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Chats" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Navigation" })).toBeInTheDocument()
  })

  it("reads no server chat history on a non-chat route", () => {
    useSettingMock.mockImplementation((setting: { key?: string } | string) => {
      const key = typeof setting === "string" ? setting : setting?.key
      if (key === "tldw:sidebar:recentCollapsed") return [false, vi.fn()]
      if (key === "tldw:sidebar:activeTab") return ["server", vi.fn()]
      if (key === "tldw:sidebar:shortcutsCollapsed") return [false, vi.fn()]
      if (key === "tldw:sidebar:shortcutSelection") return [[], vi.fn()]
      return [null, vi.fn()]
    })

    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(
      useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
    ).toBe(false)
  })

  it("shows the chat list on the chat route", () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} />
      </MemoryRouter>
    )

    expect(screen.getByRole("heading", { level: 2, name: "Chats" })).toBeInTheDocument()
    expect(screen.getByTestId("chat-sidebar-new-chat")).toBeInTheDocument()
    expect(screen.getByTestId("chat-sidebar-search")).toBeInTheDocument()
  })
})

vi.mock("../ChatSidebar/LocalChatList", () => ({ LocalChatList: () => <div data-testid="local-chat-list" /> }))
