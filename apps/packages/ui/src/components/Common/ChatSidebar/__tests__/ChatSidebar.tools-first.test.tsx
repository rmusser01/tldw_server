// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
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
})
