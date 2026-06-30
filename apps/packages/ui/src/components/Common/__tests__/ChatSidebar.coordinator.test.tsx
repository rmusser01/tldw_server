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
  useSetting: (...args: unknown[]) => useSettingMock(...args)
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
})
