// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatSidebar } from "../../ChatSidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const useSettingMock = vi.hoisted(() => vi.fn())
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
  useSetting: (...args: unknown[]) => useSettingMock(...args)
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
    typeof selector === "function" ? selector({ temporaryChat: false }) : { temporaryChat: false }
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

describe("ChatSidebar lazy history loading", () => {
  beforeEach(() => {
    useSettingMock.mockReset()
    useServerChatHistoryMock.mockClear()
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
})
