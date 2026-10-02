// @vitest-environment jsdom
import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidepanelChatSidebar } from "../Sidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import type { HistoryInfo } from "@/db/dexie/types"

const useServerChatHistoryMock = vi.hoisted(() =>
  vi.fn((_options?: unknown) => ({ data: [], isLoading: false }))
)
const fullTextSearchChatHistoriesMock = vi.hoisted(() =>
  vi.fn<(...args: unknown[]) => Promise<HistoryInfo[]>>(async () => [])
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValueOrOptions?: string | { defaultValue?: string }) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : defaultValueOrOptions?.defaultValue || _key
  })
}))

vi.mock("antd", () => ({
  message: {
    success: vi.fn(),
    error: vi.fn()
  },
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Modal: ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null)
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: vi.fn()
  })
}))

vi.mock("@/store/sidepanel-chat-tabs", () => ({
  useSidepanelChatTabsStore: (
    selector?: (state: {
      togglePinned: ReturnType<typeof vi.fn>
      renameTab: ReturnType<typeof vi.fn>
      setStatus: ReturnType<typeof vi.fn>
    }) => unknown
  ) =>
    typeof selector === "function"
      ? selector({
          togglePinned: vi.fn(),
          renameTab: vi.fn(),
          setStatus: vi.fn()
        })
      : {}
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector?: (state: { mode: string }) => unknown) =>
    typeof selector === "function" ? selector({ mode: "pro" }) : { mode: "pro" }
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  useServerChatHistory: (...args: unknown[]) => useServerChatHistoryMock(...args)
}))

vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: vi.fn(
    function MockPageAssistDatabase(this: { fullTextSearchChatHistories: typeof fullTextSearchChatHistoriesMock }) {
      this.fullTextSearchChatHistories = fullTextSearchChatHistoriesMock
    }
  )
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: (
    selector?: (state: {
      getFoldersForConversation: (conversationId: string) => string[]
      uiPrefs: { showFolders: boolean }
      folderApiAvailable: boolean
    }) => unknown
  ) =>
    typeof selector === "function"
      ? selector({
          getFoldersForConversation: () => [],
          uiPrefs: { showFolders: true },
          folderApiAvailable: true
        })
      : {
          getFoldersForConversation: () => [],
          uiPrefs: { showFolders: true },
          folderApiAvailable: true
        }
}))

vi.mock("@/hooks/useBulkChatOperations", () => ({
  useBulkChatOperations: () => ({
    openBulkFolderPicker: vi.fn(),
    openBulkTagPicker: vi.fn(),
    applyBulkDelete: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [288, vi.fn()]
}))

vi.mock("../ModeToggle", () => ({
  ModeToggle: () => null
}))

vi.mock("../ConversationContextMenu", () => ({
  ConversationContextMenu: ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  )
}))

vi.mock("../FolderPickerModal", () => ({
  FolderPickerModal: () => null
}))

describe("SidepanelChatSidebar coordinator integration", () => {
  beforeEach(() => {
    useServerChatHistoryMock.mockClear()
    fullTextSearchChatHistoriesMock.mockClear()
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

  it("tracks extension route ownership and server-history engagement", async () => {
    const { rerender } = render(
      <SidepanelChatSidebar
        open
        variant="docked"
        tabs={[]}
        activeTabId={null}
        onSelectTab={vi.fn()}
        onCloseTab={vi.fn()}
        onNewTab={vi.fn()}
        searchQuery=""
        onSearchQueryChange={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(useChatSurfaceCoordinatorStore.getState().routeId).toBe("chat")
      expect(useChatSurfaceCoordinatorStore.getState().surface).toBe("extension")
      expect(
        useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
      ).toBe(true)
      expect(
        useChatSurfaceCoordinatorStore.getState().engagedPanels["server-history"]
      ).toBe(false)
    })

    rerender(
      <SidepanelChatSidebar
        open
        variant="docked"
        tabs={[]}
        activeTabId={null}
        onSelectTab={vi.fn()}
        onCloseTab={vi.fn()}
        onNewTab={vi.fn()}
        searchQuery="quota"
        onSearchQueryChange={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(
        useChatSurfaceCoordinatorStore.getState().engagedPanels["server-history"]
      ).toBe(true)
    })

    rerender(
      <SidepanelChatSidebar
        open={false}
        variant="docked"
        tabs={[]}
        activeTabId={null}
        onSelectTab={vi.fn()}
        onCloseTab={vi.fn()}
        onNewTab={vi.fn()}
        searchQuery="quota"
        onSearchQueryChange={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(
        useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
      ).toBe(false)
    })
  })
})

const snapshot = { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" as const }, userId: 1 } }
const sidebarProps = { open: true, variant: "docked" as const, tabs: [], activeTabId: null, onSelectTab: () => {}, onCloseTab: () => {}, onNewTab: () => {}, searchQuery: "private", onSearchQueryChange: () => {} }

it("shows only verified-owner local search titles", async () => {
  fullTextSearchChatHistoriesMock.mockResolvedValue([
    { id: "a", title: "Alice private", server_scope_key: "alice", createdAt: 1, is_rag: false },
    { id: "b", title: "Bob private", server_scope_key: "bob", createdAt: 2, is_rag: false },
    { id: "legacy", title: "Unowned private", createdAt: 3, is_rag: false }
  ])
  render(<SidepanelChatSidebar {...sidebarProps} owner={{ snapshot, ownerKey: "alice", isCurrent: () => true }} />)
  await screen.findByText("Alice private")
  expect(screen.queryByText("Bob private")).not.toBeInTheDocument()
  expect(screen.queryByText("Unowned private")).not.toBeInTheDocument()
})

it("does not publish a held local search after the requesting owner is revoked", async () => {
  let finish!: (value: HistoryInfo[]) => void
  let current = true
  fullTextSearchChatHistoriesMock.mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValue([])
  const view = render(<SidepanelChatSidebar {...sidebarProps} owner={{ snapshot, ownerKey: "alice", isCurrent: () => current }} />)
  await waitFor(() => expect(finish).toBeTypeOf("function"))
  current = false
  view.rerender(<SidepanelChatSidebar {...sidebarProps} owner={{ snapshot, ownerKey: "bob", isCurrent: () => true }} />)
  await act(async () => { finish([{ id: "a", title: "Alice private", server_scope_key: "alice", createdAt: 1, is_rag: false }]) })
  expect(screen.queryByText("Alice private")).not.toBeInTheDocument()
})
