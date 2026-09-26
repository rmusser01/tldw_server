// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const historyState = vi.hoisted(() => ({
  value: {
    data: [],
    total: 0,
    status: "success",
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: false,
    isShowingStaleData: false,
    isServerPagedResult: false
  }
}))

const storeState = vi.hoisted(() => ({
  value: {
    serverChatId: null as string | null,
    setServerChatTitle: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatTopic: vi.fn()
  }
}))

const antdMessage = vi.hoisted(() => ({
  error: vi.fn(),
  success: vi.fn()
}))

const mocks = vi.hoisted(() => ({
  invalidateQueries: vi.fn(),
  deleteChat: vi.fn(),
  restoreChat: vi.fn(),
  confirmDanger: vi.fn(),
  clearChat: vi.fn(),
  selectServerChat: vi.fn(),
  setChatTypeFilter: vi.fn(),
  chatTypeFilter: "all",
  setPinnedChatIds: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValueOrOptions?: string | { defaultValue?: string }) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : defaultValueOrOptions?.defaultValue || _key
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useMutation: () => ({
    mutate: vi.fn(),
    isPending: false
  }),
  useQueryClient: () => ({
    invalidateQueries: mocks.invalidateQueries
  })
}))

vi.mock("antd", () => ({
  Empty: ({ description }: { description?: React.ReactNode }) => (
    <div>{description}</div>
  ),
  Skeleton: () => <div>loading</div>,
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} />
  ),
  Modal: ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null),
  Select: () => <div data-testid="chat-filter-select" />,
  message: antdMessage
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [[], mocks.setPinnedChatIds]
}))

vi.mock("wxt/browser", () => ({
  browser: {}
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    isConnected: true
  })
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: () => historyState.value
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [mocks.chatTypeFilter, mocks.setChatTypeFilter]
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => mocks.clearChat
}))

vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => mocks.selectServerChat
}))

vi.mock("@/hooks/useBulkChatOperations", () => ({
  useBulkChatOperations: () => ({
    openBulkFolderPicker: vi.fn(),
    openBulkTagPicker: vi.fn(),
    applyBulkDelete: vi.fn()
  })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: typeof storeState.value) => unknown) =>
    typeof selector === "function" ? selector(storeState.value) : storeState.value
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: () => ({
    folderApiAvailable: true
  })
}))

vi.mock("@/store/data-tables", () => ({
  useDataTablesStore: () => ({
    resetWizard: vi.fn(),
    addSource: vi.fn(),
    setWizardStep: vi.fn()
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    deleteChat: (...args: unknown[]) =>
      (mocks.deleteChat as (...args: unknown[]) => unknown)(...args),
    restoreChat: (...args: unknown[]) =>
      (mocks.restoreChat as (...args: unknown[]) => unknown)(...args),
    updateChat: vi.fn()
  }
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: vi.fn()
}))

vi.mock("@/utils/data-tables-prefill", () => ({
  queueDataTablesPrefill: vi.fn()
}))

vi.mock("@/utils/data-tables-create-flow", () => ({
  startCreateTableFromChat: vi.fn()
}))

vi.mock("../ServerChatRow", () => ({
  ServerChatRow: ({
    chat,
    onDeleteChat,
    onSelectChat
  }: {
    chat: { id: string; title: string }
    onDeleteChat: (chat: { id: string; title: string }) => void | Promise<void>
    onSelectChat: (chat: { id: string; title: string }) => void
  }) => (
    <div>
      <span>{chat.title}</span>
      <button onClick={() => onSelectChat(chat)}>Select {chat.title}</button>
      <button onClick={() => void onDeleteChat(chat)}>delete</button>
    </div>
  )
}))

import { ServerChatList } from "../ServerChatList"

const createChat = (overrides: Partial<{ id: string; title: string; version: number }> = {}) => ({
  id: "chat-1",
  title: "Recovered chat",
  created_at: "2026-03-08T00:00:00.000Z",
  updated_at: "2026-03-08T00:01:00.000Z",
  version: 4,
  ...overrides
})

describe("ServerChatList reliability states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.chatTypeFilter = "all"
    storeState.value.serverChatId = null
    historyState.value = {
      data: [],
      total: 0,
      status: "success",
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: false,
      isShowingStaleData: false,
      isServerPagedResult: false
    }
    mocks.deleteChat.mockResolvedValue(undefined)
    mocks.restoreChat.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    useChatSurfaceCoordinatorStore.setState({
      routeId: "chat",
      surface: "webui",
      visiblePanels: {
        "server-history": true,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      },
      engagedPanels: {
        "server-history": true,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      }
    })
  })

  it.each([false, true])("reports an accepted row selection, including the current conversation: %s", (alreadySelected) => {
    const chat = createChat()
    historyState.value.data = [chat]
    historyState.value.total = 1
    historyState.value.hasUsableData = true
    if (alreadySelected) storeState.value.serverChatId = chat.id
    const onConversationSelected = vi.fn()
    render(<ServerChatList searchQuery="" onConversationSelected={onConversationSelected} />)
    fireEvent.click(screen.getByRole("button", { name: "Select Recovered chat" }))
    expect(onConversationSelected).toHaveBeenCalledTimes(1)
    expect(mocks.selectServerChat).toHaveBeenCalledTimes(alreadySelected ? 0 : 1)
    if (!alreadySelected) {
      expect(mocks.selectServerChat.mock.invocationCallOrder[0]).toBeLessThan(onConversationSelected.mock.invocationCallOrder[0])
    }
  })

  it.each(["bulk", "trash"])("does not report conversation selection for %s actions", (mode) => {
    historyState.value.data = [createChat()]
    historyState.value.total = 1
    historyState.value.hasUsableData = true
    if (mode === "trash") mocks.chatTypeFilter = "trash"
    const onConversationSelected = vi.fn()
    render(<ServerChatList searchQuery="" selectionMode={mode === "bulk"} onConversationSelected={onConversationSelected} />)
    fireEvent.click(screen.getByRole("button", { name: "Select Recovered chat" }))
    expect(onConversationSelected).not.toHaveBeenCalled()
    expect(mocks.selectServerChat).not.toHaveBeenCalled()
  })

  it("shows a recoverable refresh warning when old chat data is still usable", () => {
    historyState.value = {
      data: [createChat()],
      total: 1,
      status: "error",
      isLoading: false,
      sidebarRefreshState: "recoverable-error",
      hasUsableData: true,
      isShowingStaleData: true,
      isServerPagedResult: false
    }

    render(<ServerChatList searchQuery="" />)

    expect(
      screen.getByText(/Showing saved chats from the last successful refresh\./)
    ).toBeInTheDocument()
    expect(screen.getByText("Recovered chat")).toBeInTheDocument()
  })

  it("shows an unavailable message instead of an empty state when refresh failed without usable data", () => {
    historyState.value = {
      data: [],
      total: 0,
      status: "error",
      isLoading: false,
      sidebarRefreshState: "recoverable-error",
      hasUsableData: false,
      isShowingStaleData: false,
      isServerPagedResult: false
    }

    render(<ServerChatList searchQuery="" />)

    expect(
      screen.getByText("Unable to refresh server chats right now. Try again shortly.")
    ).toBeInTheDocument()
    expect(screen.queryByText("No server chats yet")).not.toBeInTheDocument()
  })

  it("shows conflict-specific delete feedback when delete still fails after retry", async () => {
    historyState.value = {
      data: [createChat()],
      total: 1,
      status: "success",
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
      isServerPagedResult: false
    }
    mocks.deleteChat.mockRejectedValueOnce(
      Object.assign(new Error("409 conflict: expected version mismatch"), {
        status: 409
      })
    )

    render(<ServerChatList searchQuery="" />)

    fireEvent.click(screen.getByRole("button", { name: "delete" }))

    await waitFor(() => {
      expect(antdMessage.error).toHaveBeenCalledWith(
        "This chat changed on the server. Refresh and try again."
      )
    })
  })

  it("keeps server-paged search rows visible when navigating beyond page 1", async () => {
    historyState.value = {
      data: [createChat({ id: "chat-26", title: "Search page 2 row" })],
      total: 50,
      status: "success",
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
      isServerPagedResult: true
    }

    render(<ServerChatList searchQuery="quota" />)

    expect(screen.getByText("Search page 2 row")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByText("Search page 2 row")).toBeInTheDocument()
      expect(screen.getByDisplayValue("2")).toBeInTheDocument()
    })
  })
})
