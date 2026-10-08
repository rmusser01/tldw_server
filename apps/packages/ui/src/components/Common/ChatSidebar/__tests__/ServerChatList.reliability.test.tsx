// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
    isServerPagedResult: false,
    isFetching: false,
    refetch: vi.fn()
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
  setPinnedChatIds: vi.fn(),
  pins: [] as string[],
  realRow: false,
  connected: true,
  ownerCurrent: true,
  verificationState: "ready" as "ready" | "pending" | "error",
  retryVerification: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string; id?: string; title?: string }
    ) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : (defaultValueOrOptions?.defaultValue || _key).replace(
            /{{(id|title)}}/g,
            (_match, key: "id" | "title") => defaultValueOrOptions?.[key] || ""
          )
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
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Dropdown: ({ children }: { children: React.ReactNode }) => <>{children}</>,
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
  useStorage: () => [mocks.pins, mocks.setPinnedChatIds]
}))

vi.mock("wxt/browser", () => ({
  browser: {}
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))

vi.mock("@/hooks/useChatDraftOwner", () => {
  const isCurrent = () => mocks.ownerCurrent
  const requestScope = {
    config: { serverUrl: "http://chat.test", authMode: "multi-user" },
    userId: "A"
  }
  return {
    useChatDraftOwner: () => ({
      ownerKey: "A",
      serverScopeKey: "A",
      requestScope,
      isCurrent,
      verificationState: mocks.verificationState,
      retryVerification: mocks.retryVerification
    })
  }
})

vi.mock("@/store/connection", () => ({
  useConnectionStore: Object.assign(() => {}, {
    getState: () => ({ state: { isConnected: mocks.connected } })
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    isConnected: mocks.connected
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
  useStoreMessageOption: Object.assign(
    (selector?: (state: typeof storeState.value) => unknown) =>
      typeof selector === "function"
        ? selector(storeState.value)
        : storeState.value,
    { getState: () => storeState.value }
  )
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

vi.mock("../ServerChatRow", async () => {
  const actual =
    await vi.importActual<typeof import("../ServerChatRow")>("../ServerChatRow")
  const stub = ({
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
  return {
    ServerChatRow: (
      props: React.ComponentProps<typeof actual.ServerChatRow>
    ) => (mocks.realRow ? <actual.ServerChatRow {...props} /> : stub(props))
  }
})

import { ServerChatList } from "../ServerChatList"
import { usePlaygroundSessionStore } from "@/store/playground-session"
beforeEach(() => {
  mocks.realRow = false
  mocks.connected = true
  mocks.ownerCurrent = true
  mocks.verificationState = "ready"
  mocks.retryVerification.mockClear()
  historyState.value.refetch?.mockClear()
})

const createChat = (
  overrides: Partial<{ id: string; title: string; version: number }> = {}
) => ({
  id: "chat-1",
  title: "Recovered chat",
  created_at: "2026-03-08T00:00:00.000Z",
  updated_at: "2026-03-08T00:01:00.000Z",
  version: 4,
  createdAtMs: Date.parse("2026-03-08T00:00:00.000Z"),
  updatedAtMs: Date.parse("2026-03-08T00:01:00.000Z"),
  ...overrides
})

describe("ServerChatList reliability states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.selectServerChat.mockReset()
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

  it.each([false, true])(
    "reports an accepted row selection, including the current conversation: %s",
    (alreadySelected) => {
      const chat = createChat()
      historyState.value.data = [chat]
      historyState.value.total = 1
      historyState.value.hasUsableData = true
      if (alreadySelected) storeState.value.serverChatId = chat.id
      const onConversationSelected = vi.fn()
      render(
        <ServerChatList
          searchQuery=""
          onConversationSelected={onConversationSelected}
        />
      )
      fireEvent.click(
        screen.getByRole("button", { name: "Select Recovered chat" })
      )
      expect(onConversationSelected).toHaveBeenCalledTimes(1)
      expect(mocks.selectServerChat).toHaveBeenCalledTimes(1)
      expect(mocks.selectServerChat.mock.invocationCallOrder[0]).toBeLessThan(
        onConversationSelected.mock.invocationCallOrder[0]
      )
    }
  )

  it("does not close the history list or report selection when the transition is declined", () => {
    historyState.value.data = [createChat()]
    historyState.value.total = 1
    historyState.value.hasUsableData = true
    mocks.selectServerChat.mockReturnValueOnce(false)
    const onConversationSelected = vi.fn()
    render(<ServerChatList searchQuery="" onConversationSelected={onConversationSelected} />)
    fireEvent.click(screen.getByRole("button", { name: "Select Recovered chat" }))
    expect(mocks.selectServerChat).toHaveBeenCalledOnce()
    expect(onConversationSelected).not.toHaveBeenCalled()
  })

  it("opens a saved chat read-only in temporary mode without management actions (TASK-13389)", () => {
    historyState.value.data = [createChat()]
    historyState.value.total = 1
    historyState.value.hasUsableData = true
    const onConversationSelected = vi.fn()
    render(
      <ServerChatList
        searchQuery=""
        readOnly
        onConversationSelected={onConversationSelected}
      />
    )

    expect(screen.queryByRole("button", { name: "common:pin" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "More actions: Recovered chat" })
    ).toBeNull()

    fireEvent.click(
      screen.getByRole("button", { name: "Select Recovered chat" })
    )
    expect(mocks.selectServerChat).toHaveBeenCalledTimes(1)
    expect(onConversationSelected).toHaveBeenCalledTimes(1)
  })

  it.each(["bulk", "trash"])(
    "does not report conversation selection for %s actions",
    (mode) => {
      historyState.value.data = [createChat()]
      historyState.value.total = 1
      historyState.value.hasUsableData = true
      if (mode === "trash") mocks.chatTypeFilter = "trash"
      const onConversationSelected = vi.fn()
      render(
        <ServerChatList
          searchQuery=""
          selectionMode={mode === "bulk"}
          onConversationSelected={onConversationSelected}
        />
      )
      fireEvent.click(
        screen.getByRole("button", { name: "Select Recovered chat" })
      )
      expect(onConversationSelected).not.toHaveBeenCalled()
      expect(mocks.selectServerChat).not.toHaveBeenCalled()
    }
  )

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
      screen.getByText(
        "Could not refresh Server history. Try again; cached copies remain in On this device."
      )
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

describe("owned visible Server results for combined discovery", () => {
  beforeEach(() => {
    historyState.value = {
      data: [createChat()],
      total: 1,
      status: "success",
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
      isServerPagedResult: true,
      isFetching: false,
      refetch: vi.fn()
    }
    mocks.selectServerChat.mockClear()
    mocks.chatTypeFilter = "all"
    useChatSurfaceCoordinatorStore.getState().markPanelEngaged("server-history")
  })
  it("reports only exact currently successful visible scoped identities", async () => {
    const visible = vi.fn()
    render(
      <ServerChatList searchQuery="needle" onVisibleChatsChange={visible} />
    )
    await waitFor(() =>
      expect(visible).toHaveBeenCalledWith(
        expect.objectContaining({ serverScopeKey: "A", query: "needle" })
      )
    )
    const receipt = visible.mock.calls.at(-1)![0]
    expect([...receipt.ids]).toEqual(["chat-1"])
    expect(receipt.isCurrent()).toBe(true)
    mocks.connected = false
    expect(receipt.isCurrent()).toBe(false)
  })
  it.each(["pending", "error"] as const)(
    "does not expose server rows before owner verification (%s)",
    (state) => {
      mocks.verificationState = state
      render(<ServerChatList searchQuery="needle" />)
      expect(
        screen.queryByRole("button", { name: "Select Recovered chat" })
      ).toBeNull()
      if (state === "error") {
        fireEvent.click(screen.getByRole("button", { name: "Retry" }))
        expect(mocks.retryVerification).toHaveBeenCalledOnce()
      }
    }
  )
  it("rejects selection after imperative owner loss without waiting for rerender", () => {
    render(<ServerChatList searchQuery="needle" />)
    mocks.ownerCurrent = false
    fireEvent.click(
      screen.getByRole("button", { name: "Select Recovered chat" })
    )
    expect(mocks.selectServerChat).not.toHaveBeenCalled()
  })
  it("keeps a failed refresh distinct, retryable and unable to hide a cached copy", async () => {
    historyState.value.status = "error"
    historyState.value.sidebarRefreshState = "hard-error"
    const visible = vi.fn()
    render(
      <ServerChatList searchQuery="needle" onVisibleChatsChange={visible} />
    )
    expect(screen.getByText("Recovered chat")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(historyState.value.refetch).toHaveBeenCalledOnce()
    await waitFor(() => expect(visible).toHaveBeenLastCalledWith(null))
    fireEvent.click(
      screen.getByRole("button", { name: "Select Recovered chat" })
    )
    expect(mocks.selectServerChat).not.toHaveBeenCalled()
  })
  it("does not cut off a full page when counts fall back and can return from a later empty page", async () => {
    historyState.value.data = Array.from({ length: 25 }, (_, i) =>
      createChat({ id: `chat-${i}`, title: `Page row ${i}` })
    )
    historyState.value.total = 25
    const view = render(<ServerChatList searchQuery="needle" />)
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    historyState.value.data = []
    historyState.value.total = 0
    historyState.value.hasUsableData = false
    view.rerender(<ServerChatList searchQuery="needle" />)
    expect(screen.getByRole("button", { name: "Previous" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Previous" }))
  })
  it("rejects foreign workspace responses rather than exposing or deduplicating them", async () => {
    historyState.value.data = [
      { ...createChat(), scope_type: "workspace", workspace_id: "W" }
    ]
    const visible = vi.fn()
    render(
      <ServerChatList searchQuery="needle" onVisibleChatsChange={visible} />
    )
    expect(
      screen.queryByRole("button", { name: "Select Recovered chat" })
    ).toBeNull()
    await waitFor(() =>
      expect(visible).toHaveBeenLastCalledWith(
        expect.objectContaining({ ids: new Set() })
      )
    )
  })
})

it("keeps the filter and bounded Next reachable when the current page has no visible scoped rows", () => {
  mocks.connected = true
  mocks.ownerCurrent = true
  mocks.verificationState = "ready"
  mocks.chatTypeFilter = "character"
  historyState.value = {
    data: [{ ...createChat(), scope_type: "workspace", workspace_id: "W" }],
    total: 50,
    status: "success",
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: true,
    isShowingStaleData: false,
    isServerPagedResult: true,
    isFetching: false,
    refetch: vi.fn()
  }
  useChatSurfaceCoordinatorStore.getState().markPanelEngaged("server-history")
  render(<ServerChatList searchQuery="needle" />)
  expect(screen.getByTestId("chat-filter-select")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Next" })).toBeEnabled()
})

describe("retired Server history management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.chatTypeFilter = "all"
    historyState.value = {
      data: [createChat()],
      total: 1,
      status: "success",
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
      isServerPagedResult: true,
      isFetching: false,
      refetch: vi.fn()
    }
    mocks.deleteChat.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    useChatSurfaceCoordinatorStore.getState().markPanelEngaged("server-history")
  })
  it("disables the real row checkbox and bulk actions after a selected page becomes stale", () => {
    mocks.realRow = true
    const view = render(<ServerChatList searchQuery="needle" selectionMode />)
    fireEvent.click(screen.getByRole("checkbox"))
    expect(screen.getByRole("button", { name: "Move to trash" })).toBeEnabled()
    historyState.value.status = "error"
    historyState.value.sidebarRefreshState = "hard-error"
    view.rerender(<ServerChatList searchQuery="needle" selectionMode />)
    expect(screen.getByRole("checkbox")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Move to trash" })).toBeDisabled()
    fireEvent.click(screen.getByRole("checkbox"))
    fireEvent.click(screen.getByRole("button", { name: "Move to trash" }))
    expect(mocks.deleteChat).not.toHaveBeenCalled()
  })
  it("rejects a held confirmation after imperative account loss", async () => {
    let confirm!: (value: boolean) => void
    mocks.confirmDanger.mockImplementationOnce(
      () =>
        new Promise<boolean>((resolve) => {
          confirm = resolve
        })
    )
    render(<ServerChatList searchQuery="needle" />)
    fireEvent.click(screen.getByRole("button", { name: "delete" }))
    mocks.ownerCurrent = false
    await act(async () => confirm(true))
    expect(mocks.deleteChat).not.toHaveBeenCalled()
    expect(mocks.clearChat).not.toHaveBeenCalled()
  })
  it("dispatches against the captured account and does not clear a newer view on a late acknowledgement", async () => {
    let acknowledge!: () => void
    mocks.deleteChat.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          acknowledge = resolve
        })
    )
    storeState.value.serverChatId = "chat-1"
    render(<ServerChatList searchQuery="needle" />)
    fireEvent.click(screen.getByRole("button", { name: "delete" }))
    await waitFor(() =>
      expect(mocks.deleteChat).toHaveBeenCalledWith(
        "chat-1",
        expect.objectContaining({
          requestScope: expect.objectContaining({ userId: "A" }),
          scope: { type: "global" }
        })
      )
    )
    act(() => usePlaygroundSessionStore.getState().markSourceSelectionIntent())
    await act(async () => acknowledge())
    expect(mocks.clearChat).not.toHaveBeenCalled()
  })
})

it("distinguishes real Server rows and checkboxes when their titles and timestamps match", () => {
  mocks.realRow = true
  mocks.chatTypeFilter = "all"
  historyState.value = {
    data: [
      createChat({ id: "A", title: "Same title" }),
      createChat({ id: "B", title: "Same title" })
    ],
    total: 2,
    status: "success",
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: true,
    isShowingStaleData: false,
    isServerPagedResult: true,
    isFetching: false,
    refetch: vi.fn()
  }
  useChatSurfaceCoordinatorStore.getState().markPanelEngaged("server-history")
  render(<ServerChatList searchQuery="needle" selectionMode />)
  expect(
    screen.getByRole("button", { name: "Same title · Chat A" })
  ).toBeInTheDocument()
  expect(
    screen.getByRole("button", { name: "Same title · Chat B" })
  ).toBeInTheDocument()
  expect(
    screen.getByRole("checkbox", {
      name: "Toggle selection: Same title · Chat A"
    })
  ).toBeInTheDocument()
  expect(screen.getByText("Chat B")).toBeVisible()
})
it("retires a held bulk restore after New and makes a fresh operation usable", async () => {
  mocks.realRow = true
  mocks.chatTypeFilter = "trash"
  historyState.value = {
    data: [createChat()],
    total: 1,
    status: "success",
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: true,
    isShowingStaleData: false,
    isServerPagedResult: true,
    isFetching: false,
    refetch: vi.fn()
  }
  useChatSurfaceCoordinatorStore.getState().markPanelEngaged("server-history")
  mocks.restoreChat.mockClear()
  mocks.clearChat.mockClear()
  let acknowledge!: (value: unknown) => void
  mocks.restoreChat.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        acknowledge = resolve
      })
  )
  render(<ServerChatList searchQuery="needle" selectionMode />)
  fireEvent.click(screen.getByRole("checkbox"))
  fireEvent.click(screen.getByRole("button", { name: "Restore" }))
  await waitFor(() => expect(mocks.restoreChat).toHaveBeenCalledTimes(1))
  act(() => usePlaygroundSessionStore.getState().markSourceSelectionIntent())
  await act(async () => acknowledge(createChat()))
  fireEvent.click(screen.getByRole("button", { name: "Select all" }))
  expect(screen.getByRole("button", { name: "Restore" })).toBeEnabled()
  expect(mocks.clearChat).not.toHaveBeenCalled()
})
