// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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
  success: vi.fn(),
  warning: vi.fn()
}))

const mocks = vi.hoisted(() => ({
  invalidateQueries: vi.fn(),
  deleteChat: vi.fn(),
  removeMirror: vi.fn(),
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
        | { defaultValue?: string; id?: string; title?: string; query?: string }
    ) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : (defaultValueOrOptions?.defaultValue || _key).replace(
            /{{(id|title|query)}}/g,
            (_match, key: "id" | "title" | "query") => defaultValueOrOptions?.[key] || ""
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
  Modal: ({ open, children, title, onOk, onCancel }: { open?: boolean; children?: React.ReactNode; title?: string; onOk?: () => void; onCancel?: () => void }) => open ? <div role="dialog" aria-label={title}>{children}<button onClick={onOk}>Confirm deletion</button><button onClick={onCancel}>Cancel</button></div> : null,
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

vi.mock("@/db/dexie/server-chat-mirror", () => ({ removeServerChatMirror: (...args: unknown[]) => mocks.removeMirror(...args) }))
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

const HOUR = 60 * 60 * 1000
const NOW = new Date("2026-10-19T15:00:00")

const chatAt = (id: string, title: string, updatedAtMs: number) =>
  ({
    ...createChat({ id, title }),
    updated_at: new Date(updatedAtMs).toISOString(),
    updatedAtMs
  })

const showHistory = (data: unknown[]) => {
  historyState.value = {
    data,
    total: data.length,
    status: "success",
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: data.length > 0,
    isShowingStaleData: false,
    isServerPagedResult: false
  } as typeof historyState.value
}

/** Render the content of the latest success toast so its buttons can be used. */
const renderLatestSuccessToast = () => {
  const payload = antdMessage.success.mock.calls.at(-1)?.[0] as
    | { content?: React.ReactNode }
    | string
    | undefined
  const content = typeof payload === "string" ? payload : payload?.content
  return render(<div data-testid="toast">{content}</div>)
}

describe("Server chat history list (CS-08, CS-10)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.chatTypeFilter = "all"
    mocks.pins = []
    storeState.value.serverChatId = null
    mocks.deleteChat.mockResolvedValue(undefined)
    mocks.restoreChat.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    mocks.removeMirror.mockResolvedValue(true)
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

  afterEach(() => {
    vi.useRealTimers()
  })

  it("groups chats under Today, This week and Older (CS-08)", () => {
    vi.useFakeTimers({ toFake: ["Date"] })
    vi.setSystemTime(NOW)
    showHistory([
      chatAt("chat-today", "Morning standup", NOW.getTime() - 2 * HOUR),
      chatAt("chat-week", "Tuesday planning", NOW.getTime() - 3 * 24 * HOUR),
      chatAt("chat-old", "Spring retrospective", NOW.getTime() - 40 * 24 * HOUR)
    ])

    render(<ServerChatList searchQuery="" />)

    const headings = screen.getAllByTestId("server-chat-recency-group")
    expect(headings.map((heading) => heading.textContent)).toEqual([
      "Today",
      "This week",
      "Older"
    ])
    const follows = (a: HTMLElement, b: HTMLElement) =>
      Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING)
    expect(follows(headings[0], screen.getByText("Morning standup"))).toBe(true)
    expect(follows(screen.getByText("Morning standup"), headings[1])).toBe(true)
    expect(follows(headings[1], screen.getByText("Tuesday planning"))).toBe(true)
    expect(follows(headings[2], screen.getByText("Spring retrospective"))).toBe(true)
  })

  it("confirms a move to Trash and offers Undo and View Trash (CS-10)", async () => {
    showHistory([createChat({ id: "chat-1", title: "Recovered chat", version: 4 })])
    render(<ServerChatList searchQuery="" />)

    fireEvent.click(screen.getByRole("button", { name: "delete" }))
    await waitFor(() => expect(antdMessage.success).toHaveBeenCalled())

    const toast = renderLatestSuccessToast()
    expect(toast.getByTestId("toast")).toHaveTextContent("Moved to Trash")
    fireEvent.click(toast.getByRole("button", { name: "Undo" }))
    await waitFor(() => expect(mocks.restoreChat).toHaveBeenCalledWith("chat-1", expect.any(Object)))
    await waitFor(() =>
      expect(antdMessage.success).toHaveBeenLastCalledWith("Chat restored.")
    )

    fireEvent.click(toast.getByRole("button", { name: "View Trash" }))
    expect(mocks.setChatTypeFilter).toHaveBeenCalledWith("trash")
  })

  it("confirms a permanent delete (CS-10)", async () => {
    mocks.chatTypeFilter = "trash"
    showHistory([createChat({ id: "chat-1", title: "Old chat" })])
    render(<ServerChatList searchQuery="" />)

    fireEvent.click(screen.getByRole("button", { name: "delete" }))

    await waitFor(() =>
      expect(antdMessage.success).toHaveBeenCalledWith("Chat permanently deleted.")
    )
  })

  it("names the search that found nothing and offers to clear it (CS-10)", () => {
    showHistory([])
    const onClearSearch = vi.fn()
    render(<ServerChatList searchQuery="zebrafinch" onClearSearch={onClearSearch} />)

    expect(screen.getByText("No chats match “zebrafinch”")).toBeInTheDocument()
    // The filter stays reachable above the empty result.
    expect(screen.getByTestId("chat-filter-select")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }))
    expect(onClearSearch).toHaveBeenCalledTimes(1)
  })
})
