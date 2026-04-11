import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { MemoryRouter } from "react-router-dom"
import { ConnectionPhase } from "@/types/connection"
import { CHAT_PATH, LOREBOOK_DEBUG_FOCUS } from "@/routes/route-paths"
import { ChatPane } from "../ChatPane"

const mockCheckConnectionOnce = vi.fn()
const mockSaveWorkspaceChatSession = vi.fn()
const mockGetWorkspaceChatSession = vi.fn()
const mockFocusSourceById = vi.fn()
const mockFocusSourceByMediaId = vi.fn()
const mockClearChatFocusTarget = vi.fn()
const mockCaptureToCurrentNote = vi.fn()
const mockSetMessages = vi.fn()
const mockSetHistory = vi.fn()
const mockSetHistoryId = vi.fn()
const mockSetServerChatId = vi.fn()
const mockSetStreaming = vi.fn()
const mockSetIsProcessing = vi.fn()
const mockStopStreamingRequest = vi.fn()
const mockOnSubmit = vi.fn()
const mockRegenerateLastMessage = vi.fn()
const mockDeleteMessage = vi.fn()
const mockEditMessage = vi.fn()
const mockUseMessageOption = vi.fn()
const { mockScheduleWorkspaceUndoAction, mockUndoWorkspaceAction } = vi.hoisted(
  () => ({
    mockScheduleWorkspaceUndoAction: vi.fn(),
    mockUndoWorkspaceAction: vi.fn()
  })
)

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    isChecking: false,
    lastError: null
  },
  checkOnce: mockCheckConnectionOnce
}

const workspaceSessions = new Map<
  string,
  {
    messages: Array<{ isBot: boolean; name: string; message: string; sources: any[] }>
    history: Array<{ role: "user" | "assistant" | "system"; content: string }>
    historyId: string | null
    serverChatId: string | null
  }
>()

const workspaceStoreState = {
  sources: [] as Array<{ id: string; mediaId: number; title: string; type: string }>,
  selectedSourceIds: [] as string[],
  getSelectedSources: () => [] as Array<{ id: string; title: string }>,
  getSelectedMediaIds: () => [] as number[],
  setSelectedSourceIds: vi.fn(),
  focusSourceById: mockFocusSourceById,
  focusSourceByMediaId: mockFocusSourceByMediaId,
  chatFocusTarget: null as { messageId: string; token: number } | null,
  clearChatFocusTarget: mockClearChatFocusTarget,
  captureToCurrentNote: mockCaptureToCurrentNote,
  workspaceId: "workspace-a",
  workspaceChatReferenceId: "workspace-a",
  saveWorkspaceChatSession: mockSaveWorkspaceChatSession,
  getWorkspaceChatSession: mockGetWorkspaceChatSession
}

const messageOptionState = {
  messages: [] as Array<{
    id: string
    isBot: boolean
    name: string
    message: string
    sources: any[]
  }>,
  setMessages: mockSetMessages,
  history: [] as Array<{ role: "user" | "assistant" | "system"; content: string }>,
  setHistory: mockSetHistory,
  streaming: false,
  setStreaming: mockSetStreaming,
  isProcessing: false,
  setIsProcessing: mockSetIsProcessing,
  onSubmit: mockOnSubmit,
  stopStreamingRequest: mockStopStreamingRequest,
  regenerateLastMessage: mockRegenerateLastMessage,
  deleteMessage: mockDeleteMessage,
  editMessage: mockEditMessage,
  historyId: null as string | null,
  setHistoryId: mockSetHistoryId,
  serverChatId: null as string | null,
  setServerChatId: mockSetServerChatId
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  })
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (
    selector: (state: typeof connectionStoreState) => unknown
  ) => selector(connectionStoreState)
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: {
      setRagMediaIds: (ids: number[] | null) => void
      setChatMode: (mode: string) => void
      setFileRetrievalEnabled: (enabled: boolean) => void
      ragTopK: number
      setRagTopK: (value: number | null) => void
      ragAdvancedOptions: Record<string, unknown>
      setRagAdvancedOptions: (opts: Record<string, unknown>) => void
      selectedModel: string | null
    }) => unknown
  ) =>
    selector({
      setRagMediaIds: vi.fn(),
      setChatMode: vi.fn(),
      setFileRetrievalEnabled: vi.fn(),
      ragTopK: 8,
      setRagTopK: vi.fn(),
      ragAdvancedOptions: {},
      setRagAdvancedOptions: vi.fn(),
      selectedModel: null
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (
    selector: (state: { apiProvider?: string }) => unknown
  ) => selector({ apiProvider: undefined })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: (...args: unknown[]) => {
    mockUseMessageOption(...args)
    return messageOptionState
  }
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({
    message,
    onDeleteMessage,
    currentMessageIndex
  }: {
    message: string
    onDeleteMessage?: () => Promise<void> | void
    currentMessageIndex?: number
  }) => (
    <div data-testid="playground-message">
      <span>{message}</span>
      {onDeleteMessage && (
        <button
          type="button"
          aria-label={`delete-message-${currentMessageIndex ?? -1}`}
          onClick={() => void onDeleteMessage(currentMessageIndex ?? -1)}
        >
          Delete
        </button>
      )}
    </div>
  )
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title }: { title: string }) => <div>{title}</div>
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: mockScheduleWorkspaceUndoAction,
  undoWorkspaceAction: mockUndoWorkspaceAction
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceChatNoSourcesHint: () =>
    "Select sources from the Sources pane, then ask questions."
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getChatLorebookDiagnostics: vi.fn(async () => ({
      chat_id: "chat",
      total_turns_with_diagnostics: 0,
      turns: [],
      page: 1,
      size: 8
    }))
  }
}))

function renderChatPane() {
  return render(
    <MemoryRouter>
      <ChatPane />
    </MemoryRouter>
  )
}

describe("ChatPane Stage 1 reliability and controls", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceSessions.clear()
    mockUndoWorkspaceAction.mockReturnValue(true)
    mockScheduleWorkspaceUndoAction.mockImplementation(
      ({
        apply
      }: {
        apply: () => void
        undo: () => void
      }) => {
        apply()
        return { id: "undo-1", expiresAt: Date.now() + 10000 }
      }
    )

    connectionStoreState.state.phase = ConnectionPhase.CONNECTED
    connectionStoreState.state.isChecking = false
    connectionStoreState.state.lastError = null

    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceChatReferenceId = "workspace-a"
    workspaceStoreState.sources = []
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.getSelectedSources = () => []
    workspaceStoreState.getSelectedMediaIds = () => []
    mockFocusSourceById.mockReset()
    mockFocusSourceByMediaId.mockReset()
    workspaceStoreState.chatFocusTarget = null

    messageOptionState.messages = []
    messageOptionState.history = []
    messageOptionState.historyId = null
    messageOptionState.serverChatId = null
    messageOptionState.streaming = false
    messageOptionState.isProcessing = false
    mockOnSubmit.mockResolvedValue(undefined)
    mockUseMessageOption.mockReset()

    mockGetWorkspaceChatSession.mockImplementation((workspaceId: string) => {
      return workspaceSessions.get(workspaceId) ?? null
    })
  })

  it("shows stop button while streaming and triggers stopStreamingRequest", () => {
    messageOptionState.streaming = true

    renderChatPane()

    const stopButton = screen.getByRole("button", { name: "Stop" })
    fireEvent.click(stopButton)

    expect(mockStopStreamingRequest).toHaveBeenCalledTimes(1)
  })

  it("submits the composer with Cmd/Ctrl+Enter", async () => {
    renderChatPane()

    const textarea = screen.getByPlaceholderText("Type a message...")
    fireEvent.change(textarea, { target: { value: "Shortcut submission" } })
    fireEvent.keyDown(textarea, { key: "Enter", metaKey: true })

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        message: "Shortcut submission",
        image: ""
      })
    })
  })

  it("passes workspace scope into the shared chat hook", () => {
    renderChatPane()

    expect(mockUseMessageOption).toHaveBeenCalledWith({
      scope: { type: "workspace", workspaceId: "workspace-a" }
    })
  })

  it("keeps transcript as a scrollable flex region anchored above the composer", () => {
    renderChatPane()

    const transcript = screen.getByRole("log", { name: "Chat messages" })
    expect((transcript as HTMLElement).className).toContain("overflow-y-auto")
    expect((transcript as HTMLElement).className).toContain("flex-1")
    expect((transcript as HTMLElement).className).toContain("min-h-0")
  })

  it("clears chat with confirmation and persists empty session state", () => {
    messageOptionState.messages = [
      {
        id: "m1",
        isBot: false,
        name: "You",
        message: "hello",
        sources: []
      }
    ]
    messageOptionState.history = [{ role: "user", content: "hello" }]
    messageOptionState.historyId = "history-a"
    messageOptionState.serverChatId = "server-chat-a"

    const confirmSpy = vi
      .spyOn(Modal, "confirm")
      .mockImplementation((config) => {
        config.onOk?.()
        return {
          destroy: vi.fn(),
          update: vi.fn()
        } as any
      })

    renderChatPane()
    fireEvent.click(screen.getByRole("button", { name: "Clear chat" }))

    expect(confirmSpy).toHaveBeenCalled()
    expect(mockSetMessages).toHaveBeenCalledWith([])
    expect(mockSetHistory).toHaveBeenCalledWith([])
    expect(mockSetHistoryId).toHaveBeenCalledWith(null, {
      preserveServerChatId: true
    })
    expect(mockSetServerChatId).toHaveBeenCalledWith(null)
    expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalledTimes(1)
    expect(mockSaveWorkspaceChatSession).toHaveBeenCalledWith("workspace-a", {
      messages: [],
      history: [],
      historyId: null,
      serverChatId: null
    })
  })

  it("restores previous chat session when clear-chat undo runs", () => {
    messageOptionState.messages = [
      {
        id: "m1",
        isBot: false,
        name: "You",
        message: "hello",
        sources: []
      }
    ]
    messageOptionState.history = [{ role: "user", content: "hello" }]
    messageOptionState.historyId = "history-a"
    messageOptionState.serverChatId = "server-chat-a"

    const confirmSpy = vi
      .spyOn(Modal, "confirm")
      .mockImplementation((config) => {
        config.onOk?.()
        return {
          destroy: vi.fn(),
          update: vi.fn()
        } as any
      })

    renderChatPane()
    fireEvent.click(screen.getByRole("button", { name: "Clear chat" }))

    expect(confirmSpy).toHaveBeenCalled()
    const scheduledConfig = mockScheduleWorkspaceUndoAction.mock.calls[0]?.[0] as
      | { undo: () => void }
      | undefined
    expect(scheduledConfig).toBeDefined()

    scheduledConfig?.undo()

    expect(mockSetMessages).toHaveBeenLastCalledWith([
      {
        id: "m1",
        isBot: false,
        name: "You",
        message: "hello",
        sources: []
      }
    ])
    expect(mockSetHistory).toHaveBeenLastCalledWith([
      { role: "user", content: "hello" }
    ])
    expect(mockSetHistoryId).toHaveBeenLastCalledWith("history-a", {
      preserveServerChatId: true
    })
    expect(mockSetServerChatId).toHaveBeenLastCalledWith("server-chat-a")
    expect(mockSaveWorkspaceChatSession).toHaveBeenLastCalledWith("workspace-a", {
      messages: [
        {
          id: "m1",
          isBot: false,
          name: "You",
          message: "hello",
          sources: []
        }
      ],
      history: [{ role: "user", content: "hello" }],
      historyId: "history-a",
      serverChatId: "server-chat-a"
    })
  })

  it("routes message deletion through undo-managed restore flow", async () => {
    messageOptionState.messages = [
      {
        id: "m1",
        isBot: false,
        name: "You",
        message: "First",
        sources: []
      },
      {
        id: "m2",
        isBot: true,
        name: "Assistant",
        message: "Second",
        sources: []
      }
    ]
    messageOptionState.history = [
      { role: "user", content: "First" },
      { role: "assistant", content: "Second" }
    ]
    messageOptionState.historyId = "history-a"
    messageOptionState.serverChatId = "server-chat-a"
    mockDeleteMessage.mockResolvedValue(undefined)

    renderChatPane()
    fireEvent.click(screen.getByRole("button", { name: "delete-message-0" }))

    await waitFor(() => {
      expect(mockDeleteMessage).toHaveBeenCalledWith(0)
      expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalledTimes(1)
      expect(mockSaveWorkspaceChatSession).toHaveBeenCalledWith("workspace-a", {
        messages: [
          {
            id: "m2",
            isBot: true,
            name: "Assistant",
            message: "Second",
            sources: []
          }
        ],
        history: [{ role: "assistant", content: "Second" }],
        historyId: "history-a",
        serverChatId: "server-chat-a"
      })
    })

    const scheduledConfig = mockScheduleWorkspaceUndoAction.mock.calls[0]?.[0] as
      | { undo: () => void }
      | undefined
    expect(scheduledConfig).toBeDefined()
    scheduledConfig?.undo()

    expect(mockSetMessages).toHaveBeenLastCalledWith([
      {
        id: "m1",
        isBot: false,
        name: "You",
        message: "First",
        sources: []
      },
      {
        id: "m2",
        isBot: true,
        name: "Assistant",
        message: "Second",
        sources: []
      }
    ])
    expect(mockSetHistory).toHaveBeenLastCalledWith([
      { role: "user", content: "First" },
      { role: "assistant", content: "Second" }
    ])
    expect(mockSetHistoryId).toHaveBeenLastCalledWith("history-a", {
      preserveServerChatId: true
    })
    expect(mockSetServerChatId).toHaveBeenLastCalledWith("server-chat-a")
  })

  it("shows connection banner and retries connection check", () => {
    connectionStoreState.state.phase = ConnectionPhase.ERROR
    connectionStoreState.state.lastError = "server-unreachable"

    renderChatPane()

    expect(screen.getByText(/Unable to reach server/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(mockCheckConnectionOnce).toHaveBeenCalledTimes(1)
  })

  it("disables the composer after a submit failure surfaces the retry banner", async () => {
    mockOnSubmit.mockRejectedValueOnce(new Error("offline"))

    renderChatPane()

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Check the latest notes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      expect(screen.getByText(/Unable to reach server/)).toBeInTheDocument()
    })

    expect(screen.getByRole("textbox")).toBeDisabled()
    expect(screen.getByRole("button", { name: /Server disconnected/i })).toBeDisabled()
  })

  it("saves previous workspace chat and restores next workspace chat on switch", () => {
    messageOptionState.messages = [
      {
        id: "a-1",
        isBot: false,
        name: "You",
        message: "Workspace A message",
        sources: []
      }
    ]
    messageOptionState.history = [
      {
        role: "user",
        content: "Workspace A message"
      }
    ]
    messageOptionState.historyId = "history-a"
    messageOptionState.serverChatId = "server-chat-a"

    workspaceStoreState.workspaceChatReferenceId = "session-a"

    workspaceSessions.set("workspace-b::session-b", {
      messages: [
        {
          isBot: true,
          name: "Assistant",
          message: "Workspace B response",
          sources: []
        }
      ],
      history: [{ role: "assistant", content: "Workspace B response" }],
      historyId: "history-b",
      serverChatId: "server-chat-b"
    })

    const { rerender } = renderChatPane()

    workspaceStoreState.workspaceId = "workspace-b"
    workspaceStoreState.workspaceChatReferenceId = "session-b"
    rerender(
      <MemoryRouter>
        <ChatPane />
      </MemoryRouter>
    )

    expect(mockSaveWorkspaceChatSession).toHaveBeenCalledWith("workspace-a::session-a", {
      messages: [
        {
          id: "a-1",
          isBot: false,
          name: "You",
          message: "Workspace A message",
          sources: []
        }
      ],
      history: [{ role: "user", content: "Workspace A message" }],
      historyId: "history-a",
      serverChatId: "server-chat-a"
    })
    expect(mockSetMessages).toHaveBeenCalledWith([
      {
        isBot: true,
        name: "Assistant",
        message: "Workspace B response",
        sources: []
      }
    ])
    expect(mockSetHistory).toHaveBeenCalledWith([
      { role: "assistant", content: "Workspace B response" }
    ])
    expect(mockSetHistoryId).toHaveBeenCalledWith("history-b", {
      preserveServerChatId: true
    })
    expect(mockSetServerChatId).toHaveBeenCalledWith("server-chat-b")
  })

  it("renders full diagnostics link to a valid chat route", () => {
    messageOptionState.messages = [
      {
        id: "m1",
        isBot: true,
        name: "Assistant",
        message: "Diagnostics available",
        sources: []
      }
    ]

    renderChatPane()

    const link = screen.getByRole("link", {
      name: "Open full lorebook diagnostics"
    })
    const href = link.getAttribute("href")
    expect(href).toBeTruthy()

    const parsed = new URL(String(href), "https://example.local")
    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get("focus")).toBe(LOREBOOK_DEBUG_FOCUS)
    expect(parsed.searchParams.get("from")).toBe("workspace-playground")
  })
})
