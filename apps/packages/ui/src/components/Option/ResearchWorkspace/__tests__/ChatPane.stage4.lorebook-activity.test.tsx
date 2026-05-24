import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
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

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    isChecking: false,
    lastError: null
  },
  checkOnce: mockCheckConnectionOnce
}

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

const { tldwClientMock } = vi.hoisted(() => ({
  tldwClientMock: {
    getChatLorebookDiagnostics: vi.fn()
  }
}))

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
    }) => unknown
  ) =>
    selector({
      setRagMediaIds: vi.fn(),
      setChatMode: vi.fn(),
      setFileRetrievalEnabled: vi.fn(),
      ragTopK: 8,
      setRagTopK: vi.fn(),
      ragAdvancedOptions: {},
      setRagAdvancedOptions: vi.fn()
    })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({ message }: { message: string }) => (
    <div data-testid="playground-message">{message}</div>
  )
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title }: { title: string }) => <div>{title}</div>
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceChatNoSourcesHint: () =>
    "Select sources from the Sources pane, then ask questions."
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

describe("ChatPane Stage 4 lorebook activity", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionStoreState.state.phase = ConnectionPhase.CONNECTED
    connectionStoreState.state.isChecking = false
    connectionStoreState.state.lastError = null

    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceChatReferenceId = "workspace-a"
    workspaceStoreState.sources = []
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.getSelectedSources = () => []
    workspaceStoreState.getSelectedMediaIds = () => []
    workspaceStoreState.chatFocusTarget = null

    messageOptionState.messages = [
      {
        id: "bot-1",
        isBot: true,
        name: "Assistant",
        message: "First answer",
        sources: []
      }
    ]
    messageOptionState.history = []
    messageOptionState.historyId = "history-a"
    messageOptionState.serverChatId = "server-chat-a"
    messageOptionState.streaming = false
    messageOptionState.isProcessing = false

    mockGetWorkspaceChatSession.mockReturnValue(null)
  })

  it("renders turn-level lorebook activity from diagnostics data", async () => {
    tldwClientMock.getChatLorebookDiagnostics.mockResolvedValueOnce({
      chat_id: "server-chat-a",
      total_turns_with_diagnostics: 5,
      turns: [
        {
          turn_number: 12,
          message_preview: "Assistant summary for turn 12",
          diagnostics: [{ entry_id: 1 }, { entry_id: 2 }]
        },
        {
          turn_number: 11,
          message_preview: "Assistant summary for turn 11",
          diagnostics: [{ entry_id: 3 }]
        }
      ],
      page: 1,
      size: 8
    })

    render(<ChatPane />)

    expect(await screen.findByText("Lorebook Activity")).toBeInTheDocument()
    expect(await screen.findByText("Turn 12: 2 entries fired")).toBeInTheDocument()
    expect(screen.getByText("Assistant summary for turn 12")).toBeInTheDocument()
    expect(
      screen.getByText("Showing 2 of 5 turns with diagnostics.")
    ).toBeInTheDocument()
  })

  it("hides detailed diagnostics when diagnostics API is forbidden", async () => {
    tldwClientMock.getChatLorebookDiagnostics.mockRejectedValueOnce(
      new Error("403 forbidden")
    )

    render(<ChatPane />)

    expect(
      await screen.findByText("Lorebook activity is unavailable for this account.")
    ).toBeInTheDocument()
  })

  it("caps rendered turn cards for long diagnostic responses", async () => {
    tldwClientMock.getChatLorebookDiagnostics.mockResolvedValueOnce({
      chat_id: "server-chat-a",
      total_turns_with_diagnostics: 120,
      turns: Array.from({ length: 120 }, (_, index) => ({
        turn_number: index + 1,
        message_preview: `Preview ${index + 1}`,
        diagnostics: [{ entry_id: index + 1 }]
      })),
      page: 1,
      size: 8
    })

    render(<ChatPane />)

    await waitFor(() => {
      expect(tldwClientMock.getChatLorebookDiagnostics).toHaveBeenCalled()
    })

    const turnCards = screen.getAllByText(/entries fired/)
    expect(turnCards.length).toBeLessThanOrEqual(8)
  })
})
