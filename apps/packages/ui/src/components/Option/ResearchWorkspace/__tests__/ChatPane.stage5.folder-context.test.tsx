import { render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import { ChatPane } from "../ChatPane"

const {
  mockCheckConnectionOnce,
  mockSetRagMediaIds,
  mockSetChatMode,
  mockSetFileRetrievalEnabled,
  mockMessageInfo,
  mockGetModels,
  workspaceStoreState,
  messageOptionState
} = vi.hoisted(() => {
  const checkConnectionOnce = vi.fn()
  const setRagMediaIds = vi.fn()
  const setChatMode = vi.fn()
  const setFileRetrievalEnabled = vi.fn()
  const messageInfo = vi.fn()
  const getModels = vi.fn()

  const workspaceState = {
    sources: [] as Array<{
      id: string
      mediaId: number
      title: string
      type: "pdf" | "video" | "audio" | "website" | "document" | "text"
      addedAt?: Date
      url?: string
    }>,
    selectedSourceIds: [] as string[],
    selectedSourceFolderIds: [] as string[],
    getSelectedSources: () =>
      [] as Array<{ id: string; title: string; mediaId?: number; type?: string }>,
    getSelectedMediaIds: () => [] as number[],
    getEffectiveSelectedSources: () =>
      [] as Array<{ id: string; title: string; mediaId?: number; type?: string }>,
    getEffectiveSelectedMediaIds: () => [] as number[],
    setSelectedSourceIds: vi.fn(),
    focusSourceById: vi.fn(),
    focusSourceByMediaId: vi.fn(),
    chatFocusTarget: null as { messageId: string; token: number } | null,
    clearChatFocusTarget: vi.fn(),
    captureToCurrentNote: vi.fn(),
    workspaceId: "workspace-a",
    workspaceChatReferenceId: "workspace-a",
    saveWorkspaceChatSession: vi.fn(),
    getWorkspaceChatSession: vi.fn(() => null),
    openAddSourceModal: vi.fn(),
    createNewWorkspace: vi.fn(),
    setCurrentNote: vi.fn(),
    switchWorkspace: vi.fn()
  }

  const optionState = {
    messages: [] as Array<{
      id: string
      isBot: boolean
      name: string
      message: string
      sources: any[]
      generationInfo?: any
    }>,
    setMessages: vi.fn(),
    history: [] as Array<{ role: "user" | "assistant" | "system"; content: string }>,
    setHistory: vi.fn(),
    streaming: false,
    setStreaming: vi.fn(),
    isProcessing: false,
    setIsProcessing: vi.fn(),
    onSubmit: vi.fn(),
    stopStreamingRequest: vi.fn(),
    regenerateLastMessage: vi.fn(),
    deleteMessage: vi.fn(),
    editMessage: vi.fn(),
    createChatBranch: vi.fn(),
    historyId: null as string | null,
    setHistoryId: vi.fn(),
    serverChatId: null as string | null,
    setServerChatId: vi.fn()
  }

  return {
    mockCheckConnectionOnce: checkConnectionOnce,
    mockSetRagMediaIds: setRagMediaIds,
    mockSetChatMode: setChatMode,
    mockSetFileRetrievalEnabled: setFileRetrievalEnabled,
    mockMessageInfo: messageInfo,
    mockGetModels: getModels,
    workspaceStoreState: workspaceState,
    messageOptionState: optionState
  }
})

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    isChecking: false,
    lastError: null
  },
  checkOnce: mockCheckConnectionOnce
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
      setSelectedModel: (model: string | null) => void
      selectedModel: string | null
    }) => unknown
  ) =>
    selector({
      setRagMediaIds: mockSetRagMediaIds,
      setChatMode: mockSetChatMode,
      setFileRetrievalEnabled: mockSetFileRetrievalEnabled,
      ragTopK: 8,
      setRagTopK: vi.fn(),
      ragAdvancedOptions: {},
      setRagAdvancedOptions: vi.fn(),
      setSelectedModel: vi.fn(),
      selectedModel: null
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (
    selector: (state: { apiProvider?: string }) => unknown
  ) => selector({ apiProvider: undefined })
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
    "Select sources from the Sources pane, then ask questions.",
  getWorkspaceChatSourcesExplainer: () =>
    "Selected sources keep answers grounded in this workspace."
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: vi.fn(),
  undoWorkspaceAction: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getModels: mockGetModels,
    getChatLorebookDiagnostics: vi.fn(async () => ({
      chat_id: "chat",
      total_turns_with_diagnostics: 0,
      turns: [],
      page: 1,
      size: 8
    }))
  }
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [])
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      useMessage: () => [
        {
          open: vi.fn(),
          warning: vi.fn(),
          destroy: vi.fn(),
          info: mockMessageInfo,
          success: vi.fn(),
          error: vi.fn()
        },
        <></>
      ]
    }
  }
})

function renderChatPane() {
  return render(
    <MemoryRouter>
      <ChatPane />
    </MemoryRouter>
  )
}

describe("ChatPane Stage 5 folder-derived context", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    workspaceStoreState.sources = [
      {
        id: "source-folder",
        mediaId: 101,
        title: "Folder Source",
        type: "pdf"
      }
    ]
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.selectedSourceFolderIds = ["folder-1"]
    workspaceStoreState.getSelectedSources = () => []
    workspaceStoreState.getSelectedMediaIds = () => []
    workspaceStoreState.getEffectiveSelectedSources = () => [
      workspaceStoreState.sources[0]
    ]
    workspaceStoreState.getEffectiveSelectedMediaIds = () => [101]

    messageOptionState.messages = []
    messageOptionState.history = []
    messageOptionState.historyId = null
    messageOptionState.serverChatId = null
    messageOptionState.streaming = false
    messageOptionState.isProcessing = false

    mockGetModels.mockResolvedValue([])
  })

  it("shows folder-derived sources in the context indicator and syncs RAG scope", async () => {
    renderChatPane()

    expect(await screen.findByText("Folder Source")).toBeTruthy()

    await waitFor(() => {
      expect(mockSetRagMediaIds).toHaveBeenLastCalledWith([101])
    })

    expect(mockSetChatMode).toHaveBeenLastCalledWith("rag")
    expect(mockSetFileRetrievalEnabled).toHaveBeenLastCalledWith(true)
  })
})
