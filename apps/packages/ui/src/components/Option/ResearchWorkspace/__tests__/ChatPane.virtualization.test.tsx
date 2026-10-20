// Stage 3 (WP-04) virtualized ChatPane transcript: windowed rendering over a
// 1,000-message fixture through the REAL @tanstack/react-virtual instance
// (jsdom prototype stubs per the VirtualChatTimeline.lifecycle test pattern),
// streaming last-row semantics under windowing, per-message-id retrieval
// diagnostics memoization, focus-target reveal into the virtualized range,
// and source guards for the removed `msg-${idx}` key fallback.
import { readFileSync } from "node:fs"
import path from "node:path"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import { ChatPane } from "../ChatPane"

const buildRetrievalDiagnosticsSpy = vi.hoisted(() => vi.fn())

vi.mock("../ChatPane/retrieval-diagnostics", async (importActual) => {
  const actual =
    await importActual<typeof import("../ChatPane/retrieval-diagnostics")>()
  return {
    ...actual,
    buildRetrievalDiagnostics: buildRetrievalDiagnosticsSpy.mockImplementation(
      (
        sources: unknown[] | undefined,
        generationInfo: unknown
      ) => actual.buildRetrievalDiagnostics(sources, generationInfo)
    )
  }
})

const mockCheckConnectionOnce = vi.fn()
const mockSaveWorkspaceChatSession = vi.fn()
const mockGetWorkspaceChatSession = vi.fn()
const mockFocusSourceById = vi.fn()
const mockFocusSourceByMediaId = vi.fn()
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
const mockClearChatFocusTarget = vi.fn()

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    isChecking: false,
    lastError: null,
    isConnected: true,
    errorKind: "none" as "none" | "partial",
    knowledgeStatus: "ready" as const
  },
  checkOnce: mockCheckConnectionOnce
}

const workspaceStoreState = {
  sources: [] as Array<Record<string, unknown>>,
  sourceFolders: [] as Array<{ id: string; parentFolderId: string | null }>,
  sourceFolderMemberships: [] as Array<{ sourceId: string; folderId: string }>,
  selectedSourceFolderIds: [] as string[],
  selectedSourceIds: [] as string[],
  getSelectedSources: () => [] as Array<{ id: string; title: string }>,
  getSelectedMediaIds: () => [] as number[],
  getEffectiveSelectedMediaIds: () => [] as number[],
  setSelectedSourceIds: vi.fn(),
  focusSourceById: mockFocusSourceById,
  focusSourceByMediaId: mockFocusSourceByMediaId,
  chatFocusTarget: null as { messageId: string; token: number } | null,
  clearChatFocusTarget: mockClearChatFocusTarget,
  captureToCurrentNote: vi.fn(),
  workspaceId: "workspace-a",
  workspaceChatReferenceId: "workspace-a",
  saveWorkspaceChatSession: mockSaveWorkspaceChatSession,
  getWorkspaceChatSession: mockGetWorkspaceChatSession
}

const messageOptionState = {
  messages: [] as Array<Record<string, unknown>>,
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
      defaultValueOrOptions?: string | { defaultValue?: string }
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

// The REAL useSmartScroll runs here: its containerRef feeds the
// virtualizer's scroll element, proving the production wiring end to end.

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
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setRagMediaIds: vi.fn(),
      setChatMode: vi.fn(),
      setFileRetrievalEnabled: vi.fn(),
      ragTopK: 8,
      setRagTopK: vi.fn(),
      ragAdvancedOptions: {},
      setRagAdvancedOptions: vi.fn(),
      selectedModel: "test-model",
      setSelectedModel: vi.fn()
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: { apiProvider?: string }) => unknown) =>
    selector({ apiProvider: undefined })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({
    message,
    isStreaming
  }: {
    message: string
    isStreaming?: boolean
  }) => (
    <div data-testid="playground-message" data-is-streaming={isStreaming ? "true" : "false"}>
      <span>{message}</span>
    </div>
  )
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title }: { title: React.ReactNode }) => <div>{title}</div>
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: vi.fn(),
  undoWorkspaceAction: vi.fn()
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceChatNoSourcesHint: () => "Select sources from the Sources pane.",
  getWorkspaceChatSourcesExplainer: () => "Selected sources keep answers grounded."
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

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [
    {
      api_provider: "test",
      id: "test-model",
      label: "Test Model",
      value: "test-model"
    }
  ])
}))

type FixtureMessage = {
  id: string
  isBot: boolean
  name: string
  message: string
  sources: Array<Record<string, unknown>>
  generationInfo?: Record<string, unknown>
  createdAt: number
}

const buildMessages = (count: number, lastMessageText?: string): FixtureMessage[] =>
  Array.from({ length: count }, (_unused, index) => {
    const isBot = index % 2 === 1
    return {
      id: `m-${index}`,
      isBot,
      name: isBot ? "Assistant" : "User",
      message:
        index === count - 1 && lastMessageText !== undefined
          ? lastMessageText
          : `content ${index}`,
      sources: isBot ? [{ name: `src-${index}` }] : [],
      generationInfo: isBot ? { retrieval: { chunks_retrieved: 3 } } : undefined,
      createdAt: 1_700_000_000_000 + index
    }
  })

const renderChatPane = () =>
  render(
    <MemoryRouter>
      <ChatPane />
    </MemoryRouter>
  )

const transcriptScrollContainer = () =>
  document.querySelector('[role="log"][aria-label="Chat messages"]') as HTMLElement

// jsdom has no layout: give the virtualizer a measurable viewport (via
// offsetHeight/offsetWidth) and measurable rows (via getBoundingClientRect),
// plus a recordable element.scrollTo, per the VirtualChatTimeline lifecycle
// test pattern. 160px rows in a 640px viewport -> ~4-row window + overscan 3.

// jsdom does not implement Element.scrollIntoView; the focus-target reveal
// path calls it, so install a no-op when absent.
const scrollIntoViewDescriptor = Object.getOwnPropertyDescriptor(
  Element.prototype,
  "scrollIntoView"
)
const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(
  Element.prototype,
  "scrollHeight"
)
const clientHeightDescriptor = Object.getOwnPropertyDescriptor(
  Element.prototype,
  "clientHeight"
)
const scrollDescriptor = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "scrollTo"
)
const scrollToCalls: Array<{ element: HTMLElement; top: number }> = []

beforeAll(() => {
  if (!scrollIntoViewDescriptor) {
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      value: () => {}
    })
  }
  Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    configurable: true,
    value(this: HTMLElement, options?: ScrollToOptions | number) {
      const top =
        typeof options === "number" ? options : (options?.top ?? 0)
      scrollToCalls.push({ element: this, top })
      if (typeof options === "object") this.scrollTop = options.top ?? 0
    }
  })
  // jsdom reports scrollHeight/clientHeight of 0: the virtualizer clamps
  // scrollToIndex to scrollHeight - clientHeight, so give the scroll
  // container a realistic scrollable extent (restored in afterAll).
  Object.defineProperty(Element.prototype, "scrollHeight", {
    configurable: true,
    get: () => 200_000
  })
  Object.defineProperty(Element.prototype, "clientHeight", {
    configurable: true,
    get: () => 640
  })
})

// vitest's restoreMocks (config) restores and detaches every spy right
// before each test body runs, i.e. AFTER user beforeEach hooks: create the
// layout spies afresh inside each test that needs a measurable viewport.
const applyLayoutStubs = () => {
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(640)
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(800)
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    () => ({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 800,
      bottom: 160,
      width: 800,
      height: 160,
      toJSON() {
        return {}
      }
    })
  )
}

afterEach(() => {
  scrollToCalls.length = 0
})

describe("ChatPane virtualized transcript (Stage 3, WP-04)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    buildRetrievalDiagnosticsSpy.mockClear()

    connectionStoreState.state.phase = ConnectionPhase.CONNECTED
    connectionStoreState.state.isChecking = false
    connectionStoreState.state.lastError = null

    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceChatReferenceId = "workspace-a"
    workspaceStoreState.sources = []
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.sourceFolders = []
    workspaceStoreState.sourceFolderMemberships = []
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.chatFocusTarget = null
    workspaceStoreState.getSelectedSources = () => []
    workspaceStoreState.getSelectedMediaIds = () => []
    workspaceStoreState.getEffectiveSelectedMediaIds = () => []

    messageOptionState.messages = []
    messageOptionState.history = []
    messageOptionState.historyId = null
    messageOptionState.serverChatId = null
    messageOptionState.streaming = false
    messageOptionState.isProcessing = false
    mockOnSubmit.mockResolvedValue(undefined)
    mockGetWorkspaceChatSession.mockReturnValue(null)
  })

  it("renders a bounded window for a 1,000-message transcript, not 1,000 shells", async () => {
    applyLayoutStubs()
    messageOptionState.messages = buildMessages(1000)
    renderChatPane()

    await waitFor(() => {
      expect(document.querySelectorAll("[data-chat-message-id]").length).toBeGreaterThan(
        0
      )
    })
    const renderedRows = document.querySelectorAll("[data-chat-message-id]").length
    // Plan success criterion: <= ~20 shells; e2e bound is < 60. The windowed
    // subset with overscan 3 must stay far below the fixture size.
    expect(renderedRows).toBeLessThan(60)
    expect(renderedRows).toBeLessThan(1000)
    // The top-of-transcript window renders the first messages.
    expect(
      document.querySelector('[data-chat-message-id="msg:m-0"]')
    ).not.toBeNull()
  })

  it("keeps the streaming last row mounted, updating and flagged at the window end", async () => {
    applyLayoutStubs()
    messageOptionState.messages = buildMessages(1000)
    messageOptionState.streaming = true
    const page = renderChatPane()

    // Pin the window to the transcript end (programmatic scrollTo does not
    // fire scroll events in jsdom).
    const container = transcriptScrollContainer()
    container.scrollTop = 10_000_000
    fireEvent.scroll(container)

    await waitFor(() => {
      expect(
        document.querySelector('[data-chat-message-id="msg:m-999"]')
      ).not.toBeNull()
    })

    const lastRow = document
      .querySelector('[data-chat-message-id="msg:m-999"]')
      ?.querySelector('[data-testid="playground-message"]')
    expect(lastRow?.getAttribute("data-is-streaming")).toBe("true")
    // Earlier rows in the end window are not flagged (Stage 2 narrowing).
    const earlierRow = document
      .querySelector('[data-chat-message-id="msg:m-998"]')
      ?.querySelector('[data-testid="playground-message"]')
    expect(earlierRow?.getAttribute("data-is-streaming")).toBe("false")

    // Streaming flush: only the last message object is replaced (Stage 1
    // semantics). The mounted last row must show the new text.
    const nextMessages = buildMessages(1000)
    nextMessages[999] = {
      ...nextMessages[999],
      message: "streamed final tokens arrive here"
    }
    messageOptionState.messages = nextMessages
    page.rerender(
      <MemoryRouter>
        <ChatPane />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText("streamed final tokens arrive here")).toBeInTheDocument()
    })
  })

  it("computes retrieval diagnostics once per message id across streaming re-renders", async () => {
    applyLayoutStubs()
    const baseMessages = buildMessages(1000)
    messageOptionState.messages = baseMessages
    const page = renderChatPane()

    await waitFor(() => {
      expect(document.querySelectorAll("[data-chat-message-id]").length).toBeGreaterThan(
        0
      )
    })
    // Pin the window to the transcript end where the streamed row lives.
    const container = transcriptScrollContainer()
    container.scrollTop = 10_000_000
    fireEvent.scroll(container)
    await waitFor(() => {
      expect(
        document.querySelector('[data-chat-message-id="msg:m-999"]')
      ).not.toBeNull()
    })
    const initialCalls = buildRetrievalDiagnosticsSpy.mock.calls.length
    // Only the windowed bot rows computed diagnostics (window + overscan),
    // not all 500 bot messages.
    expect(initialCalls).toBeGreaterThan(0)
    expect(initialCalls).toBeLessThan(60)

    // Three streaming flushes that replace only the last message object
    // (Stage 1 semantics; older entries keep referential identity): every
    // bot row's diagnostics — the replaced streaming row included, since
    // text-only flushes keep sources/generationInfo identity — must come
    // from the per-id cache.
    for (let flush = 1; flush <= 3; flush += 1) {
      const nextMessages = baseMessages.slice()
      nextMessages[999] = {
        ...baseMessages[999],
        message: `flush ${flush}`
      }
      messageOptionState.messages = nextMessages
      page.rerender(
        <MemoryRouter>
          <ChatPane />
        </MemoryRouter>
      )
    }
    await waitFor(() => {
      expect(screen.getByText("flush 3")).toBeInTheDocument()
    })
    expect(buildRetrievalDiagnosticsSpy.mock.calls.length).toBe(initialCalls)
  })

  it("reveals a chat focus target that lives outside the virtualized window", async () => {
    applyLayoutStubs()
    messageOptionState.messages = buildMessages(1000)
    const page = renderChatPane()

    await waitFor(() => {
      expect(document.querySelectorAll("[data-chat-message-id]").length).toBeGreaterThan(
        0
      )
    })

    // Focus targets arrive after mount in production (a global-search result
    // click), not during the initial auto-scroll commit.
    workspaceStoreState.chatFocusTarget = { messageId: "msg:m-950", token: 1 }
    page.rerender(
      <MemoryRouter>
        <ChatPane />
      </MemoryRouter>
    )

    const container = transcriptScrollContainer()
    // The virtualizer drove the container toward the target index (~950 of
    // 1000 rows). jsdom does not emit scroll events for programmatic
    // scrollTo, so replay each programmatic scroll as a scroll event (what a
    // real browser does asynchronously) until scrollToIndex's verify loop
    // lands the window on the target row.
    await waitFor(
      () => {
        const containerCalls = scrollToCalls.filter(
          (entry) => entry.element === container && entry.top > 100_000
        )
        expect(containerCalls.length).toBeGreaterThan(0)
        fireEvent.scroll(container)
        if (
          document.querySelector('[data-chat-message-id="msg:m-950"]') === null
        ) {
          throw new Error("target row not mounted yet")
        }
      },
      { timeout: 4000, interval: 16 }
    )

    // The target row was revealed and the focus target consumed.
    await waitFor(() => {
      expect(mockClearChatFocusTarget).toHaveBeenCalled()
    })
  })

  it("auto-scrolls to the bottom when a streaming update lands (useSmartScroll wiring preserved)", async () => {
    messageOptionState.messages = buildMessages(6)
    messageOptionState.streaming = true
    const page = renderChatPane()

    const nextMessages = buildMessages(6)
    nextMessages[5] = { ...nextMessages[5], message: "fresh streamed text" }
    messageOptionState.messages = nextMessages
    page.rerender(
      <MemoryRouter>
        <ChatPane />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText("fresh streamed text")).toBeInTheDocument()
    })
    const container = transcriptScrollContainer()
    await waitFor(() => {
      expect(
        scrollToCalls.some(
          (call) => call.element === container && typeof call.top === "number"
        )
      ).toBe(true)
    })
  })
})

describe("ChatPane transcript source guards (Stage 3, WP-04)", () => {
  it("drops the msg-${idx} key fallback and virtualizes through stableMessageId", () => {
    const source = readFileSync(
      path.resolve(__dirname, "..", "ChatPane", "index.tsx"),
      "utf8"
    )
    expect(source).not.toMatch(/msg-\$\{idx\}/)
    expect(source).toContain("useVirtualizer")
    expect(source).toContain("stableMessageId")
    expect(source).toContain("getRowRetrievalDiagnostics")
  })
})

afterAll(() => {
  if (scrollDescriptor) {
    Object.defineProperty(HTMLElement.prototype, "scrollTo", scrollDescriptor)
  } else {
    delete (HTMLElement.prototype as Partial<HTMLElement>).scrollTo
  }
  // scrollIntoView/scrollHeight/clientHeight accessors are defined by
  // jsdom, so the captured descriptors always exist to restore.
  if (scrollIntoViewDescriptor) {
    Object.defineProperty(Element.prototype, "scrollIntoView", scrollIntoViewDescriptor)
  }
  if (scrollHeightDescriptor) {
    Object.defineProperty(Element.prototype, "scrollHeight", scrollHeightDescriptor)
  }
  if (clientHeightDescriptor) {
    Object.defineProperty(Element.prototype, "clientHeight", clientHeightDescriptor)
  }
  vi.restoreAllMocks()
})
