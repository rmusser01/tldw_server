import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import axe from "axe-core"
import { WorkspacePlayground } from "../index"

const { mockGetMediaDetails, mockBgRequest } = vi.hoisted(() => ({
  mockGetMediaDetails: vi.fn(),
  mockBgRequest: vi.fn()
}))

const { mockScheduleWorkspaceUndoAction, mockUndoWorkspaceAction } = vi.hoisted(
  () => ({
    mockScheduleWorkspaceUndoAction: vi.fn(),
    mockUndoWorkspaceAction: vi.fn()
  })
)

const testState = {
  isMobile: false,
  storeHydrated: true,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  workspaceId: "workspace-1",
  workspaceTag: "workspace:test",
  initializeWorkspace: vi.fn(),
  createNewWorkspace: vi.fn(),
  addSources: vi.fn(),
  setSelectedSourceIds: vi.fn(),
  captureToCurrentNote: vi.fn(),
  clearCurrentNote: vi.fn(),
  setCurrentNote: vi.fn(),
  loadNote: vi.fn(),
  selectedSourceIds: [] as string[],
  generatedArtifacts: [] as Array<{ id: string }>,
  isGeneratingOutput: false,
  generatingOutputType: null as string | null,
  setLeftPaneCollapsed: vi.fn(),
  setRightPaneCollapsed: vi.fn(),
  focusSourceById: vi.fn(() => true),
  focusChatMessageById: vi.fn(() => true),
  focusWorkspaceNote: vi.fn(),
  setSourceStatusByMediaId: vi.fn(),
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    addedAt: Date
    status?: "processing" | "ready" | "error"
    url?: string
  }>,
  workspaceChatSessions: {} as Record<string, { messages: any[] }>,
  currentNote: {
    id: 7 as number | undefined,
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
  }
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
  useMobile: () => testState.isMobile
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: (state: typeof testState) => unknown) =>
    selector(testState),
  createWorkspaceStorage: () => ({
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
    removeItem: vi.fn()
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: mockGetMediaDetails
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/utils/workspace-playground-prefill", () => ({
  consumeWorkspacePlaygroundPrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue("")
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: mockScheduleWorkspaceUndoAction,
  undoWorkspaceAction: mockUndoWorkspaceAction
}))

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: () => <div data-testid="workspace-header" />
}))

vi.mock("../SourcesPane", () => ({
  SourcesPane: () => <div data-testid="workspace-sources-pane">Sources</div>
}))

vi.mock("../ChatPane", () => ({
  ChatPane: () => <div data-testid="workspace-chat-pane">Chat</div>
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: ({ activeOperations }: { activeOperations?: string[] }) => (
    <div data-testid="workspace-status-bar">
      {activeOperations && activeOperations.length > 0 && (
        <div data-testid="workspace-statusbar-activity">
          {activeOperations.join(" \u2022 ")}
        </div>
      )}
    </div>
  )
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("WorkspacePlayground stage 3 global navigation", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query.includes("min-width: 1024px"),
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockUndoWorkspaceAction.mockReturnValue(true)
    mockScheduleWorkspaceUndoAction.mockImplementation(
      (config: { apply?: () => void }) => {
        config.apply?.()
        return { id: "workspace-undo-1", expiresAt: Date.now() + 10000 }
      }
    )
    testState.isMobile = false
    testState.storeHydrated = true
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.workspaceId = "workspace-1"
    testState.workspaceTag = "workspace:test"
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.isGeneratingOutput = false
    testState.generatingOutputType = null
    testState.sources = []
    testState.setSourceStatusByMediaId = vi.fn()
    testState.workspaceChatSessions = {}
    testState.currentNote = {
      id: 7,
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }
    testState.loadNote = vi.fn()
    mockGetMediaDetails.mockResolvedValue({})
    mockBgRequest.mockImplementation(async (request: { path: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/")) {
        return []
      }
      return { notes: [] }
    })
  })

  it("opens and closes workspace search with keyboard shortcuts", async () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "k", altKey: true })

    const dialog = await screen.findByRole("dialog", { name: "Search workspace" })
    expect(dialog).toBeInTheDocument()

    fireEvent.keyDown(
      within(dialog).getByPlaceholderText("Search sources, chat, and notes..."),
      { key: "Escape" }
    )

    await waitFor(() => {
      const dialog = screen.queryByRole("dialog", { name: "Search workspace" })
      if (!dialog) {
        expect(dialog).not.toBeInTheDocument()
        return
      }
      expect(dialog).toHaveClass("ant-zoom-leave")
    })
  })

  it("closes workspace search when Escape is pressed inside the search input", async () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "k", altKey: true })

    const dialog = await screen.findByRole("dialog", { name: "Search workspace" })
    expect(dialog).toBeInTheDocument()

    const searchInput = screen.getByPlaceholderText(/Search sources, chat, and notes/i)
    searchInput.focus()
    fireEvent.keyDown(searchInput, { key: "Escape" })

    await waitFor(() => {
      const nextDialog = screen.queryByRole("dialog", { name: "Search workspace" })
      if (!nextDialog) {
        expect(nextDialog).not.toBeInTheDocument()
        return
      }
      expect(nextDialog).toHaveClass("ant-zoom-leave")
    })
  })

  it("routes pane focus shortcuts and workspace creation shortcuts", () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "1", altKey: true })
    expect(testState.setLeftPaneCollapsed).toHaveBeenCalledWith(false)

    fireEvent.keyDown(window, { key: "3", altKey: true })
    expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)

    fireEvent.keyDown(window, { key: "N", altKey: true, shiftKey: true })
    expect(testState.createNewWorkspace).toHaveBeenCalledTimes(1)
  })

  it("starts a new note draft with Alt+N", () => {
    testState.currentNote = {
      id: undefined,
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }

    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "n", altKey: true })

    expect(testState.clearCurrentNote).toHaveBeenCalledTimes(1)
    return waitFor(() => {
      expect(testState.focusWorkspaceNote).toHaveBeenCalledWith("title")
    })
  })

  it("uses undo-managed clear flow for non-empty notes from Alt+N", async () => {
    testState.currentNote = {
      id: 9,
      title: "Draft note",
      content: "Important draft",
      keywords: ["draft"],
      isDirty: true
    }

    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "n", altKey: true })

    fireEvent.click(
      await screen.findByRole("button", { name: "New note" })
    )

    await waitFor(() => {
      expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalledTimes(1)
      expect(testState.clearCurrentNote).toHaveBeenCalledTimes(1)
    })
  })

  it("provides skip links and labeled complementary landmarks", () => {
    render(<WorkspacePlayground />)

    expect(
      screen.getByRole("link", { name: "Skip to chat content" })
    ).toHaveAttribute("href", "#workspace-main-content")
    expect(
      screen.getByRole("link", { name: "Skip to sources panel" })
    ).toHaveAttribute("href", "#workspace-sources-panel")
    expect(
      screen.getByRole("link", { name: "Skip to studio panel" })
    ).toHaveAttribute("href", "#workspace-studio-panel")

    expect(screen.getByRole("main")).toHaveAttribute(
      "id",
      "workspace-main-content"
    )
    expect(
      screen.getByRole("complementary", { name: "Sources panel" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("complementary", { name: "Studio panel" })
    ).toBeInTheDocument()
  })

  it("has no axe-core violations for landmark and naming rules", async () => {
    const { container } = render(<WorkspacePlayground />)
    const results = await axe.run(container, {
      runOnly: {
        type: "rule",
        values: [
          "landmark-one-main",
          "region",
          "button-name",
          "link-name",
          "aria-required-attr",
          "aria-valid-attr",
          "aria-valid-attr-value"
        ]
      }
    })

    expect(results.violations).toEqual([])
  })

  it("routes source search selection to source focus", async () => {
    testState.sources = [
      {
        id: "source-climate",
        mediaId: 101,
        title: "Climate Action Report",
        type: "pdf",
        addedAt: new Date("2026-02-18T09:00:00.000Z")
      }
    ]

    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "k", altKey: true })
    const searchInput = await screen.findByPlaceholderText(
      "Search sources, chat, and notes..."
    )
    fireEvent.change(searchInput, { target: { value: "climate" } })

    fireEvent.click(await screen.findByRole("button", { name: /Climate Action Report/ }))

    await waitFor(() => {
      expect(testState.focusSourceById).toHaveBeenCalledWith("source-climate")
    })
  })

  it("routes chat and note selections to their focus targets", async () => {
    testState.workspaceChatSessions = {
      "workspace-1": {
        messages: [
          {
            id: "assistant-msg-1",
            isBot: true,
            name: "Assistant",
            message: "Retrieval confidence is moderate for source B.",
            sources: []
          }
        ]
      }
    }
    testState.currentNote = {
      id: 3,
      title: "Confidence tracker",
      content: "Track confidence changes over time.",
      keywords: ["confidence"],
      isDirty: false
    }

    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "k", altKey: true })
    const searchInput = await screen.findByPlaceholderText(
      "Search sources, chat, and notes..."
    )

    fireEvent.change(searchInput, { target: { value: "moderate" } })
    fireEvent.click(await screen.findByRole("button", { name: /Assistant message/ }))

    await waitFor(() => {
      expect(testState.focusChatMessageById).toHaveBeenCalledWith(
        "msg:assistant-msg-1"
      )
    })

    fireEvent.keyDown(window, { key: "k", altKey: true })
    const noteSearchInput = await screen.findByPlaceholderText(
      "Search sources, chat, and notes..."
    )

    fireEvent.change(noteSearchInput, { target: { value: "confidence tracker" } })
    fireEvent.click(await screen.findByRole("button", { name: /Confidence tracker/ }))

    await waitFor(() => {
      expect(testState.focusWorkspaceNote).toHaveBeenCalledWith("title")
    })
  })

  it("loads and focuses non-current note results selected from global search", async () => {
    testState.currentNote = {
      id: 3,
      title: "Current draft",
      content: "Current draft content",
      keywords: [],
      isDirty: true
    }

    mockBgRequest.mockImplementation(async (request: { path: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/")) {
        return [
          {
            id: 88,
            title: "Workspace confidence note",
            content: "Detailed confidence notes",
            keywords: ["workspace:test", "confidence"]
          }
        ]
      }
      if (path.endsWith("/api/v1/notes/88")) {
        return {
          id: 88,
          title: "Workspace confidence note",
          content: "Detailed confidence notes",
          keywords: [{ keyword: "workspace:test" }, { keyword: "confidence" }],
          version: 2
        }
      }
      return { notes: [] }
    })

    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "k", altKey: true })
    const searchInput = await screen.findByPlaceholderText(
      "Search sources, chat, and notes..."
    )
    fireEvent.change(searchInput, {
      target: { value: "workspace confidence note" }
    })
    fireEvent.click(
      await screen.findByRole("button", { name: /Workspace confidence note/ })
    )

    await waitFor(() => {
      expect(testState.loadNote).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 88,
          title: "Workspace confidence note",
          content: "Detailed confidence notes",
          keywords: ["workspace:test", "confidence"],
          version: 2
        })
      )
      expect(testState.focusWorkspaceNote).toHaveBeenCalledWith("title")
    })
  })

  it("shows a brief transition cue when workspace id changes", () => {
    vi.useFakeTimers()

    const { rerender } = render(<WorkspacePlayground />)
    expect(
      screen.queryByTestId("workspace-switch-transition")
    ).not.toBeInTheDocument()

    act(() => {
      testState.workspaceId = "workspace-2"
      rerender(<WorkspacePlayground />)
    })

    expect(screen.getByTestId("workspace-switch-transition")).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(500)
    })

    expect(
      screen.queryByTestId("workspace-switch-transition")
    ).not.toBeInTheDocument()

    vi.useRealTimers()
  })

  it("keeps processing sources in processing while vector indexing is still pending", async () => {
    testState.sources = [
      {
        id: "source-processing",
        mediaId: 808,
        title: "Queued Source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]
    mockGetMediaDetails.mockResolvedValue({
      content: {
        text: "Processed transcript text"
      },
      vector_processing_status: "pending"
    })

    render(<WorkspacePlayground />)

    await waitFor(() => {
      expect(mockGetMediaDetails).toHaveBeenCalledWith(
        808,
        expect.objectContaining({
          include_content: true
        })
      )
    })

    expect(testState.setSourceStatusByMediaId).not.toHaveBeenCalledWith(
      808,
      "ready"
    )
  })

  it("promotes processing sources to ready when polling detects completed vector-ready content", async () => {
    testState.sources = [
      {
        id: "source-processing",
        mediaId: 808,
        title: "Queued Source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]
    mockGetMediaDetails.mockResolvedValue({
      content: {
        text: "Processed transcript text"
      },
      vector_processing_status: "completed"
    })

    render(<WorkspacePlayground />)

    await waitFor(() => {
      expect(mockGetMediaDetails).toHaveBeenCalledWith(
        808,
        expect.objectContaining({
          include_content: true
        })
      )
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        808,
        "ready"
      )
    })
  })

  it("promotes processing sources to ready when a later vector status indicates completion", async () => {
    testState.sources = [
      {
        id: "source-processing",
        mediaId: 808,
        title: "Queued Source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]
    mockGetMediaDetails.mockResolvedValue({
      vector_processing: "pending",
      processing: {
        vector_processing_status: "completed"
      }
    })

    render(<WorkspacePlayground />)

    await waitFor(() => {
      expect(mockGetMediaDetails).toHaveBeenCalledWith(
        808,
        expect.objectContaining({
          include_content: true
        })
      )
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        808,
        "ready"
      )
    })
  })

  it("shows an activity rail when sources are processing or outputs are generating", () => {
    testState.sources = [
      {
        id: "source-processing",
        mediaId: 808,
        title: "Queued Source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]
    testState.isGeneratingOutput = true
    testState.generatingOutputType = "summary"

    render(<WorkspacePlayground />)

    const rail = screen.getByTestId("workspace-statusbar-activity")
    expect(rail).toBeInTheDocument()
    expect(rail).toHaveTextContent("Processing 1 source")
    expect(rail).toHaveTextContent("Generating summary")
  })

  it("marks processing sources as error after repeated non-transient polling failures", async () => {
    vi.useFakeTimers()
    testState.sources = [
      {
        id: "source-processing-error",
        mediaId: 909,
        title: "Broken Source",
        type: "video",
        status: "processing",
        addedAt: new Date("2026-02-18T12:30:00.000Z")
      }
    ]

    const error = new Error("Malformed metadata") as Error & { status?: number }
    error.status = 400
    mockGetMediaDetails.mockRejectedValue(error)

    render(<WorkspacePlayground />)

    await act(async () => {
      await Promise.resolve()
    })
    expect(mockGetMediaDetails).toHaveBeenCalledTimes(1)
    expect(testState.setSourceStatusByMediaId).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(5000)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
      909,
      "error",
      "Malformed metadata"
    )

    vi.useRealTimers()
  })
})
