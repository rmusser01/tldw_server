import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import React from "react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import axe from "axe-core"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { ResearchWorkspace } from "../index"

const {
  mockGetMediaDetails,
  mockUpsertWorkspace,
  mockGetWorkspaceSources,
  mockAddWorkspaceSource,
  mockUpdateWorkspaceSourceSelection,
  mockGetWorkspaceContext,
  mockGetWorkspaceSourcesStatus,
  mockGetWorkspaceCapabilities,
  mockCreateWorkspaceMigration,
  mockPutWorkspaceMigrationChunk,
  mockFinalizeWorkspaceMigration,
  mockGetWorkspaceMigration,
  mockAckWorkspaceMigrationClientDelete,
  mockRunResearchWorkspaceMigration,
  mockChatPaneProps,
  mockBgRequest
} = vi.hoisted(() => ({
  mockGetMediaDetails: vi.fn(),
  mockUpsertWorkspace: vi.fn(),
  mockGetWorkspaceSources: vi.fn(),
  mockAddWorkspaceSource: vi.fn(),
  mockUpdateWorkspaceSourceSelection: vi.fn(),
  mockGetWorkspaceContext: vi.fn(),
  mockGetWorkspaceSourcesStatus: vi.fn(),
  mockGetWorkspaceCapabilities: vi.fn(),
  mockCreateWorkspaceMigration: vi.fn(),
  mockPutWorkspaceMigrationChunk: vi.fn(),
  mockFinalizeWorkspaceMigration: vi.fn(),
  mockGetWorkspaceMigration: vi.fn(),
  mockAckWorkspaceMigrationClientDelete: vi.fn(),
  mockRunResearchWorkspaceMigration: vi.fn(),
  mockChatPaneProps: [] as any[],
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
  workspaceName: "New Research",
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
    getMediaDetails: mockGetMediaDetails,
    upsertWorkspace: mockUpsertWorkspace,
    getWorkspaceSources: mockGetWorkspaceSources,
    addWorkspaceSource: mockAddWorkspaceSource,
    updateWorkspaceSourceSelection: mockUpdateWorkspaceSourceSelection,
    getWorkspaceContext: undefined,
    getWorkspaceSourcesStatus: mockGetWorkspaceSourcesStatus,
    getWorkspaceCapabilities: mockGetWorkspaceCapabilities,
    createWorkspaceMigration: mockCreateWorkspaceMigration,
    putWorkspaceMigrationChunk: mockPutWorkspaceMigrationChunk,
    finalizeWorkspaceMigration: mockFinalizeWorkspaceMigration,
    getWorkspaceMigration: mockGetWorkspaceMigration,
    ackWorkspaceMigrationClientDelete: mockAckWorkspaceMigrationClientDelete
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/store/workspace-migration", () => ({
  runResearchWorkspaceMigration: mockRunResearchWorkspaceMigration
}))

vi.mock("@/utils/research-workspace-prefill", () => ({
  consumeResearchWorkspacePrefill: vi.fn().mockResolvedValue(null),
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
  ChatPane: (props: any) => {
    mockChatPaneProps.push(props)
    return <div data-testid="workspace-chat-pane">Chat</div>
  }
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: ({
    activeOperations,
    statusMessages
  }: {
    activeOperations?: string[]
    statusMessages?: string[]
  }) => (
    <div data-testid="workspace-status-bar">
      {statusMessages && statusMessages.length > 0 && (
        <div data-testid="workspace-statusbar-notice">
          {statusMessages.join(" ")}
        </div>
      )}
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

const sourceStatusSummary = {
  total: 1,
  selected: 1,
  queryable: 1,
  partially_queryable: 0,
  processing: 0,
  failed: 0,
  missing: 0
}

const readiness = {
  metadata_ready: true,
  text_extracted: true,
  fts_ready: true,
  vector_ready: true,
  citation_ready: true,
  summary_ready: false,
  tool_accessible: true
}

const makeStatusSource = (overrides: Record<string, unknown> = {}) => ({
  id: "source-ready",
  workspace_id: "workspace-1",
  media_id: 101,
  title: "Ready Source",
  source_type: "pdf",
  selected: true,
  state: "queryable",
  status_reason: "source_queryable",
  readiness,
  progress_percent: 100,
  progress_message: "Ready for grounded questions.",
  job: null,
  updated_at: "2026-05-23T12:00:00Z",
  ...overrides
})

const makeStatusPayload = (overrides: Record<string, unknown> = {}) => ({
  workspace_id: "workspace-1",
  sources: [makeStatusSource()],
  summary: sourceStatusSummary,
  ...overrides
})

const makeCapabilitiesPayload = (overrides: Record<string, unknown> = {}) => ({
  workspace_id: "workspace-1",
  workspace_kind: "research_workspace",
  access_level: "owner",
  source_summary: sourceStatusSummary,
  workspace_services: {},
  allowed_actions: {},
  ...overrides
})

const makeContextPayload = (overrides: Record<string, unknown> = {}) => ({
  workspace_id: "workspace-1",
  workspace_kind: "research_workspace",
  schema_version: 1,
  generated_at: "2026-05-25T00:00:00Z",
  workspace: {
    id: "workspace-1",
    name: "New Research",
    archived: false,
    study_materials_policy: "workspace",
    deleted: false,
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null,
    created_at: "2026-05-23T12:00:00Z",
    last_modified: "2026-05-23T12:00:00Z",
    version: 1
  },
  sources: {
    items: [makeStatusSource()],
    summary: sourceStatusSummary
  },
  capabilities: makeCapabilitiesPayload(),
  services: {},
  allowed_actions: {},
  active_jobs: [],
  partial_errors: [],
  ...overrides
})

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve
    reject = nextReject
  })
  return { promise, resolve, reject }
}

describe("ResearchWorkspace stage 3 global navigation", () => {
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
    window.localStorage.clear()
    ;(tldwClient as unknown as { getWorkspaceContext?: unknown }).getWorkspaceContext =
      undefined
    mockRunResearchWorkspaceMigration.mockResolvedValue({
      status: "not_needed",
      migrationId: null,
      manifestHash: "a".repeat(64),
      serverMigration: null,
      localDeletionEligibility: null,
      deletedSurfaceIds: [],
      message: "No legacy Research Workspace content was discovered."
    })
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
    testState.workspaceName = "New Research"
    testState.workspaceTag = "workspace:test"
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.isGeneratingOutput = false
    testState.generatingOutputType = null
    testState.sources = []
    testState.setSourceStatusByMediaId = vi.fn()
    testState.workspaceChatSessions = {}
    mockChatPaneProps.length = 0
    testState.currentNote = {
      id: 7,
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }
    testState.loadNote = vi.fn()
    mockGetMediaDetails.mockResolvedValue({})
    mockUpsertWorkspace.mockResolvedValue({
      id: "workspace-1",
      name: "New Research",
      archived: false,
      study_materials_policy: "workspace",
      deleted: false,
      banner_title: null,
      banner_subtitle: null,
      banner_color: null,
      audio_provider: null,
      audio_model: null,
      audio_voice: null,
      audio_speed: null,
      created_at: "2026-05-23T12:00:00Z",
      last_modified: "2026-05-23T12:00:00Z",
      version: 1
    })
    mockGetWorkspaceSources.mockResolvedValue([])
    mockAddWorkspaceSource.mockImplementation(
      async (_workspaceId: string, source: Record<string, unknown>) => ({
        ...source,
        workspace_id: "workspace-1",
        added_at: "2026-05-23T12:00:00Z",
        version: 1
      })
    )
    mockUpdateWorkspaceSourceSelection.mockResolvedValue(undefined)
    mockGetWorkspaceSourcesStatus.mockResolvedValue({
      workspace_id: "workspace-1",
      sources: [],
      summary: {
        total: 0,
        selected: 0,
        queryable: 0,
        partially_queryable: 0,
        processing: 0,
        failed: 0,
        missing: 0
      }
    })
    mockGetWorkspaceCapabilities.mockResolvedValue({
      workspace_id: "workspace-1",
      workspace_kind: "research_workspace",
      access_level: "owner",
      source_summary: {
        total: 0,
        selected: 0,
        queryable: 0,
        partially_queryable: 0,
        processing: 0,
        failed: 0,
        missing: 0
      },
      workspace_services: {},
      allowed_actions: {}
    })
    mockBgRequest.mockImplementation(async (request: { path: string }) => {
      const path = String(request.path)
      if (path.includes("/api/v1/notes/search/")) {
        return []
      }
      return { notes: [] }
    })
  })

  it("runs legacy workspace migration once and shows a compact recovery status when local data is retained", async () => {
    const legacyPayload = JSON.stringify({
      workspaces: [{ id: "legacy-workspace", name: "Legacy Workspace" }]
    })
    window.localStorage.setItem("tldw-workspace", legacyPayload)
    mockRunResearchWorkspaceMigration.mockResolvedValueOnce({
      status: "finalized_not_delete_eligible",
      migrationId: "research-workspace-workspace-1-abcd",
      manifestHash: "b".repeat(64),
      serverMigration: {
        id: "research-workspace-workspace-1-abcd",
        status: "finalized",
        client_delete_eligible: false
      },
      localDeletionEligibility: {
        eligible: true,
        blockingSurfaces: [],
        unknownSurfaces: [],
        retainedLocalSurfaces: []
      },
      deletedSurfaceIds: [],
      message:
        "Server receipt was saved. Local data is retained until server deletion eligibility is available."
    })

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockRunResearchWorkspaceMigration).toHaveBeenCalledTimes(1)
    })

    const migrationInput = mockRunResearchWorkspaceMigration.mock.calls[0]?.[0]
    expect(migrationInput).toEqual(
      expect.objectContaining({
        targetWorkspaceId: "workspace-1",
        targetWorkspaceName: "New Research",
        discoveredLocalStorageKeys: ["tldw-workspace"]
      })
    )
    await expect(
      migrationInput.readLocalStorageValue("tldw-workspace")
    ).resolves.toBe(legacyPayload)
    expect(migrationInput.deleteLocalStorageValue).toEqual(expect.any(Function))
    expect(migrationInput.writeLocalStorageValue).toEqual(expect.any(Function))

    const notice = await screen.findByTestId("workspace-statusbar-notice")
    expect(notice).toHaveTextContent("Legacy workspace data found")
    expect(notice).toHaveTextContent("Server receipt saved")
    expect(notice).toHaveTextContent(
      "Local data retained until server deletion eligibility is available"
    )
    expect(notice).toHaveTextContent("Review recovery details")
    expect(screen.queryByText(/workspace-playground/i)).not.toBeInTheDocument()
    expect(screen.queryByTestId("workspace-trust-panel")).not.toBeInTheDocument()
  })

  it("includes legacy IndexedDB offload stores when local split storage points to them", async () => {
    window.localStorage.setItem(
      "tldw-workspace",
      JSON.stringify({
        schema: "workspace_split_v1",
        state: { workspaceIds: ["workspace-1"] }
      })
    )
    window.localStorage.setItem(
      "tldw-workspace:workspace:workspace-1:chat",
      JSON.stringify({
        offloadType: "workspace_chat_session_v1",
        key: "workspace:workspace-1:chat",
        historyId: null,
        serverChatId: null,
        updatedAt: 1
      })
    )
    window.localStorage.setItem(
      "tldw-workspace:workspace:workspace-1:snapshot",
      JSON.stringify({
        generatedArtifacts: [
          {
            id: "artifact-1",
            __tldwArtifactPayloadRef: {
              offloadType: "workspace_artifact_payload_v1",
              key: "workspace:workspace-1:artifact:artifact-1",
              fields: ["content"],
              updatedAt: 1
            }
          }
        ]
      })
    )

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockRunResearchWorkspaceMigration).toHaveBeenCalledTimes(1)
    })

    expect(mockRunResearchWorkspaceMigration.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        discoveredIndexedDbStores: expect.arrayContaining([
          {
            databaseName: "tldw-workspace-storage",
            storeName: "workspace-chat-sessions"
          },
          {
            databaseName: "tldw-workspace-storage",
            storeName: "workspace-artifact-payloads"
          }
        ])
      })
    )
  })

  it("explains that local data is retained when the server receipt is saved but local inventory blocks deletion", async () => {
    window.localStorage.setItem(
      "tldw-workspace",
      JSON.stringify({
        workspaces: [{ id: "legacy-workspace", name: "Legacy Workspace" }]
      })
    )
    mockRunResearchWorkspaceMigration.mockResolvedValueOnce({
      status: "blocked",
      migrationId: "research-workspace-workspace-1-blocked",
      manifestHash: "d".repeat(64),
      serverMigration: {
        id: "research-workspace-workspace-1-blocked",
        status: "finalized",
        client_delete_eligible: false
      },
      localDeletionEligibility: {
        eligible: false,
        blockingSurfaces: [],
        unknownSurfaces: [
          {
            id: "unknown:localStorage:tldw:research-workspace:unknown",
            kind: "local_storage",
            key: "tldw:research-workspace:unknown",
            deletionPolicy: "unknown_blocks_deletion"
          }
        ],
        retainedLocalSurfaces: []
      },
      deletedSurfaceIds: [],
      message:
        "Server receipt was saved, but local deletion is blocked by the legacy inventory gate."
    })

    render(<ResearchWorkspace />)

    const notice = await screen.findByTestId("workspace-statusbar-notice")
    expect(notice).toHaveTextContent("Server receipt saved")
    expect(notice).toHaveTextContent("Local data retained")
    expect(notice).toHaveTextContent("Review recovery details")
  })

  it("settles migration status when React StrictMode remounts effects", async () => {
    const migrationDeferred = createDeferred<{
      status: string
      migrationId: string
      manifestHash: string
      serverMigration: {
        id: string
        status: string
        client_delete_eligible: boolean
      }
      localDeletionEligibility: {
        eligible: boolean
        blockingSurfaces: unknown[]
        unknownSurfaces: unknown[]
        retainedLocalSurfaces: unknown[]
      }
      deletedSurfaceIds: string[]
      message: string
    }>()
    window.localStorage.setItem(
      "tldw-workspace",
      JSON.stringify({
        workspaces: [{ id: "legacy-workspace", name: "Legacy Workspace" }]
      })
    )
    mockRunResearchWorkspaceMigration.mockReturnValueOnce(
      migrationDeferred.promise
    )

    render(
      <React.StrictMode>
        <ResearchWorkspace />
      </React.StrictMode>
    )

    await waitFor(() => {
      expect(mockRunResearchWorkspaceMigration).toHaveBeenCalledTimes(1)
    })

    await act(async () => {
      migrationDeferred.resolve({
        status: "finalized_not_delete_eligible",
        migrationId: "research-workspace-workspace-1-strict",
        manifestHash: "c".repeat(64),
        serverMigration: {
          id: "research-workspace-workspace-1-strict",
          status: "finalized",
          client_delete_eligible: false
        },
        localDeletionEligibility: {
          eligible: true,
          blockingSurfaces: [],
          unknownSurfaces: [],
          retainedLocalSurfaces: []
        },
        deletedSurfaceIds: [],
        message:
          "Server receipt was saved. Local data is retained until server deletion eligibility is available."
      })
      await Promise.resolve()
    })

    const notice = await screen.findByTestId("workspace-statusbar-notice")
    expect(notice).toHaveTextContent("Server receipt saved")
    expect(notice).toHaveTextContent(
      "Local data retained until server deletion eligibility is available"
    )
  })

  it("opens and closes workspace search with keyboard shortcuts", async () => {
    render(<ResearchWorkspace />)

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
    render(<ResearchWorkspace />)

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
    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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
    render(<ResearchWorkspace />)

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
    const { container } = render(<ResearchWorkspace />)
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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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

    const { rerender } = render(<ResearchWorkspace />)
    expect(
      screen.queryByTestId("workspace-switch-transition")
    ).not.toBeInTheDocument()

    act(() => {
      testState.workspaceId = "workspace-2"
      rerender(<ResearchWorkspace />)
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

    render(<ResearchWorkspace />)

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

  it("does not let media-detail fallback override a partial workspace status projection", async () => {
    const partialReadiness = {
      metadata_ready: true,
      text_extracted: true,
      fts_ready: true,
      vector_ready: false,
      citation_ready: true,
      summary_ready: false,
      tool_accessible: true
    }
    testState.sources = [
      {
        id: "source-partial",
        mediaId: 808,
        title: "Partially queryable source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce(
      makeStatusPayload({
        sources: [
          makeStatusSource({
            id: "source-partial",
            media_id: 808,
            title: "Partially queryable source",
            state: "partially_queryable",
            status_reason: "vector_index_pending",
            readiness: partialReadiness,
            progress_percent: 75,
            progress_message:
              "Text search is available while vector indexing continues."
          })
        ],
        summary: {
          total: 1,
          selected: 1,
          queryable: 0,
          partially_queryable: 1,
          processing: 1,
          failed: 0,
          missing: 0
        }
      })
    )
    mockGetMediaDetails.mockResolvedValue({
      content: {
        text: "Extracted source text exists before vector indexing finishes."
      }
    })

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        808,
        "processing",
        "Text search is available while vector indexing continues.",
        partialReadiness,
        expect.objectContaining({
          lifecycleState: "partially_queryable",
          statusReason: "vector_index_pending",
          sourceOfTruth: "workspace-status-projection",
          progressPercent: 75,
          progressMessage:
            "Text search is available while vector indexing continues.",
          stale: false,
          retryEligible: false
        })
      )
    })
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

  it("renders backend status projection and reconciles source statuses fail-closed", async () => {
    testState.sources = [
      {
        id: "source-ready",
        mediaId: 101,
        title: "Ready Source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-05-23T12:00:00.000Z")
      },
      {
        id: "source-indexing",
        mediaId: 102,
        title: "Indexing Source",
        type: "website",
        status: "ready",
        addedAt: new Date("2026-05-23T12:01:00.000Z")
      },
      {
        id: "source-missing",
        mediaId: 103,
        title: "Missing Source",
        type: "document",
        status: "processing",
        addedAt: new Date("2026-05-23T12:02:00.000Z")
      }
    ]
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce({
      workspace_id: "workspace-1",
      sources: [
        {
          id: "source-ready",
          workspace_id: "workspace-1",
          media_id: 101,
          title: "Ready Source",
          source_type: "pdf",
          selected: true,
          state: "queryable",
          status_reason: "source_queryable",
          readiness: {
            metadata_ready: true,
            text_extracted: true,
            fts_ready: true,
            vector_ready: true,
            citation_ready: true,
            summary_ready: false,
            tool_accessible: true
          },
          progress_percent: 100,
          progress_message: "Ready for grounded questions.",
          job: null,
          updated_at: "2026-05-23T12:00:00Z"
        },
        {
          id: "source-indexing",
          workspace_id: "workspace-1",
          media_id: 102,
          title: "Indexing Source",
          source_type: "web",
          selected: true,
          state: "indexing",
          status_reason: "job_indexing",
          readiness: {
            metadata_ready: true,
            text_extracted: true,
            fts_ready: true,
            vector_ready: false,
            citation_ready: true,
            summary_ready: false,
            tool_accessible: true
          },
          progress_percent: 82,
          progress_message: "Indexing",
          job: null,
          updated_at: "2026-05-23T12:01:00Z"
        },
        {
          id: "source-missing",
          workspace_id: "workspace-1",
          media_id: 103,
          title: "Missing Source",
          source_type: "doc",
          selected: true,
          state: "missing_media",
          status_reason: "media_not_found",
          readiness: {
            metadata_ready: false,
            text_extracted: false,
            fts_ready: false,
            vector_ready: false,
            citation_ready: false,
            summary_ready: false,
            tool_accessible: false
          },
          progress_percent: 0,
          progress_message: "Media item is missing.",
          job: null,
          updated_at: "2026-05-23T12:02:00Z"
        }
      ],
      summary: {
        total: 3,
        selected: 3,
        queryable: 1,
        partially_queryable: 0,
        processing: 1,
        failed: 0,
        missing: 1
      }
    })
    mockGetWorkspaceCapabilities.mockResolvedValueOnce({
      workspace_id: "workspace-1",
      workspace_kind: "research_workspace",
      access_level: "owner",
      source_summary: {
        total: 3,
        selected: 3,
        queryable: 1,
        partially_queryable: 0,
        processing: 1,
        failed: 0,
        missing: 1
      },
      workspace_services: {
        mcp: {
          state: "not_configured",
          reason_code: "no_workspace_mcp_binding",
          management_surface: "mcp_hub"
        },
        acp: {
          state: "not_configured",
          reason_code: "no_workspace_acp_binding",
          management_surface: "acp_workspace"
        },
        sandbox: {
          state: "not_configured",
          reason_code: "no_workspace_sandbox_binding",
          management_surface: "sandbox_settings"
        },
        provider: {
          state: "unknown",
          reason_code: "provider_not_evaluated",
          management_surface: "model_settings"
        }
      },
      allowed_actions: {
        ask_grounded_questions: {
          allowed: true,
          reason_code: null
        },
        run_mcp_tools: {
          allowed: false,
          reason_code: "mcp_not_configured"
        }
      }
    })

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledWith("workspace-1")
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledWith("workspace-1")
      expect(mockChatPaneProps.at(-1)?.workspaceCapabilities).toMatchObject({
        workspace_id: "workspace-1",
        workspace_services: expect.objectContaining({
          mcp: expect.objectContaining({
            state: "not_configured",
            management_surface: "mcp_hub"
          }),
          acp: expect.objectContaining({
            state: "not_configured",
            management_surface: "acp_workspace"
          }),
          sandbox: expect.objectContaining({
            state: "not_configured",
            management_surface: "sandbox_settings"
          }),
          provider: expect.objectContaining({
            state: "unknown",
            management_surface: "model_settings"
          })
        })
      })
    })

    await waitFor(() => {
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        101,
        "ready",
        "Ready for grounded questions.",
        expect.objectContaining({
          text_extracted: true,
          vector_ready: true
        }),
        expect.objectContaining({
          lifecycleState: "queryable",
          statusReason: "source_queryable",
          sourceOfTruth: "workspace-status-projection",
          progressPercent: 100,
          progressMessage: "Ready for grounded questions.",
          stale: false,
          retryEligible: false
        })
      )
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        102,
        "processing",
        "Indexing",
        expect.objectContaining({
          text_extracted: true,
          vector_ready: false
        }),
        expect.objectContaining({
          lifecycleState: "indexing",
          statusReason: "job_indexing",
          sourceOfTruth: "workspace-status-projection",
          progressPercent: 82,
          progressMessage: "Indexing",
          stale: false,
          retryEligible: false
        })
      )
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        103,
        "error",
        "Media item is missing.",
        expect.objectContaining({
          text_extracted: false,
          tool_accessible: false
        }),
        expect.objectContaining({
          lifecycleState: "missing_media",
          statusReason: "media_not_found",
          sourceOfTruth: "workspace-status-projection",
          progressPercent: 0,
          progressMessage: "Media item is missing.",
          stale: false,
          retryEligible: true
        })
      )
    })
  })

  it("reconciles source status when capabilities fail independently", async () => {
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce(makeStatusPayload())
    mockGetWorkspaceCapabilities.mockRejectedValueOnce(
      new Error("Capabilities unavailable")
    )

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        101,
        "ready",
        "Ready for grounded questions.",
        expect.objectContaining({
          text_extracted: true,
          vector_ready: true
        }),
        expect.objectContaining({
          lifecycleState: "queryable",
          statusReason: "source_queryable",
          sourceOfTruth: "workspace-status-projection"
        })
      )
    })
  })

  it("uses the canonical workspace context envelope when available", async () => {
    ;(tldwClient as unknown as { getWorkspaceContext: typeof mockGetWorkspaceContext }).getWorkspaceContext =
      mockGetWorkspaceContext
    mockGetWorkspaceContext.mockResolvedValueOnce(
      makeContextPayload({
        sources: {
          items: [
            makeStatusSource({
              media_id: 404,
              progress_message: "Context source ready"
            })
          ],
          summary: sourceStatusSummary
        },
        partial_errors: [
          {
            scope: "jobs",
            code: "jobs_unavailable",
            message: "Jobs status is temporarily unavailable."
          }
        ]
      })
    )

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceContext).toHaveBeenCalledWith("workspace-1")
    })
    expect(mockGetWorkspaceSourcesStatus).not.toHaveBeenCalled()
    expect(mockGetWorkspaceCapabilities).not.toHaveBeenCalled()
    expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
      404,
      "ready",
      "Context source ready",
      expect.objectContaining({
        text_extracted: true,
        vector_ready: true
      }),
      expect.objectContaining({
        lifecycleState: "queryable",
        sourceOfTruth: "workspace-status-projection",
        progressMessage: "Context source ready"
      })
    )
  })

  it("bootstraps the server workspace and source rows before status projection calls", async () => {
    testState.sources = [
      {
        id: "source-ready",
        mediaId: 101,
        title: "Ready Source",
        type: "pdf",
        status: "processing",
        url: "https://example.test/ready.pdf",
        addedAt: new Date("2026-05-23T12:00:00.000Z")
      }
    ]

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledWith("workspace-1")
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledWith("workspace-1")
    })

    expect(mockUpsertWorkspace).toHaveBeenCalledWith("workspace-1", {
      name: "New Research",
      study_materials_policy: "workspace"
    })
    expect(mockGetWorkspaceSources).toHaveBeenCalledWith("workspace-1")
    expect(mockAddWorkspaceSource).toHaveBeenCalledWith("workspace-1", {
      id: "source-ready",
      media_id: 101,
      title: "Ready Source",
      source_type: "pdf",
      url: "https://example.test/ready.pdf",
      position: 0,
      selected: false
    })
    expect(mockUpdateWorkspaceSourceSelection).toHaveBeenCalledWith(
      "workspace-1",
      []
    )

    expect(
      mockUpsertWorkspace.mock.invocationCallOrder[0]
    ).toBeLessThan(mockGetWorkspaceSources.mock.invocationCallOrder[0])
    expect(
      mockGetWorkspaceSources.mock.invocationCallOrder[0]
    ).toBeLessThan(mockAddWorkspaceSource.mock.invocationCallOrder[0])
    expect(
      mockAddWorkspaceSource.mock.invocationCallOrder[0]
    ).toBeLessThan(mockGetWorkspaceSourcesStatus.mock.invocationCallOrder[0])
    expect(
      mockAddWorkspaceSource.mock.invocationCallOrder[0]
    ).toBeLessThan(mockGetWorkspaceCapabilities.mock.invocationCallOrder[0])
  })

  it("persists the canonical local source selection during server bootstrap", async () => {
    testState.sources = [
      {
        id: "source-unselected",
        mediaId: 111,
        title: "Unselected Source",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-05-23T12:00:00.000Z")
      },
      {
        id: "source-selected",
        mediaId: 222,
        title: "Selected Source",
        type: "website",
        status: "ready",
        addedAt: new Date("2026-05-23T12:01:00.000Z")
      }
    ]
    testState.selectedSourceIds = ["source-selected"]

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledWith("workspace-1")
    })

    expect(mockAddWorkspaceSource).toHaveBeenNthCalledWith(1, "workspace-1", {
      id: "source-unselected",
      media_id: 111,
      title: "Unselected Source",
      source_type: "pdf",
      url: null,
      position: 0,
      selected: false
    })
    expect(mockAddWorkspaceSource).toHaveBeenNthCalledWith(2, "workspace-1", {
      id: "source-selected",
      media_id: 222,
      title: "Selected Source",
      source_type: "website",
      url: null,
      position: 1,
      selected: true
    })
    expect(mockUpdateWorkspaceSourceSelection).toHaveBeenCalledWith(
      "workspace-1",
      ["source-selected"]
    )
    expect(
      mockUpdateWorkspaceSourceSelection.mock.invocationCallOrder[0]
    ).toBeLessThan(mockGetWorkspaceSourcesStatus.mock.invocationCallOrder[0])
  })

  it("continues source status projection when server bootstrap fails", async () => {
    mockUpsertWorkspace.mockRejectedValueOnce(new Error("database locked"))
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce(makeStatusPayload())
    mockGetWorkspaceCapabilities.mockResolvedValueOnce(makeCapabilitiesPayload())

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledWith("workspace-1")
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledWith("workspace-1")
    })

    expect(mockGetWorkspaceSources).not.toHaveBeenCalled()
    expect(screen.queryByTestId("workspace-trust-panel")).not.toBeInTheDocument()
    expect(screen.queryByText("Workspace server sync unavailable")).not.toBeInTheDocument()
    expect(screen.queryByText("database locked")).not.toBeInTheDocument()
  })

  it("ignores source status projection responses for a previous workspace", async () => {
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce(
      makeStatusPayload({
        workspace_id: "workspace-other",
        sources: [
          makeStatusSource({
            workspace_id: "workspace-other",
            media_id: 201,
            progress_message: "Wrong workspace"
          })
        ]
      })
    )
    mockGetWorkspaceCapabilities.mockResolvedValueOnce(
      makeCapabilitiesPayload({ workspace_id: "workspace-other" })
    )

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledWith("workspace-1")
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledWith("workspace-1")
    })

    expect(testState.setSourceStatusByMediaId).not.toHaveBeenCalledWith(
      201,
      expect.any(String),
      expect.any(String)
    )
  })

  it("fails closed for unknown lifecycle states and ignores nullable media ids", async () => {
    mockGetWorkspaceSourcesStatus.mockResolvedValueOnce(
      makeStatusPayload({
        sources: [
          makeStatusSource({
            id: "source-null-media",
            media_id: null,
            progress_message: "No media id"
          }),
          makeStatusSource({
            id: "source-unknown-state",
            media_id: 302,
            state: "unknown",
            status_reason: "unrecognized_lifecycle_state",
            progress_message: null
          })
        ],
        summary: {
          total: 2,
          selected: 2,
          queryable: 0,
          partially_queryable: 0,
          processing: 0,
          failed: 1,
          missing: 0
        }
      })
    )
    mockGetWorkspaceCapabilities.mockResolvedValueOnce(makeCapabilitiesPayload())

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(testState.setSourceStatusByMediaId).toHaveBeenCalledWith(
        302,
        "error",
        "unrecognized_lifecycle_state",
        expect.objectContaining({
          text_extracted: true,
          vector_ready: true
        }),
        expect.objectContaining({
          lifecycleState: "unknown",
          statusReason: "unrecognized_lifecycle_state",
          sourceOfTruth: "workspace-status-projection",
          stale: false,
          retryEligible: false
        })
      )
    })
    expect(testState.setSourceStatusByMediaId).not.toHaveBeenCalledWith(
      null,
      expect.any(String),
      expect.any(String)
    )
  })

  it("does not start overlapping source status projection polls", async () => {
    vi.useFakeTimers()
    try {
      const statusDeferred = createDeferred<ReturnType<typeof makeStatusPayload>>()
      const capabilitiesDeferred =
        createDeferred<ReturnType<typeof makeCapabilitiesPayload>>()
      mockGetWorkspaceSourcesStatus.mockReturnValueOnce(statusDeferred.promise)
      mockGetWorkspaceCapabilities.mockReturnValueOnce(
        capabilitiesDeferred.promise
      )

      render(<ResearchWorkspace />)

      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })
      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(1)
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledTimes(1)

      await act(async () => {
        vi.advanceTimersByTime(5000)
        await Promise.resolve()
      })

      expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(1)
      expect(mockGetWorkspaceCapabilities).toHaveBeenCalledTimes(1)

      statusDeferred.resolve(makeStatusPayload())
      capabilitiesDeferred.resolve(makeCapabilitiesPayload())
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })
    } finally {
      vi.useRealTimers()
    }
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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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

    render(<ResearchWorkspace />)

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
