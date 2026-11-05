import React from "react"
import { act, render as renderUI } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { ResearchWorkspace } from "../index"

// WP-42: the workspace's two 5 s pollers (server status projection and
// per-source readiness checks) collapse into one scheduler that only ticks
// while sources are processing and the page is visible.

const render: typeof renderUI = (ui, options) =>
  renderUI(ui, {
    wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
    ...options
  })

const {
  mockGetMediaDetails,
  mockUpsertWorkspace,
  mockGetWorkspaceSources,
  mockAddWorkspaceSource,
  mockUpdateWorkspaceSourceSelection,
  mockGetWorkspaceSourcesStatus,
  mockGetWorkspaceCapabilities,
  mockCreateWorkspaceMigration,
  mockPutWorkspaceMigrationChunk,
  mockFinalizeWorkspaceMigration,
  mockGetWorkspaceMigration,
  mockAckWorkspaceMigrationClientDelete,
  mockRunResearchWorkspaceMigration,
  mockChatPaneProps,
  mockStatusBarProps,
  mockBgRequest,
  mockReadResearchWebCaptures
} = vi.hoisted(() => ({
  mockGetMediaDetails: vi.fn(),
  mockUpsertWorkspace: vi.fn(),
  mockGetWorkspaceSources: vi.fn(),
  mockAddWorkspaceSource: vi.fn(),
  mockUpdateWorkspaceSourceSelection: vi.fn(),
  mockGetWorkspaceSourcesStatus: vi.fn(),
  mockGetWorkspaceCapabilities: vi.fn(),
  mockCreateWorkspaceMigration: vi.fn(),
  mockPutWorkspaceMigrationChunk: vi.fn(),
  mockFinalizeWorkspaceMigration: vi.fn(),
  mockGetWorkspaceMigration: vi.fn(),
  mockAckWorkspaceMigrationClientDelete: vi.fn(),
  mockRunResearchWorkspaceMigration: vi.fn(),
  mockChatPaneProps: [] as any[],
  mockStatusBarProps: [] as any[],
  mockBgRequest: vi.fn(),
  mockReadResearchWebCaptures: vi.fn()
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
  restoreServerWorkspace: vi.fn(),
  workspaceSnapshots: {},
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
      defaultValueOrOptions?: string | { defaultValue?: string }
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

vi.mock("@/store/workspace", async original => ({
  ...await original<typeof import("@/store/workspace")>(),
  useWorkspaceStore: Object.assign(
    (selector: (state: typeof testState) => unknown) => selector(testState),
    { getState: () => testState }
  ),
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
  runResearchWorkspaceMigration: mockRunResearchWorkspaceMigration,
  RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX: "tldw:research-workspace:migration:tombstone"
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: unknown, options: { signal: AbortSignal }) => ({
    scopeKey: "workspace-alice",
    requestScope: { config: { serverUrl: "https://workspace.test", authMode: "multi-user" }, userId: "alice" },
    scopeSignal: options.signal,
    scopeInvalidatedSignal: options.signal,
    release: vi.fn()
  })
}))

vi.mock("@/utils/research-workspace-prefill", async original => ({
  ...await original<typeof import("@/utils/research-workspace-prefill")>(),
  readResearchWebCaptures: mockReadResearchWebCaptures
}))

vi.mock("@/utils/use-research-workspace-prefill", () => ({
  useResearchWorkspacePrefill: () => ({
    attached: 0,
    pending: 0,
    failed: 0,
    importing: false,
    error: null,
    retry: vi.fn(),
  }),
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
  WorkspaceStatusBar: (props: any) => {
    mockStatusBarProps.push(props)
    return <div data-testid="workspace-status-bar" />
  }
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const ensureLocalStorage = () => {
  if (window.localStorage && typeof window.localStorage.clear === "function") {
    window.localStorage.clear()
    return
  }
  const storage = new Map<string, string>()
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      clear: () => storage.clear(),
      getItem: (key: string) => storage.get(key) ?? null,
      key: (index: number) => Array.from(storage.keys())[index] ?? null,
      removeItem: (key: string) => {
        storage.delete(key)
      },
      setItem: (key: string, value: string) => {
        storage.set(key, String(value))
      },
      get length() {
        return storage.size
      }
    }
  })
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

const makeStatusSource = (overrides: Record<string, unknown> = {}) => ({
  id: "source-1",
  workspace_id: "workspace-1",
  media_id: 808,
  title: "Source",
  source_type: "pdf",
  selected: true,
  state: "queryable",
  status_reason: "source_queryable",
  readiness: {},
  progress_percent: 100,
  progress_message: "Ready.",
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

const makeCapabilitiesPayload = () => ({
  workspace_id: "workspace-1",
  workspace_kind: "research_workspace",
  access_level: "owner",
  source_summary: sourceStatusSummary,
  workspace_services: {},
  allowed_actions: {}
})

const setVisibility = (state: DocumentVisibilityState) => {
  vi.spyOn(document, "visibilityState", "get").mockReturnValue(state)
  document.dispatchEvent(new Event("visibilitychange"))
}

const settle = async () => {
  await act(async () => {
    await Promise.resolve()
    await vi.advanceTimersByTimeAsync(1)
  })
}

const waitForProjection = async () => {
  for (let i = 0; i < 20 && mockGetWorkspaceSourcesStatus.mock.calls.length === 0; i += 1) {
    await settle()
  }
  expect(mockGetWorkspaceSourcesStatus.mock.calls.length).toBeGreaterThan(0)
}

const originalMatchMedia = window.matchMedia

describe("ResearchWorkspace source status polling (WP-42)", () => {
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
    vi.useFakeTimers({
      toFake: ["setInterval", "clearInterval", "setTimeout", "clearTimeout"]
    })
    vi.clearAllMocks()
    mockReadResearchWebCaptures.mockReset().mockResolvedValue([])
    ensureLocalStorage()
    delete (window as Window & {
      __tldwResearchWorkspaceFreshInitialization?: Set<string>
    }).__tldwResearchWorkspaceFreshInitialization
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
    testState.storeHydrated = true
    testState.workspaceId = "workspace-1"
    testState.workspaceName = "New Research"
    testState.selectedSourceIds = []
    testState.isGeneratingOutput = false
    testState.generatingOutputType = null
    testState.sources = []
    testState.setSourceStatusByMediaId = vi.fn()
    testState.workspaceChatSessions = {}
    mockChatPaneProps.length = 0
    mockStatusBarProps.length = 0
    mockGetMediaDetails.mockReset().mockResolvedValue({})
    mockUpsertWorkspace.mockReset().mockResolvedValue({
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
    mockGetWorkspaceSources.mockReset().mockResolvedValue({ items: [] })
    mockGetWorkspaceSourcesStatus.mockReset().mockResolvedValue(makeStatusPayload())
    mockGetWorkspaceCapabilities.mockReset().mockResolvedValue(makeCapabilitiesPayload())
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("issues no periodic status requests when no source is processing", async () => {
    testState.sources = [
      {
        id: "source-1",
        mediaId: 808,
        title: "Ready source",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]

    render(<ResearchWorkspace />)
    await waitForProjection()
    const projectionCalls = mockGetWorkspaceSourcesStatus.mock.calls.length
    expect(projectionCalls).toBe(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(projectionCalls)
    expect(mockGetMediaDetails).not.toHaveBeenCalled()
  })

  it("keeps the 5 s projection and readiness cadence while a source is processing", async () => {
    testState.sources = [
      {
        id: "source-1",
        mediaId: 808,
        title: "Processing source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]

    render(<ResearchWorkspace />)
    await waitForProjection()
    expect(mockGetMediaDetails).toHaveBeenCalledWith(
      808,
      expect.objectContaining({ include_content: true })
    )
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(4)
    expect(mockGetMediaDetails.mock.calls.length).toBeGreaterThanOrEqual(4)
  })

  it("issues no polls while the document is hidden and catches up once when visible", async () => {
    testState.sources = [
      {
        id: "source-1",
        mediaId: 808,
        title: "Processing source",
        type: "pdf",
        status: "processing",
        addedAt: new Date("2026-02-18T12:00:00.000Z")
      }
    ]

    render(<ResearchWorkspace />)
    await waitForProjection()
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(1)

    setVisibility("hidden")
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(1)

    setVisibility("visible")
    await settle()
    expect(mockGetWorkspaceSourcesStatus).toHaveBeenCalledTimes(2)
  })
})
