import type { ReactNode } from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { getDesignSystemState } from "@/design-system"
import {
  clearWorkspaceUndoActionsForTests,
  getWorkspaceUndoPendingCount
} from "../undo-manager"
import { WorkspaceHeader } from "../WorkspaceHeader"
import { ConnectionPhase, type ConnectionState } from "@/types/connection"
import {
  FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS,
  FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY
} from "@/utils/feature-rollout"

const mockNavigate = vi.fn()
const mockSwitchWorkspace = vi.fn()
const mockCreateNewWorkspace = vi.fn()
const mockExportWorkspaceBundle = vi.fn()
const mockImportWorkspaceBundle = vi.fn()
const mockDuplicateWorkspace = vi.fn()
const mockArchiveWorkspace = vi.fn()
const mockRestoreArchivedWorkspace = vi.fn()
const mockDeleteWorkspace = vi.fn()
const mockCreateWorkspaceCollection = vi.fn()
const mockDeleteWorkspaceCollection = vi.fn()
const mockAssignWorkspaceToCollection = vi.fn()
const mockSaveCurrentWorkspace = vi.fn()
const mockSetWorkspaceName = vi.fn()
const mockSetWorkspaceBanner = vi.fn()
const mockClearWorkspaceBannerImage = vi.fn()
const mockResetWorkspaceBanner = vi.fn()
const mockSetCurrentNote = vi.fn()
const mockCaptureUndoSnapshot = vi.fn()
const mockRestoreUndoSnapshot = vi.fn()
const mockCreateWorkspaceExportZipBlob = vi.fn()
const mockCreateWorkspaceExportZipFilename = vi.fn()
const mockParseWorkspaceImportFile = vi.fn()
const mockNormalizeWorkspaceBannerImage = vi.fn()
const mockTrackResearchWorkspaceTelemetry = vi.fn()
const mockGetResearchWorkspaceTelemetryState = vi.fn()
const mockResetResearchWorkspaceTelemetryState = vi.fn()
const mockAddArtifact = vi.fn()
const workspaceContextMocks = vi.hoisted(() => ({
  useActiveWorkspaceContext: vi.fn()
}))
const translationMock = vi.hoisted(() => ({
  keys: [] as string[]
}))
const {
  mockStartTutorial,
  mockMessageApi,
  mockListPersonaProfiles,
  mockGetWorkspace,
  mockPatchWorkspace
} = vi.hoisted(() => ({
  mockStartTutorial: vi.fn(),
  mockMessageApi: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  mockListPersonaProfiles: vi.fn(),
  mockGetWorkspace: vi.fn(),
  mockPatchWorkspace: vi.fn()
}))

const now = new Date("2026-02-18T12:00:00.000Z")

const mockStoreState = {
  workspaceName: "Alpha Research",
  workspaceId: "workspace-alpha",
  workspaceTag: "workspace:alpha-research",
  workspaceBanner: {
    title: "Alpha Banner",
    subtitle: "Alpha subtitle",
    image: null as null | {
      dataUrl: string
      mimeType: "image/jpeg" | "image/png" | "image/webp"
      width: number
      height: number
      bytes: number
      updatedAt: Date
    }
  },
  sources: [
    {
      id: "source-1",
      mediaId: 101,
      title: "Alpha Whitepaper",
      type: "pdf",
      addedAt: new Date("2026-02-17T11:00:00.000Z"),
      url: "https://example.com/alpha-whitepaper"
    }
  ],
  generatedArtifacts: [] as Array<any>,
  assistantDefaults: null,
  effectiveAssistantDefault: null,
  setWorkspaceName: mockSetWorkspaceName,
  setWorkspaceBanner: mockSetWorkspaceBanner,
  clearWorkspaceBannerImage: mockClearWorkspaceBannerImage,
  resetWorkspaceBanner: mockResetWorkspaceBanner,
  setCurrentNote: mockSetCurrentNote,
  savedWorkspaces: [
    {
      id: "workspace-alpha",
      name: "Alpha Research",
      tag: "workspace:alpha-research",
      collectionId: "collection-topic-a",
      createdAt: new Date("2026-02-10T10:00:00.000Z"),
      lastAccessedAt: now,
      sourceCount: 3
    },
    {
      id: "workspace-beta",
      name: "Beta Deep Dive",
      tag: "workspace:beta-deep-dive",
      collectionId: null,
      createdAt: new Date("2026-02-09T10:00:00.000Z"),
      lastAccessedAt: new Date("2026-02-18T11:00:00.000Z"),
      sourceCount: 5
    },
    {
      id: "workspace-gamma",
      name: "Gamma Notes",
      tag: "workspace:gamma-notes",
      collectionId: null,
      createdAt: new Date("2026-02-08T10:00:00.000Z"),
      lastAccessedAt: new Date("2026-02-18T09:00:00.000Z"),
      sourceCount: 2
    }
  ],
  archivedWorkspaces: [],
  workspaceCollections: [
    {
      id: "collection-topic-a",
      name: "Topic A",
      description: null,
      createdAt: new Date("2026-02-01T10:00:00.000Z"),
      updatedAt: new Date("2026-02-01T10:00:00.000Z")
    }
  ],
  createNewWorkspace: mockCreateNewWorkspace,
  exportWorkspaceBundle: mockExportWorkspaceBundle,
  importWorkspaceBundle: mockImportWorkspaceBundle,
  createWorkspaceCollection: mockCreateWorkspaceCollection,
  deleteWorkspaceCollection: mockDeleteWorkspaceCollection,
  assignWorkspaceToCollection: mockAssignWorkspaceToCollection,
  switchWorkspace: mockSwitchWorkspace,
  duplicateWorkspace: mockDuplicateWorkspace,
  archiveWorkspace: mockArchiveWorkspace,
  restoreArchivedWorkspace: mockRestoreArchivedWorkspace,
  deleteWorkspace: mockDeleteWorkspace,
  saveCurrentWorkspace: mockSaveCurrentWorkspace,
  addArtifact: mockAddArtifact,
  captureUndoSnapshot: mockCaptureUndoSnapshot,
  restoreUndoSnapshot: mockRestoreUndoSnapshot
}

const mockConnectionStoreState: { state: ConnectionState } = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    serverUrl: "http://127.0.0.1:8000",
    lastCheckedAt: Date.now(),
    lastError: null as string | null,
    lastStatusCode: 200 as number | null,
    isConnected: true,
    isChecking: false,
    consecutiveFailures: 0,
    offlineBypass: false,
    knowledgeStatus: "ready" as const,
    knowledgeLastCheckedAt: Date.now(),
    knowledgeError: null as string | null,
    mode: "normal" as const,
    configStep: "none" as const,
    errorKind: "none" as const,
    hasCompletedFirstRun: true,
    userPersona: null,
    lastConfigUpdatedAt: Date.now(),
    checksSinceConfigChange: 0
  }
}

const connectionConfigState: {
  loading: boolean
  config: {
    serverUrl: string
    authMode: "single-user"
    apiKey: string
    accessToken: string
  } | null
} = {
  loading: false,
  config: null
}

const registryStateOverrides = {
  missingDegraded: false,
  degradedLabel: "Degraded"
}

const fetchMockState = {
  fetch: vi.fn()
}
const ACP_PROJECTS_FOR_ALPHA_URL =
  "http://127.0.0.1:8000/api/v1/agent-orchestration/projects?canonical_workspace_id=workspace-alpha&canonical_workspace_source=research_workspace"
const ACP_SESSIONS_FOR_ALPHA_URL =
  "http://127.0.0.1:8000/api/v1/acp/sessions?workspace_id=workspace-alpha&limit=6"

const sourceSummaryFixture = {
  total: 0,
  selected: 0,
  queryable: 0,
  partially_queryable: 0,
  processing: 0,
  failed: 0,
  missing: 0
}

const makeActiveWorkspaceContext = (
  overrides: Record<string, unknown> = {}
) => ({
  state: "ready",
  workspaceId: "workspace-alpha",
  workspace: {
    id: "workspace-alpha",
    name: "Canonical Alpha",
    label: "Canonical Alpha",
    profile: "research",
    archived: false,
    deleted: false,
    studyMaterialsPolicy: "workspace",
    statusLabel: "Active",
    version: 3,
    lastModified: "2026-02-18T12:00:00.000Z"
  },
  attentionState: "ready",
  resolution: { status: "complete", partial_errors: [] },
  projectRoot: null,
  sourceSummary: sourceSummaryFixture,
  capabilities: null,
  allowedActions: {},
  partialErrors: [],
  recovery: {
    reasonCode: "allowed",
    severity: "info",
    message: "Workspace action is available.",
    nextStepLabel: null,
    nextStepHref: null
  },
  ...overrides
})

const createWorkspaceApiResponse = (
  overrides: Record<string, unknown> = {}
) => ({
  id: "workspace-alpha",
  name: "Alpha Research",
  archived: false,
  studyMaterialsPolicy: "workspace",
  workspaceProfile: "research",
  deleted: false,
  bannerTitle: "Alpha Banner",
  bannerSubtitle: "Alpha subtitle",
  bannerColor: null,
  audioProvider: null,
  audioModel: null,
  audioVoice: null,
  audioSpeed: null,
  createdAt: "2026-02-10T10:00:00.000Z",
  lastModified: "2026-02-18T12:00:00.000Z",
  version: 7,
  assistantDefaults: null,
  effectiveAssistantDefault: {
    status: "none",
    source: "none",
    assistantKind: null,
    assistantId: null,
    label: null,
    personaMemoryMode: null,
    degradedReason: null
  },
  ...overrides
})

const makeActiveWorkspaceHookResult = (
  contextOverrides: Record<string, unknown> = {}
) => ({
  context: makeActiveWorkspaceContext(contextOverrides),
  loading: false,
  error: null,
  refresh: vi.fn()
})

const interpolateTranslation = (
  value: string,
  interpolationValues?: Record<string, unknown>
) =>
  Object.entries(interpolationValues ?? {}).reduce(
    (text, [name, replacement]) =>
      text.split(`{{${name}}}`).join(String(replacement)),
    value
  )

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          },
      interpolationValues?: Record<string, unknown>
    ) => {
      translationMock.keys.push(key)
      if (typeof defaultValueOrOptions === "string") {
        return interpolateTranslation(defaultValueOrOptions, interpolationValues)
      }
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      ...actual.message,
      useMessage: () => [mockMessageApi, null] as const
    }
  }
})

vi.mock("@/store/workspace", () => {
  const useWorkspaceStore = ((
    selector: (state: typeof mockStoreState) => unknown
  ) => selector(mockStoreState)) as ((
    selector: (state: typeof mockStoreState) => unknown
  ) => unknown) & {
    getState: () => typeof mockStoreState
    setState: (
      update:
        | Partial<typeof mockStoreState>
        | ((state: typeof mockStoreState) => Partial<typeof mockStoreState>)
    ) => void
  }

  useWorkspaceStore.getState = () => mockStoreState
  useWorkspaceStore.setState = (update) => {
    const next = typeof update === "function" ? update(mockStoreState) : update
    Object.assign(mockStoreState, next)
  }

  return { useWorkspaceStore }
})

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: (
    selector: (state: { startTutorial: typeof mockStartTutorial }) => unknown
  ) => selector({ startTutorial: mockStartTutorial })
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (
    selector: (state: typeof mockConnectionStoreState) => unknown
  ) => selector(mockConnectionStoreState)
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: connectionConfigState.config,
    loading: connectionConfigState.loading
  })
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        if (key === "degraded" && registryStateOverrides.missingDegraded) {
          return undefined as unknown as ReturnType<
            typeof actual.getDesignSystemState
          >
        }
        const state = actual.getDesignSystemState(key)
        return {
          ...state,
          label:
            key === "degraded"
              ? registryStateOverrides.degradedLabel
              : state.label
        }
      }
    )
  }
})

vi.mock("@/store/workspace-bundle", async () => {
  const actual = await vi.importActual<typeof import("@/store/workspace-bundle")>(
    "@/store/workspace-bundle"
  )
  return {
    ...actual,
    createWorkspaceExportZipBlob: (...args: unknown[]) =>
      mockCreateWorkspaceExportZipBlob(...args),
    createWorkspaceExportZipFilename: (...args: unknown[]) =>
      mockCreateWorkspaceExportZipFilename(...args),
    parseWorkspaceImportFile: (...args: unknown[]) =>
      mockParseWorkspaceImportFile(...args)
  }
})

vi.mock("../workspace-banner-image", () => ({
  normalizeWorkspaceBannerImage: (...args: unknown[]) =>
    mockNormalizeWorkspaceBannerImage(...args),
  WorkspaceBannerImageNormalizationError: class WorkspaceBannerImageNormalizationError extends Error {
    code: string
    constructor(code: string, message: string) {
      super(message)
      this.code = code
    }
  }
}))

vi.mock("@/services/workspace-context", () => workspaceContextMocks)

vi.mock("../WorkspaceSandboxDiagnosticsPanel", () => ({
  WorkspaceSandboxDiagnosticsPanel: ({ workspaceId }: { workspaceId: string }) => (
    <div data-testid="workspace-sandbox-diagnostics-panel">
      Sandbox diagnostics for {workspaceId}
    </div>
  )
}))

vi.mock("@/utils/research-workspace-telemetry", async () => {
  const actual =
    await vi.importActual<typeof import("@/utils/research-workspace-telemetry")>(
      "@/utils/research-workspace-telemetry"
    )
  return {
    ...actual,
    trackResearchWorkspaceTelemetry: (...args: unknown[]) =>
      mockTrackResearchWorkspaceTelemetry(...args),
    getResearchWorkspaceTelemetryState: (...args: unknown[]) =>
      mockGetResearchWorkspaceTelemetryState(...args),
    resetResearchWorkspaceTelemetryState: (...args: unknown[]) =>
      mockResetResearchWorkspaceTelemetryState(...args)
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listPersonaProfiles: (...args: unknown[]) => mockListPersonaProfiles(...args),
    getWorkspace: (...args: unknown[]) => mockGetWorkspace(...args),
    patchWorkspace: (...args: unknown[]) => mockPatchWorkspace(...args)
  }
}))

if (!(globalThis as unknown as { ResizeObserver?: unknown }).ResizeObserver) {
  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const ensureLocalStorage = () => {
  if (window.localStorage && typeof window.localStorage.clear === "function") {
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

describe("WorkspaceHeader workspace browser modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAddArtifact.mockReset()
    translationMock.keys = []
    workspaceContextMocks.useActiveWorkspaceContext.mockReturnValue(
      makeActiveWorkspaceHookResult()
    )
    ensureLocalStorage()
    window.localStorage.clear()
    clearWorkspaceUndoActionsForTests()
    registryStateOverrides.missingDegraded = false
    registryStateOverrides.degradedLabel = "Degraded"
    mockStoreState.workspaceId = "workspace-alpha"
    mockStoreState.workspaceName = "Alpha Research"
    mockStoreState.workspaceTag = "workspace:alpha-research"
    mockStoreState.generatedArtifacts = []
    mockStoreState.assistantDefaults = null
    mockStoreState.effectiveAssistantDefault = null
    connectionConfigState.loading = false
    connectionConfigState.config = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key",
      accessToken: ""
    }
    mockStoreState.savedWorkspaces = [
      {
        id: "workspace-alpha",
        name: "Alpha Research",
        tag: "workspace:alpha-research",
        collectionId: "collection-topic-a",
        createdAt: new Date("2026-02-10T10:00:00.000Z"),
        lastAccessedAt: now,
        sourceCount: 3
      },
      {
        id: "workspace-beta",
        name: "Beta Deep Dive",
        tag: "workspace:beta-deep-dive",
        collectionId: null,
        createdAt: new Date("2026-02-09T10:00:00.000Z"),
        lastAccessedAt: new Date("2026-02-18T11:00:00.000Z"),
        sourceCount: 5
      },
      {
        id: "workspace-gamma",
        name: "Gamma Notes",
        tag: "workspace:gamma-notes",
        collectionId: null,
        createdAt: new Date("2026-02-08T10:00:00.000Z"),
        lastAccessedAt: new Date("2026-02-18T09:00:00.000Z"),
        sourceCount: 2
      }
    ]
    mockStoreState.workspaceCollections = [
      {
        id: "collection-topic-a",
        name: "Topic A",
        description: null,
        createdAt: new Date("2026-02-01T10:00:00.000Z"),
        updatedAt: new Date("2026-02-01T10:00:00.000Z")
      }
    ]
    mockCaptureUndoSnapshot.mockReturnValue({
      workspaceId: "workspace-alpha",
      workspaceName: "Alpha Research"
    })
    mockConnectionStoreState.state = {
      ...mockConnectionStoreState.state,
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      isChecking: false,
      errorKind: "none",
      knowledgeStatus: "ready",
      lastError: null,
      knowledgeError: null
    }
    mockExportWorkspaceBundle.mockReturnValue({
      format: "tldw.research-workspace.bundle",
      schemaVersion: 1,
      exportedAt: "2026-02-18T12:00:00.000Z",
      workspace: {
        name: "Alpha Research",
        tag: "workspace:alpha-research",
        createdAt: "2026-02-10T10:00:00.000Z",
        snapshot: {
          workspaceName: "Alpha Research",
          workspaceTag: "workspace:alpha-research",
          workspaceCreatedAt: "2026-02-10T10:00:00.000Z",
          sources: [],
          selectedSourceIds: [],
          generatedArtifacts: [],
          notes: "",
          currentNote: {
            title: "",
            content: "",
            keywords: [],
            isDirty: false
          },
          workspaceBanner: {
            title: "",
            subtitle: "",
            image: null
          },
          leftPaneCollapsed: false,
          rightPaneCollapsed: false,
          audioSettings: {
            provider: "tldw",
            model: "kokoro",
            voice: "af_heart",
            speed: 1,
            format: "mp3"
          }
        }
      }
    })
    mockImportWorkspaceBundle.mockReturnValue("workspace-imported")
    mockCreateWorkspaceExportZipBlob.mockResolvedValue(
      new Blob(["zip-bytes"], { type: "application/zip" })
    )
    mockCreateWorkspaceExportZipFilename.mockReturnValue("alpha.workspace.zip")
    mockParseWorkspaceImportFile.mockResolvedValue({
      format: "tldw.research-workspace.bundle",
      schemaVersion: 1,
      exportedAt: "2026-02-18T12:00:00.000Z",
      workspace: {
        name: "Imported",
        tag: "workspace:imported",
        createdAt: "2026-02-18T10:00:00.000Z",
        snapshot: {
          workspaceName: "Imported",
          workspaceTag: "workspace:imported",
          workspaceCreatedAt: "2026-02-18T10:00:00.000Z",
          sources: [],
          selectedSourceIds: [],
          generatedArtifacts: [],
          notes: "",
          currentNote: {
            title: "",
            content: "",
            keywords: [],
            isDirty: false
          },
          workspaceBanner: {
            title: "",
            subtitle: "",
            image: null
          },
          leftPaneCollapsed: false,
          rightPaneCollapsed: false,
          audioSettings: {
            provider: "tldw",
            model: "kokoro",
            voice: "af_heart",
            speed: 1,
            format: "mp3"
          }
        }
      }
    })
    mockGetResearchWorkspaceTelemetryState.mockResolvedValue({
      version: 1,
      counters: {
        status_viewed: 3,
        citation_provenance_opened: 1,
        token_cost_rendered: 2,
        diagnostics_toggled: 1,
        quota_warning_seen: 0,
        conflict_modal_opened: 1,
        undo_triggered: 2,
        operation_cancelled: 1,
        artifact_rehydrated_failed: 0,
        source_status_polled: 5,
        source_status_ready: 4,
        connectivity_state_changed: 2,
        confusion_retry_burst: 0,
        confusion_refresh_loop: 0,
        confusion_duplicate_submission: 0
      },
      last_event_at: Date.parse("2026-02-20T01:23:45.000Z"),
      recent_events: [
        {
          type: "status_viewed",
          at: Date.parse("2026-02-20T01:22:00.000Z"),
          details: { workspace_id: "workspace-alpha" }
        },
        {
          type: "operation_cancelled",
          at: Date.parse("2026-02-20T01:23:45.000Z"),
          details: { scope: "chat" }
        },
        {
          type: "confusion_retry_burst",
          at: Date.parse("2026-02-20T01:24:12.000Z"),
          details: { retry_count: 3, window_ms: 30000 }
        }
      ]
    })
    mockResetResearchWorkspaceTelemetryState.mockResolvedValue(undefined)
    mockNormalizeWorkspaceBannerImage.mockResolvedValue({
      dataUrl: "data:image/webp;base64,banner",
      mimeType: "image/webp",
      width: 1200,
      height: 400,
      bytes: 16000,
      updatedAt: new Date("2026-02-25T10:00:00.000Z")
    })
    mockListPersonaProfiles.mockResolvedValue([
      {
        id: "persona-lit-reviewer",
        name: "Literature Reviewer",
        character_card_id: null,
        origin_character_id: null,
        buddy_summary: null,
        metadata: null
      },
      {
        id: "persona-methods",
        name: "Methods Auditor",
        character_card_id: null,
        origin_character_id: null,
        buddy_summary: null,
        metadata: null
      }
    ])
    mockGetWorkspace.mockResolvedValue(createWorkspaceApiResponse())
    mockPatchWorkspace.mockImplementation(
      async (_workspaceId: string, payload: Record<string, unknown>) =>
        createWorkspaceApiResponse({
          version: 8,
          assistantDefaults:
            "assistantDefaults" in payload ? payload.assistantDefaults : null
        })
    )
    connectionConfigState.config = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key",
      accessToken: ""
    }
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      throw new Error(`unexpected fetch: ${String(input)}`)
    })
    vi.stubGlobal("fetch", fetchMockState.fetch)
  })

  it("keeps mobile header actions within a shrinkable wrapping row", () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        hideToggles
      />
    )

    expect(screen.getByTestId("workspace-header")).toHaveClass(
      "min-w-0",
      "flex-wrap"
    )
    expect(screen.getByRole("heading", { name: "Alpha Research" })).toHaveClass(
      "truncate"
    )
    expect(screen.getByTestId("workspace-header-actions")).toHaveClass(
      "min-w-0",
      "flex-wrap",
      "overflow-hidden"
    )
    expect(screen.getByTestId("workspace-workspaces-button")).toHaveClass(
      "min-w-0",
      "max-w-full"
    )
  })

  it("shows feedback when replaying the workspace tour from the header", () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByTestId("workspace-help-tour-button"))

    expect(mockStartTutorial).toHaveBeenCalledWith("research-workspace-basics")
    expect(mockMessageApi.info).toHaveBeenCalledWith(
      "Tour started. Follow the highlighted steps."
    )
  })

  it("exposes workspace search as a first-class header action without credentials", () => {
    const onOpenSearch = vi.fn()
    connectionConfigState.config = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "",
      accessToken: ""
    }

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        onOpenSearch={onOpenSearch}
        hideToggles
      />
    )

    const searchButton = screen.getByRole("button", {
      name: /search workspace/i
    })

    expect(searchButton).toBeVisible()
    expect(searchButton).toHaveTextContent(/K/)

    fireEvent.click(searchButton)

    expect(onOpenSearch).toHaveBeenCalledTimes(1)
  })

  it("renders server-authoritative workspace context", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    expect(
      await screen.findByText("Server Workspace")
    ).toBeInTheDocument()
    expect(screen.getByText("Canonical Alpha")).toBeInTheDocument()
    expect(screen.getByText("Server context ready")).toBeInTheDocument()
    expect(translationMock.keys).toContain(
      "playground:workspace.serverContextReady"
    )
    expect(translationMock.keys).not.toContain(
      "playground:workspace.Servercontextready"
    )
    expect(workspaceContextMocks.useActiveWorkspaceContext).toHaveBeenCalledWith(
      expect.objectContaining({ workspaceId: "workspace-alpha" })
    )
  })

  it("passes the server context refresh version into the workspace context hook", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        serverContextRefreshVersion={7}
      />
    )

    expect(
      await screen.findByText("Server Workspace")
    ).toBeInTheDocument()
    expect(workspaceContextMocks.useActiveWorkspaceContext).toHaveBeenCalledWith(
      expect.objectContaining({
        workspaceId: "workspace-alpha",
        refreshKey: 7
      })
    )
  })

  it("renders shared recovery copy when server workspace context fails", async () => {
    workspaceContextMocks.useActiveWorkspaceContext.mockReturnValue(
      makeActiveWorkspaceHookResult({
        state: "error",
        workspaceId: null,
        workspace: null,
        recovery: {
          reasonCode: "workspace_context_error",
          severity: "error",
          message: "Server Workspace context is unavailable right now.",
          nextStepLabel: "Open Workspaces",
          nextStepHref: "#/workspaces"
        }
      })
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    expect(
      await screen.findByText("Server Workspace context is unavailable right now.")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open Workspaces" })
    ).toHaveAttribute("href", "#/workspaces")
  })

  it("labels archived server workspace context explicitly", async () => {
    workspaceContextMocks.useActiveWorkspaceContext.mockReturnValue(
      makeActiveWorkspaceHookResult({
        state: "ready",
        attentionState: "archived",
        workspace: {
          ...(makeActiveWorkspaceContext().workspace as Record<string, unknown>),
          archived: true,
          statusLabel: "Archived"
        },
        recovery: {
          reasonCode: "workspace_archived",
          severity: "warning",
          message: "This server Workspace is archived. Restore it before making changes.",
          nextStepLabel: "Open Workspaces",
          nextStepHref: "#/workspaces"
        }
      })
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    expect(await screen.findByText("Server context archived")).toBeInTheDocument()
    expect(screen.queryByText("Server context ready")).not.toBeInTheDocument()
  })

  it("does not filter the workspace browser list by active server context", async () => {
    workspaceContextMocks.useActiveWorkspaceContext.mockReturnValue(
      makeActiveWorkspaceHookResult({
        workspaceId: "workspace-alpha",
        workspace: {
          ...(makeActiveWorkspaceContext().workspace as Record<string, unknown>),
          id: "workspace-alpha",
          label: "Canonical Alpha"
        }
      })
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("View all workspaces"))

    const modal = await screen.findByRole("dialog", {
      name: "All Workspaces"
    })
    expect(within(modal).getByText("Alpha Research")).toBeInTheDocument()
    expect(within(modal).getByText("Beta Deep Dive")).toBeInTheDocument()
    expect(within(modal).getByText("Gamma Notes")).toBeInTheDocument()
  })

  it("opens view-all modal and filters workspaces by search query", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("View all workspaces"))

    const modal = await screen.findByRole("dialog", {
      name: "All Workspaces"
    })
    expect(modal).toBeInTheDocument()
    expect(within(modal).getByText("Beta Deep Dive")).toBeInTheDocument()
    expect(within(modal).getByText("Gamma Notes")).toBeInTheDocument()

    const searchInput = within(modal).getByLabelText("Search workspaces")
    fireEvent.change(searchInput, { target: { value: "gamma" } })

    await waitFor(() => {
      expect(within(modal).queryByText("Beta Deep Dive")).not.toBeInTheDocument()
      expect(within(modal).getByText("Gamma Notes")).toBeInTheDocument()
    })
  })

  it("opens the canonical Workspaces manager from the Workspaces dropdown", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("Manage server Workspaces"))

    expect(mockSaveCurrentWorkspace).toHaveBeenCalledTimes(1)
    expect(mockNavigate).toHaveBeenCalledWith("/workspaces")
  })

  it("switches workspace when selecting from view-all modal", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("View all workspaces"))

    const modal = await screen.findByRole("dialog", {
      name: "All Workspaces"
    })
    const targetWorkspaceRow = await within(modal).findByRole("button", {
      name: /Beta Deep Dive/
    })
    fireEvent.click(targetWorkspaceRow)

    expect(mockSwitchWorkspace).toHaveBeenCalledWith("workspace-beta")
  })

  it("renders collection groups and assigns workspaces from the browser modal", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("View all workspaces"))

    const modal = await screen.findByRole("dialog", {
      name: "All Workspaces"
    })
    expect(
      within(modal).getByLabelText("Collection group Topic A")
    ).toBeInTheDocument()
    expect(
      within(modal).getByLabelText("Collection group Unassigned")
    ).toBeInTheDocument()

    fireEvent.change(
      within(modal).getByLabelText("Collection for Beta Deep Dive"),
      { target: { value: "collection-topic-a" } }
    )

    expect(mockAssignWorkspaceToCollection).toHaveBeenCalledWith(
      "workspace-beta",
      "collection-topic-a"
    )
  })

  it("creates and deletes collections from the browser modal", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    fireEvent.click(await screen.findByText("View all workspaces"))

    const modal = await screen.findByRole("dialog", {
      name: "All Workspaces"
    })

    fireEvent.change(
      within(modal).getByLabelText("New collection name"),
      { target: { value: "Topic B" } }
    )
    fireEvent.click(within(modal).getByRole("button", { name: "Add collection" }))
    fireEvent.click(
      within(modal).getByRole("button", { name: "Delete collection Topic A" })
    )

    expect(mockCreateWorkspaceCollection).toHaveBeenCalledWith("Topic B", null)
    expect(mockDeleteWorkspaceCollection).toHaveBeenCalledWith(
      "collection-topic-a"
    )
  })

  it("exports workspace bundle from the settings menu", async () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:workspace-export-zip")
    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined)
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Export Workspace"))

    await waitFor(() => {
      expect(mockExportWorkspaceBundle).toHaveBeenCalledWith("workspace-alpha")
      expect(mockCreateWorkspaceExportZipBlob).toHaveBeenCalledTimes(1)
      expect(mockCreateWorkspaceExportZipFilename).toHaveBeenCalledTimes(1)
      expect(createObjectUrlSpy).toHaveBeenCalled()
    })
    expect(anchorClickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:workspace-export-zip")
    expect(mockMessageApi.success).toHaveBeenCalledWith(
      "Workspace exported: alpha.workspace.zip"
    )
  })

  it("opens the canonical Workspaces manager from the settings menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Manage in Workspaces"))

    expect(mockSaveCurrentWorkspace).toHaveBeenCalledTimes(1)
    expect(mockNavigate).toHaveBeenCalledWith("/workspaces")
  })

  it("opens default assistant settings and saves a read-only Persona default", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))

    const modal = await screen.findByTestId("workspace-default-assistant-modal")
    expect(within(modal).getByText("No default assistant")).toBeInTheDocument()
    expect(mockGetWorkspace).toHaveBeenCalledWith("workspace-alpha")
    expect(mockListPersonaProfiles).toHaveBeenCalledTimes(1)

    fireEvent.change(
      within(modal).getByTestId("workspace-default-assistant-select"),
      { target: { value: "persona-lit-reviewer" } }
    )
    fireEvent.click(screen.getByRole("button", { name: "Save default" }))

    await waitFor(() => {
      expect(mockPatchWorkspace).toHaveBeenCalledWith(
        "workspace-alpha",
        expect.objectContaining({
          version: 7,
          assistantDefaults: {
            assistantKind: "persona",
            assistantId: "persona-lit-reviewer",
            personaMemoryMode: "read_only",
            voice: null,
            style: null,
            toolPolicyProfileId: null
          }
        })
      )
    })
    await waitFor(() => {
      expect(mockStoreState.assistantDefaults).toEqual(
        expect.objectContaining({
          assistantKind: "persona",
          assistantId: "persona-lit-reviewer",
          personaMemoryMode: "read_only"
        })
      )
      expect(mockSaveCurrentWorkspace).toHaveBeenCalled()
    })
  })

  it("saves default assistant settings against the workspace loaded into the modal", async () => {
    const header = (
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )
    const { rerender } = render(header)

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))
    const modal = await screen.findByTestId("workspace-default-assistant-modal")

    mockStoreState.workspaceId = "workspace-beta"
    mockStoreState.workspaceName = "Beta Deep Dive"
    mockStoreState.workspaceTag = "workspace:beta-deep-dive"
    rerender(header)

    fireEvent.change(
      within(modal).getByTestId("workspace-default-assistant-select"),
      { target: { value: "persona-lit-reviewer" } }
    )
    fireEvent.click(screen.getByRole("button", { name: "Save default" }))

    await waitFor(() => {
      expect(mockPatchWorkspace).toHaveBeenCalledWith(
        "workspace-alpha",
        expect.objectContaining({
          version: 7,
          assistantDefaults: expect.objectContaining({
            assistantId: "persona-lit-reviewer"
          })
        })
      )
    })
    expect(mockPatchWorkspace).not.toHaveBeenCalledWith(
      "workspace-beta",
      expect.anything()
    )
  })

  it("clears default assistant modal state before failed reloads can reuse stale values", async () => {
    mockGetWorkspace.mockResolvedValueOnce(
      createWorkspaceApiResponse({
        version: 11,
        assistantDefaults: {
          assistantKind: "persona",
          assistantId: "persona-lit-reviewer",
          personaMemoryMode: "read_only",
          voice: null,
          style: null,
          toolPolicyProfileId: null
        },
        effectiveAssistantDefault: {
          status: "available",
          source: "workspace",
          assistantKind: "persona",
          assistantId: "persona-lit-reviewer",
          label: "Literature Reviewer",
          personaMemoryMode: "read_only",
          degradedReason: null
        }
      })
    )
    mockGetWorkspace.mockRejectedValueOnce(new Error("temporary failure"))

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))

    const modal = await screen.findByTestId("workspace-default-assistant-modal")
    await waitFor(() => {
      expect(
        within(modal).getByTestId("workspace-default-assistant-select")
      ).toHaveValue("persona-lit-reviewer")
    })

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    const defaultAssistantItems = await screen.findAllByText("Default assistant")
    const reopenedDefaultAssistantItem =
      defaultAssistantItems.find((item) =>
        item.closest(".ant-dropdown-menu")
      ) ?? defaultAssistantItems[0]
    fireEvent.click(reopenedDefaultAssistantItem)

    const reopenedModal = await screen.findByTestId(
      "workspace-default-assistant-modal"
    )
    await waitFor(() => {
      expect(
        within(reopenedModal).getByText(
          "Could not load default assistant settings."
        )
      ).toBeInTheDocument()
    })
    expect(
      within(reopenedModal).getByTestId("workspace-default-assistant-select")
    ).toHaveValue("")
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()
    expect(mockPatchWorkspace).not.toHaveBeenCalled()
  })

  it("requires confirmation before saving a read-write Persona default", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))

    const modal = await screen.findByTestId("workspace-default-assistant-modal")
    await waitFor(() => {
      expect(
        within(modal).getByTestId("workspace-default-assistant-select")
      ).toBeEnabled()
    })
    fireEvent.change(
      within(modal).getByTestId("workspace-default-assistant-select"),
      { target: { value: "persona-methods" } }
    )
    fireEvent.change(
      within(modal).getByTestId("workspace-default-assistant-memory-mode"),
      { target: { value: "read_write" } }
    )

    expect(
      within(modal).getByTestId("workspace-default-assistant-read-write-confirm")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()

    fireEvent.click(
      within(modal).getByTestId("workspace-default-assistant-read-write-confirm")
    )
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save default" })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Save default" }))

    await waitFor(() => {
      expect(mockPatchWorkspace).toHaveBeenCalledWith(
        "workspace-alpha",
        expect.objectContaining({
          version: 7,
          assistantDefaults: expect.objectContaining({
            assistantKind: "persona",
            assistantId: "persona-methods",
            personaMemoryMode: "read_write"
          }),
          confirmReadWriteAssistantDefault: true
        })
      )
    })
  })

  it("clears the Workspace default assistant", async () => {
    mockGetWorkspace.mockResolvedValueOnce(
      createWorkspaceApiResponse({
        version: 11,
        assistantDefaults: {
          assistantKind: "persona",
          assistantId: "persona-lit-reviewer",
          personaMemoryMode: "read_only",
          voice: null,
          style: null,
          toolPolicyProfileId: null
        },
        effectiveAssistantDefault: {
          status: "available",
          source: "workspace",
          assistantKind: "persona",
          assistantId: "persona-lit-reviewer",
          label: "Literature Reviewer",
          personaMemoryMode: "read_only",
          degradedReason: null
        }
      })
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))

    await screen.findByTestId("workspace-default-assistant-modal")
    fireEvent.click(screen.getByRole("button", { name: "Clear default" }))

    await waitFor(() => {
      expect(mockPatchWorkspace).toHaveBeenCalledWith("workspace-alpha", {
        version: 11,
        assistantDefaults: null
      })
    })
    await waitFor(() => {
      expect(mockStoreState.assistantDefaults).toBeNull()
      expect(mockSaveCurrentWorkspace).toHaveBeenCalled()
    })
  })

  it("redacts unavailable default assistant labels", async () => {
    mockListPersonaProfiles.mockResolvedValueOnce([
      {
        id: "persona-hidden",
        name: "Hidden Persona",
        character_card_id: null,
        origin_character_id: null,
        buddy_summary: null,
        metadata: null
      }
    ])
    mockGetWorkspace.mockResolvedValueOnce(
      createWorkspaceApiResponse({
        assistantDefaults: {
          assistantKind: "persona",
          assistantId: "persona-hidden",
          personaMemoryMode: "read_only",
          voice: null,
          style: null,
          toolPolicyProfileId: null
        },
        effectiveAssistantDefault: {
          status: "unavailable",
          source: "workspace",
          assistantKind: "persona",
          assistantId: "persona-hidden",
          label: null,
          personaMemoryMode: "read_only",
          degradedReason: "permission_denied"
        }
      })
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Default assistant"))

    const modal = await screen.findByTestId("workspace-default-assistant-modal")
    expect(within(modal).getByText("Default unavailable")).toBeInTheDocument()
    expect(within(modal).getByText("Permission denied")).toBeInTheDocument()
    expect(translationMock.keys).toContain(
      "playground:workspace.defaultAssistantDegraded.permissionDenied"
    )
    expect(within(modal).queryByText("Hidden Persona")).not.toBeInTheDocument()

    fireEvent.change(
      within(modal).getByTestId("workspace-default-assistant-select"),
      { target: { value: "" } }
    )
    expect(within(modal).queryByText("Hidden Persona")).not.toBeInTheDocument()
  })

  it("raises split workspace intent from the settings menu", async () => {
    const onOpenSplitWorkspace = vi.fn()

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        onOpenSplitWorkspace={onOpenSplitWorkspace}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Split workspace"))

    expect(onOpenSplitWorkspace).toHaveBeenCalledTimes(1)
  })

  it("reports duplicate workspace identity and offers a path back to the original", async () => {
    mockDuplicateWorkspace.mockReturnValue("workspace-alpha-copy")

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Duplicate Current Workspace"))

    expect(mockDuplicateWorkspace).toHaveBeenCalledWith("workspace-alpha")
    expect(mockMessageApi.open).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "success"
      })
    )

    const openConfig = mockMessageApi.open.mock.calls.at(-1)?.[0] as {
      content: ReactNode
    }
    const duplicateToast = render(<>{openConfig.content}</>)
    expect(
      duplicateToast.getByText("Duplicated Alpha Research. You are editing the new copy.")
    ).toBeInTheDocument()
    fireEvent.click(
      duplicateToast.getByRole("button", { name: "Open original" })
    )

    expect(mockSwitchWorkspace).toHaveBeenCalledWith("workspace-alpha")
  })

  it("dismisses the workspaces menu when opening workspace settings", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    expect(await screen.findByText("View all workspaces")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    expect(await screen.findByText("Import Workspace")).toBeInTheDocument()

    await waitFor(() => {
      const viewAllWorkspaces = screen.queryByText("View all workspaces")
      if (viewAllWorkspaces) {
        expect(viewAllWorkspaces).not.toBeVisible()
      }
    })
  })

  it("hides the split workspace menu item when no split callback is provided", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))

    expect(screen.queryByText("Split workspace")).not.toBeInTheDocument()
  })

  it("falls back to JSON export when ZIP creation fails", async () => {
    mockCreateWorkspaceExportZipBlob.mockRejectedValueOnce(
      new Error("zip unavailable")
    )
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:workspace-export")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Export Workspace"))

    await waitFor(() => {
      expect(createObjectUrlSpy).toHaveBeenCalled()
    })

    const exportedBlob = createObjectUrlSpy.mock.calls[0]?.[0] as Blob
    expect(exportedBlob.type).toContain("application/json")
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:workspace-export")
  })

  it("exports workspace citations in BibTeX format", async () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:workspace-bibtex")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Export Citations (BibTeX)"))

    expect(createObjectUrlSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:workspace-bibtex")
    expect(mockMessageApi.success).toHaveBeenCalledWith(
      expect.stringMatching(
        /^Citations exported: alpha-research-citations-\d{8}\.bib$/
      )
    )
  })

  it("creates a workspace from a template and seeds starter note content", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Literature Review"))

    expect(mockCreateNewWorkspace).toHaveBeenCalledWith(
      "Literature Review Workspace"
    )
    expect(mockSetCurrentNote).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Literature Review Plan",
        keywords: expect.arrayContaining(["literature", "evidence"])
      })
    )
  })

  it.each([
    {
      label: "Literature Review",
      workspaceName: "Literature Review Workspace",
      noteTitle: "Literature Review Plan",
      keyword: "literature",
      prompt: "Compare the strongest and weakest evidence across selected sources.",
      studioPreset: "Literature matrix"
    },
    {
      label: "Interview Analysis",
      workspaceName: "Interview Analysis Workspace",
      noteTitle: "Interview Findings",
      keyword: "interviews",
      prompt: "Summarize recurring themes and unresolved follow-ups.",
      studioPreset: "Theme synthesis"
    },
    {
      label: "Product Brief",
      workspaceName: "Product Brief Workspace",
      noteTitle: "Product Brief Draft",
      keyword: "product",
      prompt: "Draft a decision-ready product brief from the selected sources.",
      studioPreset: "Executive brief"
    }
  ])(
    "applies a documented scaffold for $label templates",
    async ({ label, workspaceName, noteTitle, keyword, prompt, studioPreset }) => {
      render(
        <WorkspaceHeader
          leftPaneOpen={true}
          rightPaneOpen={true}
          onToggleLeftPane={vi.fn()}
          onToggleRightPane={vi.fn()}
        />
      )

      fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
      fireEvent.click(await screen.findByText(label))

      expect(mockCreateNewWorkspace).toHaveBeenCalledWith(workspaceName)
      expect(mockSetCurrentNote).toHaveBeenCalledWith(
        expect.objectContaining({
          title: noteTitle,
          keywords: expect.arrayContaining([keyword, "template"]),
          isDirty: true,
          content: expect.stringContaining("## Source checklist")
        })
      )

      const scaffoldNote = mockSetCurrentNote.mock.calls.at(-1)?.[0] as {
        content: string
      }
      expect(scaffoldNote.content).toContain("## Suggested prompts")
      expect(scaffoldNote.content).toContain(prompt)
      expect(scaffoldNote.content).toContain("## Studio recommendations")
      expect(scaffoldNote.content).toContain(studioPreset)
      expect(scaffoldNote.content).toContain("## Next steps")

      expect(mockMessageApi.open).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "success"
        })
      )

      const openConfig = mockMessageApi.open.mock.calls.at(-1)?.[0] as {
        content: ReactNode
      }
      const templateToast = render(<>{openConfig.content}</>)
      expect(
        templateToast.getByText(
          `${label} template applied. Added outline, source checklist, suggested prompts, and Studio recommendations.`
        )
      ).toBeInTheDocument()
      fireEvent.click(
        templateToast.getByRole("button", { name: "Start over" })
      )

      expect(mockCreateNewWorkspace).toHaveBeenLastCalledWith()
    }
  )

  it("replays a different template by replacing the scaffold note state", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Literature Review"))
    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Product Brief"))

    expect(mockCreateNewWorkspace).toHaveBeenNthCalledWith(
      1,
      "Literature Review Workspace"
    )
    expect(mockCreateNewWorkspace).toHaveBeenNthCalledWith(
      2,
      "Product Brief Workspace"
    )
    const latestNote = mockSetCurrentNote.mock.calls.at(-1)?.[0] as {
      title: string
      content: string
    }
    expect(latestNote.title).toBe("Product Brief Draft")
    expect(latestNote.content).toContain("## Studio recommendations")
    expect(latestNote.content).toContain("Executive brief")
    expect(latestNote.content).not.toContain("Literature matrix")
  })

  it("imports workspace bundle file from the workspace menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    const input = screen.getByTestId("workspace-import-input")
    const file = new File(["{}"], "workspace.json", {
      type: "application/json"
    })

    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => {
      expect(mockParseWorkspaceImportFile).toHaveBeenCalledWith(file)
      expect(mockImportWorkspaceBundle).toHaveBeenCalledTimes(1)
    })
  })

  it("opens a visible import dialog, rejects unsafe files, and names imported workspaces", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Import Workspace"))

    const modal = await screen.findByRole("dialog", {
      name: "Import Workspace"
    })
    const fileInput = within(modal).getByLabelText("Workspace bundle file")
    expect(fileInput).toHaveAttribute(
      "accept",
      ".json,.workspace.json,.zip,.workspace.zip"
    )

    const unsafeFile = new File(["plain"], "notes.txt", {
      type: "text/plain"
    })
    fireEvent.change(fileInput, { target: { files: [unsafeFile] } })

    expect(mockMessageApi.error).toHaveBeenCalledWith(
      "Choose a .workspace.zip or workspace JSON export."
    )
    expect(mockParseWorkspaceImportFile).not.toHaveBeenCalled()

    const validFile = new File(["{}"], "workspace.workspace.json", {
      type: "application/json"
    })
    fireEvent.change(fileInput, { target: { files: [validFile] } })

    await waitFor(() => {
      expect(mockParseWorkspaceImportFile).toHaveBeenCalledWith(validFile)
      expect(mockImportWorkspaceBundle).toHaveBeenCalledTimes(1)
    })
    expect(mockMessageApi.success).toHaveBeenCalledWith(
      "Workspace imported: Imported"
    )
  })

  it("opens Customize banner modal from settings menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Customize banner"))

    const modal = await screen.findByRole("dialog", {
      name: "Customize banner"
    })
    expect(modal).toBeInTheDocument()
    expect(within(modal).getByLabelText("Banner title")).toHaveValue("Alpha Banner")
    expect(
      within(modal).getByLabelText("Banner subtitle")
    ).toHaveValue("Alpha subtitle")
  })

  it("saves title, subtitle, and image into workspace store", async () => {
    const normalizedImage = {
      dataUrl: "data:image/webp;base64,saved-banner",
      mimeType: "image/webp" as const,
      width: 1400,
      height: 420,
      bytes: 21000,
      updatedAt: new Date("2026-02-25T11:00:00.000Z")
    }
    mockNormalizeWorkspaceBannerImage.mockResolvedValueOnce(normalizedImage)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Customize banner"))

    const modal = await screen.findByRole("dialog", {
      name: "Customize banner"
    })
    fireEvent.change(within(modal).getByLabelText("Banner title"), {
      target: { value: "Updated Banner" }
    })
    fireEvent.change(
      within(modal).getByLabelText("Banner subtitle"),
      {
        target: { value: "Updated subtitle" }
      }
    )

    const file = new File(["banner"], "banner.png", { type: "image/png" })
    fireEvent.change(screen.getByTestId("workspace-banner-upload-input"), {
      target: { files: [file] }
    })

    await waitFor(() => {
      expect(mockNormalizeWorkspaceBannerImage).toHaveBeenCalledWith(file)
    })

    fireEvent.click(within(modal).getByRole("button", { name: "Save" }))

    expect(mockSetWorkspaceBanner).toHaveBeenCalledWith({
      title: "Updated Banner",
      subtitle: "Updated subtitle",
      image: normalizedImage
    })
  })

  it("resets banner fields", async () => {
    const confirmSpy = vi
      .spyOn(Modal, "confirm")
      .mockImplementation((config) => {
        config.onOk?.()
        return {
          destroy: vi.fn(),
          update: vi.fn()
        } as any
      })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Customize banner"))

    const modal = await screen.findByRole("dialog", {
      name: "Customize banner"
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Reset banner" }))

    expect(confirmSpy).toHaveBeenCalled()
    expect(mockResetWorkspaceBanner).toHaveBeenCalledTimes(1)
  })

  it("archives current workspace with undo availability", async () => {
    const confirmSpy = vi
      .spyOn(Modal, "confirm")
      .mockImplementation((config) => {
        config.onOk?.()
        return {
          destroy: vi.fn(),
          update: vi.fn()
        } as any
      })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Archive Current Workspace"))

    expect(confirmSpy).toHaveBeenCalled()
    expect(mockArchiveWorkspace).toHaveBeenCalledWith("workspace-alpha")
    expect(getWorkspaceUndoPendingCount()).toBeGreaterThan(0)

    const openConfig = mockMessageApi.open.mock.calls.at(-1)?.[0] as
      | { content: ReactNode; btn?: unknown; type?: string }
      | undefined
    expect(openConfig?.type).toBe("warning")

    const undoToast = render(<>{openConfig?.content}</>)
    expect(undoToast.getByText("Workspace archived.")).toBeInTheDocument()
    expect(
      undoToast.getByRole("button", { name: "Undo" })
    ).toBeInTheDocument()
    expect(openConfig).not.toHaveProperty("btn")
  })

  it("accepts ZIP workspace imports via the hidden file input", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    const input = screen.getByTestId("workspace-import-input")
    expect(input).toHaveAttribute(
      "accept",
      ".json,.workspace.json,.zip,.workspace.zip"
    )

    const zipFile = new File(["zip"], "workspace.workspace.zip", {
      type: "application/zip"
    })
    fireEvent.change(input, { target: { files: [zipFile] } })

    await waitFor(() => {
      expect(mockParseWorkspaceImportFile).toHaveBeenCalledWith(zipFile)
      expect(mockImportWorkspaceBundle).toHaveBeenCalled()
    })
  })

  it("opens keyboard shortcuts cheat sheet from workspace menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Keyboard Shortcuts"))

    const shortcutsModal = await screen.findByRole("dialog", {
      name: "Keyboard Shortcuts"
    })
    expect(shortcutsModal).toBeInTheDocument()
    expect(within(shortcutsModal).getByText("Search workspace")).toBeInTheDocument()
    expect(within(shortcutsModal).getByText("Focus sources pane")).toBeInTheDocument()
    expect(within(shortcutsModal).getByText("Focus chat pane")).toBeInTheDocument()
    expect(within(shortcutsModal).getByText("Focus studio pane")).toBeInTheDocument()
  })

  it("opens telemetry summary modal from settings menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Telemetry summary"))

    const telemetryModal = await screen.findByRole("dialog", {
      name: "Telemetry summary"
    })
    expect(telemetryModal).toBeInTheDocument()
    expect(mockGetResearchWorkspaceTelemetryState).toHaveBeenCalledTimes(1)
    expect(
      within(telemetryModal).getByTestId(
        "workspace-telemetry-counter-status_viewed"
      )
    ).toHaveTextContent("3")
    expect(
      within(telemetryModal).getByTestId(
        "workspace-telemetry-counter-connectivity_state_changed"
      )
    ).toHaveTextContent("2")
  })

  it("resets telemetry summary state from the telemetry modal", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Telemetry summary"))

    const telemetryModal = await screen.findByRole("dialog", {
      name: "Telemetry summary"
    })
    fireEvent.click(within(telemetryModal).getByRole("button", { name: "Reset" }))

    await waitFor(() => {
      expect(mockResetResearchWorkspaceTelemetryState).toHaveBeenCalledTimes(1)
      expect(mockGetResearchWorkspaceTelemetryState).toHaveBeenCalledTimes(2)
    })
  })

  it("exports telemetry summary and confusion CSV from telemetry modal", async () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:workspace-telemetry")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Telemetry summary"))

    const telemetryModal = await screen.findByRole("dialog", {
      name: "Telemetry summary"
    })
    fireEvent.click(
      within(telemetryModal).getByRole("button", { name: "Export JSON" })
    )
    fireEvent.click(
      within(telemetryModal).getByRole("button", {
        name: "Export confusion CSV"
      })
    )

    await waitFor(() => {
      expect(createObjectUrlSpy).toHaveBeenCalledTimes(2)
    })

    const firstBlob = createObjectUrlSpy.mock.calls[0]?.[0] as Blob
    const secondBlob = createObjectUrlSpy.mock.calls[1]?.[0] as Blob
    expect(firstBlob.type).toContain("application/json")
    expect(secondBlob.type).toContain("text/csv")
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:workspace-telemetry")
  })

  it("uses the design-system registry label without coupling telemetry to display copy", async () => {
    mockConnectionStoreState.state = {
      ...mockConnectionStoreState.state,
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      errorKind: "partial",
      knowledgeStatus: "ready"
    }

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(mockTrackResearchWorkspaceTelemetry).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "connectivity_state_changed",
          to: "degraded"
        })
      )
    })
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith("degraded")
  })

  it("falls back when the degraded design-system registry entry is unavailable", async () => {
    registryStateOverrides.missingDegraded = true
    mockConnectionStoreState.state = {
      ...mockConnectionStoreState.state,
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      errorKind: "partial",
      knowledgeStatus: "ready"
    }

    expect(() => {
      render(
        <WorkspaceHeader
          leftPaneOpen={true}
          rightPaneOpen={true}
          onToggleLeftPane={vi.fn()}
          onToggleRightPane={vi.fn()}
        />
      )
    }).not.toThrow()

    await waitFor(() => {
      expect(mockTrackResearchWorkspaceTelemetry).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "connectivity_state_changed",
          to: "degraded"
        })
      )
    })
  })

  it("loads rollout execution controls from localStorage in telemetry modal", async () => {
    window.localStorage.setItem(
      FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
      "subject-ops-42"
    )
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1,
      "10"
    )
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_workspace_status_guardrails_v1,
      "50"
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Telemetry summary"))

    const telemetryModal = await screen.findByRole("dialog", {
      name: "Telemetry summary"
    })
    expect(
      within(telemetryModal).getByTestId("workspace-rollout-subject-id")
    ).toHaveTextContent("subject-ops-42")
    expect(
      within(telemetryModal).getByTestId(
        "workspace-rollout-percentage-research_workspace_provenance_v1"
      )
    ).toHaveTextContent("10%")
    expect(
      within(telemetryModal).getByTestId(
        "workspace-rollout-percentage-research_workspace_status_guardrails_v1"
      )
    ).toHaveTextContent("50%")
  })

  it("persists rollout preset updates from telemetry modal controls", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Telemetry summary"))

    const telemetryModal = await screen.findByRole("dialog", {
      name: "Telemetry summary"
    })
    const provenanceControl = within(telemetryModal).getByTestId(
      "workspace-rollout-control-research_workspace_provenance_v1"
    )
    fireEvent.click(within(provenanceControl).getByRole("button", { name: "10%" }))

    await waitFor(() => {
      expect(
        window.localStorage.getItem(
          FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1
        )
      ).toBe("10")
    })
    expect(
      within(provenanceControl).getByTestId(
        "workspace-rollout-percentage-research_workspace_provenance_v1"
      )
    ).toHaveTextContent("10%")
  })

  // Storage and connection indicators moved to WorkspaceStatusBar component

  it("hides telemetry menu when rollout flags are disabled", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        statusGuardrailsEnabled={false}
        provenanceEnabled={false}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    expect(screen.queryByText("Telemetry summary")).not.toBeInTheDocument()
  })

  it("creates an ACP-backed agent task for the current workspace", async () => {
    fetchMockState.fetch.mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : undefined

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/workspaces/canonical-bridge"
        ) {
          expect(init?.method).toBe("POST")
          expect((init?.headers as Record<string, string>)["X-API-KEY"]).toBe(
            "test-api-key"
          )
          expect(body).toMatchObject({
            canonical_workspace_id: "workspace-alpha",
            canonical_workspace_source: "research_workspace",
            root_path: "/Users/macbook-dev/src/alpha",
            metadata: {
              created_from: "research_workspace",
              canonical_workspace_id: "workspace-alpha"
            }
          })
          return {
            ok: true,
            json: async () => ({
              id: 33,
              name: "Alpha Research execution",
              root_path: "/Users/macbook-dev/src/alpha",
              canonical_workspace: {
                acp_workspace_id: 33,
                canonical_workspace_id: "workspace-alpha",
                canonical_workspace_source: "research_workspace",
                link_status: "linked"
              }
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects"
        ) {
          expect(init?.method).toBe("POST")
          expect(body).toMatchObject({
            name: "Alpha Research agent work",
            workspace_id: 33,
            metadata: {
              created_from: "research_workspace",
              canonical_workspace_id: "workspace-alpha",
              acp_workspace_id: 33
            }
          })
          return {
            ok: true,
            json: async () => ({
              id: 44,
              name: "Alpha Research agent work",
              workspace_id: 33
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
        ) {
          expect(init?.method).toBe("POST")
          expect(body).toMatchObject({
            title: "Summarize workspace blockers",
            description: "Review the current sources and identify blockers.",
            agent_type: "codex",
            metadata: {
              created_from: "research_workspace",
              canonical_workspace_id: "workspace-alpha",
              acp_workspace_id: 33
            }
          })
          return {
            ok: true,
            json: async () => ({
              id: 55,
              project_id: 44,
              title: "Summarize workspace blockers"
            })
          } as Response
        }

        throw new Error(`unexpected fetch: ${url}`)
      }
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Create agent task"))

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    fireEvent.change(within(modal).getByLabelText("Execution root path"), {
      target: { value: "/Users/macbook-dev/src/alpha" }
    })
    fireEvent.change(within(modal).getByLabelText("Task title"), {
      target: { value: "Summarize workspace blockers" }
    })
    fireEvent.change(within(modal).getByLabelText("Task description"), {
      target: { value: "Review the current sources and identify blockers." }
    })
    fireEvent.change(within(modal).getByLabelText("Agent type"), {
      target: { value: "codex" }
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Create task" }))

    await waitFor(() => {
      expect(fetchMockState.fetch).toHaveBeenCalledTimes(3)
      expect(within(modal).getByText("Agent task created")).toBeInTheDocument()
      expect(within(modal).getByText("ACP workspace #33")).toBeInTheDocument()
    })

    fireEvent.click(within(modal).getByRole("button", { name: "Open Agent Tasks" }))
    expect(mockNavigate).toHaveBeenCalledWith("/agent-tasks?workspace=workspace-alpha")
  })

  it("stores workspace task context metadata on created agent tasks", async () => {
    fetchMockState.fetch.mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : undefined

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/workspaces/canonical-bridge"
        ) {
          return {
            ok: true,
            json: async () => ({
              id: 33,
              canonical_workspace: {
                acp_workspace_id: 33
              }
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects"
        ) {
          expect(body).toMatchObject({
            metadata: {
              created_from: "research_workspace",
              canonical_workspace_id: "workspace-alpha",
              acp_workspace_id: 33,
              research_workspace_task_context: {
                entrypoint: "chat",
                selectedSourceIds: ["source-1"]
              }
            }
          })
          return {
            ok: true,
            json: async () => ({
              id: 44,
              workspace_id: 33
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
        ) {
          expect(body).toMatchObject({
            title: "Investigate chat thread",
            description: "Use selected sources.",
            metadata: {
              research_workspace_task_context: {
                entrypoint: "chat",
                selectedSourceIds: ["source-1"]
              }
            }
          })
          return {
            ok: true,
            json: async () => ({
              id: 55
            })
          } as Response
        }

        throw new Error(`unexpected fetch: ${url}`)
      }
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        agentTaskHandoffOpenSignal={1}
        agentTaskPrefill={{
          title: "Investigate chat thread",
          description: "Use selected sources.",
          metadata: {
            entrypoint: "chat",
            selectedSourceIds: ["source-1"]
          }
        }}
      />
    )

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    fireEvent.change(within(modal).getByLabelText("Execution root path"), {
      target: { value: "/Users/macbook-dev/src/alpha" }
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Create task" }))

    await waitFor(() => {
      expect(fetchMockState.fetch).toHaveBeenCalledTimes(3)
      expect(within(modal).getByText("Agent task created")).toBeInTheDocument()
    })
  })

  it("explains agent tasks are governed by ACP sandbox and approvals", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        agentTaskHandoffOpenSignal={1}
      />
    )

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    expect(
      within(modal).getByText(/ACP capabilities, sandbox checks, and approvals/i)
    ).toBeInTheDocument()
    expect(
      within(modal).getByText(/observable events, artifacts, diagnostics, and results/i)
    ).toBeInTheDocument()
  })

  it("shows recent ACP run history for the current workspace and opens diagnostics", async () => {
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (
        url ===
        ACP_PROJECTS_FOR_ALPHA_URL
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 44,
              name: "Alpha agent work",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            },
            {
              id: 77,
              name: "Beta agent work",
              canonical_workspace: {
                canonical_workspace_id: "workspace-beta",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 55,
              project_id: 44,
              title: "Summarize workspace blockers",
              status: "complete",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/55"
      ) {
        return {
          ok: true,
          json: async () => ({
            id: 55,
            project_id: 44,
            title: "Summarize workspace blockers",
            status: "complete",
            runs: [
              {
                id: 88,
                task_id: 55,
                status: "completed",
                agent_type: "codex",
                result_summary: "Identified two release blockers.",
                started_at: "2026-05-13T13:00:00.000Z",
                completed_at: "2026-05-13T13:05:00.000Z",
                session: {
                  session_id: "sess-alpha",
                  available: true,
                  links: {
                    diagnostics: "/api/v1/acp/sessions/sess-alpha/diagnostics",
                    artifacts: "/api/v1/acp/sessions/sess-alpha/artifacts",
                    audit: "/api/v1/acp/sessions/sess-alpha/audit"
                  }
                },
                history: {
                  audit_event_count: 2,
                  artifact_count: 1,
                  diagnostic_count: 3,
                  result: {
                    preview: "Workspace run preview"
                  }
                }
              }
            ]
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Alpha agent work")).toBeInTheDocument()
    expect(
      within(modal).getByText("Summarize workspace blockers")
    ).toBeInTheDocument()
    expect(within(modal).getByText("sess-alpha")).toBeInTheDocument()
    expect(within(modal).getByText("1 artifacts/files")).toBeInTheDocument()
    expect(within(modal).getByText("3 diagnostics/warnings")).toBeInTheDocument()
    expect(
      within(modal).getByText("Identified two release blockers.")
    ).toBeInTheDocument()

    fireEvent.click(within(modal).getByRole("button", { name: "Open diagnostics" }))
    expect(mockNavigate).toHaveBeenNthCalledWith(
      1,
      "/acp-playground?session=sess-alpha&view=diagnostics"
    )

    fireEvent.click(within(modal).getByRole("button", { name: "Open Agent Tasks" }))
    expect(mockNavigate).toHaveBeenNthCalledWith(
      2,
      "/agent-tasks?workspace=workspace-alpha"
    )
    expect(mockNavigate).toHaveBeenCalledTimes(2)
  })

  it("saves ACP run results as traceable Studio artifacts", async () => {
    mockStoreState.generatedArtifacts = [
      {
        id: "artifact-prior",
        type: "report",
        title: "Agent result: Synthesize the workspace",
        status: "completed",
        version: 1,
        artifactVersionId: "acp-run-99-v1",
        rootArtifactId: "acp-run-99",
        content: "Prior saved result.",
        createdAt: new Date("2026-05-13T13:05:00.000Z")
      }
    ]
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (url === ACP_PROJECTS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => [
            {
              id: 66,
              name: "Alpha agent work",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/66/tasks"
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 77,
              project_id: 66,
              title: "Synthesize the workspace",
              status: "complete",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/77"
      ) {
        return {
          ok: true,
          json: async () => ({
            id: 77,
            project_id: 66,
            title: "Synthesize the workspace",
            status: "complete",
            runs: [
              {
                id: 99,
                task_id: 77,
                status: "completed",
                agent_type: "codex",
                result_summary: "Completed synthesis with two follow-up actions.",
                started_at: "2026-05-13T13:00:00.000Z",
                completed_at: "2026-05-13T13:05:00.000Z",
                session: {
                  session_id: "sess-99",
                  available: true,
                  links: {
                    diagnostics: "/api/v1/acp/sessions/sess-99/diagnostics",
                    artifacts: "/api/v1/acp/sessions/sess-99/artifacts",
                    audit: "/api/v1/acp/sessions/sess-99/audit"
                  }
                },
                history: {
                  audit_event_count: 2,
                  artifact_count: 1,
                  diagnostic_count: 3,
                  event_count: 4,
                  result: {
                    preview: "Completed synthesis from result preview."
                  }
                }
              }
            ]
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Alpha agent work")).toBeInTheDocument()
    expect(
      within(modal).getByLabelText(
        /Observable activity: 1 artifacts\/files, 3 diagnostics\/warnings, 2 audit\/approvals, 4 events\/tool activity/i
      )
    ).toBeInTheDocument()

    fireEvent.click(within(modal).getByRole("button", { name: "Save to Studio" }))

    expect(mockAddArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "report",
        status: "completed",
        title: "Agent result: Synthesize the workspace",
        content: expect.stringContaining("Completed synthesis"),
        version: 2,
        artifactVersionId: "acp-run-99-v2",
        rootArtifactId: "acp-run-99",
        previousVersionId: "acp-run-99-v1",
        projectId: "66",
        taskId: "77",
        ownerScope: "research_workspace",
        ownerId: "workspace-alpha",
        producerMetadata: expect.objectContaining({
          producerType: "acp_agent_task",
          runId: "99",
          sessionId: "sess-99",
          taskId: "77",
          projectId: "66",
          producerId: "77",
          links: expect.objectContaining({
            diagnostics: "/api/v1/acp/sessions/sess-99/diagnostics",
            artifacts: "/api/v1/acp/sessions/sess-99/artifacts",
            audit: "/api/v1/acp/sessions/sess-99/audit"
          })
        }),
        versionMetadata: expect.objectContaining({
          revisionReason: "Saved from ACP run history"
        }),
        data: expect.objectContaining({
          acpRun: expect.objectContaining({
            artifactCount: 1,
            diagnosticCount: 3,
            auditEventCount: 2,
            eventCount: 4
          })
        })
      })
    )
    expect(mockSaveCurrentWorkspace).toHaveBeenCalled()
    expect(mockMessageApi.success).toHaveBeenCalledWith(
      "Agent result saved to Studio outputs."
    )
  })

  it("creates distinct ACP artifact versions when a completed run is saved repeatedly", async () => {
    mockStoreState.generatedArtifacts = [
      {
        id: "artifact-prior-v3",
        type: "report",
        title: "Agent result: Versioned synthesis",
        status: "completed",
        artifactVersionId: "acp-run-101-v3",
        rootArtifactId: "acp-run-101",
        content: "Earlier saved result.",
        createdAt: new Date("2026-05-13T13:20:00.000Z")
      }
    ]
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (url === ACP_PROJECTS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => [
            {
              id: 68,
              name: "Alpha versioned results",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/68/tasks"
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 79,
              project_id: 68,
              title: "Versioned synthesis",
              status: "complete",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/79"
      ) {
        return {
          ok: true,
          json: async () => ({
            id: 79,
            project_id: 68,
            title: "Versioned synthesis",
            status: "complete",
            runs: [
              {
                id: 101,
                task_id: 79,
                status: "completed",
                result_summary: "Completed versioned synthesis.",
                completed_at: "2026-05-13T13:25:00.000Z",
                session: {
                  session_id: "sess-101",
                  available: true
                },
                history: {
                  event_count: 2
                }
              }
            ]
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })
    mockAddArtifact.mockImplementation((artifact) => {
      const savedArtifact = {
        ...artifact,
        id: `artifact-${mockAddArtifact.mock.calls.length}`,
        createdAt: new Date()
      }
      mockStoreState.generatedArtifacts = [
        savedArtifact,
        ...mockStoreState.generatedArtifacts
      ]
      return savedArtifact
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    const saveButton = await within(modal).findByRole("button", {
      name: "Save to Studio"
    })

    fireEvent.click(saveButton)
    fireEvent.click(saveButton)

    expect(mockAddArtifact).toHaveBeenCalledTimes(2)
    expect(mockAddArtifact.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        artifactVersionId: "acp-run-101-v4",
        previousVersionId: "acp-run-101-v3"
      })
    )
    expect(mockAddArtifact.mock.calls[1]?.[0]).toEqual(
      expect.objectContaining({
        artifactVersionId: "acp-run-101-v5",
        previousVersionId: "acp-run-101-v4"
      })
    )
  })

  it("saves completed ACP run results even when the session is no longer retained", async () => {
    mockStoreState.generatedArtifacts = undefined as any
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (url === ACP_PROJECTS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => [
            {
              id: 67,
              name: "Alpha retained results",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/67/tasks"
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 78,
              project_id: 67,
              title: "Write retained summary",
              status: "complete",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/78"
      ) {
        return {
          ok: true,
          json: async () => ({
            id: 78,
            project_id: 67,
            title: "Write retained summary",
            status: "complete",
            runs: [
              {
                id: 100,
                task_id: 78,
                session_id: null,
                status: "completed",
                result_summary: "Completed retained summary.",
                completed_at: "not-a-date",
                session: null,
                history: {
                  audit_event_count: 0,
                  artifact_count: 0,
                  diagnostic_count: 0,
                  event_count: 2
                }
              }
            ]
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Alpha retained results")).toBeInTheDocument()

    fireEvent.click(within(modal).getByRole("button", { name: "Save to Studio" }))

    expect(mockAddArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Agent result: Write retained summary",
        content: "Completed retained summary.",
        artifactVersionId: "acp-run-100-v1",
        rootArtifactId: "acp-run-100",
        producerMetadata: expect.objectContaining({
          producerType: "acp_agent_task",
          runId: "100",
          sessionId: undefined
        })
      })
    )
    const savedArtifact = mockAddArtifact.mock.calls[0]?.[0]
    expect(savedArtifact.completedAt).toBeInstanceOf(Date)
    expect(Number.isNaN(savedArtifact.completedAt.getTime())).toBe(false)
  })

  it("shows direct workspace ACP sessions when Agent Tasks history has no runs", async () => {
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({
            sessions: [
              {
                session_id: "direct-session-alpha",
                user_id: 1,
                agent_type: "codex",
                name: "Codex workspace session",
                status: "active",
                created_at: "2026-05-13T13:00:00.000Z",
                last_activity_at: "2026-05-13T13:10:00.000Z",
                message_count: 4,
                usage: {
                  prompt_tokens: 12,
                  completion_tokens: 8,
                  total_tokens: 20
                },
                tags: [],
                has_websocket: false,
                workspace_id: "workspace-alpha",
                sandbox_session_id: "sandbox-session-1",
                sandbox_run_id: "sandbox-run-1",
                workspace_context: {
                  workspace_id: "workspace-alpha",
                  mcp_server_count: 1,
                  mcp_server_names: ["filesystem"],
                  sandbox_session_id: "sandbox-session-1",
                  sandbox_run_id: "sandbox-run-1",
                  agent_type: "codex",
                  runtime_backend: "acp_downstream",
                  entrypoint_strategy: "external_acp_adapter",
                  adapter_source: "zed-industries/codex-acp",
                  adapter_package: "@zed-industries/codex-acp",
                  adapter_version: "0.15.0",
                  support_state: "supported_with_caveats",
                  verification_level: "live_e2e_tested"
                }
              }
            ],
            total: 1
          })
        } as Response
      }

      if (url === ACP_PROJECTS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => []
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Direct ACP sessions")).toBeInTheDocument()
    expect(within(modal).getByText("Codex workspace session")).toBeInTheDocument()
    expect(within(modal).getByText("direct-session-alpha")).toBeInTheDocument()
    expect(within(modal).getByText("4 messages")).toBeInTheDocument()
    expect(within(modal).getByText("1 MCP server")).toBeInTheDocument()
    expect(fetchMockState.fetch).toHaveBeenCalledWith(
      ACP_SESSIONS_FOR_ALPHA_URL,
      expect.objectContaining({
        headers: expect.objectContaining({ "X-API-KEY": "test-api-key" })
      })
    )

    fireEvent.click(within(modal).getByRole("button", { name: "Open diagnostics" }))
    expect(mockNavigate).toHaveBeenCalledWith(
      "/acp-playground?session=direct-session-alpha&view=diagnostics"
    )
  })

  it("surfaces direct ACP session fetch errors when Agent Tasks history has no runs", async () => {
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: false,
          status: 503,
          json: async () => ({
            detail: "Direct ACP session history is temporarily unavailable"
          })
        } as Response
      }

      if (url === ACP_PROJECTS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => []
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(
      await within(modal).findByText(
        "Direct ACP session history is temporarily unavailable"
      )
    ).toBeInTheDocument()
  })

  it("opens sandbox diagnostics from the settings menu", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Sandbox diagnostics"))

    const modal = await screen.findByRole("dialog", {
      name: "Sandbox diagnostics"
    })
    expect(within(modal).getByTestId("workspace-sandbox-diagnostics-panel")).toHaveTextContent(
      "Sandbox diagnostics for workspace-alpha"
    )
    expect(modal).not.toHaveTextContent(/workspace trust/i)
  })

  it("aborts ACP run history requests when the modal closes", async () => {
    let capturedSignal: AbortSignal | undefined
    fetchMockState.fetch.mockImplementation(
      async (_input: RequestInfo | URL, init?: RequestInit) => {
        capturedSignal = init?.signal as AbortSignal | undefined
        return new Promise<Response>(() => undefined)
      }
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    await waitFor(() => expect(capturedSignal).toBeInstanceOf(AbortSignal))

    const closeButtons = within(modal).getAllByRole("button", { name: "Close" })
    fireEvent.click(closeButtons[closeButtons.length - 1])

    expect(capturedSignal?.aborted).toBe(true)
  })

  it("prioritizes newest workspace tasks before fetching ACP run detail", async () => {
    const detailRequests: string[] = []

    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (
        url ===
        ACP_PROJECTS_FOR_ALPHA_URL
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 44,
              name: "Alpha agent work",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
      ) {
        return {
          ok: true,
          json: async () =>
            Array.from({ length: 13 }, (_value, index) => {
              const id = index + 1
              return {
                id,
                project_id: 44,
                title: `Task ${id}`,
                status: "complete",
                created_at: `2026-05-13T13:${String(id).padStart(2, "0")}:00.000Z`,
                updated_at: `2026-05-13T13:${String(id).padStart(2, "0")}:30.000Z`,
                canonical_workspace: {
                  canonical_workspace_id: "workspace-alpha"
                }
              }
            })
        } as Response
      }

      if (
        url.startsWith(
          "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/"
        )
      ) {
        detailRequests.push(url)
        const id = Number(url.split("/").pop())
        return {
          ok: true,
          json: async () => ({
            id,
            project_id: 44,
            title: `Task ${id}`,
            status: "complete",
            runs:
              id === 13
                ? [
                    {
                      id: 130,
                      task_id: 13,
                      status: "completed",
                      result_summary: "Newest task run",
                      completed_at: "2026-05-13T14:00:00.000Z",
                      session: {
                        session_id: "sess-newest",
                        links: {
                          diagnostics: "/api/v1/acp/sessions/sess-newest/diagnostics"
                        }
                      },
                      history: {
                        artifact_count: 0,
                        diagnostic_count: 1,
                        audit_event_count: 0
                      }
                    }
                  ]
                : []
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Newest task run")).toBeInTheDocument()
    expect(detailRequests).toContain(
      "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/13"
    )
    expect(detailRequests).not.toContain(
      "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/1"
    )
  })

  it("shows an empty ACP run history state for workspaces without runs", async () => {
    fetchMockState.fetch.mockResolvedValue({
      ok: true,
      json: async () => []
    } as Response)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(
      await within(modal).findByText("No ACP runs linked to this workspace yet")
    ).toBeInTheDocument()
    expect(
      within(modal).queryByText("Workspace setup needs attention")
    ).not.toBeInTheDocument()
  })

  it("shows ACP run history load errors without setup guidance", async () => {
    fetchMockState.fetch.mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        detail: "Agent orchestration is temporarily unavailable"
      })
    } as Response)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(
      await within(modal).findByText(
        "Agent orchestration is temporarily unavailable"
      )
    ).toBeInTheDocument()
    expect(
      within(modal).queryByText("Workspace setup needs attention")
    ).not.toBeInTheDocument()
  })

  it("surfaces missing task errors instead of treating them as unsupported orchestration", async () => {
    fetchMockState.fetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url === ACP_SESSIONS_FOR_ALPHA_URL) {
        return {
          ok: true,
          json: async () => ({ sessions: [], total: 0 })
        } as Response
      }

      if (
        url ===
        ACP_PROJECTS_FOR_ALPHA_URL
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 44,
              name: "Alpha agent work",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha",
                link_status: "linked"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
      ) {
        return {
          ok: true,
          json: async () => [
            {
              id: 55,
              project_id: 44,
              title: "Deleted task",
              status: "complete",
              canonical_workspace: {
                canonical_workspace_id: "workspace-alpha"
              }
            }
          ]
        } as Response
      }

      if (
        url ===
        "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/55"
      ) {
        return {
          ok: false,
          status: 404,
          json: async () => ({
            detail: "Task not found"
          })
        } as Response
      }

      throw new Error(`unexpected fetch: ${url}`)
    })

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(await within(modal).findByText("Task not found")).toBeInTheDocument()
    expect(
      within(modal).queryByText("Agent orchestration is not available on this server.")
    ).not.toBeInTheDocument()
  })

  it("shows unsupported ACP run history state without setup guidance", async () => {
    fetchMockState.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({
        detail: "Not Found"
      })
    } as Response)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("ACP run history"))

    const modal = await screen.findByRole("dialog", {
      name: "ACP run history"
    })
    expect(
      await within(modal).findByText(
        "Agent orchestration is not available on this server."
      )
    ).toBeInTheDocument()
    expect(
      within(modal).queryByText("Workspace setup needs attention")
    ).not.toBeInTheDocument()
  })

  it("surfaces ACP bridge setup failures without creating project or task records", async () => {
    fetchMockState.fetch.mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({
        detail: {
          code: "workspace_root_not_allowed",
          message: "Root path is outside the configured ACP allowlist."
        }
      })
    } as Response)

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Create agent task"))

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    fireEvent.change(within(modal).getByLabelText("Execution root path"), {
      target: { value: "/private/not-allowed" }
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Create task" }))

    await waitFor(() => {
      expect(fetchMockState.fetch).toHaveBeenCalledTimes(1)
      expect(
        within(modal).getByText("Root path is outside the configured ACP allowlist.")
      ).toBeInTheDocument()
    })
  })

  it("waits for connection configuration before enabling task creation", async () => {
    connectionConfigState.loading = true
    connectionConfigState.config = null

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Create agent task"))

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    const createTaskButton = within(modal).getByRole("button", {
      name: "Create task"
    })

    expect(createTaskButton).toBeDisabled()
    fireEvent.click(createTaskButton)
    expect(fetchMockState.fetch).not.toHaveBeenCalled()
  })

  it("rolls back a created ACP project when task creation fails", async () => {
    fetchMockState.fetch.mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/workspaces/canonical-bridge"
        ) {
          return {
            ok: true,
            json: async () => ({
              id: 33,
              canonical_workspace: {
                acp_workspace_id: 33
              }
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects"
        ) {
          return {
            ok: true,
            json: async () => ({
              id: 44,
              name: "Alpha Research agent work",
              workspace_id: 33
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
        ) {
          return {
            ok: false,
            status: 500,
            json: async () => ({
              detail: "Task creation failed."
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44"
        ) {
          expect(init?.method).toBe("DELETE")
          return {
            ok: true,
            json: async () => ({})
          } as Response
        }

        throw new Error(`unexpected fetch: ${url}`)
      }
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Create agent task"))

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    fireEvent.change(within(modal).getByLabelText("Execution root path"), {
      target: { value: "/Users/macbook-dev/src/alpha" }
    })
    fireEvent.change(within(modal).getByLabelText("Task title"), {
      target: { value: "Summarize workspace blockers" }
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Create task" }))

    await waitFor(() => {
      expect(fetchMockState.fetch).toHaveBeenCalledTimes(4)
      expect(within(modal).getByText("Task creation failed.")).toBeInTheDocument()
    })
    expect(
      fetchMockState.fetch.mock.calls.some(
        ([input, init]) =>
          String(input) ===
            "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44" &&
          init?.method === "DELETE"
      )
    ).toBe(true)
    expect(within(modal).queryByText("Agent task created")).not.toBeInTheDocument()
  })

  it("keeps the task handoff modal open while submission is in flight", async () => {
    let resolveBridge: (response: Response) => void = () => {}

    fetchMockState.fetch.mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/workspaces/canonical-bridge"
        ) {
          return new Promise<Response>((resolve) => {
            resolveBridge = resolve
          })
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects"
        ) {
          return {
            ok: true,
            json: async () => ({
              id: 44,
              workspace_id: 33
            })
          } as Response
        }

        if (
          url ===
          "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/44/tasks"
        ) {
          expect(init?.method).toBe("POST")
          return {
            ok: true,
            json: async () => ({
              id: 55
            })
          } as Response
        }

        throw new Error(`unexpected fetch: ${url}`)
      }
    )

    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(await screen.findByText("Create agent task"))

    const modal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    fireEvent.change(within(modal).getByLabelText("Execution root path"), {
      target: { value: "/Users/macbook-dev/src/alpha" }
    })
    const createTaskButton = within(modal).getByRole("button", {
      name: "Create task"
    })
    fireEvent.click(createTaskButton)

    const cancelButton = within(modal).getByRole("button", { name: "Cancel" })
    await waitFor(() => {
      expect(cancelButton).toBeDisabled()
      expect(createTaskButton).toBeDisabled()
    })
    fireEvent.click(cancelButton)
    expect(
      screen.getByRole("dialog", { name: "Create agent task" })
    ).toBeInTheDocument()

    resolveBridge({
      ok: true,
      json: async () => ({
        id: 33,
        canonical_workspace: {
          acp_workspace_id: 33
        }
      })
    } as Response)

    await waitFor(() => {
      expect(within(modal).getByText("Agent task created")).toBeInTheDocument()
    })
  })

  it("does not reuse web-clip agent-task prefill for later manual task creation", async () => {
    render(
      <WorkspaceHeader
        leftPaneOpen={true}
        rightPaneOpen={true}
        onToggleLeftPane={vi.fn()}
        onToggleRightPane={vi.fn()}
        agentTaskHandoffOpenSignal={1}
        agentTaskPrefill={{
          title: "Review captured page: Example Story",
          description: "Captured excerpt:\nAlpha body copy"
        }}
      />
    )

    const handoffModal = await screen.findByRole("dialog", {
      name: "Create agent task"
    })
    expect(within(handoffModal).getByLabelText("Task title")).toHaveValue(
      "Review captured page: Example Story"
    )
    expect(within(handoffModal).getByLabelText("Task description")).toHaveValue(
      "Captured excerpt:\nAlpha body copy"
    )

    fireEvent.click(within(handoffModal).getByRole("button", { name: "Cancel" }))
    await waitFor(() => {
      const dialog = screen.queryByRole("dialog", { name: "Create agent task" })
      if (!dialog) {
        expect(dialog).not.toBeInTheDocument()
        return
      }
      expect(dialog).toHaveClass("ant-zoom-leave")
    })

    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Create agent task" })
    )

    const manualModal = await waitFor(() => {
      const activeDialog = screen
        .getAllByRole("dialog", { name: "Create agent task" })
        .find((dialog) => !dialog.classList.contains("ant-zoom-leave"))
      expect(activeDialog).toBeDefined()
      return activeDialog as HTMLElement
    })
    expect(within(manualModal).getByLabelText("Task title")).not.toHaveValue(
      "Review captured page: Example Story"
    )
    expect(within(manualModal).getByLabelText("Task description")).toHaveValue("")
  })
})
