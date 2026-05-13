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
const mockTrackWorkspacePlaygroundTelemetry = vi.fn()
const mockGetWorkspacePlaygroundTelemetryState = vi.fn()
const mockResetWorkspacePlaygroundTelemetryState = vi.fn()
const registryStateOverrides = vi.hoisted(() => ({
  degradedLabel: "Registry Degraded",
  missingDegraded: false
}))
const connectionConfigState = vi.hoisted(
  (): {
    loading: boolean
    config: {
      serverUrl: string
      authMode: "single-user"
      apiKey: string
      accessToken: string
    } | null
  } => ({
    loading: false,
    config: {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user" as const,
      apiKey: "test-api-key",
      accessToken: ""
    }
  })
)
const fetchMockState = vi.hoisted(() => ({
  fetch: vi.fn()
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

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof mockStoreState) => unknown
  ) => selector(mockStoreState)
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

vi.mock("@/utils/workspace-playground-telemetry", async () => {
  const actual =
    await vi.importActual<typeof import("@/utils/workspace-playground-telemetry")>(
      "@/utils/workspace-playground-telemetry"
    )
  return {
    ...actual,
    trackWorkspacePlaygroundTelemetry: (...args: unknown[]) =>
      mockTrackWorkspacePlaygroundTelemetry(...args),
    getWorkspacePlaygroundTelemetryState: (...args: unknown[]) =>
      mockGetWorkspacePlaygroundTelemetryState(...args),
    resetWorkspacePlaygroundTelemetryState: (...args: unknown[]) =>
      mockResetWorkspacePlaygroundTelemetryState(...args)
  }
})

if (!(globalThis as unknown as { ResizeObserver?: unknown }).ResizeObserver) {
  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("WorkspaceHeader workspace browser modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    clearWorkspaceUndoActionsForTests()
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
      format: "tldw.workspace-playground.bundle",
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
      format: "tldw.workspace-playground.bundle",
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
    mockGetWorkspacePlaygroundTelemetryState.mockResolvedValue({
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
    mockResetWorkspacePlaygroundTelemetryState.mockResolvedValue(undefined)
    registryStateOverrides.degradedLabel = "Registry Degraded"
    registryStateOverrides.missingDegraded = false
    mockNormalizeWorkspaceBannerImage.mockResolvedValue({
      dataUrl: "data:image/webp;base64,banner",
      mimeType: "image/webp",
      width: 1200,
      height: 400,
      bytes: 16000,
      updatedAt: new Date("2026-02-25T10:00:00.000Z")
    })
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

    const searchInput = within(modal).getByPlaceholderText(
      "Search workspaces by name or tag"
    )
    fireEvent.change(searchInput, { target: { value: "gamma" } })

    await waitFor(() => {
      expect(within(modal).queryByText("Beta Deep Dive")).not.toBeInTheDocument()
      expect(within(modal).getByText("Gamma Notes")).toBeInTheDocument()
    })
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
      within(modal).getByPlaceholderText("New collection name"),
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
    })
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
    expect(within(modal).getByTestId("workspace-banner-title-input")).toHaveValue(
      "Alpha Banner"
    )
    expect(
      within(modal).getByTestId("workspace-banner-subtitle-input")
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
    fireEvent.change(within(modal).getByTestId("workspace-banner-title-input"), {
      target: { value: "Updated Banner" }
    })
    fireEvent.change(
      within(modal).getByTestId("workspace-banner-subtitle-input"),
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
    expect(mockGetWorkspacePlaygroundTelemetryState).toHaveBeenCalledTimes(1)
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
      expect(mockResetWorkspacePlaygroundTelemetryState).toHaveBeenCalledTimes(1)
      expect(mockGetWorkspacePlaygroundTelemetryState).toHaveBeenCalledTimes(2)
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
      expect(mockTrackWorkspacePlaygroundTelemetry).toHaveBeenCalledWith(
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
      expect(mockTrackWorkspacePlaygroundTelemetry).toHaveBeenCalledWith(
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
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_studio_provenance_v1,
      "10"
    )
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_studio_status_guardrails_v1,
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
        "workspace-rollout-percentage-research_studio_provenance_v1"
      )
    ).toHaveTextContent("10%")
    expect(
      within(telemetryModal).getByTestId(
        "workspace-rollout-percentage-research_studio_status_guardrails_v1"
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
      "workspace-rollout-control-research_studio_provenance_v1"
    )
    fireEvent.click(within(provenanceControl).getByRole("button", { name: "10%" }))

    await waitFor(() => {
      expect(
        window.localStorage.getItem(
          FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_studio_provenance_v1
        )
      ).toBe("10")
    })
    expect(
      within(provenanceControl).getByTestId(
        "workspace-rollout-percentage-research_studio_provenance_v1"
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
            canonical_workspace_source: "workspace_playground",
            root_path: "/Users/macbook-dev/src/alpha",
            metadata: {
              created_from: "workspace_playground",
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
                canonical_workspace_source: "workspace_playground",
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
              created_from: "workspace_playground",
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
              created_from: "workspace_playground",
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
    expect(mockNavigate).toHaveBeenCalledWith("/agent-tasks")
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
})
