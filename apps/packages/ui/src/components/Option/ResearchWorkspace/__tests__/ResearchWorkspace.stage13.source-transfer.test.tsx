import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi
} from "vitest"
import { ResearchWorkspace } from "../index"
import { clearWorkspaceUndoActionsForTests } from "../undo-manager"

const mockInitializeWorkspace = vi.fn()
const mockCreateNewWorkspace = vi.fn()
const mockAddSources = vi.fn()
const mockSetSelectedSourceIds = vi.fn()
const mockCaptureToCurrentNote = vi.fn()
const mockCaptureUndoSnapshot = vi.fn()
const mockRestoreUndoSnapshot = vi.fn()
const mockClearCurrentNote = vi.fn()
const mockSetCurrentNote = vi.fn()
const mockLoadNote = vi.fn()
const mockDuplicateWorkspace = vi.fn()
const mockSetLeftPaneCollapsed = vi.fn()
const mockSetRightPaneCollapsed = vi.fn()
const mockFocusSourceById = vi.fn()
const mockFocusChatMessageById = vi.fn()
const mockFocusWorkspaceNote = vi.fn()
const mockSetSourceStatusByMediaId = vi.fn()
const mockTransferSourcesBetweenWorkspaces = vi.fn()
const mockSwitchWorkspace = vi.fn()
const mockMessageApi = {
  open: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  destroy: vi.fn()
}
const { workspaceStorage, workspaceStorageItems } = vi.hoisted(() => {
  const storageItems = new Map<string, string>()
  const storage = {
    getItem: vi.fn((key: string) => storageItems.get(key) ?? null),
    setItem: vi.fn(async (key: string, value: string) => {
      storageItems.set(key, value)
    }),
    removeItem: vi.fn(async (key: string) => {
      storageItems.delete(key)
    })
  }

  return {
    workspaceStorage: storage,
    workspaceStorageItems: storageItems
  }
})

const destinationWorkspaceId = "workspace-destination"
const archivedWorkspaceId = "workspace-archived"

const testState = {
  isMobile: false,
  storeHydrated: true,
  workspaceId: "workspace-current",
  workspaceName: "Current Workspace",
  workspaceTag: "workspace:current",
  workspaceBanner: {
    title: "",
    subtitle: "",
    image: null
  },
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  sources: [
    {
      id: "source-a",
      mediaId: 101,
      title: "Alpha Source",
      type: "pdf" as const,
      status: "ready" as const,
      addedAt: new Date("2026-03-28T00:00:00.000Z")
    },
    {
      id: "source-b",
      mediaId: 102,
      title: "Beta Source",
      type: "website" as const,
      status: "ready" as const,
      addedAt: new Date("2026-03-28T00:00:00.000Z")
    },
    {
      id: "source-c",
      mediaId: 103,
      title: "Gamma Processing Source",
      type: "document" as const,
      status: "processing" as const,
      addedAt: new Date("2026-03-28T00:00:00.000Z")
    }
  ],
  selectedSourceIds: ["source-a", "source-b"] as string[],
  selectedSourceFolderIds: [] as string[],
  sourceFolders: [] as Array<{ id: string }>,
  sourceFolderMemberships: [] as Array<{ sourceId: string; folderId: string }>,
  activeFolderId: null as string | null,
  sourceSearchQuery: "",
  generatedArtifacts: [] as Array<{ id: string }>,
  currentNote: {
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
  },
  workspaceChatSessions: {} as Record<
    string,
    { messages: Array<{ message: string; sources: unknown[]; isBot: boolean; name: string }> }
  >,
  savedWorkspaces: [
    {
      id: "workspace-current",
      name: "Current Workspace",
      tag: "workspace:current",
      collectionId: null,
      createdAt: new Date("2026-03-28T00:00:00.000Z"),
      lastAccessedAt: new Date("2026-03-28T00:00:00.000Z"),
      sourceCount: 2
    },
    {
      id: destinationWorkspaceId,
      name: "Destination Workspace",
      tag: "workspace:destination",
      collectionId: null,
      createdAt: new Date("2026-03-28T00:00:00.000Z"),
      lastAccessedAt: new Date("2026-03-28T00:00:00.000Z"),
      sourceCount: 2
    }
  ],
  archivedWorkspaces: [
    {
      id: archivedWorkspaceId,
      name: "Archived Workspace",
      tag: "workspace:archived",
      collectionId: null,
      createdAt: new Date("2026-03-28T00:00:00.000Z"),
      lastAccessedAt: new Date("2026-03-28T00:00:00.000Z"),
      sourceCount: 1
    }
  ],
  workspaceSnapshots: {
    [destinationWorkspaceId]: {
      workspaceId: destinationWorkspaceId,
      workspaceName: "Destination Workspace",
      workspaceTag: "workspace:destination",
      workspaceCreatedAt: new Date("2026-03-28T00:00:00.000Z"),
      workspaceChatReferenceId: destinationWorkspaceId,
      studyMaterialsPolicy: null,
      sources: [
        {
          id: "destination-source-a",
          mediaId: 101,
          title: "Existing Alpha",
          type: "pdf" as const,
          status: "ready" as const,
          addedAt: new Date("2026-03-28T00:00:00.000Z")
        },
        {
          id: "destination-source-b",
          mediaId: 102,
          title: "Existing Beta",
          type: "website" as const,
          status: "ready" as const,
          addedAt: new Date("2026-03-28T00:00:00.000Z")
        }
      ],
      selectedSourceIds: [],
      sourceFolders: [],
      sourceFolderMemberships: [],
      selectedSourceFolderIds: [],
      activeFolderId: null,
      generatedArtifacts: [],
      notes: [],
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
      audioSettings: {}
    }
  } as Record<string, unknown>,
  initializeWorkspace: mockInitializeWorkspace,
  createNewWorkspace: mockCreateNewWorkspace,
  addSources: mockAddSources,
  setSelectedSourceIds: mockSetSelectedSourceIds,
  captureToCurrentNote: mockCaptureToCurrentNote,
  captureUndoSnapshot: mockCaptureUndoSnapshot,
  clearCurrentNote: mockClearCurrentNote,
  setCurrentNote: mockSetCurrentNote,
  loadNote: mockLoadNote,
  duplicateWorkspace: mockDuplicateWorkspace,
  setLeftPaneCollapsed: mockSetLeftPaneCollapsed,
  setRightPaneCollapsed: mockSetRightPaneCollapsed,
  focusSourceById: mockFocusSourceById,
  focusChatMessageById: mockFocusChatMessageById,
  focusWorkspaceNote: mockFocusWorkspaceNote,
  setSourceStatusByMediaId: mockSetSourceStatusByMediaId,
  restoreUndoSnapshot: mockRestoreUndoSnapshot,
  getEffectiveSelectedSources: () =>
    testState.sources.filter((source) =>
      testState.selectedSourceIds.includes(source.id)
    ),
  isGeneratingOutput: false,
  generatingOutputType: null as string | null,
  transferSourcesBetweenWorkspaces: mockTransferSourcesBetweenWorkspaces,
  switchWorkspace: mockSwitchWorkspace
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            count?: number
            defaultValue?: string
            workspaceName?: string
          },
      interpolationValues?: {
        count?: number
        workspaceName?: string
      }
    ) => {
      const template =
        typeof defaultValueOrOptions === "string"
          ? defaultValueOrOptions
          : defaultValueOrOptions?.defaultValue || key
      return template
        .replace(
          "{{count}}",
          String(
            interpolationValues?.count ?? defaultValueOrOptions?.count ?? ""
          )
        )
        .replace(
          "{{workspaceName}}",
          String(
            interpolationValues?.workspaceName ??
              defaultValueOrOptions?.workspaceName ??
              ""
          )
        )
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => testState.isMobile
}))

vi.mock("@/hooks/useFeatureFlags", () => ({
  FEATURE_FLAGS: {
    RESEARCH_WORKSPACE_PROVENANCE_V1: "research-workspace-provenance-v1",
    RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1: "research-workspace-status-guardrails-v1"
  },
  useFeatureFlag: () => [true]
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: (selector: (state: { startTutorial: () => void }) => unknown) =>
    selector({ startTutorial: vi.fn() })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: (state: typeof testState) => unknown) =>
    selector(testState),
  createWorkspaceStorage: () => workspaceStorage
}))

vi.mock("@/utils/research-workspace-prefill", () => ({
  consumeResearchWorkspacePrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue("")
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue([])
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({}),
    getCurrentUserStorageQuota: vi.fn().mockResolvedValue({}),
    getCurrentUserProfile: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      ...actual.message,
      useMessage: () => [
        mockMessageApi,
        <div key="message-context" data-testid="workspace-message-context" />
      ]
    }
  }
})

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: (props: {
    onOpenSplitWorkspace?: () => void
  }) => (
    <div data-testid="workspace-header">
      <button
        type="button"
        onClick={() => props.onOpenSplitWorkspace?.()}
      >
        Split workspace
      </button>
    </div>
  )
}))

const launchHiddenSelectionTransfer = {
  entryPoint: "sources" as const,
  selectedSourceIds: ["source-a", "source-b"],
  eligibleSelectedSourceIds: ["source-a"],
  totalSelectedCount: 2,
  hiddenSelectedCount: 1,
  ineligibleSelectedCount: 1
}

const launchConflictTransfer = {
  entryPoint: "sources" as const,
  selectedSourceIds: ["source-a", "source-b"],
  eligibleSelectedSourceIds: ["source-a", "source-b"],
  totalSelectedCount: 2,
  hiddenSelectedCount: 0,
  ineligibleSelectedCount: 0
}

let nextLaunchPayload = launchHiddenSelectionTransfer

vi.mock("../SourcesPane", () => ({
  SourcesPane: (props: {
    onOpenTransferSources?: (payload: typeof launchHiddenSelectionTransfer) => void
  }) => (
    <div data-testid="workspace-sources-pane">
      <button
        type="button"
        onClick={() => props.onOpenTransferSources?.(nextLaunchPayload)}
      >
        Launch transfer
      </button>
    </div>
  )
}))

vi.mock("../ChatPane", () => ({
  ChatPane: () => <div data-testid="workspace-chat-pane">Chat</div>
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => <div data-testid="workspace-status-bar" />
}))

if (!(globalThis as { ResizeObserver?: unknown }).ResizeObserver) {
  ;(globalThis as { ResizeObserver?: typeof ResizeObserver }).ResizeObserver =
    class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as typeof ResizeObserver
}

describe("ResearchWorkspace stage 13 source transfer", () => {
  const originalMatchMedia = window.matchMedia
  const mockUndoSnapshot = {
    workspaceId: "workspace-current",
    workspaceName: "Current Workspace",
    sources: [],
    sourceFolders: [],
    sourceFolderMemberships: []
  }

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStorageItems.clear()
    clearWorkspaceUndoActionsForTests()
    testState.isMobile = false
    testState.storeHydrated = true
    testState.workspaceId = "workspace-current"
    testState.workspaceName = "Current Workspace"
    testState.leftPaneCollapsed = false
    testState.selectedSourceIds = ["source-a", "source-b"]
    nextLaunchPayload = launchHiddenSelectionTransfer
    mockCaptureUndoSnapshot.mockReturnValue(mockUndoSnapshot)
    mockTransferSourcesBetweenWorkspaces.mockReturnValue({
      originWorkspaceId: "workspace-current",
      destinationWorkspaceId,
      destinationWasCreated: false,
      transferredMediaIds: [101, 102],
      transferredDestinationSourceIds: [
        "destination-source-a",
        "destination-source-b"
      ],
      removedOriginSourceIds: ["source-a", "source-b"],
      newlyEmptiedOriginFolderIds: ["folder-empty"],
      conflictsResolved: [101, 102],
      conflictsSkipped: [],
      originSnapshot: {
        workspaceId: "workspace-current",
        sources: [],
        sourceFolders: [],
        sourceFolderMemberships: []
      },
      destinationSnapshot: {
        workspaceId: destinationWorkspaceId,
        sources: [],
        sourceFolders: [],
        sourceFolderMemberships: []
      }
    })
  })

  afterEach(() => {
    clearWorkspaceUndoActionsForTests()
  })

  it("shows hidden and ineligible summaries and excludes current and archived workspaces from the destination picker", async () => {
    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Launch transfer" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    expect(
      screen.getByRole("radio", { name: "Destination Workspace" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("radio", { name: "Current Workspace" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("radio", { name: "Archived Workspace" })
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "Destination Workspace" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    expect(screen.getByText("Selected: 2")).toBeInTheDocument()
    expect(
      screen.getByText(/selected sources are hidden by current filters/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/processing or errored sources are excluded/i)
    ).toBeInTheDocument()
  })

  it("opens split mode from the workspace settings shortcut in create new workspace mode", async () => {
    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Split workspace" }))

    expect(
      await screen.findByRole("dialog", { name: "Transfer sources" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("radio", { name: "Create a new workspace" })
    ).toBeChecked()
    expect(screen.getByPlaceholderText("New Research")).toBeInTheDocument()
    expect(mockMessageApi.info).not.toHaveBeenCalled()
  })

  it("shows the ineligible summary for processing selections from the header path", async () => {
    testState.selectedSourceIds = ["source-a", "source-c"]

    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Split workspace" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    expect(
      screen.getByText(/1 ready sources will transfer/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/1 processing or errored sources are excluded/i)
    ).toBeInTheDocument()
    expect(mockMessageApi.info).not.toHaveBeenCalled()
  })

  it("treats an all-ineligible header selection like no transferable selection", async () => {
    const originalMatchMedia = window.matchMedia
    try {
      window.matchMedia = vi.fn().mockImplementation((query: string) => ({
        matches: true,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))

      testState.selectedSourceIds = ["source-c"]
      testState.leftPaneCollapsed = true

      render(<ResearchWorkspace />)

      fireEvent.click(
        await screen.findByRole("button", { name: "Split workspace" })
      )

      expect(mockSetLeftPaneCollapsed).toHaveBeenCalledWith(false)
      expect(mockMessageApi.info).toHaveBeenCalledTimes(1)
      expect(screen.queryByRole("dialog", { name: "Transfer sources" })).toBeNull()
    } finally {
      window.matchMedia = originalMatchMedia
    }
  })

  it("reveals the Sources pane and shows an info message when nothing is selected on desktop", async () => {
    const originalMatchMedia = window.matchMedia
    try {
      window.matchMedia = vi.fn().mockImplementation((query: string) => ({
        matches: true,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))

      testState.selectedSourceIds = []
      testState.leftPaneCollapsed = true

      render(<ResearchWorkspace />)

      fireEvent.click(
        await screen.findByRole("button", { name: "Split workspace" })
      )

      expect(mockSetLeftPaneCollapsed).toHaveBeenCalledWith(false)
      expect(mockMessageApi.info).toHaveBeenCalledTimes(1)
      expect(screen.queryByRole("dialog", { name: "Transfer sources" })).toBeNull()
    } finally {
      window.matchMedia = originalMatchMedia
    }
  })

  it("moves to the Sources tab instead of doing nothing on mobile when nothing is selected", async () => {
    testState.isMobile = true
    testState.selectedSourceIds = []

    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Split workspace" }))

    expect(mockMessageApi.info).toHaveBeenCalledTimes(1)
    expect(await screen.findByTestId("workspace-sources-pane")).toBeInTheDocument()
    expect(screen.queryByRole("dialog", { name: "Transfer sources" })).toBeNull()
  })

  it("collects conflict resolutions, move cleanup policy, and rewinds the modal state when the transfer is undone", async () => {
    nextLaunchPayload = launchConflictTransfer

    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Launch transfer" }))
    fireEvent.click(screen.getByRole("radio", { name: "Move selected sources" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("radio", { name: "Destination Workspace" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    expect(screen.getByText("Alpha Source")).toBeInTheDocument()
    expect(screen.getByText("Beta Source")).toBeInTheDocument()
    expect(
      screen.getByRole("checkbox", {
        name: "Apply to all remaining conflicts"
      })
    ).toBeInTheDocument()

    const alphaConflictCard = screen.getByText("Alpha Source").closest("div")
    expect(alphaConflictCard).not.toBeNull()
    expect(
      within(alphaConflictCard as HTMLElement).getByRole("radio", {
        name: "Skip"
      })
    ).toBeInTheDocument()
    expect(
      within(alphaConflictCard as HTMLElement).getByRole("radio", {
        name: "Merge folder memberships"
      })
    ).toBeInTheDocument()
    expect(
      within(alphaConflictCard as HTMLElement).getByRole("radio", {
        name: "Replace transferred folder memberships"
      })
    ).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Apply to all remaining conflicts"
      })
    )
    fireEvent.click(
      within(alphaConflictCard as HTMLElement).getByRole("radio", {
        name: "Merge folder memberships"
      })
    )
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    expect(
      screen.getByRole("radio", { name: "Keep empty folders" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("radio", { name: "Delete emptied folders" })
    ).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("radio", { name: "Delete emptied folders" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Transfer sources" }))

    expect(mockTransferSourcesBetweenWorkspaces).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "move",
        destination: {
          kind: "existing",
          workspaceId: destinationWorkspaceId
        },
        selectedSourceIds: ["source-a", "source-b"],
        emptyFolderPolicy: "delete-empty-folders",
        conflictResolutions: {
          101: "merge-folders",
          102: "merge-folders"
        }
      })
    )
    expect(mockCaptureUndoSnapshot).toHaveBeenCalledTimes(1)
    expect(mockMessageApi.open).toHaveBeenCalledTimes(1)

    const undoMessageConfig = mockMessageApi.open.mock.calls[0]?.[0] as
      | { btn?: React.ReactElement; key?: string; type?: string }
      | undefined
    expect(undoMessageConfig?.type).toBe("warning")
    expect(undoMessageConfig?.btn).toBeTruthy()

    if (React.isValidElement(undoMessageConfig?.btn)) {
      act(() => {
        undoMessageConfig.btn?.props?.onClick?.()
      })
    } else {
      throw new Error("Expected the transfer undo button to be rendered")
    }

    expect(mockRestoreUndoSnapshot).toHaveBeenCalledWith(mockUndoSnapshot)
    expect(mockMessageApi.success).toHaveBeenCalledWith("Transfer undone.")
    expect(mockMessageApi.destroy).toHaveBeenCalledWith(undoMessageConfig?.key)
    expect(
      screen.queryByRole("button", { name: "Open destination" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("radio", { name: "Delete emptied folders" })
    ).toBeChecked()
  })

  it("offers an open-destination follow-up for existing workspaces after a successful transfer", async () => {
    nextLaunchPayload = launchConflictTransfer

    render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Launch transfer" }))
    fireEvent.click(screen.getByRole("radio", { name: "Move selected sources" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("radio", { name: "Destination Workspace" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Apply to all remaining conflicts"
      })
    )
    fireEvent.click(
      within(
        screen.getByText("Alpha Source").closest("div") as HTMLElement
      ).getByRole("radio", {
        name: "Merge folder memberships"
      })
    )
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(
      screen.getByRole("radio", { name: "Delete emptied folders" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Transfer sources" }))

    expect(
      screen.getByRole("button", { name: "Open destination" })
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open destination" }))

    expect(mockSwitchWorkspace).toHaveBeenCalledWith(destinationWorkspaceId)
  })

  it("keeps the success state visible after destination metadata changes and rerender", async () => {
    nextLaunchPayload = launchConflictTransfer
    const { rerender } = render(<ResearchWorkspace />)

    fireEvent.click(await screen.findByRole("button", { name: "Launch transfer" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("radio", { name: "Destination Workspace" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Apply to all remaining conflicts"
      })
    )
    fireEvent.click(
      within(
        screen.getByText("Alpha Source").closest("div") as HTMLElement
      ).getByRole("radio", {
        name: "Merge folder memberships"
      })
    )
    fireEvent.click(screen.getByRole("button", { name: "Transfer sources" }))

    expect(
      screen.getByRole("button", { name: "Open destination" })
    ).toBeInTheDocument()

    testState.savedWorkspaces = testState.savedWorkspaces.map((workspace) =>
      workspace.id === destinationWorkspaceId
        ? {
            ...workspace,
            name: "Destination Workspace Renamed",
            sourceCount: 3,
            lastAccessedAt: new Date("2026-03-29T00:00:00.000Z")
          }
        : workspace
    )
    testState.workspaceSnapshots = {
      ...testState.workspaceSnapshots,
      [destinationWorkspaceId]: {
        ...(testState.workspaceSnapshots[destinationWorkspaceId] as Record<
          string,
          unknown
        >),
        sources: [
          ...(
            (
              testState.workspaceSnapshots[destinationWorkspaceId] as {
                sources?: unknown[]
              }
            )?.sources || []
          ),
          {
            id: "destination-source-c",
            mediaId: 103,
            title: "Gamma Source",
            type: "pdf",
            status: "ready",
            addedAt: new Date("2026-03-29T00:00:00.000Z")
          }
        ]
      }
    }

    rerender(<ResearchWorkspace />)

    expect(
      screen.getByRole("button", { name: "Open destination" })
    ).toBeInTheDocument()
  })
})
