import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { ResearchWorkspace } from "../index"

const savedViewMocks = vi.hoisted(() => ({
  useSourceSavedViews: vi.fn(),
  controller: {
    available: true,
    generation: 7,
    views: [],
    loading: false,
    listError: null,
    activeViewId: null,
    activeSnapshot: null,
    currentSignature: null,
    modified: false,
    serializationIssues: [],
    duplicateConflict: null,
    limitState: null,
    versionConflict: null,
    mutation: null,
    busy: false,
    mutationError: null,
    announcement: null,
    canRetryMutation: false,
    canRetryVersion: false,
    load: vi.fn(),
    retry: vi.fn(),
    retryMutation: vi.fn(),
    retryVersionConflict: vi.fn(),
    applyView: vi.fn(),
    createView: vi.fn(),
    confirmReplace: vi.fn(),
    replaceView: vi.fn(),
    resetView: vi.fn(),
    deleteView: vi.fn()
  }
}))

const testState = {
  isMobile: false,
  storeHydrated: true,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  workspaceId: "workspace-1",
  initializeWorkspace: vi.fn(),
  addSources: vi.fn(),
  setSelectedSourceIds: vi.fn(),
  captureToCurrentNote: vi.fn(),
  setLeftPaneCollapsed: vi.fn(),
  setRightPaneCollapsed: vi.fn(),
  selectedSourceIds: [] as string[],
  generatedArtifacts: [] as Array<{ id: string }>,
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    addedAt: Date
  }>,
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
  focusSourceById: vi.fn(),
  focusChatMessageById: vi.fn(),
  focusWorkspaceNote: vi.fn(),
  setSourceStatusByMediaId: vi.fn()
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
  createWorkspaceStorage: () => ({
    getItem: vi.fn().mockResolvedValue("1")
  }),
  useWorkspaceStore: (
    selector: (state: {
      storeHydrated?: boolean
      workspaceId: string | null
      initializeWorkspace: () => void
      addSources: (
        sources: Array<{ mediaId: number; title: string; type: string }>
      ) => unknown
      setSelectedSourceIds: (ids: string[]) => void
      captureToCurrentNote: (input: {
        title?: string
        content: string
        mode?: "append" | "replace"
      }) => void
      leftPaneCollapsed: boolean
      rightPaneCollapsed: boolean
      setLeftPaneCollapsed: (collapsed: boolean) => void
      setRightPaneCollapsed: (collapsed: boolean) => void
      selectedSourceIds: string[]
      generatedArtifacts: Array<{ id: string }>
      sources: Array<{
        id: string
        mediaId: number
        title: string
        type: "pdf" | "video" | "audio" | "website" | "document" | "text"
        addedAt: Date
      }>
      currentNote: {
        title: string
        content: string
        keywords: string[]
        isDirty: boolean
      }
      workspaceChatSessions: Record<
        string,
        { messages: Array<{ message: string; sources: unknown[]; isBot: boolean; name: string }> }
      >
      focusSourceById: (id: string) => boolean
      focusChatMessageById: (messageId: string) => boolean
      focusWorkspaceNote: (field?: "title" | "content") => void
      setSourceStatusByMediaId: (
        mediaId: number,
        status: "processing" | "ready" | "error",
        statusMessage?: string
      ) => void
    }) => unknown
  ) =>
    selector({
      storeHydrated: testState.storeHydrated,
      workspaceId: testState.workspaceId,
      initializeWorkspace: testState.initializeWorkspace,
      addSources: testState.addSources,
      setSelectedSourceIds: testState.setSelectedSourceIds,
      captureToCurrentNote: testState.captureToCurrentNote,
      leftPaneCollapsed: testState.leftPaneCollapsed,
      rightPaneCollapsed: testState.rightPaneCollapsed,
      setLeftPaneCollapsed: testState.setLeftPaneCollapsed,
      setRightPaneCollapsed: testState.setRightPaneCollapsed,
      selectedSourceIds: testState.selectedSourceIds,
      generatedArtifacts: testState.generatedArtifacts,
      sources: testState.sources,
      currentNote: testState.currentNote,
      workspaceChatSessions: testState.workspaceChatSessions,
      focusSourceById: testState.focusSourceById,
      focusChatMessageById: testState.focusChatMessageById,
      focusWorkspaceNote: testState.focusWorkspaceNote,
      setSourceStatusByMediaId: testState.setSourceStatusByMediaId
    })
}))

vi.mock("@/utils/research-workspace-prefill", () => ({
  consumeResearchWorkspacePrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue("")
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("../SourcesPane/use-source-saved-views", () => ({
  useSourceSavedViews: savedViewMocks.useSourceSavedViews
}))

vi.mock("../SourcesPane/SourceViewControls", () => ({
  SourceViewOverlayHost: (props: {
    controller: unknown
    request: { kind: string } | null
  }) => (
    <div data-testid="source-view-overlay-host">
      {props.request && (
        <div role="dialog" aria-label="Source view overlay">
          {props.request.kind}
        </div>
      )}
    </div>
  )
}))

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: (props: { onToggleLeftPane: () => void }) => (
    <div data-testid="workspace-header">
      <button type="button" onClick={props.onToggleLeftPane}>
        Toggle sources
      </button>
    </div>
  )
}))

vi.mock("../SourcesPane", () => ({
  SourcesPane: (props: {
    sourceListViewState?: { sort?: string }
    onPatchSourceListViewState?: (patch: { sort: string }) => void
    sourceSavedViewsController?: unknown
    onOpenSourceViewOverlay?: (request: {
      id: number
      kind: "save"
      generation: number
      invoker: HTMLElement
    }) => void
  }) => (
    <div data-testid="workspace-sources-pane">
      <div data-testid="source-list-sort-state">
        {props.sourceListViewState?.sort ?? "missing"}
      </div>
      <button
        type="button"
        onClick={() => props.onPatchSourceListViewState?.({ sort: "name_asc" })}
      >
        Patch source list sort
      </button>
      <button
        type="button"
        aria-label="Open mock saved view"
        onClick={(event) =>
          props.onOpenSourceViewOverlay?.({
            id: 1,
            kind: "save",
            generation: 7,
            invoker: event.currentTarget
          })
        }
      >
        Open saved view
      </button>
      <div data-testid="saved-view-controller-identity">
        {props.sourceSavedViewsController === savedViewMocks.controller
          ? "shared"
          : "different"}
      </div>
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

describe("ResearchWorkspace source list view state", () => {
  const originalMatchMedia = window.matchMedia

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
    testState.isMobile = false
    testState.storeHydrated = true
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.workspaceId = "workspace-1"
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.sources = []
    testState.currentNote = {
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }
    testState.workspaceChatSessions = {}
    testState.setSourceStatusByMediaId = vi.fn()
    savedViewMocks.useSourceSavedViews.mockReset()
    savedViewMocks.useSourceSavedViews.mockReturnValue(savedViewMocks.controller)
  })

  it("preserves source list view state across sources pane remounts", async () => {
    const { rerender } = render(<ResearchWorkspace />)

    expect(await screen.findByTestId("source-list-sort-state")).toHaveTextContent(
      "manual"
    )

    fireEvent.click(screen.getByRole("button", { name: "Patch source list sort" }))

    expect(screen.getByTestId("source-list-sort-state")).toHaveTextContent("name_asc")

    testState.leftPaneCollapsed = true
    rerender(<ResearchWorkspace />)
    expect(screen.queryByTestId("workspace-sources-pane")).not.toBeInTheDocument()

    testState.leftPaneCollapsed = false
    rerender(<ResearchWorkspace />)
    expect(await screen.findByTestId("source-list-sort-state")).toHaveTextContent(
      "name_asc"
    )
  })

  it("shares one saved-view controller and one overlay host across desktop and drawer panes", async () => {
    testState.isMobile = true
    const { rerender } = render(<ResearchWorkspace />)
    fireEvent.click(screen.getByRole("button", { name: "Toggle sources" }))

    testState.isMobile = false
    rerender(<ResearchWorkspace />)

    const panes = await screen.findAllByTestId("workspace-sources-pane")
    expect(panes).toHaveLength(2)
    expect(screen.getAllByTestId("saved-view-controller-identity")).toHaveLength(2)
    expect(
      screen.getAllByTestId("saved-view-controller-identity").every(
        (node) => node.textContent === "shared"
      )
    ).toBe(true)
    expect(screen.getAllByTestId("source-view-overlay-host")).toHaveLength(1)

    fireEvent.click(screen.getAllByRole("button", { name: "Open mock saved view" })[1]!)
    expect(
      screen.getAllByRole("dialog", { name: "Source view overlay" })
    ).toHaveLength(1)
  })

  it("passes the raw nullable workspace identity to the single page controller", () => {
    const { rerender } = render(<ResearchWorkspace />)
    expect(savedViewMocks.useSourceSavedViews).toHaveBeenCalledWith(
      "workspace-1",
      expect.any(Object),
      expect.any(Function)
    )

    testState.workspaceId = null
    rerender(<ResearchWorkspace />)

    expect(savedViewMocks.useSourceSavedViews).toHaveBeenLastCalledWith(
      null,
      expect.any(Object),
      expect.any(Function)
    )
    expect(savedViewMocks.useSourceSavedViews).not.toHaveBeenCalledWith(
      "local",
      expect.anything(),
      expect.anything()
    )
  })
})
