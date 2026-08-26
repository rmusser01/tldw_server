import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSourceSavedViewResponse } from "@/services/tldw/domains/workspace-api"
import type { WorkspaceSourceSavedViewStateV1 } from "@/types/workspace-source-saved-view"
import { ResearchWorkspace } from "../index"
import type { SourceListViewState } from "../SourcesPane/source-list-view"
import type { SourceSavedViewsController } from "../SourcesPane/use-source-saved-views"

const savedViewHarness = vi.hoisted(() => ({
  hookInvocations: vi.fn(),
  paneControllers: [] as unknown[],
  controllerIds: new WeakMap<object, number>(),
  nextControllerId: 1,
  rows: [] as WorkspaceSourceSavedViewResponse[],
  nextId: 1,
  upsertWorkspace: vi.fn(),
  getWorkspaceSources: vi.fn(),
  addWorkspaceSource: vi.fn(),
  listWorkspaceSourceViews: vi.fn(),
  createWorkspaceSourceView: vi.fn(),
  updateWorkspaceSourceView: vi.fn(),
  deleteWorkspaceSourceView: vi.fn()
}))

const wireState = (
  overrides: Partial<WorkspaceSourceSavedViewStateV1> = {}
): WorkspaceSourceSavedViewStateV1 => ({
  type_filters: [],
  status_filters: [],
  review_state_filters: [],
  lifecycle_state_filters: [],
  date_field: "added_at",
  date_from: null,
  date_to: null,
  require_url: false,
  require_file_size: false,
  require_duration: false,
  require_page_count: false,
  file_size_min: null,
  file_size_max: null,
  duration_min: null,
  duration_max: null,
  page_count_min: null,
  page_count_max: null,
  sort: "manual",
  ...overrides
})

const validView = (
  workspaceId: string,
  overrides: Partial<WorkspaceSourceSavedViewResponse> = {}
): WorkspaceSourceSavedViewResponse => ({
  id: `${workspaceId}-view-${savedViewHarness.nextId}`,
  workspace_id: workspaceId,
  name: "Current filters",
  schema_version: 1,
  version: 1,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
  valid: true,
  invalid_reason: null,
  state: wireState(),
  ...overrides
})

const latestController = (): SourceSavedViewsController =>
  savedViewHarness.paneControllers.at(-1)! as SourceSavedViewsController

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

const openSavedViewCommands = async (
  user: ReturnType<typeof userEvent.setup>,
  viewName: string
) => {
  const submenu = await screen.findByRole("menuitem", {
    name: new RegExp(`^${viewName}`)
  })
  await user.hover(submenu)
  await screen.findByRole("menuitem", {
    name: `Apply saved view ${viewName}`
  })
}

const expectDialogClosingOrGone = (name: string) => {
  const dialog = screen.queryByRole("dialog", { name })
  if (dialog) expect(dialog).toHaveClass("ant-zoom-leave")
}

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
  sourceSearchQuery: "",
  activeFolderId: null as string | null,
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
      sourceSearchQuery: string
      activeFolderId: string | null
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
      sourceSearchQuery: testState.sourceSearchQuery,
      activeFolderId: testState.activeFolderId,
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
    getMediaDetails: vi.fn().mockResolvedValue({}),
    upsertWorkspace: savedViewHarness.upsertWorkspace,
    getWorkspaceSources: savedViewHarness.getWorkspaceSources,
    addWorkspaceSource: savedViewHarness.addWorkspaceSource,
    listWorkspaceSourceViews: savedViewHarness.listWorkspaceSourceViews,
    createWorkspaceSourceView: savedViewHarness.createWorkspaceSourceView,
    updateWorkspaceSourceView: savedViewHarness.updateWorkspaceSourceView,
    deleteWorkspaceSourceView: savedViewHarness.deleteWorkspaceSourceView
  }
}))

vi.mock("../SourcesPane/use-source-saved-views", async () => {
  const actual = await vi.importActual<
    typeof import("../SourcesPane/use-source-saved-views")
  >("../SourcesPane/use-source-saved-views")
  return {
    ...actual,
    useSourceSavedViews: (...args: Parameters<typeof actual.useSourceSavedViews>) => {
      savedViewHarness.hookInvocations(...args)
      return actual.useSourceSavedViews(...args)
    }
  }
})

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: (props: { onToggleLeftPane: () => void }) => (
    <div data-testid="workspace-header">
      <button type="button" onClick={props.onToggleLeftPane}>
        Toggle sources
      </button>
    </div>
  )
}))

vi.mock("../SourcesPane", async () => {
  const { SourceViewControls } = await vi.importActual<
    typeof import("../SourcesPane/SourceViewControls")
  >("../SourcesPane/SourceViewControls")
  const { useWorkspaceStore } = await import("@/store/workspace")
  return {
    SourcesPane: (props: {
      sourceListViewState?: SourceListViewState
      onPatchSourceListViewState?: (patch: Partial<SourceListViewState>) => void
      onApplySourceListViewState?: (state: SourceListViewState) => void
      sourceSavedViewsController?: SourceSavedViewsController
      onOpenSourceViewOverlay?: React.ComponentProps<
        typeof SourceViewControls
      >["onOpenOverlay"]
    }) => {
      const observedSourceSearchQuery = useWorkspaceStore(
        (state) => state.sourceSearchQuery
      )
      const observedActiveFolderId = useWorkspaceStore(
        (state) => state.activeFolderId
      )
      if (props.sourceSavedViewsController) {
        savedViewHarness.paneControllers.push(props.sourceSavedViewsController)
      }
      const controllerId = props.sourceSavedViewsController
        ? savedViewHarness.controllerIds.get(props.sourceSavedViewsController) ??
          savedViewHarness.nextControllerId++
        : null
      if (
        props.sourceSavedViewsController &&
        !savedViewHarness.controllerIds.has(props.sourceSavedViewsController)
      ) {
        savedViewHarness.controllerIds.set(
          props.sourceSavedViewsController,
          controllerId!
        )
      }
      return (
        <div
        data-testid="workspace-sources-pane"
        data-sources-focus-target
        role="region"
        aria-label="Sources"
        tabIndex={-1}
      >
        <div data-testid="source-list-sort-state">
          {props.sourceListViewState?.sort ?? "missing"}
        </div>
        <div data-testid="source-list-type-state">
          {props.sourceListViewState?.typeFilters.join(",") || "none"}
        </div>
        <div data-testid="source-list-expanded-state">
          {props.sourceListViewState?.expanded ? "expanded" : "collapsed"}
        </div>
        <div data-testid="workspace-source-search-state">
          {observedSourceSearchQuery || "none"}
        </div>
        <div data-testid="workspace-active-folder-state">
          {observedActiveFolderId || "none"}
        </div>
        <button
          type="button"
          onClick={() => props.onPatchSourceListViewState?.({ sort: "name_asc" })}
        >
          Patch source list sort
        </button>
        <button
          type="button"
          onClick={() =>
            props.onPatchSourceListViewState?.({
              expanded: true,
              typeFilters: ["website"],
              sort: "name_asc"
            })
          }
        >
          Set current filters
        </button>
        <button
          type="button"
          onClick={() => props.onPatchSourceListViewState?.({ expanded: true })}
        >
          Expand source filters
        </button>
        {props.sourceSavedViewsController &&
          props.sourceListViewState &&
          props.onApplySourceListViewState &&
          props.onOpenSourceViewOverlay && (
            <SourceViewControls
              controller={props.sourceSavedViewsController}
              sourceListViewState={props.sourceListViewState}
              onApplySourceListViewState={props.onApplySourceListViewState}
              onOpenOverlay={props.onOpenSourceViewOverlay}
            />
          )}
        <div
          data-testid="saved-view-controller-identity"
          data-controller-id={controllerId ?? ""}
        >
          {props.sourceSavedViewsController ? "captured" : "missing"}
        </div>
      </div>
      )
    }
  }
})

vi.mock("../ChatPane", () => ({
  ChatPane: () => <div data-testid="workspace-chat-pane">Chat</div>
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: (props: {
    workspaceContextStatus?: { detail?: string } | null
  }) => (
    <div data-testid="workspace-status-bar">
      {props.workspaceContextStatus?.detail || ""}
    </div>
  )
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
    testState.sourceSearchQuery = ""
    testState.activeFolderId = null
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
    savedViewHarness.nextId = 1
    savedViewHarness.rows = []
    savedViewHarness.paneControllers = []
    savedViewHarness.controllerIds = new WeakMap<object, number>()
    savedViewHarness.nextControllerId = 1
    savedViewHarness.hookInvocations.mockClear()
    savedViewHarness.upsertWorkspace.mockImplementation(
      async (workspaceId: string) => ({ id: workspaceId })
    )
    savedViewHarness.getWorkspaceSources.mockResolvedValue([])
    savedViewHarness.addWorkspaceSource.mockResolvedValue({})
    savedViewHarness.listWorkspaceSourceViews.mockImplementation(
      async (workspaceId: string) => ({
        items: savedViewHarness.rows.filter(
          (view) => view.workspace_id === workspaceId
        )
      })
    )
    savedViewHarness.createWorkspaceSourceView.mockImplementation(
      async (
        workspaceId: string,
        body: {
          name: string
          schema_version: number
          state: WorkspaceSourceSavedViewStateV1
        }
      ) => {
        const view = validView(workspaceId, {
          id: `${workspaceId}-view-${savedViewHarness.nextId}`,
          name: body.name,
          schema_version: body.schema_version,
          state: body.state
        })
        savedViewHarness.nextId += 1
        savedViewHarness.rows = [view, ...savedViewHarness.rows]
        return view
      }
    )
    savedViewHarness.updateWorkspaceSourceView.mockImplementation(
      async (
        workspaceId: string,
        viewId: string,
        body: {
          name?: string
          schema_version: number
          state: WorkspaceSourceSavedViewStateV1
        }
      ) => {
        const existing = savedViewHarness.rows.find(
          (view) => view.workspace_id === workspaceId && view.id === viewId
        )
        if (!existing) throw new Error("Saved view not found")
        const updated = validView(workspaceId, {
          ...existing,
          name: body.name ?? existing.name,
          schema_version: body.schema_version,
          state: body.state,
          version: existing.version + 1
        })
        savedViewHarness.rows = savedViewHarness.rows.map((view) =>
          view.id === viewId ? updated : view
        )
        return updated
      }
    )
    savedViewHarness.deleteWorkspaceSourceView.mockImplementation(
      async (workspaceId: string, viewId: string) => {
        savedViewHarness.rows = savedViewHarness.rows.filter(
          (view) => view.workspace_id !== workspaceId || view.id !== viewId
        )
      }
    )
  })

  it("loads saved views after upsert but before source reconciliation finishes", async () => {
    const upsert = deferred<{ id: string }>()
    const sourceRows = deferred<never[]>()
    savedViewHarness.upsertWorkspace.mockReturnValue(upsert.promise)
    savedViewHarness.getWorkspaceSources.mockReturnValue(sourceRows.promise)

    render(<ResearchWorkspace />)

    await waitFor(() =>
      expect(savedViewHarness.upsertWorkspace).toHaveBeenCalledWith(
        "workspace-1",
        expect.any(Object)
      )
    )
    expect(savedViewHarness.listWorkspaceSourceViews).not.toHaveBeenCalled()

    upsert.resolve({ id: "workspace-1" })

    await waitFor(() =>
      expect(savedViewHarness.listWorkspaceSourceViews).toHaveBeenCalledWith(
        "workspace-1"
      )
    )
    expect(savedViewHarness.getWorkspaceSources).toHaveBeenCalledWith(
      "workspace-1"
    )

    sourceRows.resolve([])
  })

  it("does not let a late workspace A upsert unlock saved views for workspace B", async () => {
    const upsertA = deferred<{ id: string }>()
    const upsertB = deferred<{ id: string }>()
    savedViewHarness.upsertWorkspace.mockImplementation(
      (workspaceId: string) =>
        workspaceId === "workspace-1" ? upsertA.promise : upsertB.promise
    )

    const { rerender } = render(<ResearchWorkspace />)
    await waitFor(() =>
      expect(savedViewHarness.upsertWorkspace).toHaveBeenCalledWith(
        "workspace-1",
        expect.any(Object)
      )
    )

    testState.workspaceId = "workspace-2"
    rerender(<ResearchWorkspace />)
    await waitFor(() =>
      expect(savedViewHarness.upsertWorkspace).toHaveBeenCalledWith(
        "workspace-2",
        expect.any(Object)
      )
    )

    await act(async () => upsertA.resolve({ id: "workspace-1" }))
    expect(savedViewHarness.listWorkspaceSourceViews).not.toHaveBeenCalled()
    expect(savedViewHarness.hookInvocations).not.toHaveBeenCalledWith(
      "workspace-2",
      true,
      expect.anything(),
      expect.anything()
    )

    await act(async () => upsertB.resolve({ id: "workspace-2" }))
    await waitFor(() =>
      expect(savedViewHarness.listWorkspaceSourceViews).toHaveBeenCalledWith(
        "workspace-2"
      )
    )
  })

  it("keeps saved views blocked and shows workspace recovery when upsert fails", async () => {
    savedViewHarness.upsertWorkspace.mockRejectedValue(
      new Error("workspace parent unavailable")
    )

    render(<ResearchWorkspace />)

    await waitFor(() =>
      expect(screen.getAllByTestId("workspace-status-bar")[0]).toHaveTextContent(
        "Workspace server sync unavailable"
      )
    )
    expect(savedViewHarness.listWorkspaceSourceViews).not.toHaveBeenCalled()
    expect(screen.queryByText(/Source view not found/i)).not.toBeInTheDocument()
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

  it("shares one controller and one real overlay host across desktop and drawer panes", async () => {
    const user = userEvent.setup()
    testState.isMobile = true
    const { rerender } = render(<ResearchWorkspace />)
    expect(
      savedViewHarness.hookInvocations.mock.calls.every(
        ([workspaceId]) => workspaceId === "workspace-1"
      )
    ).toBe(true)
    await user.click(screen.getByRole("button", { name: "Toggle sources" }))

    testState.isMobile = false
    rerender(<ResearchWorkspace />)

    const controllerIdentities = await screen.findAllByTestId(
      "saved-view-controller-identity"
    )
    expect(controllerIdentities).toHaveLength(2)
    const panes = screen.getAllByTestId("workspace-sources-pane")
    expect(panes).toHaveLength(2)
    const simultaneousControllerIds = controllerIdentities.map((identity) =>
      identity.getAttribute("data-controller-id")
    )
    expect(new Set(simultaneousControllerIds)).toEqual(
      new Set([simultaneousControllerIds[0]])
    )
    expect(screen.getAllByTestId("source-view-overlay-host")).toHaveLength(1)

    const invokers = screen.getAllByRole("button", { name: "Save source view" })
    expect(invokers).toHaveLength(2)
    await user.click(invokers[0]!)
    expect(screen.getAllByRole("textbox", { name: "View name" })).toHaveLength(1)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(document.activeElement).toBe(invokers[0]))

    await user.click(invokers[1]!)
    expect(screen.getAllByRole("textbox", { name: "View name" })).toHaveLength(1)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(document.activeElement).toBe(invokers[1]))
  })

  it("passes the raw nullable workspace identity and readiness to the single page controller", async () => {
    const { rerender } = render(<ResearchWorkspace />)
    await waitFor(() =>
      expect(savedViewHarness.hookInvocations).toHaveBeenCalledWith(
        "workspace-1",
        true,
        expect.any(Object),
        expect.any(Function)
      )
    )

    testState.workspaceId = null
    rerender(<ResearchWorkspace />)

    expect(savedViewHarness.hookInvocations).toHaveBeenLastCalledWith(
      null,
      false,
      expect.any(Object),
      expect.any(Function)
    )
    expect(savedViewHarness.hookInvocations).not.toHaveBeenCalledWith(
      "local",
      expect.anything(),
      expect.anything(),
      expect.anything()
    )
    expect(savedViewHarness.listWorkspaceSourceViews).not.toHaveBeenCalledWith(
      "local"
    )
    expect(screen.getByRole("button", { name: "Save source view" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    expect(screen.queryByRole("dialog", { name: "Save source view" })).not.toBeInTheDocument()
  })

  it.each([
    { label: "workspace B", nextWorkspaceId: "workspace-2" },
    { label: "no workspace", nextWorkspaceId: null }
  ])(
    "synchronously closes an A Save overlay for $label and discards its draft",
    async ({ nextWorkspaceId }) => {
      const user = userEvent.setup()
      const { rerender } = render(<ResearchWorkspace />)
      await user.click(screen.getByRole("button", { name: "Save source view" }))
      const dialog = await screen.findByRole("dialog", { name: "Save source view" })
      await user.type(within(dialog).getByRole("textbox", { name: "View name" }), "Draft A")

      testState.workspaceId = nextWorkspaceId
      rerender(<ResearchWorkspace />)

      expectDialogClosingOrGone("Save source view")
      const closingDialog = screen.queryByRole("dialog", {
        name: "Save source view"
      })
      if (closingDialog) {
        expect(
          within(closingDialog).getByRole("button", { name: "Save" })
        ).toBeDisabled()
      }
      expect(savedViewHarness.createWorkspaceSourceView).not.toHaveBeenCalled()
    }
  )

  it.each([
    { label: "workspace B", nextWorkspaceId: "workspace-2" },
    { label: "no workspace", nextWorkspaceId: null }
  ])(
    "discards an A replacement confirmation for $label",
    async ({ nextWorkspaceId }) => {
      const user = userEvent.setup()
      savedViewHarness.rows = [validView("workspace-1")]
      const { rerender } = render(<ResearchWorkspace />)
      await user.click(screen.getByRole("button", { name: "Source views" }))
      await openSavedViewCommands(user, "Current filters")
      await user.click(
        screen.getByRole("menuitem", {
          name: "Replace saved view Current filters"
        })
      )
      await screen.findByRole("dialog", {
        name: "Replace saved view?"
      })

      testState.workspaceId = nextWorkspaceId
      rerender(<ResearchWorkspace />)

      expectDialogClosingOrGone("Replace saved view?")
      expect(savedViewHarness.updateWorkspaceSourceView).not.toHaveBeenCalled()
    }
  )

  it("restores focus to the real Sources tab after the invoking mobile pane unmounts", async () => {
    const user = userEvent.setup()
    testState.isMobile = true
    render(<ResearchWorkspace />)
    await user.click(screen.getByRole("tab", { name: /Sources/ }))
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    expect(await screen.findByRole("dialog", { name: "Save source view" })).toBeInTheDocument()

    const sourcesTab = screen.getByRole("tab", { name: /Sources/ })
    await user.click(screen.getByRole("tab", { name: /Chat/ }))
    expect(screen.queryByRole("button", { name: "Save source view" })).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => expect(document.activeElement).toBe(sourcesTab))
  })

  it("persists a created view through the server boundary and restores it only when reselected", async () => {
    const user = userEvent.setup()
    testState.sourceSearchQuery = "Alpha"
    testState.activeFolderId = "folder-1"
    testState.selectedSourceIds = ["source-1"]
    const first = render(<ResearchWorkspace />)
    await user.click(
      await screen.findByRole("button", { name: "Set current filters" })
    )
    expect(screen.getByTestId("source-list-type-state")).toHaveTextContent("website")
    expect(screen.getByTestId("source-list-expanded-state")).toHaveTextContent(
      "expanded"
    )

    const saveViewButton = screen.getByRole("button", {
      name: "Save source view"
    })
    await waitFor(() => expect(saveViewButton).toBeEnabled())
    await user.click(saveViewButton)
    const saveDialog = await screen.findByRole("dialog", { name: "Save source view" })
    await user.type(within(saveDialog).getByRole("textbox", { name: "View name" }), "Current filters")
    await user.click(within(saveDialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(savedViewHarness.createWorkspaceSourceView).toHaveBeenCalledTimes(1)
    )
    expect(savedViewHarness.rows).toEqual([
      expect.objectContaining({
        name: "Current filters",
        state: expect.objectContaining({
          type_filters: ["website"],
          sort: "name_asc"
        })
      })
    ])
    await waitFor(() => expectDialogClosingOrGone("Save source view"))

    first.unmount()
    savedViewHarness.paneControllers = []
    const remounted = render(<ResearchWorkspace />)
    await waitFor(() => expect(latestController().views).toHaveLength(1))
    expect(latestController().activeViewId).toBeNull()
    expect(screen.getByTestId("source-list-type-state")).toHaveTextContent("none")
    expect(screen.getByTestId("source-list-sort-state")).toHaveTextContent("manual")
    await user.click(screen.getByRole("button", { name: "Expand source filters" }))
    await user.click(screen.getByRole("button", { name: "Source views" }))
    expect(
      await screen.findByRole("menuitem", { name: /Current filters/ })
    ).toBeInTheDocument()
    await openSavedViewCommands(user, "Current filters")
    await user.click(
      screen.getByRole("menuitem", { name: "Apply saved view Current filters" })
    )
    expect(latestController().activeViewId).toBe(savedViewHarness.rows[0]?.id)
    expect(screen.getByTestId("source-list-type-state")).toHaveTextContent("website")
    expect(screen.getByTestId("source-list-sort-state")).toHaveTextContent("name_asc")
    expect(screen.getByTestId("source-list-expanded-state")).toHaveTextContent(
      "expanded"
    )
    expect(screen.getByTestId("workspace-source-search-state")).toHaveTextContent(
      "Alpha"
    )
    expect(screen.getByTestId("workspace-active-folder-state")).toHaveTextContent(
      "folder-1"
    )
    expect(testState.sourceSearchQuery).toBe("Alpha")
    expect(testState.activeFolderId).toBe("folder-1")
    expect(testState.selectedSourceIds).toEqual(["source-1"])

    remounted.unmount()
  })
})
