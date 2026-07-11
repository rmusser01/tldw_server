import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSourceSavedViewResponse } from "@/services/tldw/domains/workspace-api"
import { ResearchWorkspace } from "../index"
import type { SourceListViewState } from "../SourcesPane/source-list-view"
import type { SourceSavedViewsController } from "../SourcesPane/use-source-saved-views"

const savedViewMocks = vi.hoisted(() => ({
  useSourceSavedViews: vi.fn(),
  generation: 0,
  controllers: [] as Array<{
    workspaceId: string | null
    controller: unknown
  }>
}))

const validView = (workspaceId: string): WorkspaceSourceSavedViewResponse => ({
  id: `${workspaceId}-view`,
  workspace_id: workspaceId,
  name: "Saved PDFs",
  schema_version: 1,
  version: 1,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
  valid: true,
  invalid_reason: null,
  state: {
    type_filters: ["pdf"],
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
    sort: "name_asc"
  }
})

const latestController = (): SourceSavedViewsController =>
  savedViewMocks.controllers.at(-1)!.controller as SourceSavedViewsController

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
  return {
    SourcesPane: (props: {
      sourceListViewState?: SourceListViewState
      onPatchSourceListViewState?: (patch: Partial<SourceListViewState>) => void
      onApplySourceListViewState?: (state: SourceListViewState) => void
      sourceSavedViewsController?: SourceSavedViewsController
      onOpenSourceViewOverlay?: React.ComponentProps<
        typeof SourceViewControls
      >["onOpenOverlay"]
    }) => (
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
        <button
          type="button"
          onClick={() => props.onPatchSourceListViewState?.({ sort: "name_asc" })}
        >
          Patch source list sort
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
        <div data-testid="saved-view-controller-identity">
          {savedViewMocks.controllers.some(
            ({ controller }) => controller === props.sourceSavedViewsController
          )
            ? "shared"
            : "different"}
        </div>
      </div>
    )
  }
})

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
    savedViewMocks.generation = 0
    savedViewMocks.controllers = []
    savedViewMocks.useSourceSavedViews.mockReset()
    savedViewMocks.useSourceSavedViews.mockImplementation(
      (
        workspaceId: string | null,
        currentState: SourceListViewState,
        onApply: (state: SourceListViewState) => void
      ) => {
        const previous = savedViewMocks.controllers.at(-1)
        if (previous?.workspaceId === workspaceId) {
          const existing = previous.controller as SourceSavedViewsController
          existing.applyView.mockImplementation(
            (view: WorkspaceSourceSavedViewResponse) => {
              if (!view.valid) return
              onApply({
                ...currentState,
                typeFilters: [...view.state.type_filters],
                sort: view.state.sort
              })
            }
          )
          return existing
        }

        savedViewMocks.generation += 1
        const applyView = vi.fn((view: WorkspaceSourceSavedViewResponse) => {
          if (!view.valid) return
          onApply({
            ...currentState,
            typeFilters: [...view.state.type_filters],
            sort: view.state.sort
          })
        })
        const controller = {
          available: workspaceId !== null,
          generation: savedViewMocks.generation,
          views: workspaceId === null ? [] : [validView(workspaceId)],
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
          applyView,
          createView: vi.fn(),
          confirmReplace: vi.fn(),
          dismissDuplicateConflict: vi.fn(),
          replaceView: vi.fn(),
          resetView: vi.fn(),
          deleteView: vi.fn()
        } as unknown as SourceSavedViewsController
        savedViewMocks.controllers.push({ workspaceId, controller })
        return controller
      }
    )
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
    expect(savedViewMocks.controllers).toHaveLength(1)
    expect(
      savedViewMocks.useSourceSavedViews.mock.calls.every(
        ([workspaceId]) => workspaceId === "workspace-1"
      )
    ).toBe(true)
    await user.click(screen.getByRole("button", { name: "Toggle sources" }))

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
    expect(screen.getByRole("button", { name: "Save source view" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    expect(screen.queryByRole("dialog", { name: "Save source view" })).not.toBeInTheDocument()
  })

  it.each([
    { label: "workspace B", nextWorkspaceId: "workspace-2" },
    { label: "no workspace", nextWorkspaceId: null }
  ])(
    "synchronously closes an A overlay for $label and refuses its stale submit",
    async ({ nextWorkspaceId }) => {
      const user = userEvent.setup()
      const { rerender } = render(<ResearchWorkspace />)
      const controllerA = latestController()
      await user.click(screen.getByRole("button", { name: "Save source view" }))
      const dialog = await screen.findByRole("dialog", { name: "Save source view" })
      await user.type(within(dialog).getByRole("textbox", { name: "View name" }), "Draft A")
      const staleSubmit = within(dialog).getByRole("button", { name: "Save" })

      testState.workspaceId = nextWorkspaceId
      rerender(<ResearchWorkspace />)

      expect(screen.queryByRole("dialog", { name: "Save source view" })).not.toBeInTheDocument()
      expect(screen.queryByDisplayValue("Draft A")).not.toBeInTheDocument()
      fireEvent.click(staleSubmit)
      expect(controllerA.createView).not.toHaveBeenCalled()
      expect(latestController().createView).not.toHaveBeenCalled()
    }
  )

  it.each([
    { label: "workspace B", nextWorkspaceId: "workspace-2" },
    { label: "no workspace", nextWorkspaceId: null }
  ])(
    "discards an A replacement confirmation for $label and refuses its stale action",
    async ({ nextWorkspaceId }) => {
      const user = userEvent.setup()
      const { rerender } = render(<ResearchWorkspace />)
      const controllerA = latestController()
      await user.click(screen.getByRole("button", { name: "Source views" }))
      await user.click(
        screen.getByRole("button", { name: "Replace saved view Saved PDFs" })
      )
      const confirmation = await screen.findByRole("alertdialog", {
        name: "Replace saved view?"
      })
      const staleReplace = within(confirmation).getByRole("button", {
        name: "Replace"
      })

      testState.workspaceId = nextWorkspaceId
      rerender(<ResearchWorkspace />)

      expect(
        screen.queryByRole("alertdialog", { name: "Replace saved view?" })
      ).not.toBeInTheDocument()
      fireEvent.click(staleReplace)
      expect(controllerA.replaceView).not.toHaveBeenCalled()
      expect(latestController().replaceView).not.toHaveBeenCalled()
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

  it("lists saved views after remount and only restores one when explicitly reselected", async () => {
    const user = userEvent.setup()
    const first = render(<ResearchWorkspace />)
    const firstController = latestController()
    expect(firstController.applyView).not.toHaveBeenCalled()
    expect(firstController.activeViewId).toBeNull()

    await user.click(screen.getByRole("button", { name: "Save source view" }))
    const saveDialog = await screen.findByRole("dialog", { name: "Save source view" })
    await user.type(within(saveDialog).getByRole("textbox", { name: "View name" }), "Current filters")
    await user.click(within(saveDialog).getByRole("button", { name: "Save" }))
    expect(firstController.createView).toHaveBeenCalledWith("Current filters")
    await user.click(within(saveDialog).getByRole("button", { name: "Cancel" }))

    first.unmount()
    savedViewMocks.controllers = []
    const remounted = render(<ResearchWorkspace />)
    const reloadedController = latestController()
    expect(reloadedController.applyView).not.toHaveBeenCalled()
    expect(reloadedController.activeViewId).toBeNull()
    await user.click(screen.getByRole("button", { name: "Source views" }))
    expect(await screen.findByRole("menuitem", { name: /Saved PDFs/ })).toBeInTheDocument()
    await user.click(screen.getByRole("menuitem", { name: /Saved PDFs/ }))
    expect(reloadedController.applyView).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("source-list-sort-state")).toHaveTextContent("name_asc")

    remounted.unmount()
  })
})
