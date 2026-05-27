import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import { SourcesPane } from "../SourcesPane"
import { WORKSPACE_SOURCE_DRAG_TYPE } from "../drag-source"
const { mockScheduleWorkspaceUndoAction, mockUndoWorkspaceAction } = vi.hoisted(
  () => ({
    mockScheduleWorkspaceUndoAction: vi.fn(),
    mockUndoWorkspaceAction: vi.fn()
  })
)

const { mockAddMedia, mockGetWorkspaceSourcePreview } = vi.hoisted(() => ({
  mockAddMedia: vi.fn(),
  mockGetWorkspaceSourcePreview: vi.fn()
}))

const mockToggleSourceSelection = vi.fn()
const mockSelectAllSources = vi.fn()
const mockDeselectAllSources = vi.fn()
const mockSetSourceSearchQuery = vi.fn()
const mockOpenAddSourceModal = vi.fn()
const mockRemoveSource = vi.fn()
const mockRestoreSource = vi.fn()
const mockReorderSource = vi.fn()
const mockClearSourceFocusTarget = vi.fn()

const defaultSources: WorkspaceSource[] = [
  {
    id: "s1",
    mediaId: 1,
    title: "Source One",
    type: "pdf" as const,
    status: "ready" as const,
    addedAt: new Date("2026-02-18T00:00:00.000Z")
  },
  {
    id: "s2",
    mediaId: 2,
    title: "Source Two",
    type: "video" as const,
    status: "processing" as const,
    addedAt: new Date("2026-02-18T00:00:00.000Z")
  }
]

const workspaceStoreState = {
  workspaceId: "workspace-1",
  sources: [...defaultSources] as WorkspaceSource[],
  selectedSourceIds: [] as string[],
  sourceSearchQuery: "",
  sourceFocusTarget: null as { sourceId: string; token: number } | null,
  toggleSourceSelection: mockToggleSourceSelection,
  selectAllSources: mockSelectAllSources,
  deselectAllSources: mockDeselectAllSources,
  setSourceSearchQuery: mockSetSourceSearchQuery,
  clearSourceFocusTarget: mockClearSourceFocusTarget,
  openAddSourceModal: mockOpenAddSourceModal,
  removeSource: mockRemoveSource,
  restoreSource: mockRestoreSource,
  reorderSource: mockReorderSource,
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          },
      options?: Record<string, unknown>
    ) => {
      const interpolationValues = {
        ...(typeof defaultValueOrOptions === "object" ? defaultValueOrOptions : {}),
        ...(options || {})
      }
      const interpolate = (value: string) =>
        value.replace(/\{\{(\w+)\}\}/g, (_match, key) =>
          interpolationValues[key] !== undefined
            ? String(interpolationValues[key])
            : `{{${key}}}`
        )
      if (typeof defaultValueOrOptions === "string") {
        return interpolate(defaultValueOrOptions)
      }
      if (defaultValueOrOptions?.defaultValue) {
        return interpolate(defaultValueOrOptions.defaultValue)
      }
      return _key
    }
  })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    addMedia: mockAddMedia,
    getWorkspaceSourcePreview: mockGetWorkspaceSourcePreview
  }
}))

vi.mock("../SourcesPane/AddSourceModal", () => ({
  AddSourceModal: () => <div data-testid="add-source-modal" />
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: mockScheduleWorkspaceUndoAction,
  undoWorkspaceAction: mockUndoWorkspaceAction
}))

describe("SourcesPane Stage 2 source highlighting", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    mockGetWorkspaceSourcePreview.mockResolvedValue({
      workspace_id: "workspace-1",
      source_id: "s1",
      media_id: 1,
      title: "Source One",
      source_type: "pdf",
      url: null,
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
      content_available: true,
      preview_mode: "available",
      unavailable_reason: null,
      text_preview: "Default captured source preview text.",
      text_total_chars: 37,
      text_truncated: false,
      snippets: [],
      generated_at: "2026-05-25T00:00:00Z"
    })
    mockUndoWorkspaceAction.mockReturnValue(true)
    mockScheduleWorkspaceUndoAction.mockImplementation(
      ({
        apply
      }: {
        apply: () => void
        undo: () => void
      }) => {
        apply()
        return { id: "undo-1", expiresAt: Date.now() + 10000 }
      }
    )
    workspaceStoreState.sources = [...defaultSources]
    workspaceStoreState.sourceSearchQuery = ""
    workspaceStoreState.sourceFocusTarget = null

    mockSetSourceSearchQuery.mockImplementation((value: string) => {
      workspaceStoreState.sourceSearchQuery = value
    })
    mockClearSourceFocusTarget.mockImplementation(() => {
      workspaceStoreState.sourceFocusTarget = null
    })
    mockReorderSource.mockImplementation((sourceId: string, targetIndex: number) => {
      const currentIndex = workspaceStoreState.sources.findIndex((source) => source.id === sourceId)
      if (currentIndex < 0) return
      const next = [...workspaceStoreState.sources]
      const [moved] = next.splice(currentIndex, 1)
      next.splice(targetIndex, 0, moved)
      workspaceStoreState.sources = next
    })
  })

  it("labels the primary source intake action as Add Sources", () => {
    render(<SourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Add Sources" }))

    expect(mockOpenAddSourceModal).toHaveBeenCalledWith("existing")
  })

  it("surfaces partial workspace context errors as a compact source warning", () => {
    render(<SourcesPane statusProjectionError="Jobs service unavailable" />)

    expect(
      screen.getByRole("button", { name: "Source status warning" })
    ).toBeInTheDocument()
  })

  it("keeps advanced controls scrollable without replacing the source list region", () => {
    render(
      <SourcesPane
        sourceListViewState={{
          sort: "manual",
          dateField: "addedAt",
          dateFrom: null,
          dateTo: null,
          statusFilters: [],
          typeFilters: [],
          requireUrl: false,
          requireFileSize: false,
          requireDuration: false,
          requirePageCount: false,
          fileSizeMin: null,
          fileSizeMax: null,
          durationMin: null,
          durationMax: null,
          pageCountMin: null,
          pageCountMax: null,
          expanded: true
        }}
      />
    )

    expect(screen.getByTestId("workspace-sources-pane-root")).toHaveClass(
      "min-h-0",
      "overflow-hidden"
    )
    expect(screen.getByTestId("sources-management-controls")).toHaveClass(
      "shrink-0",
      "overflow-y-auto"
    )
    expect(screen.getByTestId("sources-list-region")).toHaveClass(
      "min-h-0",
      "flex-1",
      "overflow-y-auto"
    )
    expect(screen.getByText("2 source(s)")).toBeInTheDocument()
  })

  it("scrolls to and highlights a focused source target", () => {
    vi.useFakeTimers()
    const scrollSpy = vi.fn()
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView
    HTMLElement.prototype.scrollIntoView = scrollSpy

    try {
      workspaceStoreState.sourceFocusTarget = { sourceId: "s2", token: 1 }

      const { container } = render(<SourcesPane />)

      act(() => {
        vi.advanceTimersByTime(0)
      })

      expect(scrollSpy).toHaveBeenCalledTimes(1)
      expect(mockClearSourceFocusTarget).toHaveBeenCalledTimes(1)
      expect(
        container
          .querySelector('[data-source-id="s2"]')
          ?.getAttribute("data-highlighted")
      ).toBe("true")

      act(() => {
        vi.advanceTimersByTime(1800)
      })
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView
      vi.useRealTimers()
    }
  })

  it("clears active source search when focused source is filtered out", () => {
    workspaceStoreState.sourceSearchQuery = "no-match"
    workspaceStoreState.sourceFocusTarget = { sourceId: "s1", token: 2 }

    render(<SourcesPane />)

    expect(mockSetSourceSearchQuery).toHaveBeenCalledWith("")
  })

  it("marks source rows as draggable and sets workspace drag payload", () => {
    render(<SourcesPane />)

    const sourceRow = screen
      .getByText("Source One")
      .closest('[data-source-id="s1"]') as HTMLElement
    expect(sourceRow).toBeTruthy()
    expect(sourceRow).toHaveAttribute("draggable", "true")

    const setData = vi.fn()
    fireEvent.dragStart(sourceRow, {
      dataTransfer: {
        effectAllowed: "",
        setData
      }
    })

    expect(setData).toHaveBeenCalledWith(
      WORKSPACE_SOURCE_DRAG_TYPE,
      expect.stringContaining('"sourceId":"s1"')
    )
    expect(setData).toHaveBeenCalledWith("text/plain", "Source One")
  })

  it("reorders sources by drag-and-drop within the source list", () => {
    workspaceStoreState.sources = [
      {
        ...defaultSources[0],
        status: "ready" as const
      },
      {
        ...defaultSources[1],
        status: "ready" as const
      }
    ]

    render(<SourcesPane />)

    const firstRow = screen
      .getByText("Source One")
      .closest('[data-source-id="s1"]') as HTMLElement
    const secondRow = screen
      .getByText("Source Two")
      .closest('[data-source-id="s2"]') as HTMLElement

    fireEvent.dragStart(firstRow, {
      dataTransfer: {
        effectAllowed: "copyMove",
        setData: vi.fn()
      }
    })
    fireEvent.dragOver(secondRow, {
      dataTransfer: {
        dropEffect: "move"
      }
    })
    fireEvent.drop(secondRow, {
      dataTransfer: {
        dropEffect: "move"
      }
    })

    expect(mockReorderSource).toHaveBeenCalledWith("s1", 1)
  })

  it("applies touch-friendly hit areas for source selection controls", () => {
    render(<SourcesPane />)

    const checkboxHitArea = screen.getByTestId("source-checkbox-hitarea-s1")
    expect(checkboxHitArea.className).toContain("[@media(hover:none)]:min-h-11")
    expect(checkboxHitArea.className).toContain("[@media(hover:none)]:min-w-11")
  })

  it("keeps remove action visible for keyboard focus and touch devices", () => {
    render(<SourcesPane />)

    const removeButton = screen.getByTestId("remove-source-s1")
    expect(removeButton.className).toContain("focus-visible:opacity-100")
    expect(removeButton.className).toContain("[@media(hover:none)]:opacity-100")
  })

  it("keeps selected-row remove action visible without hover", () => {
    workspaceStoreState.selectedSourceIds = ["s1"]

    render(<SourcesPane />)

    const removeButton = screen.getByTestId("remove-source-s1")
    expect(removeButton.className).toContain("opacity-100")
  })

  it("provides keyboard-accessible reorder buttons", () => {
    render(<SourcesPane />)

    const moveUp = screen.getByTestId("move-source-up-s1")
    const moveDown = screen.getByTestId("move-source-down-s1")
    expect(moveUp).toBeDisabled()
    expect(moveDown).toBeEnabled()

    fireEvent.click(moveDown)
    expect(mockReorderSource).toHaveBeenCalledWith("s1", 1)
  })

  it("offers keyboard users a confirmation path before remove", async () => {
    render(<SourcesPane />)

    const removeButton = screen.getByTestId("remove-source-s1")
    fireEvent.keyDown(removeButton, { key: "Enter" })

    expect(await screen.findByText("Remove source?")).toBeInTheDocument()
    expect(mockRemoveSource).not.toHaveBeenCalled()

    const confirmButton = screen
      .getAllByRole("button", { name: "Remove" })
      .find((button) => button.className.includes("ant-btn-primary"))
    expect(confirmButton).toBeTruthy()
    if (confirmButton) {
      fireEvent.click(confirmButton)
    }

    await waitFor(() => {
      expect(mockRemoveSource).toHaveBeenCalledWith("s1")
    })
  })

  it("keeps keyboard focus order logical within each source row", () => {
    render(<SourcesPane />)

    const checkbox = screen
      .getByTestId("source-checkbox-hitarea-s1")
      .querySelector("input[type='checkbox']") as HTMLInputElement | null
    const removeButton = screen.getByTestId("remove-source-s1")

    expect(checkbox).toBeTruthy()
    if (checkbox) {
      const relation = checkbox.compareDocumentPosition(removeButton)
      expect(relation & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

      checkbox.focus()
      expect(checkbox).toHaveFocus()
    }

    removeButton.focus()
    expect(removeButton).toHaveFocus()
  })

  it("shows processing status and disables selection for non-ready sources", () => {
    render(<SourcesPane />)

    expect(screen.getAllByText("Processing").length).toBeGreaterThan(0)

    const processingHitArea = screen.getByTestId("source-checkbox-hitarea-s2")
    const checkboxInput = processingHitArea.querySelector(
      "input[type='checkbox']"
    ) as HTMLInputElement | null
    expect(checkboxInput).toBeTruthy()
    expect(checkboxInput?.disabled).toBe(true)
  })

  it("surfaces backend processing detail text when available", () => {
    workspaceStoreState.sources = [
      defaultSources[0],
      {
        ...defaultSources[1],
        status: "processing" as const,
        statusMessage: "Processing... chunks extracted: 45/120"
      }
    ]

    render(<SourcesPane />)

    expect(
      screen.getByText("Processing... chunks extracted: 45/120")
    ).toBeInTheDocument()
  })

  it("opens a compact source status drilldown for processing diagnostics", () => {
    workspaceStoreState.sources = [
      {
        ...defaultSources[1],
        status: "processing" as const,
        statusMessage: "Indexing chunks 45/120",
        readiness: {
          metadata_ready: true,
          text_extracted: true,
          fts_ready: true,
          vector_ready: false,
          citation_ready: false,
          summary_ready: false,
          tool_accessible: true
        },
        statusDetails: {
          lifecycleState: "indexing",
          statusReason: "job_indexing",
          sourceOfTruth: "workspace-status-projection",
          updatedAt: new Date("2026-05-23T12:01:00.000Z"),
          stale: false,
          retryEligible: false,
          progressPercent: 38,
          progressMessage: "Indexing chunks 45/120",
          job: {
            id: 44,
            uuid: "job-index-44",
            status: "running",
            jobType: "workspace_source_index",
            progressPercent: 38,
            progressMessage: "Indexing chunks 45/120",
            errorMessage: null
          }
        }
      } as WorkspaceSource
    ]

    render(<SourcesPane />)

    fireEvent.click(
      screen.getByRole("button", {
        name: "View source status details for Source Two"
      })
    )

    const dialog = screen.getByRole("dialog", {
      name: "Source status details"
    })
    expect(dialog).toHaveTextContent("Source Two")
    expect(dialog).toHaveTextContent("Lifecycle")
    expect(dialog).toHaveTextContent("Indexing")
    expect(dialog).toHaveTextContent("Status reason")
    expect(dialog).toHaveTextContent("job_indexing")
    expect(dialog).toHaveTextContent("Source of truth")
    expect(dialog).toHaveTextContent("Server workspace status projection")
    expect(dialog).toHaveTextContent("Last refresh")
    expect(dialog).toHaveTextContent("Progress")
    expect(dialog).toHaveTextContent("38%")
    expect(dialog).toHaveTextContent("Retry eligibility")
    expect(dialog).toHaveTextContent("Retry not available while processing")
    expect(dialog).toHaveTextContent("Stale state")
    expect(dialog).toHaveTextContent("Fresh status")
    expect(dialog).toHaveTextContent("Media ID")
    expect(dialog).toHaveTextContent("2")
    expect(dialog).toHaveTextContent("Source ID")
    expect(dialog).toHaveTextContent("s2")
    expect(dialog).toHaveTextContent("Next action")
    expect(dialog).toHaveTextContent("Wait for indexing to finish")
    expect(dialog).not.toHaveTextContent("workspace-playground")
  })

  it("surfaces source metadata preview when metadata is available", () => {
    workspaceStoreState.sources = [
      {
        ...defaultSources[0],
        fileSize: 2 * 1024 * 1024,
        duration: 125,
        pageCount: 10
      }
    ]

    render(<SourcesPane />)

    expect(screen.getByText("2 MB • 2m 5s")).toBeInTheDocument()
  })

  it("renders source thumbnail previews when thumbnail metadata is available", () => {
    workspaceStoreState.sources = [
      {
        ...defaultSources[0],
        thumbnailUrl: "https://example.com/thumb.jpg"
      }
    ]

    render(<SourcesPane />)

    const thumbnail = screen.getByTestId("source-thumbnail-s1")
    expect(thumbnail).toBeInTheDocument()
    expect(thumbnail).toHaveAttribute("src", "https://example.com/thumb.jpg")
  })

  it("prefers API-origin created date metadata when available", () => {
    workspaceStoreState.sources = [
      {
        ...defaultSources[0],
        sourceCreatedAt: new Date("2025-01-01T00:00:00.000Z")
      }
    ]

    render(<SourcesPane />)

    expect(screen.getByText(/^Created /)).toBeInTheDocument()
    expect(screen.queryByText(/^Added /)).not.toBeInTheDocument()
  })

  it("supports source preview annotations create, edit, and delete with undo parity", async () => {
    render(<SourcesPane />)

    fireEvent.click(screen.getByTestId("preview-source-s1"))
    expect(
      await screen.findByText("Source preview and annotations")
    ).toBeInTheDocument()

    fireEvent.change(
      screen.getByPlaceholderText("Highlighted excerpt (optional)"),
      {
        target: { value: "Key highlighted excerpt" }
      }
    )
    fireEvent.change(screen.getByPlaceholderText("Annotation note"), {
      target: { value: "Initial annotation" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add annotation" }))

    expect(await screen.findByText("Initial annotation")).toBeInTheDocument()
    expect(screen.getByText(/Key highlighted excerpt/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    fireEvent.change(screen.getByPlaceholderText("Annotation note"), {
      target: { value: "Edited annotation" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save annotation" }))

    expect(await screen.findByText("Edited annotation")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Delete" }))
    await waitFor(() => {
      expect(screen.getByText("No local annotations yet.")).toBeInTheDocument()
    })

    expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalledTimes(1)
    const scheduledConfig = mockScheduleWorkspaceUndoAction.mock.calls[0]?.[0] as
      | { undo: () => void }
      | undefined
    expect(scheduledConfig).toBeDefined()
    scheduledConfig?.undo()

    await waitFor(() => {
      expect(screen.getByText("Edited annotation")).toBeInTheDocument()
    })
  }, 10000)

  it("loads captured source content and evidence snippets in the source preview", async () => {
    mockGetWorkspaceSourcePreview.mockResolvedValueOnce({
      workspace_id: "workspace-1",
      source_id: "s1",
      media_id: 1,
      title: "Source One",
      source_type: "pdf",
      url: null,
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
      content_available: true,
      preview_mode: "available",
      unavailable_reason: null,
      text_preview: "Captured source text that the user can inspect.",
      text_total_chars: 2400,
      text_truncated: true,
      snippets: [
        {
          id: "chunk-0",
          source_id: "s1",
          media_id: 1,
          kind: "chunk",
          text: "Chunk evidence used for citations.",
          start_char: 0,
          end_char: 34,
          chunk_index: 0,
          chunk_uuid: "chunk-0",
          chunk_type: "text"
        }
      ],
      generated_at: "2026-05-25T00:00:00Z"
    })

    render(<SourcesPane />)

    fireEvent.click(screen.getByTestId("preview-source-s1"))

    expect(await screen.findByText("Captured content")).toBeInTheDocument()
    expect(
      screen.getByText("Captured source text that the user can inspect.")
    ).toBeInTheDocument()
    expect(screen.getByText("Evidence snippets")).toBeInTheDocument()
    expect(screen.getByText("Chunk evidence used for citations.")).toBeInTheDocument()
    expect(
      screen.getByText("Showing first 47 of 2,400 characters.")
    ).toBeInTheDocument()
    expect(mockGetWorkspaceSourcePreview).toHaveBeenCalledWith("workspace-1", "s1", {
      max_chars: 3000,
      chunk_limit: 3
    })
  })

  it("explains when captured content is pending instead of replacing inspection with annotations", async () => {
    mockGetWorkspaceSourcePreview.mockResolvedValueOnce({
      workspace_id: "workspace-1",
      source_id: "s2",
      media_id: 2,
      title: "Source Two",
      source_type: "video",
      url: null,
      state: "extracting",
      status_reason: "extraction_pending",
      readiness: {
        metadata_ready: true,
        text_extracted: false,
        fts_ready: false,
        vector_ready: false,
        citation_ready: false,
        summary_ready: false,
        tool_accessible: true
      },
      content_available: false,
      preview_mode: "pending",
      unavailable_reason: "extraction_pending",
      text_preview: null,
      text_total_chars: null,
      text_truncated: false,
      snippets: [],
      generated_at: "2026-05-25T00:00:00Z"
    })

    render(<SourcesPane />)

    fireEvent.click(screen.getByTestId("preview-source-s2"))

    expect(await screen.findByText("Captured content")).toBeInTheDocument()
    expect(
      screen.getByText("Text extraction has not completed yet.")
    ).toBeInTheDocument()
    expect(screen.getByText("Local highlights & annotations")).toBeInTheDocument()
    expect(
      screen.getByText("Saved in this browser for this workspace.")
    ).toBeInTheDocument()
  })

  it("shows source preview failure details and supports retry", async () => {
    mockGetWorkspaceSourcePreview
      .mockRejectedValueOnce(new Error("Preview endpoint returned 503"))
      .mockResolvedValueOnce({
        workspace_id: "workspace-1",
        source_id: "s1",
        media_id: 1,
        title: "Source One",
        source_type: "pdf",
        url: null,
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
        content_available: true,
        preview_mode: "available",
        unavailable_reason: null,
        text_preview: "Retry loaded captured text.",
        text_total_chars: 27,
        text_truncated: false,
        snippets: [],
        generated_at: "2026-05-25T00:00:00Z"
      })

    render(<SourcesPane />)

    fireEvent.click(screen.getByTestId("preview-source-s1"))

    expect(await screen.findByText("Source preview could not load.")).toBeInTheDocument()
    expect(screen.getByText("Preview endpoint returned 503")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry preview" }))

    expect(await screen.findByText("Retry loaded captured text.")).toBeInTheDocument()
    expect(mockGetWorkspaceSourcePreview).toHaveBeenCalledTimes(2)
  })

  it("persists source annotations across modal remounts", async () => {
    const { unmount } = render(<SourcesPane />)

    fireEvent.click(screen.getByTestId("preview-source-s1"))
    await screen.findByText("Source preview and annotations")
    fireEvent.change(screen.getByPlaceholderText("Annotation note"), {
      target: { value: "Persistent annotation" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add annotation" }))
    expect(await screen.findByText("Persistent annotation")).toBeInTheDocument()

    unmount()
    render(<SourcesPane />)
    fireEvent.click(screen.getByTestId("preview-source-s1"))

    expect(await screen.findByText("Persistent annotation")).toBeInTheDocument()
  })

  it("shows selected-source action strip and previews single selected source", async () => {
    workspaceStoreState.selectedSourceIds = ["s1"]

    render(<SourcesPane />)

    expect(screen.getByTestId("sources-selected-actions")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Preview selected" }))

    expect(
      await screen.findByText("Source preview and annotations")
    ).toBeInTheDocument()
    expect(
      screen.getByPlaceholderText("Highlighted excerpt (optional)")
    ).toBeInTheDocument()
  })

  it("enables virtualized rendering when source volume crosses threshold", () => {
    workspaceStoreState.sources = Array.from({ length: 70 }, (_, index) => ({
      id: `source-${index + 1}`,
      mediaId: index + 1,
      title: `Source ${index + 1}`,
      type: "pdf" as const,
      status: "ready" as const,
      addedAt: new Date("2026-02-18T00:00:00.000Z")
    }))

    render(<SourcesPane />)

    expect(screen.getByTestId("sources-virtualized-list")).toBeInTheDocument()
  })
})
