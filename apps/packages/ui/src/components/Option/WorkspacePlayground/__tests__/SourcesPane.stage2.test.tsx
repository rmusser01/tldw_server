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
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return _key
    }
  })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
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

  it("opens Add Sources directly on My Media", () => {
    render(<SourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Add Sources" }))

    expect(mockOpenAddSourceModal).toHaveBeenCalledWith("existing")
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

    expect(screen.getByText("Created {{date}}")).toBeInTheDocument()
    expect(screen.queryByText("Added {{date}}")).not.toBeInTheDocument()
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
      expect(screen.getByText("No annotations yet.")).toBeInTheDocument()
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
