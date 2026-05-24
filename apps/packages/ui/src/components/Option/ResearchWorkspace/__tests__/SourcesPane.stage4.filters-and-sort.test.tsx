import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesPane } from "../SourcesPane"
import {
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  type SourceListViewState
} from "../SourcesPane/source-list-view"
import type { WorkspaceSource } from "@/types/workspace"

const mockToggleSourceSelection = vi.fn()
const mockToggleSourceFolderSelection = vi.fn()
const mockSelectAllSources = vi.fn()
const mockDeselectAllSources = vi.fn()
const mockSetSelectedSourceIds = vi.fn()
const mockSetSourceSearchQuery = vi.fn()
const mockClearSourceFocusTarget = vi.fn()
const mockOpenAddSourceModal = vi.fn()
const mockAddSource = vi.fn()
const mockRemoveSource = vi.fn()
const mockRemoveSources = vi.fn()
const mockRestoreSource = vi.fn()
const mockReorderSource = vi.fn()
const mockSetActiveFolder = vi.fn()
const mockAssignSourceToFolders = vi.fn()
const mockGetEffectiveSelectedSources = vi.fn()

const defaultSources: WorkspaceSource[] = [
  {
    id: "s1",
    mediaId: 101,
    title: "Alpha PDF",
    type: "pdf",
    status: "ready",
    addedAt: new Date("2026-03-11T00:00:00.000Z"),
    pageCount: 12,
    fileSize: 2_048
  },
  {
    id: "s2",
    mediaId: 102,
    title: "Bravo Website",
    type: "website",
    status: "error",
    addedAt: new Date("2026-03-12T00:00:00.000Z"),
    url: "https://example.com"
  }
]

const workspaceStoreState = {
  sources: [...defaultSources] as WorkspaceSource[],
  selectedSourceIds: [] as string[],
  sourceFolders: [] as Array<{
    id: string
    workspaceId: string
    name: string
    parentFolderId: string | null
    createdAt: Date
    updatedAt: Date
  }>,
  sourceFolderMemberships: [] as Array<{ folderId: string; sourceId: string }>,
  selectedSourceFolderIds: [] as string[],
  activeFolderId: null as string | null,
  sourceSearchQuery: "",
  sourceFocusTarget: null as { sourceId: string; token: number } | null,
  toggleSourceSelection: mockToggleSourceSelection,
  toggleSourceFolderSelection: mockToggleSourceFolderSelection,
  selectAllSources: mockSelectAllSources,
  deselectAllSources: mockDeselectAllSources,
  setSelectedSourceIds: mockSetSelectedSourceIds,
  setSourceSearchQuery: mockSetSourceSearchQuery,
  clearSourceFocusTarget: mockClearSourceFocusTarget,
  openAddSourceModal: mockOpenAddSourceModal,
  addSource: mockAddSource,
  removeSource: mockRemoveSource,
  removeSources: mockRemoveSources,
  restoreSource: mockRestoreSource,
  reorderSource: mockReorderSource,
  setActiveFolder: mockSetActiveFolder,
  assignSourceToFolders: mockAssignSourceToFolders,
  getEffectiveSelectedSources: mockGetEffectiveSelectedSources
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            count?: number
            defaultValue?: string
          },
      interpolationValues?: {
        count?: number
      }
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace(
          "{{count}}",
          String(interpolationValues?.count ?? "")
        )
      }
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          "{{count}}",
          String(defaultValueOrOptions.count ?? "")
        )
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

vi.mock("../SourcesPane/AddSourceModal", () => ({
  AddSourceModal: () => <div data-testid="add-source-modal" />
}))

const ControlledSourcesPane = () => {
  const [sourceListViewState, setSourceListViewState] = React.useState<SourceListViewState>(
    DEFAULT_SOURCE_LIST_VIEW_STATE
  )

  return (
    <SourcesPane
      sourceListViewState={sourceListViewState}
      onPatchSourceListViewState={(patch) =>
        setSourceListViewState((current) => ({ ...current, ...patch }))
      }
      onResetAdvancedSourceFilters={() =>
        setSourceListViewState((current) => ({
          ...DEFAULT_SOURCE_LIST_VIEW_STATE,
          expanded: current.expanded
        }))
      }
    />
  )
}

const getRenderedSourceTitles = (): string[] =>
  Array.from(document.querySelectorAll("[data-source-id]"))
    .map((element) => element.querySelector("p")?.textContent?.trim() || "")
    .filter(Boolean)

describe("SourcesPane stage 4 filters and sort", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.sources = [...defaultSources]
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.sourceFolders = []
    workspaceStoreState.sourceFolderMemberships = []
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.activeFolderId = null
    workspaceStoreState.sourceSearchQuery = ""
    workspaceStoreState.sourceFocusTarget = null
    mockSetSourceSearchQuery.mockImplementation((value: string) => {
      workspaceStoreState.sourceSearchQuery = value
    })
    mockSetSelectedSourceIds.mockImplementation((ids: string[]) => {
      workspaceStoreState.selectedSourceIds = ids
    })
    mockGetEffectiveSelectedSources.mockReturnValue([])
  })

  it("shows a collapsed summary and reveals advanced controls on demand", () => {
    render(<ControlledSourcesPane />)

    expect(screen.getByRole("button", { name: "Advanced" })).toBeInTheDocument()
    expect(screen.queryByText(/Sort: Added date/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Status Ready" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Sort by" }), {
      target: { value: "added_desc" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))

    expect(screen.getByText(/Status=Ready/)).toBeInTheDocument()
    expect(screen.getByText(/Sort: Added date/)).toBeInTheDocument()
  })

  it("applies advanced filters after search narrowing", () => {
    workspaceStoreState.sources = [
      {
        id: "s1",
        mediaId: 101,
        title: "Alpha Ready",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-03-11T00:00:00.000Z")
      },
      {
        id: "s2",
        mediaId: 102,
        title: "Alpha Error",
        type: "website",
        status: "error",
        addedAt: new Date("2026-03-12T00:00:00.000Z")
      }
    ]

    const { rerender } = render(<ControlledSourcesPane />)

    fireEvent.change(screen.getByPlaceholderText("Search sources..."), {
      target: { value: "Alpha" }
    })
    rerender(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Status Ready" }))

    expect(screen.getByText("Alpha Ready")).toBeInTheDocument()
    expect(screen.queryByText("Alpha Error")).not.toBeInTheDocument()
  })

  it("restores manual order when returning from a temporary sort", () => {
    workspaceStoreState.sources = [
      {
        id: "s1",
        mediaId: 101,
        title: "Zulu Ready",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-03-11T00:00:00.000Z")
      },
      {
        id: "s2",
        mediaId: 102,
        title: "Alpha Error",
        type: "website",
        status: "error",
        addedAt: new Date("2026-03-12T00:00:00.000Z")
      }
    ]

    render(<ControlledSourcesPane />)

    expect(getRenderedSourceTitles()).toEqual(["Zulu Ready", "Alpha Error"])

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Sort by" }), {
      target: { value: "name_asc" }
    })
    expect(getRenderedSourceTitles()).toEqual(["Alpha Error", "Zulu Ready"])

    fireEvent.change(screen.getByRole("combobox", { name: "Sort by" }), {
      target: { value: "manual" }
    })
    expect(getRenderedSourceTitles()).toEqual(["Zulu Ready", "Alpha Error"])
  })

  it("uses select visible when advanced filters narrow the rendered list", () => {
    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Status Ready" }))

    expect(screen.getByRole("checkbox", { name: "Select visible" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: "Select visible" }))

    expect(mockSetSelectedSourceIds).toHaveBeenCalledWith(["s1"])
  })

  it("preserves hidden direct selections when selecting visible results", () => {
    workspaceStoreState.sources = [
      {
        id: "s1",
        mediaId: 101,
        title: "Alpha PDF",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-03-11T00:00:00.000Z")
      },
      {
        id: "s2",
        mediaId: 102,
        title: "Bravo Website",
        type: "website",
        status: "ready",
        addedAt: new Date("2026-03-12T00:00:00.000Z")
      }
    ]
    workspaceStoreState.selectedSourceIds = ["s2"]
    mockGetEffectiveSelectedSources.mockReturnValue([workspaceStoreState.sources[1]])

    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Type PDF" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Select visible" }))

    expect(mockSetSelectedSourceIds).toHaveBeenCalledWith(["s1", "s2"])
  })

  it("clears only visible direct selections when toggling select visible off", () => {
    workspaceStoreState.sources = [
      {
        id: "s1",
        mediaId: 101,
        title: "Alpha PDF",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-03-11T00:00:00.000Z")
      },
      {
        id: "s2",
        mediaId: 102,
        title: "Bravo Website",
        type: "website",
        status: "ready",
        addedAt: new Date("2026-03-12T00:00:00.000Z")
      }
    ]
    workspaceStoreState.selectedSourceIds = ["s1", "s2"]
    mockGetEffectiveSelectedSources.mockReturnValue([...workspaceStoreState.sources])

    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Type PDF" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Select visible" }))

    expect(mockSetSelectedSourceIds).toHaveBeenCalledWith(["s2"])
  })

  it("warns when bulk remove includes hidden selected sources", async () => {
    workspaceStoreState.selectedSourceIds = ["s1", "s2"]
    mockGetEffectiveSelectedSources.mockReturnValue([...workspaceStoreState.sources])

    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Status Ready" }))
    fireEvent.click(screen.getByRole("button", { name: "Remove (2)" }))

    expect(
      await screen.findByText(/hidden by current filters/i)
    ).toBeInTheDocument()
  })

  it("disables reorder controls while a temporary sort is active", () => {
    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Sort by" }), {
      target: { value: "name_asc" }
    })

    expect(screen.getByTestId("move-source-down-s1")).toBeDisabled()
    expect(
      screen.getByText(/temporary sort is active\. switch back to manual order/i)
    ).toBeInTheDocument()
  })

  it("hides metadata-specific controls when no source exposes that field", () => {
    workspaceStoreState.sources = workspaceStoreState.sources.map((source) => ({
      ...source,
      pageCount: undefined,
      duration: undefined,
      fileSize: undefined
    }))

    render(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))

    expect(screen.queryByLabelText("Page count min")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Duration min")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("File size min")).not.toBeInTheDocument()
  })

  it("clears only advanced filters without clearing search text", () => {
    const { rerender } = render(<ControlledSourcesPane />)

    fireEvent.change(screen.getByPlaceholderText("Search sources..."), {
      target: { value: "Alpha" }
    })
    rerender(<ControlledSourcesPane />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Status Ready" }))
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    rerender(<ControlledSourcesPane />)

    expect(screen.getByDisplayValue("Alpha")).toBeInTheDocument()
    expect(screen.getByRole("checkbox", { name: "Status Ready" })).not.toBeChecked()
  })
})
