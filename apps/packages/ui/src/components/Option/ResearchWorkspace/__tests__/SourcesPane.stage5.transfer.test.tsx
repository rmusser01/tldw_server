import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesPane } from "../SourcesPane"
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
    id: "source-ready-visible",
    mediaId: 101,
    title: "Ready Source",
    type: "pdf",
    status: "ready",
    addedAt: new Date("2026-03-28T00:00:00.000Z")
  },
  {
    id: "source-processing",
    mediaId: 102,
    title: "Processing Source",
    type: "website",
    status: "processing",
    addedAt: new Date("2026-03-28T00:00:00.000Z")
  }
]

const workspaceStoreState = {
  sources: [...defaultSources] as WorkspaceSource[],
  selectedSourceIds: ["source-ready-visible", "source-processing"] as string[],
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
      key: string,
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
      const count =
        interpolationValues?.count ?? defaultValueOrOptions?.count ?? undefined
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace("{{count}}", String(count ?? ""))
      }
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          "{{count}}",
          String(count ?? "")
        )
      }
      return key
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

describe("SourcesPane stage 5 transfer launch", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.sources = [...defaultSources]
    workspaceStoreState.selectedSourceIds = [
      "source-ready-visible",
      "source-processing"
    ]
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.sourceSearchQuery = ""
    mockGetEffectiveSelectedSources.mockReturnValue([...defaultSources])
  })

  it("shows Move / Copy for effective selection and launches the shared transfer modal with the expected payload", () => {
    const mockOpenTransferSources = vi.fn()

    render(<SourcesPane onOpenTransferSources={mockOpenTransferSources} />)

    fireEvent.click(screen.getByRole("button", { name: "Move / Copy" }))

    expect(mockOpenTransferSources).toHaveBeenCalledWith({
      entryPoint: "sources",
      selectedSourceIds: ["source-ready-visible", "source-processing"],
      eligibleSelectedSourceIds: ["source-ready-visible"],
      totalSelectedCount: 2,
      hiddenSelectedCount: 0,
      ineligibleSelectedCount: 1
    })
  })

  it("hides Move / Copy when the effective selection has no eligible ready sources", () => {
    workspaceStoreState.sources = [
      {
        id: "source-processing",
        mediaId: 102,
        title: "Processing Source",
        type: "website",
        status: "processing",
        addedAt: new Date("2026-03-28T00:00:00.000Z")
      }
    ]
    workspaceStoreState.selectedSourceIds = ["source-processing"]
    mockGetEffectiveSelectedSources.mockReturnValue([...workspaceStoreState.sources])

    render(<SourcesPane onOpenTransferSources={vi.fn()} />)

    expect(
      screen.queryByRole("button", { name: "Move / Copy" })
    ).not.toBeInTheDocument()
  })
})
