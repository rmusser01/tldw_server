import { render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import type { WorkspaceSource } from "@/types/workspace"

import { SourcesPane } from "../SourcesPane"

const registryLabels = vi.hoisted(() => ({
  ready: "Registry Ready"
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "ready" ? registryLabels.ready : state.label
        }
      }
    )
  }
})

const workspaceStoreState = {
  sources: [] as WorkspaceSource[],
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
  toggleSourceSelection: vi.fn(),
  toggleSourceFolderSelection: vi.fn(),
  selectAllSources: vi.fn(),
  deselectAllSources: vi.fn(),
  setSelectedSourceIds: vi.fn(),
  setSourceSearchQuery: vi.fn(),
  clearSourceFocusTarget: vi.fn(),
  openAddSourceModal: vi.fn(),
  addSource: vi.fn(),
  removeSource: vi.fn(),
  removeSources: vi.fn(),
  restoreSource: vi.fn(),
  reorderSource: vi.fn(),
  setActiveFolder: vi.fn(),
  assignSourceToFolders: vi.fn(),
  getEffectiveSelectedSources: vi.fn(() => [])
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

describe("SourcesPane design-system state labels", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "Registry-backed source",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-03-11T00:00:00.000Z")
      }
    ]
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.sourceFolders = []
    workspaceStoreState.sourceFolderMemberships = []
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.activeFolderId = null
    workspaceStoreState.sourceSearchQuery = ""
    workspaceStoreState.sourceFocusTarget = null
    workspaceStoreState.getEffectiveSelectedSources.mockReturnValue([])
  })

  it("uses the design-system registry label for ready source status badges", () => {
    render(<SourcesPane />)

    const sourceRow = screen
      .getByText("Registry-backed source")
      .closest('[data-source-id="source-1"]') as HTMLElement

    expect(within(sourceRow).getByText(registryLabels.ready)).toBeInTheDocument()
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith("ready")
  })

  it("explains empty source storage without adding a persistent trust banner", () => {
    workspaceStoreState.sources = []

    render(<SourcesPane />)

    expect(screen.getByText(/configured local or self-hosted server/i)).toBeInTheDocument()
  })
})
