import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"

const mockCloseAddSourceModal = vi.fn()
const mockSetAddSourceModalTab = vi.fn()
const mockSetAddSourceProcessing = vi.fn()
const mockSetAddSourceError = vi.fn()
const mockAddSource = vi.fn()

let isMobile = false

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "upload" as const,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: mockCloseAddSourceModal,
  setAddSourceModalTab: mockSetAddSourceModalTab,
  setAddSourceProcessing: mockSetAddSourceProcessing,
  setAddSourceError: mockSetAddSourceError,
  addSource: mockAddSource,
  workspaceTag: "workspace:test"
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

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => isMobile
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    uploadMedia: vi.fn(),
    addMedia: vi.fn(),
    webSearch: vi.fn().mockResolvedValue({ results: [] }),
    searchMedia: vi.fn().mockResolvedValue({ items: [] }),
    listMedia: vi.fn().mockResolvedValue({ items: [] }),
    updateMediaKeywords: vi.fn()
  }
}))

describe("AddSourceModal Stage 1 mobile upload affordances", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceModalTab = "upload"
    workspaceStoreState.addSourceProcessing = false
    workspaceStoreState.addSourceError = null
    isMobile = false
  })

  it("uses desktop drag-and-drop copy outside mobile", () => {
    isMobile = false

    render(<AddSourceModal />)

    expect(screen.getByText("Click or drag files to upload")).toBeInTheDocument()
    expect(screen.queryByTestId("mobile-browse-files-button")).not.toBeInTheDocument()

    const modal = document.querySelector(".ant-modal") as HTMLElement | null
    expect(modal).toBeTruthy()
    expect(modal.style.width).toBe("600px")
  })

  it("uses touch copy and shows explicit browse button on mobile", () => {
    isMobile = true

    render(<AddSourceModal />)

    expect(screen.getByText("Tap to select files")).toBeInTheDocument()
    const browseButton = screen.getByTestId("mobile-browse-files-button")
    expect(browseButton).toBeInTheDocument()

    fireEvent.click(browseButton)
    // Assert button is actionable in the modal and does not throw when tapped.
    expect(browseButton).toBeEnabled()

    const modal = document.querySelector(".ant-modal") as HTMLElement | null
    const modalBody = document.querySelector(".ant-modal-body") as HTMLElement | null
    expect(modal).toBeTruthy()
    expect(modalBody).toBeTruthy()
    expect(modal?.style.width).toBe("100%")
    expect(modalBody?.style.maxHeight).toBe("70vh")
    expect(modalBody?.style.overflowY).toBe("auto")
  })
})
