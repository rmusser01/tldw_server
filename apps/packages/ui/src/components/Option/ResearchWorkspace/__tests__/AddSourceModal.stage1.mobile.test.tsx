import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { AddSourceTab } from "@/types/workspace"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"

const mockCloseAddSourceModal = vi.fn()
const mockSetAddSourceModalTab = vi.fn()
const mockSetAddSourceProcessing = vi.fn()
const mockSetAddSourceError = vi.fn()
const mockAddSource = vi.fn()

let isMobile = false

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "upload" as AddSourceTab,
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

  it("gives upload triggers specific accessible names", () => {
    isMobile = false
    const { unmount } = render(<AddSourceModal />)

    expect(
      screen.getByRole("button", { name: "Upload source files" })
    ).toBeInTheDocument()

    unmount()
    isMobile = true
    render(<AddSourceModal />)

    expect(
      screen.getByRole("button", { name: "Browse source files" })
    ).toBeInTheDocument()
  })

  it("supplements add-source placeholder fields with accessible names", () => {
    workspaceStoreState.addSourceModalTab = "url"
    const { unmount } = render(<AddSourceModal />)

    expect(screen.getByLabelText("Source URL")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Batch (one per line)" }))
    expect(screen.getByLabelText("Source URLs")).toBeInTheDocument()

    unmount()
    workspaceStoreState.addSourceModalTab = "paste"
    const pasteRender = render(<AddSourceModal />)

    expect(screen.getByLabelText("Pasted source title")).toBeInTheDocument()
    expect(screen.getByLabelText("Pasted source content")).toBeInTheDocument()

    pasteRender.unmount()
    workspaceStoreState.addSourceModalTab = "search"
    render(<AddSourceModal />)

    expect(screen.getByLabelText("Search the web")).toBeInTheDocument()
  })
})
