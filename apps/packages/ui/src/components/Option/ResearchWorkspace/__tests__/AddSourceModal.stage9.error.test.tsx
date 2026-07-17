import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { AddSourceTab } from "@/types/workspace"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"

const {
  mockWebSearch,
  mockUploadMedia,
  mockAddMedia,
  mockAddSource
} = vi.hoisted(() => ({
  mockWebSearch: vi.fn(),
  mockUploadMedia: vi.fn(),
  mockAddMedia: vi.fn(),
  mockAddSource: vi.fn()
}))

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "search" as AddSourceTab,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: vi.fn(),
  setAddSourceModalTab: vi.fn(),
  setAddSourceProcessing: vi.fn(),
  setAddSourceError: vi.fn(),
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
          },
      interpolationOptions?: Record<string, unknown>
    ) => {
      const renderTemplate = (value: string) =>
        value.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolationOptions?.[token] ?? "")
        )

      if (typeof defaultValueOrOptions === "string") {
        return renderTemplate(defaultValueOrOptions)
      }
      if (defaultValueOrOptions?.defaultValue) {
        return renderTemplate(defaultValueOrOptions.defaultValue)
      }
      return _key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    uploadMedia: mockUploadMedia,
    addMedia: mockAddMedia,
    webSearch: mockWebSearch,
    searchMedia: vi.fn().mockResolvedValue({ results: [] }),
    listMedia: vi.fn().mockResolvedValue({ media: [] }),
    updateMediaKeywords: vi.fn().mockResolvedValue(undefined)
  }
}))

describe("AddSourceModal Stage 1 error surfaces", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceModalTab = "search"
    workspaceStoreState.addSourceProcessing = false
    workspaceStoreState.addSourceError = null

    mockWebSearch.mockResolvedValue({
      results: [
        {
          title: "Result One",
          url: "https://example.com/one"
        },
        {
          title: "Result Two",
          url: "https://example.com/two"
        }
      ]
    })

    mockAddMedia
      .mockResolvedValueOnce({ results: [{ media_id: 9001, title: "One" }] })
      .mockRejectedValueOnce(new Error("timeout"))
    mockUploadMedia.mockResolvedValue({ media_id: 9100, title: "Pasted Source" })
    workspaceStoreState.setAddSourceError.mockImplementation((error: string | null) => {
      workspaceStoreState.addSourceError = error
    })
  })

  it("reports partial batch URL ingestion failures with actionable summary", async () => {
    render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Search the web..."), {
      target: { value: "example" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Search" }))

    expect(await screen.findByText("Result One")).toBeInTheDocument()
    expect(screen.getByText("Result Two")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Result One"))
    fireEvent.click(screen.getByText("Result Two"))
    fireEvent.click(screen.getByRole("button", { name: "Add 2 selected" }))

    await waitFor(() => expect(mockAddSource).toHaveBeenCalledTimes(1))
    expect(await screen.findByText("Queued as workspace source")).toBeInTheDocument()
    expect(screen.getByText("Media #9001")).toBeInTheDocument()
    expect(screen.getByText("Failed to import")).toBeInTheDocument()
    expect(
      screen.getByText("Request timed out. Retry, or try a smaller source.")
    ).toBeInTheDocument()
    expect(screen.getByText("Result Two").closest('[role="listitem"]')).toHaveClass(
      "bg-primary/10"
    )
  })

  it("keeps pasted text editable and gives an auth recovery path when paste upload is denied", async () => {
    workspaceStoreState.addSourceModalTab = "paste"
    const authError = new Error("Unauthorized")
    ;(authError as { status?: number }).status = 401
    mockUploadMedia.mockRejectedValueOnce(authError)

    const { rerender } = render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Give your content a title"), {
      target: { value: "Field notes" }
    })
    fireEvent.change(screen.getByPlaceholderText("Paste your text content here..."), {
      target: { value: "Important pasted notes that should stay editable." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add Text" }))

    await waitFor(() => {
      expect(workspaceStoreState.setAddSourceError).toHaveBeenCalledWith(
        "You need to finish server setup or sign in before adding sources. Your pasted text is still here."
      )
    })
    expect(screen.getByPlaceholderText("Give your content a title")).toHaveValue(
      "Field notes"
    )
    expect(screen.getByPlaceholderText("Paste your text content here...")).toHaveValue(
      "Important pasted notes that should stay editable."
    )
    rerender(<AddSourceModal />)
    expect(screen.getByRole("button", { name: "Retry after setup" })).toBeInTheDocument()
  })

  it("shows an auth-specific recovery message instead of a generic media load error", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    const authError = new Error("Missing API key")
    ;(authError as { status?: number }).status = 401
    vi.mocked(tldwClient.listMedia).mockRejectedValueOnce(authError)

    render(<AddSourceModal />)

    expect(
      await screen.findByText(
        "Sign in or finish server setup to browse your media library."
      )
    ).toBeInTheDocument()
  })
})
