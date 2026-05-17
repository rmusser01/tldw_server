import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"
import type { AddSourceTab } from "@/types/workspace"

const ADD_SOURCE_TAB_USAGE_STORAGE_KEY =
  "tldw:workspace-playground:add-source-tab-usage:v1"

const {
  mockUploadMedia,
  mockAddMedia,
  mockWebSearch,
  mockSearchMedia,
  mockListMedia,
  mockAddSource,
  mockCloseAddSourceModal
} = vi.hoisted(() => ({
  mockUploadMedia: vi.fn(),
  mockAddMedia: vi.fn(),
  mockWebSearch: vi.fn(),
  mockSearchMedia: vi.fn(),
  mockListMedia: vi.fn(),
  mockAddSource: vi.fn(),
  mockCloseAddSourceModal: vi.fn()
}))

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "upload" as AddSourceTab,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: mockCloseAddSourceModal,
  setAddSourceModalTab: vi.fn(),
  setAddSourceProcessing: vi.fn(),
  setAddSourceError: vi.fn(),
  addSource: mockAddSource,
  workspaceTag: "workspace:test"
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          },
      interpolation?: Record<string, unknown>
    ) => {
      const renderValue = (value: string) =>
        value.replace(/\{\{(\w+)\}\}/g, (_match, token) =>
          String(interpolation?.[token] ?? "")
        )
      if (typeof defaultValueOrOptions === "string") return renderValue(defaultValueOrOptions)
      if (defaultValueOrOptions?.defaultValue) return renderValue(defaultValueOrOptions.defaultValue)
      return key
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
    searchMedia: mockSearchMedia,
    listMedia: mockListMedia,
    updateMediaKeywords: vi.fn().mockResolvedValue(undefined)
  }
}))

describe("AddSourceModal Stage 2 intake and relevance", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.removeItem(ADD_SOURCE_TAB_USAGE_STORAGE_KEY)
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceError = null
    workspaceStoreState.sources = []
    workspaceStoreState.addSourceModalTab = "upload"
    mockWebSearch.mockResolvedValue({ results: [] })
    mockSearchMedia.mockResolvedValue({ results: [] })
    mockListMedia.mockResolvedValue({ media: [] })
  })

  it("uploads pasted text with an explicit document media_type", async () => {
    workspaceStoreState.addSourceModalTab = "paste"
    mockUploadMedia.mockResolvedValueOnce({
      results: [{ media_id: 9101, title: "Pasted Note" }]
    })

    render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Give your content a title"), {
      target: { value: "Pasted Note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Paste your text content here..."), {
      target: { value: "workspace pasted content" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add Text" }))

    await waitFor(() => {
      expect(mockUploadMedia).toHaveBeenCalledWith(expect.any(File), expect.any(Object))
    })

    const uploadOptions = mockUploadMedia.mock.calls[0]?.[1]
    expect(uploadOptions).toMatchObject({
      title: "Pasted Note",
      media_type: "document",
      overwrite: "false",
      perform_chunking: "true",
      generate_embeddings: "true",
      embedding_dispatch_mode: "background"
    })
    expect(uploadOptions).not.toHaveProperty("embedding_provider")
    expect(uploadOptions).not.toHaveProperty("embedding_model")
  })

  it("orders tabs as Upload, Library, URL, Paste, Search", async () => {
    render(<AddSourceModal />)

    const tabLabels = screen
      .getAllByRole("tab")
      .map((tab) => tab.textContent?.replace(/\s+/g, " ").trim())

    expect(tabLabels).toEqual([
      "Upload",
      "My Media",
      "URL",
      "Paste",
      "Search Server"
    ])
  })

  it("keeps visible tabs in a stable order despite prior usage frequency", async () => {
    window.localStorage.setItem(
      ADD_SOURCE_TAB_USAGE_STORAGE_KEY,
      JSON.stringify({
        upload: 0,
        existing: 2,
        url: 5,
        paste: 1,
        search: 9
      })
    )

    render(<AddSourceModal />)

    const tabLabels = screen
      .getAllByRole("tab")
      .map((tab) => tab.textContent?.replace(/\s+/g, " ").trim())

    expect(tabLabels).toEqual([
      "Upload",
      "My Media",
      "URL",
      "Paste",
      "Search Server"
    ])
  })

  it("shows a load error when My Media cannot load", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia.mockRejectedValueOnce(new Error("offline"))

    render(<AddSourceModal />)

    expect(await screen.findByText(/unable to load media/i)).toBeInTheDocument()
  })

  it("renders My Media items from items response shape", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia.mockResolvedValueOnce({
      items: [{ id: 701, title: "Library Item", type: "pdf" }],
      total: 1
    })

    render(<AddSourceModal />)

    expect(await screen.findByText("Library Item")).toBeInTheDocument()
    expect(screen.getByText("Showing 1 of 1")).toBeInTheDocument()
  })

  it("uses backend pagination.total_items so large media libraries can load more", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia.mockResolvedValueOnce({
      items: [
        { id: 701, title: "Library Item", type: "pdf" },
        { id: 702, title: "Second Library Item", type: "video" }
      ],
      pagination: {
        page: 1,
        per_page: 2,
        total_pages: 63,
        results_per_page: 2,
        total_items: 125
      }
    })

    render(<AddSourceModal />)

    expect(await screen.findByText("Library Item")).toBeInTheDocument()
    expect(screen.getByText("Second Library Item")).toBeInTheDocument()
    expect(screen.getByText("Showing 2 of 125")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Load more" })).toBeInTheDocument()
  })

  it("distinguishes all-added media from an empty media library", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    workspaceStoreState.sources = [{ mediaId: 701 }]
    mockListMedia.mockResolvedValueOnce({
      items: [{ id: 701, title: "Already Added", type: "pdf" }],
      total: 1
    })

    render(<AddSourceModal />)

    expect(
      await screen.findByText(/already in this workspace/i)
    ).toBeInTheDocument()
  })

  it("toggles a My Media checkbox once when clicked directly", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia.mockResolvedValueOnce({
      items: [{ id: 701, title: "Library Item", type: "pdf" }],
      total: 1
    })

    render(<AddSourceModal />)

    const checkbox = await screen.findByRole("checkbox", {
      name: /select library item/i
    })
    fireEvent.click(checkbox)
    expect(checkbox).toBeChecked()

    fireEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
  })

  it("persists updated tab usage when switching tabs", async () => {
    render(<AddSourceModal />)

    fireEvent.click(screen.getByRole("tab", { name: "Search Server" }))

    await waitFor(() => {
      const raw = window.localStorage.getItem(ADD_SOURCE_TAB_USAGE_STORAGE_KEY)
      expect(raw).toBeTruthy()
      const parsed = JSON.parse(raw || "{}")
      expect(parsed.search).toBeGreaterThan(0)
    })
  })

  it("supports batch URL ingestion with per-URL status reporting", async () => {
    workspaceStoreState.addSourceModalTab = "url"
    mockAddMedia
      .mockResolvedValueOnce({ results: [{ media_id: 8001, title: "One" }] })
      .mockRejectedValueOnce(new Error("timeout"))

    render(<AddSourceModal />)

    fireEvent.click(screen.getByRole("button", { name: "Batch (one per line)" }))
    fireEvent.change(screen.getByPlaceholderText(/article-1/), {
      target: {
        value: "https://example.com/one\nhttps://example.com/two"
      }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add URLs" }))

    await waitFor(() => {
      expect(mockAddSource).toHaveBeenCalledWith(
        expect.objectContaining({
          mediaId: 8001,
          status: "processing"
        })
      )
    })
    expect(mockAddMedia).toHaveBeenNthCalledWith(
      1,
      "https://example.com/one",
      expect.objectContaining({
        perform_chunking: "true",
        generate_embeddings: "true",
        embedding_dispatch_mode: "background"
      })
    )
    expect(mockAddMedia).toHaveBeenNthCalledWith(
      2,
      "https://example.com/two",
      expect.objectContaining({
        perform_chunking: "true",
        generate_embeddings: "true",
        embedding_dispatch_mode: "background"
      })
    )
    expect(mockAddMedia.mock.calls[0]?.[1]).not.toHaveProperty("embedding_provider")
    expect(mockAddMedia.mock.calls[0]?.[1]).not.toHaveProperty("embedding_model")
    expect(mockAddMedia.mock.calls[1]?.[1]).not.toHaveProperty("embedding_provider")
    expect(mockAddMedia.mock.calls[1]?.[1]).not.toHaveProperty("embedding_model")

    expect(screen.getByText("https://example.com/one")).toBeInTheDocument()
    expect(screen.getByText("https://example.com/two")).toBeInTheDocument()
    expect(screen.getByText("Added")).toBeInTheDocument()
    expect(screen.getByText(/Unable to reach the server|timed out/i)).toBeInTheDocument()
    expect(mockCloseAddSourceModal).not.toHaveBeenCalled()
  })

  it("normalizes metadata from URL ingestion responses", async () => {
    workspaceStoreState.addSourceModalTab = "url"
    mockAddMedia.mockResolvedValueOnce({
      results: [
        {
          media_id: 9001,
          title: "Metadata Doc",
          media_type: "pdf",
          file_size: 4096,
          page_count: 12,
          url: "https://example.com/doc"
        }
      ]
    })

    render(<AddSourceModal />)

    fireEvent.change(
      screen.getByPlaceholderText("https://example.com/article or YouTube URL"),
      {
        target: { value: "https://example.com/doc" }
      }
    )
    fireEvent.click(screen.getByRole("button", { name: "Add URL" }))

    await waitFor(() => {
      expect(mockAddSource).toHaveBeenCalledWith(
        expect.objectContaining({
          mediaId: 9001,
          type: "pdf",
          fileSize: 4096,
          pageCount: 12,
          url: "https://example.com/doc"
        })
      )
    })
    expect(mockAddMedia).toHaveBeenCalledWith(
      "https://example.com/doc",
      expect.objectContaining({
        perform_chunking: "true",
        generate_embeddings: "true",
        embedding_dispatch_mode: "background"
      })
    )
    expect(mockAddMedia.mock.calls[0]?.[1]).not.toHaveProperty("embedding_provider")
    expect(mockAddMedia.mock.calls[0]?.[1]).not.toHaveProperty("embedding_model")
  })

  it("renders search snippets and favicon hints in web results", async () => {
    workspaceStoreState.addSourceModalTab = "search"
    mockWebSearch.mockResolvedValueOnce({
      results: [
        {
          title: "Climate Result",
          url: "https://example.com/climate",
          snippet: "Key findings about climate mitigation strategies."
        }
      ]
    })

    render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Search the web..."), {
      target: { value: "climate" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Search" }))

    expect(
      await screen.findByText("Key findings about climate mitigation strategies.")
    ).toBeInTheDocument()

    const favicon = screen.getByTestId("search-result-favicon-0")
    expect(favicon).toHaveAttribute("src", expect.stringContaining("google.com/s2/favicons"))
  })

  it("supports library load-more pagination and total count text", async () => {
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia
      .mockResolvedValueOnce({
        media: [
          { id: 1, title: "Doc 1", type: "pdf" },
          { id: 2, title: "Doc 2", type: "pdf" }
        ],
        total_count: 4
      })
      .mockResolvedValueOnce({
        media: [
          { id: 3, title: "Doc 3", type: "pdf" },
          { id: 4, title: "Doc 4", type: "pdf" }
        ],
        total_count: 4
      })

    render(<AddSourceModal />)

    expect(await screen.findByText("Showing 2 of 4")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Load more" }))

    expect(await screen.findByText("Showing 4 of 4")).toBeInTheDocument()
    expect(mockListMedia).toHaveBeenNthCalledWith(1, {
      page: 1,
      results_per_page: 50,
      include_keywords: true
    })
    expect(mockListMedia).toHaveBeenNthCalledWith(2, {
      page: 2,
      results_per_page: 50,
      include_keywords: true
    })
  })
})
