import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import i18next from "i18next"
import common from "@/assets/locale/en/common.json"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"
import type { AddSourceTab } from "@/types/workspace"

const {
  mockUploadMedia,
  mockAddSource,
  mockUpdateMediaKeywords,
  mockCloseAddSourceModal,
  mockSetAddSourceError,
  mockTranslate
} = vi.hoisted(() => ({
  mockUploadMedia: vi.fn(),
  mockAddSource: vi.fn(),
  mockUpdateMediaKeywords: vi.fn(),
  mockCloseAddSourceModal: vi.fn(),
  mockSetAddSourceError: vi.fn(),
  mockTranslate: vi.fn()
}))
const pdfIngest = vi.hoisted(() => ({ run: vi.fn(), list: vi.fn(), pending: vi.fn(), attached: vi.fn(), scope: vi.fn() }))
vi.mock("@/services/tldw/workspace-pdf-ingest", () => ({
  ingestWorkspacePdf: pdfIngest.run,
  listWorkspacePdfUploads: pdfIngest.list,
  hasWorkspacePdfUploads: pdfIngest.pending,
  markWorkspacePdfAttached: pdfIngest.attached,
}))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: pdfIngest.scope }))

const workspaceStoreState = {
  workspaceId: "workspace-test",
  addSourceModalOpen: true,
  addSourceModalTab: "upload" as AddSourceTab,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: mockCloseAddSourceModal,
  setAddSourceModalTab: vi.fn(),
  setAddSourceProcessing: vi.fn(),
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
          },
      interpolation?: Record<string, unknown>
    ) => {
      if (mockTranslate.getMockImplementation()) {
        return mockTranslate(_key, defaultValueOrOptions, interpolation)
      }
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolation?.[token] ?? "")
        )
      }
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_m, token) => String(interpolation?.[token] ?? "")
        )
      }
      return _key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: Object.assign((
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState), { getState: () => workspaceStoreState })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    uploadMedia: mockUploadMedia,
    addMedia: vi.fn(),
    webSearch: vi.fn().mockResolvedValue({ results: [] }),
    searchMedia: vi.fn().mockResolvedValue({ media: [] }),
    listMedia: vi.fn().mockResolvedValue({ media: [] }),
    updateMediaKeywords: mockUpdateMediaKeywords
  }
}))

const getUploadInput = (): HTMLInputElement => {
  const element = document.querySelector(
    "input[type='file']"
  ) as HTMLInputElement | null
  if (!element) {
    throw new Error("Upload input not found")
  }
  return element
}

const expectWorkspaceIngestFields = (options: Record<string, unknown>) => {
  expect(options).toMatchObject({
    overwrite: "false",
    perform_chunking: "true",
    generate_embeddings: "true",
    embedding_dispatch_mode: "background"
  })
  expect(options).not.toHaveProperty("embedding_provider")
  expect(options).not.toHaveProperty("embedding_model")
}

describe("AddSourceModal Stage 1 ingestion safety", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceModalTab = "upload"
    workspaceStoreState.addSourceError = null
    workspaceStoreState.workspaceId = "workspace-test"
    pdfIngest.run.mockReset()
    pdfIngest.pending.mockReturnValue(false)
    pdfIngest.list.mockReturnValue([])
    pdfIngest.scope.mockResolvedValue({ requestScope: { config: { serverUrl: "https://owned.test", authMode: "multi-user" }, userId: 1 }, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() })
    mockUpdateMediaKeywords.mockResolvedValue(undefined)
  })

  it("keeps a PDF queued until the job completes and adds only its saved media", async () => {
    let finish!: (value: unknown) => void
    pdfIngest.run.mockImplementation(options => {
      options.onProgress({ id: "pdf", fileName: "long.pdf", fileSize: 4, status: "queued", message: "Queued for extraction" })
      return new Promise(resolve => { finish = resolve })
    })
    render(<AddSourceModal />)
    await act(async () => fireEvent.change(getUploadInput(), { target: { files: [new File(["long"], "long.pdf", { type: "application/pdf" })] } }))
    await waitFor(() => expect(pdfIngest.run).toHaveBeenCalledTimes(1))
    expect(screen.getByText("Queued for extraction")).toBeInTheDocument()
    expect(mockUploadMedia).not.toHaveBeenCalled()
    expect(mockAddSource).not.toHaveBeenCalled()
    await act(async () => finish({ mediaId: 922, data: { title: "Long book" }, receipt: { id: "pdf" } }))
    expect(mockAddSource).toHaveBeenCalledWith(expect.objectContaining({ mediaId: 922, type: "pdf" }))
  })

  it("does not add a completed PDF to a replacement workspace", async () => {
    let finish!: (value: unknown) => void
    pdfIngest.run.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const view = render(<AddSourceModal />)
    await act(async () => fireEvent.change(getUploadInput(), { target: { files: [new File(["long"], "long.pdf", { type: "application/pdf" })] } }))
    await waitFor(() => expect(pdfIngest.run).toHaveBeenCalledTimes(1))
    workspaceStoreState.workspaceId = "other-workspace"
    view.rerender(<AddSourceModal />)
    await act(async () => finish({ mediaId: 922, data: {}, receipt: { id: "pdf" } }))
    expect(mockAddSource).not.toHaveBeenCalled()
    expect(mockUpdateMediaKeywords).not.toHaveBeenCalled()
  })

  it("reopens a saved job and reconnects without selecting or uploading the file again", async () => {
    const receipt = { id: "saved", fileName: "book.pdf", fileSize: 4, status: "processing", tracking: { mode: "webui-direct", batchId: "batch", jobIds: [7] }, message: "Extracting PDF pages" }
    pdfIngest.pending.mockReturnValue(true)
    pdfIngest.list.mockReturnValue([receipt])
    pdfIngest.run.mockResolvedValue({ mediaId: 922, data: {}, receipt })
    render(<AddSourceModal />)
    fireEvent.click(await screen.findByRole("button", { name: "Check saved upload" }))
    await waitFor(() => expect(mockAddSource).toHaveBeenCalledWith(expect.objectContaining({ mediaId: 922 })))
    expect(pdfIngest.run).toHaveBeenCalledWith(expect.objectContaining({ receipt, file: undefined }))
    expect(mockUploadMedia).not.toHaveBeenCalled()
  })

  it.each([["interrupted", "Saved upload"], ["completed", "Ready to add"]])("recovers a %s job without reporting failure or claiming it is attached", async (status, label) => {
    const receipt = { id: "saved", fileName: "book.pdf", fileSize: 4, status, tracking: { mode: "webui-direct", batchId: "batch", jobIds: [7] }, message: "process" }
    pdfIngest.pending.mockReturnValue(true)
    pdfIngest.list.mockReturnValue([receipt])
    pdfIngest.run.mockResolvedValue({ mediaId: 922, data: {}, receipt })
    render(<AddSourceModal />)
    expect(await screen.findByText(label, { selector: "span" })).toBeInTheDocument()
    expect(screen.queryByText("Error", { selector: "span" })).not.toBeInTheDocument()
    if (status === "interrupted") expect(screen.queryByText("process")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Check saved upload" }))
    await waitFor(() => expect(mockAddSource).toHaveBeenCalledWith(expect.objectContaining({ mediaId: 922 })))
    expect(await screen.findByText("Added", { selector: "span" })).toBeInTheDocument()
    expect(mockUploadMedia).not.toHaveBeenCalled()
  })

  it("explains uncertain acceptance instead of suggesting a duplicate upload", async () => {
    pdfIngest.run.mockImplementation(async options => {
      options.onProgress({ id: "saved", fileName: "book.pdf", fileSize: 4, status: "unknown", message: "Upload acceptance could not be confirmed. Check My Media or ingestion Jobs before uploading again." })
      throw new Error("timeout")
    })
    render(<AddSourceModal />)
    await act(async () => fireEvent.change(getUploadInput(), { target: { files: [new File(["book"], "book.pdf", { type: "application/pdf" })] } }))
    await waitFor(() => expect(mockSetAddSourceError).toHaveBeenLastCalledWith(expect.stringContaining("Check My Media")))
  })

  it("renders failed uploads with the real nested error translations", async () => {
    const translations = i18next.createInstance()
    await translations.init({ lng: "en", resources: { en: { common } }, returnObjects: true })
    mockTranslate.mockImplementation((key: string, fallback: string) => translations.t(key, fallback))
    pdfIngest.run.mockRejectedValueOnce(new Error("timeout"))
    try {
      render(<AddSourceModal />)
      const file = new File(["report"], "report.pdf", { type: "application/pdf" })
      await act(async () => fireEvent.change(getUploadInput(), { target: { files: [file] } }))
      expect(await screen.findByText("Error", { selector: "span" })).toBeInTheDocument()
      expect(mockAddSource).not.toHaveBeenCalled()
    } finally {
      mockTranslate.mockReset()
    }
  })

  it("shows per-file upload progress fallback and marks added sources as processing", async () => {
    let resolveUpload: ((value: unknown) => void) | null = null
    const uploadPromise = new Promise((resolve) => {
      resolveUpload = resolve
    })
    mockUploadMedia.mockReturnValueOnce(uploadPromise)

    render(<AddSourceModal />)

    const file = new File(["report"], "report.docx", { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" })
    await act(async () => {
      fireEvent.change(getUploadInput(), { target: { files: [file] } })
    })

    expect(await screen.findByTestId("upload-progress-list")).toBeInTheDocument()
    expect(screen.getByText("Uploading")).toBeInTheDocument()
    expect(mockUploadMedia).toHaveBeenCalledWith(file, expect.any(Object))
    const uploadOptions = mockUploadMedia.mock.calls[0]?.[1]
    expect(uploadOptions).toMatchObject({
      media_type: "document"
    })
    expectWorkspaceIngestFields(uploadOptions)

    resolveUpload?.({
      results: [
        {
          media_id: 321,
          title: "Uploaded Report",
          created_at: "2026-02-18T09:00:00.000Z"
        }
      ]
    })

    await waitFor(() => {
      expect(mockAddSource).toHaveBeenCalledWith(
        expect.objectContaining({
          mediaId: 321,
          title: "Uploaded Report",
          status: "processing",
          sourceCreatedAt: expect.any(Date)
        })
      )
    })

    expect(screen.getByText("Processing")).toBeInTheDocument()
  })

  it("rejects oversized files before upload starts", async () => {
    render(<AddSourceModal />)

    const hugeFile = new File(["a"], "huge.pdf", { type: "application/pdf" })
    Object.defineProperty(hugeFile, "size", {
      configurable: true,
      value: 600 * 1024 * 1024
    })

    await act(async () => {
      fireEvent.change(getUploadInput(), { target: { files: [hugeFile] } })
    })

    expect(mockUploadMedia).not.toHaveBeenCalled()
    expect(mockSetAddSourceError).toHaveBeenCalledWith(
      expect.stringContaining("too large")
    )
  })

  it.each([
    {
      name: "archive.zip",
      file: new File(["zip"], "archive.zip", { type: "application/zip" })
    },
    {
      name: "legacy.doc",
      file: new File(["legacy"], "legacy.doc", { type: "application/msword" })
    }
  ])("rejects unsupported file type $name before upload starts", async ({ file }) => {
    render(<AddSourceModal />)

    await act(async () => {
      fireEvent.change(getUploadInput(), { target: { files: [file] } })
    })

    expect(mockUploadMedia).not.toHaveBeenCalled()
    expect(mockSetAddSourceError).toHaveBeenCalledWith(
      expect.stringContaining("not a supported file type")
    )
  })

  it("closes the paste modal after adding a source without waiting for workspace tagging", async () => {
    workspaceStoreState.addSourceModalTab = "paste"
    mockUploadMedia.mockResolvedValueOnce({
      results: [{ media_id: 9911, title: "Field Notes" }]
    })
    mockUpdateMediaKeywords.mockReturnValueOnce(new Promise(() => {}))

    render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Give your content a title"), {
      target: { value: "Field Notes" }
    })
    fireEvent.change(screen.getByPlaceholderText("Paste your text content here..."), {
      target: { value: "BULK-RECOVERY-EVIDENCE pasted note" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add Text" }))

    await waitFor(() => {
      expect(mockAddSource).toHaveBeenCalledWith(
        expect.objectContaining({
          mediaId: 9911,
          title: "Field Notes"
        })
      )
    })
    expect(mockUpdateMediaKeywords).toHaveBeenCalledWith(
      9911,
      {
        keywords: ["workspace:test"],
        mode: "add"
      },
      { suppressBackendUnavailableEvent: true }
    )
    expect(mockCloseAddSourceModal).toHaveBeenCalled()
  })
})
