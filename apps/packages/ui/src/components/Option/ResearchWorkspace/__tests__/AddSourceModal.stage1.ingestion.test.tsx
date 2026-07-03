import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"
import type { AddSourceTab } from "@/types/workspace"

const {
  mockUploadMedia,
  mockAddSource,
  mockUpdateMediaKeywords,
  mockCloseAddSourceModal,
  mockSetAddSourceError
} = vi.hoisted(() => ({
  mockUploadMedia: vi.fn(),
  mockAddSource: vi.fn(),
  mockUpdateMediaKeywords: vi.fn(),
  mockCloseAddSourceModal: vi.fn(),
  mockSetAddSourceError: vi.fn()
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
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
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
    mockUpdateMediaKeywords.mockResolvedValue(undefined)
  })

  it("shows per-file upload progress fallback and marks added sources as processing", async () => {
    let resolveUpload: ((value: unknown) => void) | null = null
    const uploadPromise = new Promise((resolve) => {
      resolveUpload = resolve
    })
    mockUploadMedia.mockReturnValueOnce(uploadPromise)

    render(<AddSourceModal />)

    const file = new File(["report"], "report.pdf", { type: "application/pdf" })
    await act(async () => {
      fireEvent.change(getUploadInput(), { target: { files: [file] } })
    })

    expect(await screen.findByTestId("upload-progress-list")).toBeInTheDocument()
    expect(screen.getByText("Uploading")).toBeInTheDocument()
    expect(mockUploadMedia).toHaveBeenCalledWith(file, expect.any(Object))
    const uploadOptions = mockUploadMedia.mock.calls[0]?.[1]
    expect(uploadOptions).toMatchObject({
      media_type: "pdf"
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
