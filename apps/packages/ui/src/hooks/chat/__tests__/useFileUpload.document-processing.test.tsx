// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { UploadedFile } from "@/db/dexie/types"
import { useFileUpload } from "../useFileUpload"

const preflightDocumentUpload = vi.hoisted(() => vi.fn())
const cancelPreparedDocumentProcessing = vi.hoisted(() => vi.fn())
let resolvePreflight: (value: unknown) => void = () => undefined

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "file-1"
}))

vi.mock("~/utils/file-processor", () => ({
  processFileUpload: vi.fn(async () => ({
    content: "data:text/plain;base64,aGVsbG8="
  }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    preflightDocumentUpload
  }
}))

vi.mock("@/services/chat-document-processing", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/chat-document-processing")>()
  return {
    ...actual,
    cancelPreparedDocumentProcessing
  }
})

const useHarness = ({
  initialContextFiles = [],
  initialUploadedFiles = []
}: {
  initialContextFiles?: UploadedFile[]
  initialUploadedFiles?: UploadedFile[]
} = {}) => {
  const [uploadedFiles, setUploadedFiles] =
    React.useState<UploadedFile[]>(initialUploadedFiles)
  const [contextFiles, setContextFiles] =
    React.useState<UploadedFile[]>(initialContextFiles)
  const upload = useFileUpload({
    maxContextFileSizeBytes: 20 * 1024 * 1024,
    maxContextFileSizeLabel: "20 MB",
    notification: { error: vi.fn() },
    t: (_key: string, fallback: string) => fallback,
    uploadedFiles,
    setUploadedFiles,
    contextFiles,
    setContextFiles
  })

  return {
    ...upload,
    uploadedFiles,
    contextFiles
  }
}

describe("useFileUpload document processing", () => {
  beforeEach(() => {
    preflightDocumentUpload.mockReset()
    cancelPreparedDocumentProcessing.mockReset()
    cancelPreparedDocumentProcessing.mockResolvedValue(undefined)
    preflightDocumentUpload.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreflight = resolve
        })
    )
  })

  it("stages documents as chat-only and applies backend preflight capabilities", async () => {
    const { result } = renderHook(() => useHarness())

    await act(async () => {
      await result.current.handleFileUpload(
        new File(["hello"], "notes.pdf", { type: "application/pdf" })
      )
    })

    await waitFor(() => {
      expect(result.current.uploadedFiles[0]).toMatchObject({
        id: "file-1",
        filename: "notes.pdf",
        processingMode: "add_to_chat",
        processingStatus: "preflighting"
      })
    })
    expect(result.current.contextFiles[0]).toMatchObject({
      id: "file-1",
      processingMode: "add_to_chat"
    })

    await act(async () => {
      resolvePreflight({
        files: [
          {
            client_id: "file-1",
            filename: "notes.pdf",
            media_type: "pdf",
            default_mode: "add_to_chat",
            max_size_bytes: 20 * 1024 * 1024,
            max_pages: 200,
            max_chat_tokens: 24000,
            modes: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: { available: true, status: "available" },
              ingest_to_library: { available: true, status: "available" }
            }
          }
        ]
      })
    })

    await waitFor(() => {
      expect(result.current.uploadedFiles[0]).toMatchObject({
        id: "file-1",
        processingStatus: "pending",
        processingCapabilities: {
          ocr_pages: { available: true, status: "available" }
        }
      })
    })
    expect(preflightDocumentUpload).toHaveBeenCalledWith({
      files: [
        {
          client_id: "file-1",
          filename: "notes.pdf",
          mime_type: "application/pdf",
          size_bytes: 5
        }
      ]
    })
  })

  it("marks uploaded documents blocked when backend preflight fails", async () => {
    preflightDocumentUpload.mockRejectedValueOnce(new Error("preflight down"))
    const { result } = renderHook(() => useHarness())

    await act(async () => {
      await result.current.handleFileUpload(
        new File(["hello"], "notes.pdf", { type: "application/pdf" })
      )
    })

    await waitFor(() => {
      expect(result.current.uploadedFiles[0]).toMatchObject({
        id: "file-1",
        filename: "notes.pdf",
        processingMode: "add_to_chat",
        processingStatus: "blocked",
        processingBlockedReason:
          "Document preflight failed. Try again or remove the file."
      })
    })
    expect(result.current.contextFiles[0]).toMatchObject({
      id: "file-1",
      processingStatus: "blocked"
    })
  })

  it("cleans up draft-backed document processing when a file is removed", async () => {
    const draftBackedFile: UploadedFile = {
      id: "file-1",
      filename: "notes.pdf",
      type: "application/pdf",
      content: "",
      size: 5,
      uploadedAt: Date.now(),
      processed: false,
      processingMode: "add_to_chat",
      processingStatus: "processing",
      processingResultRef: { kind: "draft", id: "draft-1" }
    }
    const { result } = renderHook(() =>
      useHarness({
        initialContextFiles: [draftBackedFile],
        initialUploadedFiles: [draftBackedFile]
      })
    )

    await act(async () => {
      await result.current.removeUploadedFile("file-1")
    })

    expect(cancelPreparedDocumentProcessing).toHaveBeenCalledWith([
      draftBackedFile
    ])
    expect(result.current.uploadedFiles).toEqual([])
    expect(result.current.contextFiles).toEqual([])
  })
})
