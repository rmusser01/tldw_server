import { beforeEach, describe, expect, it, vi } from "vitest"

import type { UploadedFile } from "@/db/dexie/types"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
}))

import {
  DEFAULT_DOCUMENT_PROCESSING_MODE,
  buildChatScopedRetrievalContext,
  buildIngestIdempotencyKey,
  cancelPreparedDocumentProcessing,
  computeDocumentFileFingerprint,
  dataUrlToUploadFile,
  estimateDirectChatTokens,
  hasBlockedDocumentProcessing,
  ingestDocumentToLibrary,
  normalizeDocumentPreflightResponse,
  prepareChatDocumentAttachmentsForSend,
  setBatchDocumentProcessingMode,
  setFileDocumentProcessingMode,
  waitForIngestDocumentJob,
  withDefaultDocumentDecision,
} from "@/services/chat-document-processing"
import i18n from "i18next"

const makeUploadedFile = (
  overrides: Partial<UploadedFile> = {},
): UploadedFile =>
  ({
    id: "file-1",
    filename: "notes.md",
    type: "text/markdown",
    content: "data:text/markdown;base64,I25vdGVz",
    size: 12,
    uploadedAt: 1,
    processed: false,
    ...overrides,
  }) as UploadedFile

const makeDeps = () => ({
  processDocument: vi.fn(),
  processPdf: vi.fn(),
  ingestDocument: vi.fn(),
})

describe("chat document processing service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("keeps add-to-chat chat scoped and does not create media ids", async () => {
    const deps = makeDeps()
    const file = makeUploadedFile({ processingMode: "add_to_chat" })
    deps.processDocument.mockResolvedValueOnce({
      content: "parsed notes",
      sourceName: "notes.md",
    })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [file],
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(result.contextFiles[0]).toMatchObject({
      id: "file-1",
      content: "parsed notes",
      processed: true,
      processingMode: "add_to_chat",
      processingStatus: "ready",
    })
    expect(result.requestOverrides?.ragMediaIds).toBeUndefined()
    expect(deps.ingestDocument).not.toHaveBeenCalled()
  })

  it("uses PDF OCR processing when the file selects OCR pages", async () => {
    const deps = makeDeps()
    const file = makeUploadedFile({
      filename: "scan.pdf",
      type: "application/pdf",
      processingMode: "ocr_pages",
    })
    deps.processPdf.mockResolvedValueOnce({
      content: "ocr text",
      sourceName: "scan.pdf",
    })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [file],
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(deps.processPdf).toHaveBeenCalledWith(
      expect.objectContaining({ filename: "scan.pdf" }),
      { enableOcr: true },
    )
    expect(result.contextFiles[0]).toMatchObject({
      content: "ocr text",
      processingMode: "ocr_pages",
      processingStatus: "ready",
    })
  })

  it("returns RAG media overrides for durable ingest", async () => {
    const deps = makeDeps()
    const file = makeUploadedFile({ processingMode: "ingest_to_library" })
    deps.ingestDocument.mockResolvedValueOnce({
      mediaId: 42,
      jobId: 77,
      batchId: "batch-1",
    })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [file],
      historyId: "history-1",
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(result.contextFiles).toEqual([])
    expect(result.requestOverrides).toMatchObject({
      contextFiles: [],
      ragMediaIds: [42],
      fileRetrievalEnabled: true,
    })
    expect(deps.processDocument).not.toHaveBeenCalled()
  })

  it("does not fall back to add-to-chat when ingest fails", async () => {
    const deps = makeDeps()
    const file = makeUploadedFile({ processingMode: "ingest_to_library" })
    deps.ingestDocument.mockRejectedValueOnce(new Error("ingest failed"))

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [file],
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(result.failedFiles[0]).toMatchObject({
      processingMode: "ingest_to_library",
      processingStatus: "failed",
      processingError: "ingest failed",
    })
    expect(result.contextFiles).toEqual([])
    expect(deps.processDocument).not.toHaveBeenCalled()
  })

  it("keeps mixed chat-scoped and ingest batches out of contextFiles", async () => {
    const deps = makeDeps()
    deps.processDocument.mockResolvedValueOnce({ content: "parsed notes" })
    deps.ingestDocument.mockResolvedValueOnce({ mediaId: 42 })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [
        makeUploadedFile({ id: "chat-file", processingMode: "add_to_chat" }),
        makeUploadedFile({
          id: "ingest-file",
          filename: "library.pdf",
          type: "application/pdf",
          processingMode: "ingest_to_library",
        }),
      ],
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(result.contextFiles).toEqual([])
    expect(result.requestOverrides).toMatchObject({
      contextFiles: [],
      ragMediaIds: [42],
      fileRetrievalEnabled: true,
    })
    expect("messageForModel" in (result.requestOverrides ?? {})).toBe(false)
    expect(result.requestOverrides?.documentSnippetForModel).toContain("parsed notes")
  })

  it("normalizes backend preflight capabilities onto matching files", () => {
    const file = makeUploadedFile()

    const [normalized] = normalizeDocumentPreflightResponse(
      {
        files: [
          {
            client_id: "file-1",
            filename: "notes.md",
            media_type: "document",
            default_mode: "add_to_chat",
            modes: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "OCR unavailable: server cannot render .MD pages",
              },
              ingest_to_library: { available: true, status: "available" },
            },
            max_size_bytes: 20,
            max_pages: 200,
            max_chat_tokens: 24000,
            requires_send_time_estimate: true,
          },
        ],
      },
      [file],
    )

    expect(normalized).toMatchObject({
      processingMode: "add_to_chat",
      processingStatus: "pending",
      processingCapabilities: {
        ocr_pages: {
          available: false,
          status: "unavailable",
          reason: "OCR unavailable: server cannot render .MD pages",
        },
      },
    })
  })

  it("does not borrow a blocked reason from an unrelated processing mode", () => {
    const [normalized] = normalizeDocumentPreflightResponse(
      {
        files: [
          {
            client_id: "file-1",
            filename: "notes.md",
            media_type: "document",
            default_mode: "add_to_chat",
            modes: {
              add_to_chat: { available: false, status: "unavailable" },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "OCR is unavailable for markdown",
              },
              ingest_to_library: { available: true, status: "available" },
            },
            max_size_bytes: 20,
            max_pages: 200,
            max_chat_tokens: 24000,
          },
        ],
      },
      [makeUploadedFile({ processingMode: "add_to_chat" })],
    )

    expect(normalized.processingBlockedReason).toBe(
      "This document type is unsupported.",
    )
  })

  it.each([
    ["add_to_chat", ["switch_to_ingest"]],
    ["ingest_to_library", ["switch_to_add_to_chat"]],
  ] as const)(
    "does not suggest switching to the already blocked %s mode",
    async (processingMode, expectedActions) => {
      const result = await prepareChatDocumentAttachmentsForSend({
        files: [
          makeUploadedFile({
            processingMode,
            processingCapabilities: {
              add_to_chat: {
                available: processingMode !== "add_to_chat",
                status:
                  processingMode === "add_to_chat" ? "unavailable" : "available",
              },
              ingest_to_library: {
                available: processingMode !== "ingest_to_library",
                status:
                  processingMode === "ingest_to_library"
                    ? "unavailable"
                    : "available",
              },
            },
          }),
        ],
      })

      expect(result.blockedFiles[0]?.processingRecoveryActions).toEqual(
        expectedActions,
      )
    },
  )

  it("does not suggest switching to a target mode that is also unavailable", async () => {
    const result = await prepareChatDocumentAttachmentsForSend({
      files: [
        makeUploadedFile({
          processingMode: "add_to_chat",
          processingCapabilities: {
            add_to_chat: { available: false, status: "unavailable" },
            ingest_to_library: { available: false, status: "unavailable" },
          },
        }),
      ],
    })

    expect(result.blockedFiles[0]?.processingRecoveryActions).toEqual([])
  })

  it("does not suggest a recovery mode when its capability is missing", async () => {
    const result = await prepareChatDocumentAttachmentsForSend({
      files: [
        makeUploadedFile({
          processingMode: "add_to_chat",
          processingCapabilities: {
            add_to_chat: { available: false, status: "unavailable" }
          }
        })
      ]
    })

    expect(result.blockedFiles[0]?.processingRecoveryActions).toEqual([])
  })

  it("applies default and explicit document processing decisions", () => {
    const file = makeUploadedFile({
      processingCapabilities: {
        add_to_chat: { available: true, status: "available" },
        ocr_pages: {
          available: false,
          status: "unavailable",
          reason: "OCR unavailable",
        },
        ingest_to_library: { available: true, status: "available" },
      },
    })
    const otherFile = makeUploadedFile({ id: "file-2" })

    expect(withDefaultDocumentDecision(file)).toMatchObject({
      processingMode: DEFAULT_DOCUMENT_PROCESSING_MODE,
      processingStatus: "preflighting",
    })

    const [blocked] = setBatchDocumentProcessingMode([file], "ocr_pages")
    expect(blocked).toMatchObject({
      processingMode: "ocr_pages",
      processingStatus: "blocked",
      processingBlockedReason: "OCR unavailable",
    })
    expect(hasBlockedDocumentProcessing([blocked])).toBe(true)

    const [, updatedOther] = setFileDocumentProcessingMode(
      [blocked, otherFile],
      "file-2",
      "ingest_to_library",
    )
    expect(updatedOther).toMatchObject({
      processingMode: "ingest_to_library",
      processingStatus: "pending",
    })
  })

  it("reconstructs upload files and builds chat-scoped retrieval snippets", async () => {
    const uploadFile = await dataUrlToUploadFile(makeUploadedFile())

    expect(uploadFile.name).toBe("notes.md")
    expect(uploadFile.type).toBe("text/markdown")
    await expect(uploadFile.text()).resolves.toBe("#notes")

    expect(estimateDirectChatTokens("12345678")).toBe(2)
    expect(
      buildChatScopedRetrievalContext({
        chunks: ["alpha beta", "gamma delta"],
        query: "alpha",
        tokenBudget: 10,
      }).content,
    ).toContain("alpha beta")
  })

  it("blocks unsupported preflight results instead of defaulting to add-to-chat", () => {
    const file = makeUploadedFile({
      filename: "malware.exe",
      type: "application/x-msdownload",
    })

    const [normalized] = normalizeDocumentPreflightResponse(
      {
        files: [
          {
            client_id: "file-1",
            filename: "malware.exe",
            media_type: "unsupported",
            default_mode: null,
            modes: {
              add_to_chat: {
                available: false,
                status: "unavailable",
                reason: "Unsupported document type",
              },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "Unsupported document type",
              },
              ingest_to_library: {
                available: false,
                status: "unavailable",
                reason: "Unsupported document type",
              },
            },
            max_size_bytes: 20,
            max_pages: 200,
            max_chat_tokens: 24000,
          },
        ],
      },
      [file],
    )

    expect(normalized).toMatchObject({
      processingStatus: "blocked",
      processingBlockedReason: "This document type is unsupported.",
    })
    expect(normalized.processingMode).toBeUndefined()
  })

  it("builds stable fingerprints and scoped ingest idempotency keys", async () => {
    const file = makeUploadedFile()

    const firstFingerprint = await computeDocumentFileFingerprint(file)
    const secondFingerprint = await computeDocumentFileFingerprint(file)
    const firstKey = await buildIngestIdempotencyKey({
      file,
      historyId: "history-1",
    })
    const secondKey = await buildIngestIdempotencyKey({
      file,
      historyId: "history-1",
    })

    expect(firstFingerprint).toEqual(secondFingerprint)
    expect(firstFingerprint).toMatch(/^[a-f0-9]{64}$/)
    expect(firstKey).toEqual(secondKey)
    expect(firstKey).toBe(`chat-document-ingest:history-1:${firstFingerprint}`)
  })

  it("blocks direct chat context overflow with chat-scoped recovery actions", async () => {
    const deps = makeDeps()
    deps.processDocument.mockResolvedValueOnce({ content: "word ".repeat(50) })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [makeUploadedFile({ processingMode: "add_to_chat" })],
      maxDirectChatTokens: 10,
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
    })

    expect(result.blockedFiles[0]).toMatchObject({
      processingStatus: "blocked",
      processingRecoveryActions: [
        "use_chat_scoped_retrieval",
        "switch_to_ingest",
      ],
    })
    expect(result.recoveryActions).toEqual([
      "use_chat_scoped_retrieval",
      "switch_to_ingest",
    ])
    expect(result.requestOverrides).toBeUndefined()
  })

  it("waits for accepted ingest jobs before building retrieval overrides", async () => {
    const deps = makeDeps()
    deps.ingestDocument.mockResolvedValueOnce({
      jobId: 77,
      batchId: "batch-1",
      status: "processing",
    })
    const waitForIngestJob = vi.fn().mockResolvedValue({
      mediaId: 42,
      jobId: 77,
      batchId: "batch-1",
      status: "completed",
    })

    const result = await prepareChatDocumentAttachmentsForSend({
      files: [makeUploadedFile({ processingMode: "ingest_to_library" })],
      processDocument: deps.processDocument,
      processPdf: deps.processPdf,
      ingestDocument: deps.ingestDocument,
      waitForIngestJob,
    })

    expect(result.contextFiles).toEqual([])
    expect(result.failedFiles).toEqual([])
    expect(waitForIngestJob).toHaveBeenCalledWith(77)
    expect(result.requestOverrides).toMatchObject({
      ragMediaIds: [42],
      fileRetrievalEnabled: true,
    })
    expect(result.turnMetadata).toMatchObject({
      status: "ready",
      files: [
        {
          status: "ready",
          mode: "ingest_to_library",
        },
      ],
    })
    expect(deps.processDocument).not.toHaveBeenCalled()
  })

  it("reuses an existing non-terminal ingest job before uploading again", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      job_id: 77,
      batch_id: "batch-1",
      status: "processing",
    })

    const result = await ingestDocumentToLibrary(
      makeUploadedFile({
        processingMode: "ingest_to_library",
        ingestJobId: 77,
        ingestBatchId: "batch-1",
      }),
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/77",
        method: "GET",
      }),
    )
    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(result).toMatchObject({
      jobId: 77,
      batchId: "batch-1",
      status: "processing",
    })
  })

  it("polls an accepted ingest job to its completed media id", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      status: "completed",
      result: { media_id: 42 },
    })

    const result = await waitForIngestDocumentJob(77, { pollIntervalMs: 1 })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/media/ingest/jobs/77",
      method: "GET",
    })
    expect(result).toEqual({
      mediaId: 42,
      jobId: 77,
      status: "completed",
    })
  })

  it("localizes invalid ingest job errors", async () => {
    const translate = vi
      .spyOn(i18n, "t")
      .mockReturnValue("Localized invalid ingest job" as never)

    await expect(waitForIngestDocumentJob("invalid")).rejects.toThrow(
      "Localized invalid ingest job"
    )
    expect(translate).toHaveBeenCalledWith(
      "playground:documentProcessing.invalidIngestJobId",
      "Ingest job returned an invalid job id."
    )
    translate.mockRestore()
  })

  it("cancels local drafts and active ingest jobs", async () => {
    const cancelIngestJob = vi.fn()
    const cancelIngestBatch = vi.fn()
    const deleteDraft = vi.fn()

    await cancelPreparedDocumentProcessing(
      [
        makeUploadedFile({
          ingestJobId: 77,
          ingestBatchId: "batch-1",
          processingResultRef: { kind: "draft", id: "draft-1" },
        }),
      ],
      { cancelIngestJob, cancelIngestBatch, deleteDraft },
    )

    expect(cancelIngestBatch).toHaveBeenCalledWith("batch-1")
    expect(cancelIngestJob).toHaveBeenCalledWith(77)
    expect(deleteDraft).toHaveBeenCalledWith("draft-1")
  })

  it("continues document cleanup after cancellation failures and dedupes batches", async () => {
    const cancelIngestJob = vi.fn()
    const cancelIngestBatch = vi
      .fn()
      .mockRejectedValueOnce(new Error("batch unavailable"))
    const deleteDraft = vi.fn()
    const consoleWarn = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined)

    await cancelPreparedDocumentProcessing(
      [
        makeUploadedFile({
          id: "file-1",
          ingestJobId: 77,
          ingestBatchId: "batch-1",
          processingResultRef: { kind: "draft", id: "draft-1" },
        }),
        makeUploadedFile({
          id: "file-2",
          ingestJobId: 78,
          ingestBatchId: "batch-1",
          documentDraftId: "draft-2",
        }),
      ],
      { cancelIngestJob, cancelIngestBatch, deleteDraft },
    )

    expect(cancelIngestBatch).toHaveBeenCalledTimes(1)
    expect(cancelIngestJob).toHaveBeenCalledWith(77)
    expect(cancelIngestJob).toHaveBeenCalledWith(78)
    expect(deleteDraft).toHaveBeenCalledWith("draft-1")
    expect(deleteDraft).toHaveBeenCalledWith("draft-2")
    expect(consoleWarn).toHaveBeenCalled()
    consoleWarn.mockRestore()
  })
})
