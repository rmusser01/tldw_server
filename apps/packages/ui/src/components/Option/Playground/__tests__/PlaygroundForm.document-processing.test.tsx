// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { UploadedFile } from "@/db/dexie/types"
import { usePlaygroundSubmit } from "../hooks/usePlaygroundSubmit"

const prepareChatDocumentAttachmentsForSend = vi.hoisted(() => vi.fn())

vi.mock("@/services/chat-document-processing", () => ({
  documentProcessingSelectionKey: (files: UploadedFile[]) =>
    files.map((file) => file.id).join("|"),
  prepareChatDocumentAttachmentsForSend
}))

vi.mock("@/components/Chat/composer/hooks/useComposerSubmit", () => ({
  useComposerSubmit: ({ sendMessage }: { sendMessage: (payload: any) => Promise<any> }) => ({
    dispatch: async (
      payload: any,
      options?: { afterSend?: (result: unknown) => void }
    ) => {
      const result = await sendMessage(payload)
      options?.afterSend?.(result)
      return result
    }
  })
}))

vi.mock("~/services/tldw-server", () => ({
  defaultEmbeddingModelForRag: vi.fn(async () => "embedding-model")
}))

vi.mock("@/services/search", () => ({
  getIsSimpleInternetSearch: vi.fn(async () => false)
}))

vi.mock("@/utils/rag-format", () => ({
  formatPinnedResults: vi.fn(() => "")
}))

vi.mock("@/utils/chat-model-availability", () => ({
  normalizeChatModelId: vi.fn((model: string | null | undefined) => model ?? "")
}))

vi.mock("../usage-metrics", () => ({
  projectTokenBudget: vi.fn(() => ({
    isOverLimit: false,
    isNearLimit: false
  }))
}))

const makeFile = (
  overrides: Partial<UploadedFile> = {}
): UploadedFile => ({
  id: "file-1",
  filename: "scan.pdf",
  type: "application/pdf",
  content: "data:application/pdf;base64,abc",
  size: 1024,
  uploadedAt: 1,
  processed: false,
  processingMode: "ingest_to_library",
  processingStatus: "pending",
  ...overrides
})

const makeForm = (message = "summarize") => ({
  onSubmit: (callback: (value: any) => void | Promise<void>) =>
    () => void callback({ message, image: "" }),
  setFieldValue: vi.fn(),
  setFieldError: vi.fn(),
  reset: vi.fn()
})

const t = (_key: string, fallback?: any) =>
  typeof fallback === "string" ? fallback : _key

const baseDeps = (overrides: Record<string, unknown> = {}) => ({
  form: makeForm(),
  isSending: false,
  isConnectionReady: true,
  webSearch: false,
  compareModeActive: false,
  compareSelectedModels: [],
  selectedModel: "openai:gpt-4o-mini",
  fileRetrievalEnabled: false,
  ragPinnedResults: [],
  selectedDocuments: [],
  uploadedFiles: [makeFile()],
  currentContextSnapshot: {},
  conversationTokenCount: 0,
  characterContextTokenEstimate: 0,
  pinnedSourceTokenEstimate: 0,
  resolvedMaxContext: 100_000,
  jsonMode: false,
  openUIRequestMode: false,
  sendMessage: vi.fn(async () => ({ status: "submitted" })),
  clearOpenUIRequestMode: vi.fn(),
  clearSelectedDocuments: vi.fn(),
  clearUploadedFiles: vi.fn(),
  textAreaFocus: vi.fn(),
  setLastSubmittedContext: vi.fn(),
  estimateTokensForText: vi.fn(() => 1),
  resolveSubmissionIntent: vi.fn((message: string) => ({
    message,
    handled: false,
    invalidImageCommand: false,
    isImageCommand: false
  })),
  queueSubmission: vi.fn(),
  validateSelectedChatModelsAvailability: vi.fn(() => true),
  compareModelsSupportCapability: vi.fn(() => true),
  notificationApi: {
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn()
  },
  t,
  ...overrides
})

describe("Playground document processing submit", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("prepares selected ingest documents and sends explicit retrieval overrides", async () => {
    const turnMetadata = {
      status: "ready",
      files: [
        {
          id: "file-1",
          filename: "scan.pdf",
          mode: "ingest_to_library",
          status: "ready"
        }
      ]
    }
    prepareChatDocumentAttachmentsForSend.mockResolvedValue({
      contextFiles: [],
      failedFiles: [],
      blockedFiles: [],
      recoveryActions: [],
      requestOverrides: {
        contextFiles: [],
        uploadedFiles: [],
        ragMediaIds: [42],
        fileRetrievalEnabled: true,
        documentSnippetForModel: "parsed notes",
        documentProcessing: turnMetadata
      },
      turnMetadata
    })
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const clearUploadedFiles = vi.fn()
    const { result } = renderHook(() =>
      usePlaygroundSubmit(baseDeps({ sendMessage, clearUploadedFiles }) as any)
    )

    await act(async () => {
      result.current.submitForm()
    })

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
    expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalledWith(
      expect.objectContaining({
        files: [expect.objectContaining({ filename: "scan.pdf" })],
        historyId: undefined,
        sessionId: undefined
      })
    )
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "summarize",
        requestOverrides: expect.objectContaining({
          contextFiles: [],
          uploadedFiles: [],
          ragMediaIds: [42],
          fileRetrievalEnabled: true,
          messageForModel: "summarize\n\nparsed notes",
          userMetadataExtra: {
            documentProcessing: expect.objectContaining({
              ...turnMetadata,
              status: "sending_prompt"
            })
          }
        })
      })
    )
    expect(clearUploadedFiles).toHaveBeenCalledTimes(1)
  })

  it("reserves and updates a visible document-processing turn while preparing files", async () => {
    const turnMetadata = {
      status: "ready",
      files: [
        {
          id: "file-1",
          filename: "scan.pdf",
          mode: "ingest_to_library",
          status: "ready"
        }
      ]
    }
    prepareChatDocumentAttachmentsForSend.mockResolvedValue({
      contextFiles: [],
      failedFiles: [],
      blockedFiles: [],
      recoveryActions: [],
      requestOverrides: {
        contextFiles: [],
        uploadedFiles: [],
        ragMediaIds: [42],
        fileRetrievalEnabled: true,
        documentProcessing: turnMetadata
      },
      turnMetadata
    })
    const reserveDocumentProcessingTurn = vi.fn(() => "reserved-user-message")
    const updateDocumentProcessingTurn = vi.fn()
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const { result } = renderHook(() =>
      usePlaygroundSubmit(
        baseDeps({
          sendMessage,
          reserveDocumentProcessingTurn,
          updateDocumentProcessingTurn
        }) as any
      )
    )

    await act(async () => {
      result.current.submitForm()
    })

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
    expect(reserveDocumentProcessingTurn).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "summarize",
        metadata: expect.objectContaining({
          status: "waiting_for_files",
          files: [
            expect.objectContaining({
              id: "file-1",
              filename: "scan.pdf",
              mode: "ingest_to_library",
              status: "pending"
            })
          ]
        })
      })
    )
    expect(
      reserveDocumentProcessingTurn.mock.invocationCallOrder[0]
    ).toBeLessThan(
      prepareChatDocumentAttachmentsForSend.mock.invocationCallOrder[0]
    )
    expect(updateDocumentProcessingTurn).toHaveBeenNthCalledWith(
      1,
      "reserved-user-message",
      expect.objectContaining({ status: "processing" })
    )
    expect(updateDocumentProcessingTurn).toHaveBeenNthCalledWith(
      2,
      "reserved-user-message",
      expect.objectContaining({ status: "sending_prompt" })
    )
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        requestOverrides: expect.objectContaining({
          userMessageId: "reserved-user-message",
          userMetadataExtra: {
            documentProcessing: expect.objectContaining({
              status: "sending_prompt"
            })
          }
        })
      })
    )
  })

  it("blocks send and keeps attachments when document preparation is blocked", async () => {
    const blockedFile = makeFile({
      processingStatus: "blocked",
      processingBlockedReason: "Document text is too large"
    })
    prepareChatDocumentAttachmentsForSend.mockResolvedValue({
      contextFiles: [],
      failedFiles: [],
      blockedFiles: [blockedFile],
      recoveryActions: ["switch_to_ingest"],
      requestOverrides: undefined,
      turnMetadata: {
        status: "blocked",
        files: [
          {
            id: "file-1",
            filename: "scan.pdf",
            mode: "ingest_to_library",
            status: "blocked",
            summary: "Document text is too large"
          }
        ]
      }
    })
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const form = makeForm()
    const clearUploadedFiles = vi.fn()
    const notificationApi = {
      warning: vi.fn(),
      error: vi.fn(),
      info: vi.fn()
    }
    const { result } = renderHook(() =>
      usePlaygroundSubmit(
        baseDeps({ form, sendMessage, clearUploadedFiles, notificationApi }) as any
      )
    )

    await act(async () => {
      result.current.submitForm()
    })

    await waitFor(() =>
      expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalled()
    )
    expect(sendMessage).not.toHaveBeenCalled()
    expect(form.reset).not.toHaveBeenCalled()
    expect(clearUploadedFiles).not.toHaveBeenCalled()
    expect(notificationApi.error).toHaveBeenCalled()
  })

  it("updates the reserved document-processing turn when preparation is blocked", async () => {
    const blockedMetadata = {
      status: "blocked",
      files: [
        {
          id: "file-1",
          filename: "scan.pdf",
          mode: "ingest_to_library",
          status: "blocked",
          summary: "Document text is too large"
        }
      ]
    }
    prepareChatDocumentAttachmentsForSend.mockResolvedValue({
      contextFiles: [],
      failedFiles: [],
      blockedFiles: [
        makeFile({
          processingStatus: "blocked",
          processingBlockedReason: "Document text is too large"
        })
      ],
      recoveryActions: ["switch_to_ingest"],
      requestOverrides: undefined,
      turnMetadata: blockedMetadata
    })
    const reserveDocumentProcessingTurn = vi.fn(() => "reserved-user-message")
    const updateDocumentProcessingTurn = vi.fn()
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const { result } = renderHook(() =>
      usePlaygroundSubmit(
        baseDeps({
          sendMessage,
          reserveDocumentProcessingTurn,
          updateDocumentProcessingTurn
        }) as any
      )
    )

    await act(async () => {
      result.current.submitForm()
    })

    await waitFor(() =>
      expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalled()
    )
    expect(sendMessage).not.toHaveBeenCalled()
    expect(updateDocumentProcessingTurn).toHaveBeenLastCalledWith(
      "reserved-user-message",
      blockedMetadata
    )
  })

  it("ignores duplicate submits while document preparation is pending", async () => {
    let resolvePreparation: (value: any) => void = () => undefined
    prepareChatDocumentAttachmentsForSend.mockReturnValue(
      new Promise((resolve) => {
        resolvePreparation = resolve
      })
    )
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const { result } = renderHook(() =>
      usePlaygroundSubmit(baseDeps({ sendMessage }) as any)
    )

    act(() => {
      result.current.submitForm()
      result.current.submitForm()
    })

    await waitFor(() =>
      expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalledTimes(1)
    )

    await act(async () => {
      resolvePreparation({
        contextFiles: [],
        failedFiles: [],
        blockedFiles: [],
        recoveryActions: [],
        requestOverrides: {
          contextFiles: [],
          uploadedFiles: [],
          ragMediaIds: [42],
          fileRetrievalEnabled: true
        },
        turnMetadata: { status: "ready", files: [] }
      })
      await Promise.resolve()
    })

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
  })

  it("aborts pending preparation when the selected attachments change", async () => {
    let resolvePreparation: (value: any) => void = () => undefined
    let preparationSignal: AbortSignal | undefined
    prepareChatDocumentAttachmentsForSend.mockImplementation((options: any) => {
      preparationSignal = options.signal
      return new Promise((resolve) => {
        resolvePreparation = resolve
      })
    })
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const { result, rerender } = renderHook(
      ({ files }) =>
        usePlaygroundSubmit(
          baseDeps({ uploadedFiles: files, sendMessage }) as any
        ),
      { initialProps: { files: [makeFile()] } }
    )

    act(() => {
      result.current.submitForm()
    })
    await waitFor(() =>
      expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalledTimes(1)
    )

    rerender({ files: [] })
    expect(preparationSignal?.aborted).toBe(true)

    await act(async () => {
      resolvePreparation({
        contextFiles: [],
        failedFiles: [],
        blockedFiles: [],
        recoveryActions: [],
        requestOverrides: {
          contextFiles: [],
          uploadedFiles: [],
          ragMediaIds: [42],
          fileRetrievalEnabled: true
        },
        turnMetadata: { status: "ready", files: [] }
      })
      await Promise.resolve()
    })

    expect(sendMessage).not.toHaveBeenCalled()
  })
})
