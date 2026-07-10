// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { UploadedFile } from "@/db/dexie/types"
import { usePlaygroundQueueManagement } from "../usePlaygroundQueueManagement"

const prepareChatDocumentAttachmentsForSend = vi.hoisted(() => vi.fn())
const queueMock = vi.hoisted(() => ({
  args: null as any,
  enqueue: vi.fn((input) => input)
}))

vi.mock("@/services/chat-document-processing", () => ({
  prepareChatDocumentAttachmentsForSend
}))

vi.mock("@/components/Chat/composer/hooks/useComposerQueue", () => ({
  useComposerQueue: (args: any) => {
    queueMock.args = args
    return {
      queuedRequestActions: [],
      enqueue: queueMock.enqueue,
      cancelCurrentAndRunDisabledReason: null,
      handleRunQueuedRequest: vi.fn(),
      handleRunNextQueuedRequest: vi.fn()
    }
  }
}))

vi.mock("@/utils/chat-model-availability", () => ({
  buildAvailableChatModelIds: vi.fn(() => new Set(["openai:gpt-4o-mini"])),
  findUnavailableChatModel: vi.fn(() => null),
  normalizeChatModelId: vi.fn((model: string | null | undefined) => model ?? "")
}))

vi.mock("../usage-metrics", () => ({
  projectTokenBudget: vi.fn(() => ({
    isOverLimit: false,
    isNearLimit: false
  }))
}))

const makeFile = (): UploadedFile => ({
  id: "file-1",
  filename: "scan.pdf",
  type: "application/pdf",
  content: "data:application/pdf;base64,abc",
  size: 1024,
  uploadedAt: 1,
  processed: false,
  processingMode: "ingest_to_library",
  processingStatus: "pending",
  processingCapabilities: {
    add_to_chat: { available: true, status: "available" },
    ingest_to_library: { available: true, status: "available" }
  },
  ingestIdempotencyKey: "chat-document-ingest:history:file-1"
})

const t = (_key: string, fallback?: string) => fallback || _key

const baseDeps = (overrides: Record<string, unknown> = {}) => ({
  composerModels: [],
  isConnectionReady: false,
  isSending: false,
  selectedModel: "openai:gpt-4o-mini",
  chatMode: "normal",
  webSearch: false,
  compareMode: false,
  compareModeActive: false,
  compareSelectedModels: [],
  selectedSystemPrompt: "",
  selectedQuickPrompt: null,
  toolChoice: "auto",
  useOCR: false,
  selectedDocuments: [],
  uploadedFiles: [makeFile()],
  contextFiles: [],
  documentContext: [],
  queuedMessages: [],
  setQueuedMessages: vi.fn(),
  historyId: "history-1",
  serverChatId: null,
  conversationTokenCount: 0,
  resolvedMaxContext: 100_000,
  estimateTokensForText: vi.fn(() => 1),
  characterContextTokenEstimate: 0,
  pinnedSourceTokenEstimate: 0,
  currentContextSnapshot: {},
  setLastSubmittedContext: vi.fn(),
  setSelectedModel: vi.fn(),
  setChatMode: vi.fn(),
  setWebSearch: vi.fn(),
  setCompareMode: vi.fn(),
  setCompareSelectedModels: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  setToolChoice: vi.fn(),
  setUseOCR: vi.fn(),
  compareModelsSupportCapability: vi.fn(() => true),
  sendMessage: vi.fn(),
  stopStreamingRequest: vi.fn(),
  form: { setFieldError: vi.fn(), reset: vi.fn() },
  clearSelectedDocuments: vi.fn(),
  clearUploadedFiles: vi.fn(),
  textAreaFocus: vi.fn(),
  notificationApi: {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  },
  t,
  ...overrides
})

describe("usePlaygroundQueueManagement document processing", () => {
  beforeEach(() => {
    queueMock.args = null
    queueMock.enqueue.mockClear()
    prepareChatDocumentAttachmentsForSend.mockReset()
  })

  it("queues replayable document upload metadata instead of blocking document drafts", () => {
    const { result } = renderHook(() =>
      usePlaygroundQueueManagement(baseDeps() as any)
    )

    expect(queueMock.args.isQueuedDispatchBlocked).toBe(false)

    act(() => {
      result.current.queueSubmission({
        promptText: "summarize",
        image: "",
        intent: {
          message: "summarize",
          isImageCommand: false
        }
      })
    })

    expect(queueMock.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        blockedReason: null,
        sourceContext: expect.objectContaining({
          uploadedFiles: [
            expect.objectContaining({
              filename: "scan.pdf",
              content: "data:application/pdf;base64,abc",
              processingMode: "ingest_to_library",
              ingestIdempotencyKey: "chat-document-ingest:history:file-1"
            })
          ]
        })
      })
    )
  })

  it("replays queued document uploads through send-time preparation", async () => {
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
    renderHook(() =>
      usePlaygroundQueueManagement(
        baseDeps({
          uploadedFiles: [],
          sendMessage
        }) as any
      )
    )

    await act(async () => {
      await queueMock.args.sendQueuedRequest({
        id: "queued-1",
        promptText: "summarize",
        image: "",
        snapshot: {
          selectedModel: "openai:gpt-4o-mini",
          chatMode: "normal",
          webSearch: false,
          compareMode: false,
          compareSelectedModels: [],
          selectedSystemPrompt: "",
          selectedQuickPrompt: "",
          toolChoice: "auto",
          useOCR: false
        },
        sourceContext: {
          uploadedFiles: [makeFile()]
        }
      })
    })

    expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalledWith(
      expect.objectContaining({
        files: [expect.objectContaining({ filename: "scan.pdf" })],
        historyId: "history-1"
      })
    )
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        requestOverrides: expect.objectContaining({
          contextFiles: [],
          uploadedFiles: [],
          ragMediaIds: [42],
          fileRetrievalEnabled: true,
          messageForModel: "summarize\n\nparsed notes",
          userMetadataExtra: {
            documentProcessing: turnMetadata
          }
        })
      })
    )
  })

  it("does not mutate composer state when queued document preparation is blocked", async () => {
    prepareChatDocumentAttachmentsForSend.mockResolvedValue({
      contextFiles: [],
      failedFiles: [],
      blockedFiles: [
        {
          ...makeFile(),
          processingStatus: "blocked",
          processingBlockedReason: "Ingest is unavailable"
        }
      ],
      recoveryActions: ["switch_to_add_to_chat"],
      requestOverrides: undefined,
      turnMetadata: {
        status: "blocked",
        files: []
      }
    })
    const deps = baseDeps({ uploadedFiles: [] })
    renderHook(() => usePlaygroundQueueManagement(deps as any))

    await expect(
      act(async () => {
        await queueMock.args.sendQueuedRequest({
          id: "queued-blocked",
          promptText: "summarize",
          image: "",
          snapshot: {
            selectedModel: "openai:gpt-4o-mini",
            chatMode: "normal",
            webSearch: false,
            compareMode: false,
            compareSelectedModels: [],
            selectedSystemPrompt: "",
            selectedQuickPrompt: "",
            toolChoice: "auto",
            useOCR: false
          },
          sourceContext: {
            uploadedFiles: [makeFile()]
          }
        })
      })
    ).rejects.toThrow("Ingest is unavailable")

    expect(deps.setSelectedModel).not.toHaveBeenCalled()
    expect(deps.setChatMode).not.toHaveBeenCalled()
    expect(deps.setWebSearch).not.toHaveBeenCalled()
    expect(deps.setLastSubmittedContext).not.toHaveBeenCalled()
  })
})
