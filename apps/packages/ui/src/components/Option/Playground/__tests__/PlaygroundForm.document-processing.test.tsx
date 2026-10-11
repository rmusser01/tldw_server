// @vitest-environment jsdom
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { UploadedFile } from "@/db/dexie/types"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { createDefaultActorSettings } from "@/types/actor"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { usePlaygroundSubmit, type UsePlaygroundSubmitDeps } from "../hooks/usePlaygroundSubmit"
import type { PreparedChatDocumentAttachments } from "@/services/chat-document-processing"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { defaultEmbeddingModelForRag } from "~/services/tldw-server"

const prepareChatDocumentAttachmentsForSend = vi.hoisted(() => vi.fn())
const getActorSceneForChat = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat }))

type ActorSceneResult = Awaited<ReturnType<typeof getActorSceneForChat>>

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(complete => { resolve = complete })
  return { promise, resolve }
}

vi.mock("@/services/chat-document-processing", () => ({
  documentProcessingSelectionKey: (files: UploadedFile[]) =>
    files.map((file) => file.id).join("|"),
  prepareChatDocumentAttachmentsForSend
}))

vi.mock("@/components/Chat/composer/hooks/useComposerSubmit", () => ({
  useComposerSubmit: ({ sendMessage }: { sendMessage: (payload: any) => Promise<any> }) => ({
    dispatch: async (
      payload: any,
      options?: { beforeSend?: () => void; afterSend?: (result: unknown) => void }
    ) => {
      options?.beforeSend?.()
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

const requestScope = { config: { serverUrl: "https://outer-command.example", authMode: "single-user" as const }, userId: 1 }
const historySelection = { getCurrent: () => ({ status: "idle", owner: null, view: null }), fence: () => () => true,
  draftAccount: { requestScope, isCurrent: () => true } } as unknown as HistorySelectionController

const baseDeps = (overrides: Partial<UsePlaygroundSubmitDeps> = {}) => ({
  historySelection,
  form: makeForm(),
  draftReady: true,
  restorePromptAssistAttempt: vi.fn(),
  beginPromptAssistReset: vi.fn(() => 1),
  markPromptAssistAttemptSaved: vi.fn(),
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
  sendMessage: vi.fn(async (_payload: { message: string; image: string; requestOverrides?: Record<string, unknown> }) => ({ status: "submitted" as const })),
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
    useStoreMessageOption.getState().startNewChatDraft()
    useStoreMessageOption.setState({ historyId: null, serverChatId: null })
    getActorSceneForChat.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(),
      originalRecord: null, origin: "default", profileUnavailable: false }))
  })

  it.each(["scene", "documents"].flatMap(wait => ["model settings", "enabled tools"].map(setting => ({ wait, setting }))))("refuses changed $setting during outer full $wait preparation", async ({ wait, setting }) => {
    useStoreChatModelSettings.getState().reset();
    useStoreMessageOption.setState({ selectedModel: "openai:gpt-4o-mini", toolChoice: "auto" });
    useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{ name: "outer-tool", inputSchema: { type: "object" } }] }).chatTools });
    const scene = deferred<ActorSceneResult>();
    const documents = deferred<PreparedChatDocumentAttachments>();
    const waiting = wait === "scene" ? scene.promise : documents.promise;
    const boundary = wait === "scene" ? getActorSceneForChat : prepareChatDocumentAttachmentsForSend;
    if (wait === "scene") getActorSceneForChat.mockReturnValueOnce(scene.promise);
    else prepareChatDocumentAttachmentsForSend.mockReturnValueOnce(documents.promise);
    const deps = baseDeps({ uploadedFiles: wait === "scene" ? [] : [makeFile()] });
    const view = renderHook(() => usePlaygroundSubmit(deps));
    act(() => { view.result.current.submitForm(); });
    await waitFor(() => expect(boundary).toHaveBeenCalledOnce());
    act(() => {
      if (setting === "model settings") useStoreChatModelSettings.getState().updateSetting("temperature", 0.19);
      else useMcpToolsStore.setState({ chatTools: [] });
    });
    await act(async () => {
      if (wait === "scene") scene.resolve({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false });
      else documents.resolve({ contextFiles: [], failedFiles: [], blockedFiles: [], recoveryActions: [], requestOverrides: { contextFiles: [], uploadedFiles: [] }, turnMetadata: { status: "ready", files: [] } });
      await waiting;
    });
    expect(deps.sendMessage).not.toHaveBeenCalled();
    expect(deps.form.reset).not.toHaveBeenCalled();
  });
  it.each(["scene", "web search"])("keeps newly attached files and input after cancellation during empty-file %s preparation", async (boundary) => {
    const scene = deferred<ActorSceneResult>()
    const embedding = deferred<Awaited<ReturnType<typeof defaultEmbeddingModelForRag>>>()
    const waiting = boundary === "scene" ? scene.promise : embedding.promise
    const held = boundary === "scene" ? getActorSceneForChat : vi.mocked(defaultEmbeddingModelForRag)
    if (boundary === "scene") getActorSceneForChat.mockReturnValueOnce(scene.promise)
    else vi.mocked(defaultEmbeddingModelForRag).mockReturnValueOnce(embedding.promise)
    const deps = baseDeps({ uploadedFiles: [], webSearch: boundary === "web search" })
    const newlyAttached = makeFile({ id: "new-unsent-file" })
    const originalDocuments = [{ id: 7, title: "Original selected tab" }]
    const view = renderHook(() => {
      const [files, setFiles] = React.useState<UploadedFile[]>([])
      const [documents, setDocuments] = React.useState(originalDocuments)
      const form = useSimpleForm({ initialValues: { message: "KEEP ORIGINAL INPUT", image: "" } })
      const submit = usePlaygroundSubmit({ ...deps, form, uploadedFiles: files, selectedDocuments: documents,
        clearUploadedFiles: () => setFiles([]), clearSelectedDocuments: () => setDocuments([]) })
      return { submit, form, files, documents, setFiles }
    })
    act(() => { view.result.current.submit.submitForm() })
    await waitFor(() => expect(held).toHaveBeenCalledOnce())
    act(() => { view.result.current.setFiles([newlyAttached]) })
    await act(async () => {
      if (boundary === "scene") scene.resolve({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
      else embedding.resolve("embedding-model")
      await waiting
    })
    expect(deps.sendMessage).not.toHaveBeenCalled()
    expect(view.result.current.files).toEqual([newlyAttached])
    expect(view.result.current.documents).toEqual(originalDocuments)
    expect(view.result.current.form.getValues().message).toBe("KEEP ORIGINAL INPUT")
    view.unmount()
  })

  it.each(["scene", "documents"])("allows New to send again before the old %s preparation returns", async (boundary) => {
    const scene = deferred<ActorSceneResult>()
    const documents = deferred<PreparedChatDocumentAttachments>()
    const waiting = boundary === "scene" ? scene.promise : documents.promise
    const held = boundary === "scene" ? getActorSceneForChat : prepareChatDocumentAttachmentsForSend
    if (boundary === "scene") getActorSceneForChat.mockReturnValueOnce(scene.promise)
    else prepareChatDocumentAttachmentsForSend.mockReturnValueOnce(documents.promise)
    const deps = baseDeps({ uploadedFiles: [] })
    const view = renderHook(() => {
      const [files, setFiles] = React.useState<UploadedFile[]>(boundary === "documents" ? [makeFile()] : [])
      const form = useSimpleForm({ initialValues: { message: "OLD HELD INPUT", image: "" } })
      return { files, setFiles, form, submit: usePlaygroundSubmit({ ...deps, form, uploadedFiles: files }) }
    })
    act(() => { view.result.current.submit.submitForm() })
    await waitFor(() => expect(held).toHaveBeenCalledOnce())
    try {
      act(() => {
        useStoreMessageOption.getState().startNewChatDraft()
        view.result.current.setFiles([])
        view.result.current.form.setFieldValue("message", "NEW CHAT INPUT")
      })
      act(() => { view.result.current.submit.submitForm() })
      await waitFor(() => expect(deps.sendMessage).toHaveBeenCalledOnce())
      expect(deps.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "NEW CHAT INPUT" }))
      expect(view.result.current.submit.isPreparingDocuments).toBe(false)
    } finally {
      await act(async () => {
        if (boundary === "scene") scene.resolve({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
        else documents.resolve({ contextFiles: [], failedFiles: [], blockedFiles: [], recoveryActions: [], requestOverrides: { contextFiles: [], uploadedFiles: [] }, turnMetadata: { status: "ready", files: [] } })
        await waiting
      })
      view.unmount()
    }
    expect(deps.sendMessage).toHaveBeenCalledOnce()
  })

  it("captures the original scene before document preparation and forwards private ownership separately", async () => {
    const original = { ...createDefaultActorSettings(), notes: "Original direct scene before documents" }
    let stored = original, readsBeforeDocuments = -1
    getActorSceneForChat.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored),
      originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
    prepareChatDocumentAttachmentsForSend.mockImplementation(async () => {
      readsBeforeDocuments = getActorSceneForChat.mock.calls.length
      stored = { ...original, notes: "Newer saved scene during documents" }
      return { contextFiles: [], failedFiles: [], blockedFiles: [], requestOverrides: {}, turnMetadata: { status: "ready", files: [] } }
    })
    const deps = baseDeps()
    const view = renderHook(() => usePlaygroundSubmit(deps))
    await act(async () => { view.result.current.submitForm() })
    await waitFor(() => expect(deps.sendMessage).toHaveBeenCalledOnce())
    expect(readsBeforeDocuments).toBe(1)
    expect(deps.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope,
      scene: expect.objectContaining({ settings: original }) }) }))
    expect(vi.mocked(deps.sendMessage).mock.calls[0][0].requestOverrides).not.toHaveProperty("actorSceneSend")
  })

  it.each([false, true])("keeps a newer form edit during a held scene read (undo=%s)", async (undo) => {
    let release!: (value: ActorSceneResult) => void
    const pending = new Promise<ActorSceneResult>(resolve => { release = resolve })
    getActorSceneForChat.mockReturnValueOnce(pending)
    const deps = baseDeps({ uploadedFiles: [] })
    const view = renderHook(() => {
      const form = useSimpleForm({ initialValues: { message: "Original typed question", image: "" } })
      return { form, submit: usePlaygroundSubmit({ ...deps, form }) }
    })
    act(() => { view.result.current.submit.submitForm() })
    await waitFor(() => expect(getActorSceneForChat).toHaveBeenCalledOnce())
    act(() => {
      view.result.current.form.setFieldValue("message", "Newer typed question")
      if (undo) view.result.current.form.setFieldValue("message", "Original typed question")
    })
    await act(async () => {
      release({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(),
        originalRecord: null, origin: "default", profileUnavailable: false })
      await pending
    })
    expect(deps.sendMessage).not.toHaveBeenCalled()
    expect(view.result.current.form.getValues().message).toBe(undo ? "Original typed question" : "Newer typed question")
    expect(deps.clearUploadedFiles).not.toHaveBeenCalled()
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
      usePlaygroundSubmit(baseDeps({ sendMessage, clearUploadedFiles }))
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
        })
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
        baseDeps({ form, sendMessage, clearUploadedFiles, notificationApi })
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
        })
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
      usePlaygroundSubmit(baseDeps({ sendMessage }))
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
          baseDeps({ uploadedFiles: files, sendMessage })
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


it("Full saved Roleplay reaches its sender with no live model or Scene read", async () => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({
    serverChatId: "saved-roleplay-form",
    serverChatAssistantKind: "character",
    temporaryChat: false
  })
  const owner = {
    kind: "native" as const,
    conversation_id: "saved-roleplay-form",
    owner_key: "owned-roleplay",
    scope: { type: "global" as const },
    request_scope: requestScope,
    validate_lease: () => true
  }
  const controller = {
    ...historySelection,
    getCurrent: () => ({ status: "ready", owner })
  } as unknown as HistorySelectionController
  getActorSceneForChat
    .mockReset()
    .mockRejectedValue(new Error("live Scene must not be read"))
  const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))
  const deps = baseDeps({
    historySelection: controller,
    uploadedFiles: [],
    selectedModel: "",
    sendMessage
  })
  const mounted = renderHook(() => usePlaygroundSubmit(deps))
  act(() => mounted.result.current.submitForm())
  await waitFor(() => expect(sendMessage).toHaveBeenCalledOnce())
  expect(getActorSceneForChat).not.toHaveBeenCalled()
  expect(deps.validateSelectedChatModelsAvailability).not.toHaveBeenCalled()
})


it("Full saved Roleplay refuses extra documents before ingestion or optimistic reset", async () => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({
    serverChatId: "saved-roleplay-form",
    serverChatAssistantKind: "character",
    temporaryChat: false
  })
  const owner = {
    kind: "native" as const,
    conversation_id: "saved-roleplay-form",
    owner_key: "owned-roleplay",
    scope: { type: "global" as const },
    request_scope: requestScope,
    validate_lease: () => true
  }
  const controller = {
    ...historySelection,
    getCurrent: () => ({ status: "ready", owner })
  } as unknown as HistorySelectionController
  prepareChatDocumentAttachmentsForSend.mockClear()
  const deps = baseDeps({ historySelection: controller })
  const mounted = renderHook(() => usePlaygroundSubmit(deps))
  await act(async () => {
    mounted.result.current.submitForm()
    await Promise.resolve()
  })
  expect(prepareChatDocumentAttachmentsForSend).not.toHaveBeenCalled()
  expect(deps.sendMessage).not.toHaveBeenCalled()
  expect(deps.form.reset).not.toHaveBeenCalled()
  expect(deps.notificationApi.error).toHaveBeenCalled()
})
