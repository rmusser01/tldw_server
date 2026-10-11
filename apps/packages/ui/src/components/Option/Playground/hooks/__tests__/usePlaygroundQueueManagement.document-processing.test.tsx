// @vitest-environment jsdom
import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { buildQueuedRequest } from "@/utils/chat-request-queue"
import type { UploadedFile } from "@/db/dexie/types"
import type { PreparedChatDocumentAttachments } from "@/services/chat-document-processing"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { createDefaultActorSettings } from "@/types/actor"
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import type { HistorySelectionController, HistoryLoadReceipt } from "@/hooks/chat/useHistorySelection"
import { usePlaygroundQueueManagement, type UsePlaygroundQueueManagementDeps } from "../usePlaygroundQueueManagement"

const prepareChatDocumentAttachmentsForSend = vi.hoisted(() => vi.fn())
const getActorSceneForChat = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat }))

type ActorSceneResult = Awaited<ReturnType<typeof getActorSceneForChat>>

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(complete => { resolve = complete })
  return { promise, resolve }
}
const queueMock = vi.hoisted(() => ({
  args: null as any,
  enqueue: vi.fn((input) => input)
}))

vi.mock("@/services/chat-document-processing", () => ({
  prepareChatDocumentAttachmentsForSend
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    subscribeInvalidation: vi.fn(() => () => undefined)
  }
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

const formAst = ts.createSourceFile("PlaygroundForm.tsx", readFileSync(path.resolve(__dirname, "../../PlaygroundForm.tsx"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let formModelScopeEffect: ts.ExpressionStatement | undefined
const findFormModelScopeEffect = (node: ts.Node) => {
  if (ts.isExpressionStatement(node) && ts.isCallExpression(node.expression) && node.expression.expression.getText(formAst) === "React.useEffect" && node.getText(formAst).includes("setActiveChatModelSettingsScope(selectedModelSettingsScope ?? null)")) formModelScopeEffect = node
  ts.forEachChild(node, findFormModelScopeEffect)
}
findFormModelScopeEffect(formAst)
if (!formModelScopeEffect) throw new Error("Missing actual form model-scope publication effect")
const useFormModelScope = new Function("React", "useStoreChatModelSettings", ts.transpileModule(`return function useFormModelScope(selectedModelSettingsScope) {
  const setActiveChatModelSettingsScope = useStoreChatModelSettings(state => state.setActiveSettingsScope);
  ${formModelScopeEffect.getText(formAst)}
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText)(React, useStoreChatModelSettings) as (scope: string | null) => void

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

const requestScope = { config: { serverUrl: "https://outer-command.example", authMode: "single-user" as const }, userId: 1 }
const acceptedLocal: HistoryLoadReceipt = { owner: { kind: "local", conversation_id: "history-1", profile_id: "outer-profile", owner_key: "outer-owner" },
  view: { conversation_id: "history-1", owner_key: "outer-owner" } as HistoryLoadReceipt["view"] }
const historySelection = { getCurrent: () => ({ status: "ready", ...acceptedLocal }), fence: () => () => true,
  draftAccount: { requestScope, isCurrent: () => true } } as unknown as HistorySelectionController

const baseDeps = (overrides: Record<string, unknown> = {}) => ({
  historySelection,
  composerModels: [],
  isConnectionReady: true,
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
  sendMessage: vi.fn<UsePlaygroundQueueManagementDeps["sendMessage"]>(),
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
    useStoreMessageOption.getState().startNewChatDraft()
    expect(captureChatDraftLoad("switch")(acceptedLocal)).toBe(true)
    useStoreMessageOption.setState({ historyId: "history-1", serverChatId: null })
    getActorSceneForChat.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(),
      originalRecord: null, origin: "default", profileUnavailable: false }))
    useStoreChatModelSettings.getState().reset()
    useMcpToolsStore.setState({ healthState: "healthy", chatTools: [] })
    queueMock.args = null
    queueMock.enqueue.mockClear()
    prepareChatDocumentAttachmentsForSend.mockReset()
  })

  it.each(["scene", "documents"].flatMap(boundary => ["model settings", "model scope", "enabled tools"].map(setting => ({ boundary, setting }))))("rejects changed $setting during queued full $boundary preparation", async ({ boundary, setting }) => {
    useStoreMessageOption.setState({ selectedModel: "ambient-model", toolChoice: "none" })
    useMcpToolsStore.setState({ chatTools: buildChatToolFilterState({ tools: [{ name: "queued-full-tool", inputSchema: { type: "object" } }] }).chatTools })
    const scene = deferred<ActorSceneResult>()
    const documents = deferred<PreparedChatDocumentAttachments>()
    const held = boundary === "scene" ? getActorSceneForChat : prepareChatDocumentAttachmentsForSend
    if (boundary === "scene") getActorSceneForChat.mockReturnValueOnce(scene.promise)
    else prepareChatDocumentAttachmentsForSend.mockReturnValueOnce(documents.promise)
    const core = useStoreMessageOption.getState()
    const deps = baseDeps({ uploadedFiles: [], setSelectedModel: vi.fn(core.setSelectedModel), setToolChoice: core.setToolChoice, sendMessage: vi.fn(async () => ({ status: "submitted" })) })
    const view = renderHook(() => usePlaygroundQueueManagement(deps))
    const item = buildQueuedRequest({ promptText: "KEEP FULL QUEUED INPUT", snapshot: { selectedModel: "openai:gpt-4o-mini", toolChoice: "auto" },
      sourceContext: { uploadedFiles: boundary === "documents" ? [makeFile()] : [] } })
    const pending = queueMock.args.sendQueuedRequest(item, () => {})
    await waitFor(() => expect(held).toHaveBeenCalledOnce())
    act(() => {
      if (setting === "model settings") useStoreChatModelSettings.getState().updateSetting("temperature", 0.17)
      else if (setting === "model scope") useStoreChatModelSettings.getState().setActiveSettingsScope("external:scope")
      else useMcpToolsStore.setState({ chatTools: [] })
    })
    await act(async () => {
      if (boundary === "scene") scene.resolve({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
      else documents.resolve({ contextFiles: [], failedFiles: [], blockedFiles: [], recoveryActions: [], requestOverrides: { contextFiles: [], uploadedFiles: [] }, turnMetadata: { status: "ready", files: [] } })
      await expect(pending).rejects.toThrow()
    })
    expect(deps.sendMessage).not.toHaveBeenCalled()
    expect(deps.setSelectedModel).toHaveBeenCalledOnce()
    expect(deps.setLastSubmittedContext).not.toHaveBeenCalled()
    expect(item.promptText).toBe("KEEP FULL QUEUED INPUT")
    expect(useStoreMessageOption.getState()).toMatchObject({ selectedModel: "openai:gpt-4o-mini", toolChoice: "none" })
    if (setting === "model settings") expect(useStoreChatModelSettings.getState().getEffectiveSettings().temperature).toBe(0.17)
    else if (setting === "model scope") expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("external:scope")
    else expect(useMcpToolsStore.getState().chatTools).toEqual([])
    view.unmount()
  })

  it("captures queued model scope before preparation and survives the actual form scope-publication effect", async () => {
    useStoreMessageOption.setState({ selectedModel: "openai:ambient-model", toolChoice: "none" })
    useStoreChatModelSettings.getState().setActiveSettingsScope("openai:ambient-model")
    useStoreChatModelSettings.getState().updateScopedSetting("openai:gpt-4o-mini", "temperature", 0.41)
    useMcpToolsStore.setState({ chatTools: buildChatToolFilterState({ tools: [{ name: "queued-full-tool", inputSchema: { type: "object" } }] }).chatTools })
    let release!: (value: ActorSceneResult) => void
    const waiting = new Promise<ActorSceneResult>(resolve => { release = resolve })
    getActorSceneForChat.mockReturnValueOnce(waiting)
    const core = useStoreMessageOption.getState()
    const composerModels = [{ provider: "openai", model: "gpt-4o-mini" }, { provider: "openai", model: "ambient-model" }]
    const deps = baseDeps({ composerModels, uploadedFiles: [], setSelectedModel: core.setSelectedModel, setToolChoice: core.setToolChoice,
      sendMessage: vi.fn(async () => ({ status: "submitted" })) })
    const view = renderHook(() => {
      const selectedModel = useStoreMessageOption(state => state.selectedModel)
      const settings = useStoreChatModelSettings()
      const selected = resolveModelSelectorSelection(composerModels, selectedModel, settings.apiProvider,
        { scope: settings.activeSettingsScope, providerOverride: settings.activeSettingsScope ? settings.scopedSettingsByModelKey[settings.activeSettingsScope]?.apiProvider : undefined })
      useFormModelScope(selected.settingsScope)
      return usePlaygroundQueueManagement(deps)
    })
    const item = buildQueuedRequest({ promptText: "INTENDED FULL QUEUE", snapshot: { selectedModel: "openai:gpt-4o-mini", toolChoice: "auto", selectedSystemPrompt: "INTENDED QUEUED SYSTEM" } })
    let pending!: Promise<unknown>
    act(() => { pending = queueMock.args.sendQueuedRequest(item, () => {}) })
    await waitFor(() => expect(getActorSceneForChat).toHaveBeenCalledOnce())
    try {
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:gpt-4o-mini")
      expect(useStoreChatModelSettings.getState().getEffectiveSettings().temperature).toBe(0.41)
    } finally {
      await act(async () => {
        release({ target: getActorSceneForChat.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
        await pending
      })
      view.unmount()
    }
    expect(deps.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "INTENDED FULL QUEUE", requestOverrides: expect.objectContaining({ selectedModel: "openai:gpt-4o-mini", toolChoice: "auto", selectedSystemPrompt: "INTENDED QUEUED SYSTEM" }) }))
  })

  it("captures the executing scene before queued document preparation without persisting ownership in the queue", async () => {
    const original = { ...createDefaultActorSettings(), notes: "Original queued execution scene" }
    let stored = original, readsBeforeDocuments = -1
    getActorSceneForChat.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored),
      originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
    prepareChatDocumentAttachmentsForSend.mockImplementation(async () => {
      readsBeforeDocuments = getActorSceneForChat.mock.calls.length
      stored = { ...original, notes: "Newer saved scene during documents" }
      return { contextFiles: [], failedFiles: [], blockedFiles: [], requestOverrides: {}, turnMetadata: { status: "ready", files: [] } }
    })
    const deps = baseDeps({ sendMessage: vi.fn(async () => ({ status: "submitted" })) })
    const view = renderHook(() => usePlaygroundQueueManagement(deps))
    act(() => { view.result.current.queueSubmission({ promptText: "Original queued question", image: "", intent: { message: "Original queued question", isImageCommand: false } }) })
    const enqueued = queueMock.enqueue.mock.calls[0][0]
    expect(enqueued).not.toHaveProperty("actorSceneSend")
    expect(enqueued.sourceContext).not.toHaveProperty("actorSceneSend")
    const item = buildQueuedRequest({ ...enqueued, snapshot: queueMock.args.buildQueuedRequestSnapshot() })
    await act(async () => { await queueMock.args.sendQueuedRequest(item, () => {}) })
    expect(readsBeforeDocuments).toBe(1)
    expect(deps.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope,
      scene: expect.objectContaining({ settings: original }) }) }))
    expect(deps.sendMessage.mock.calls[0][0].requestOverrides).not.toHaveProperty("actorSceneSend")
  })

  it("queues replayable document upload metadata instead of blocking document drafts", () => {
    const { result } = renderHook(() =>
      usePlaygroundQueueManagement(baseDeps())
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
        })
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
      }, () => {})
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

  it("retains composer text, files and context when document preparation refuses the intended queued model setup", async () => {
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
    useStoreMessageOption.setState({ selectedModel: "openai:ambient-model", toolChoice: "none" })
    const originalFiles = [{ ...makeFile(), id: "current-unsent-file" }]
    const originalDocuments = [{ id: 8, title: "Current unsent tab" }]
    const originalContext = { summary: "Current composer context" }
    const deps = baseDeps({ setSelectedModel: vi.fn(useStoreMessageOption.getState().setSelectedModel) })
    const view = renderHook(() => {
      const [files, setFiles] = React.useState(originalFiles)
      const [documents, setDocuments] = React.useState(originalDocuments)
      const [context, setContext] = React.useState<Record<string, unknown>>(originalContext)
      const form = useSimpleForm({ initialValues: { message: "CURRENT UNSENT INPUT", image: "" } })
      const queue = usePlaygroundQueueManagement({ ...deps, form, uploadedFiles: files, selectedDocuments: documents,
        clearUploadedFiles: () => { deps.clearUploadedFiles(); setFiles([]) },
        clearSelectedDocuments: () => { deps.clearSelectedDocuments(); setDocuments([]) },
        setLastSubmittedContext: (next: Record<string, unknown>) => { deps.setLastSubmittedContext(next); setContext(next) } })
      return { queue, form, files, documents, context }
    })

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
        }, () => {})
      })
    ).rejects.toThrow("Ingest is unavailable")

    expect(deps.setSelectedModel).toHaveBeenCalledWith("openai:gpt-4o-mini")
    expect(useStoreMessageOption.getState()).toMatchObject({ selectedModel: "openai:gpt-4o-mini", toolChoice: "none" })
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:gpt-4o-mini")
    expect(deps.setChatMode).not.toHaveBeenCalled()
    expect(deps.setWebSearch).not.toHaveBeenCalled()
    expect(deps.setCompareMode).not.toHaveBeenCalled()
    expect(deps.setCompareSelectedModels).not.toHaveBeenCalled()
    expect(deps.setSelectedSystemPrompt).not.toHaveBeenCalled()
    expect(deps.setSelectedQuickPrompt).not.toHaveBeenCalled()
    expect(deps.setToolChoice).not.toHaveBeenCalled()
    expect(deps.setUseOCR).not.toHaveBeenCalled()
    expect(deps.setLastSubmittedContext).not.toHaveBeenCalled()
    expect(deps.sendMessage).not.toHaveBeenCalled()
    expect(deps.clearUploadedFiles).not.toHaveBeenCalled()
    expect(deps.clearSelectedDocuments).not.toHaveBeenCalled()
    expect(view.result.current.form.getValues().message).toBe("CURRENT UNSENT INPUT")
    expect(view.result.current.files).toEqual(originalFiles)
    expect(view.result.current.documents).toEqual(originalDocuments)
    expect(view.result.current.context).toEqual(originalContext)
    view.unmount()
  })
})

it("rejects the queued document continuation before publishing remaining settings, context or dispatch", async () => {
  let release!: (value: unknown) => void
  const waiting = new Promise(resolve => { release = resolve })
  prepareChatDocumentAttachmentsForSend.mockReset()
  prepareChatDocumentAttachmentsForSend.mockReturnValue(waiting)
  const deps = baseDeps({ uploadedFiles: [] })
  let ready = true
  const guard = vi.fn(() => { if (!ready) throw new Error("connection_unavailable") })
  renderHook(() => usePlaygroundQueueManagement(deps as UsePlaygroundQueueManagementDeps))
  const pending = queueMock.args.sendQueuedRequest(buildQueuedRequest({ promptText: "KEEP QUEUED DOCUMENT", sourceContext: { uploadedFiles: [makeFile()] }, snapshot: { selectedModel: "openai:gpt-4o-mini" } }), guard)
  await waitFor(() => expect(prepareChatDocumentAttachmentsForSend).toHaveBeenCalledOnce())
  ready = false
  release({ blockedFiles: [], failedFiles: [], requestOverrides: {}, contextFiles: [], turnMetadata: {} })
  await expect(pending).rejects.toThrow("connection_unavailable")
  expect(deps.sendMessage).not.toHaveBeenCalled()
  expect(deps.setSelectedModel).toHaveBeenCalledWith("openai:gpt-4o-mini")
  expect(deps.setChatMode).not.toHaveBeenCalled()
  expect(deps.setToolChoice).not.toHaveBeenCalled()
  expect(deps.clearUploadedFiles).not.toHaveBeenCalled()
  expect(deps.clearSelectedDocuments).not.toHaveBeenCalled()
  expect(deps.setLastSubmittedContext).not.toHaveBeenCalled()
})

it("does not publish queued full settings or context after the model setter synchronously opens New", async () => {
  useStoreMessageOption.getState().startNewChatDraft();
  expect(captureChatDraftLoad("switch")(acceptedLocal)).toBe(true);
  useStoreMessageOption.setState({ historyId: "history-1", serverChatId: null, selectedModel: null });
  getActorSceneForChat.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }));
  const core = useStoreMessageOption.getState();
  const deps = baseDeps({ uploadedFiles: [], setSelectedModel: core.setSelectedModel, setChatMode: core.setChatMode, setWebSearch: core.setWebSearch,
    setSelectedSystemPrompt: core.setSelectedSystemPrompt, setToolChoice: core.setToolChoice, setUseOCR: core.setUseOCR });
  renderHook(() => usePlaygroundQueueManagement(deps));
  const replacement = { historyId: "replacement-full-queue", serverChatId: null, selectedModel: "replacement-model", chatMode: "rag", webSearch: true, selectedSystemPrompt: "REPLACEMENT SYSTEM PROMPT", toolChoice: "none" as const, useOCR: true };
  const unsubscribe = useStoreMessageOption.subscribe(state => {
    if (state.selectedModel !== "openai:gpt-4o-mini") return;
    unsubscribe(); state.startNewChatDraft(); useStoreMessageOption.setState(replacement);
  });
  try {
    const item = buildQueuedRequest({ promptText: "OLD FULL QUEUE", snapshot: { selectedModel: "openai:gpt-4o-mini", chatMode: "normal", webSearch: false, selectedSystemPrompt: "OLD SYSTEM", toolChoice: "auto", useOCR: false } });
    await expect(queueMock.args.sendQueuedRequest(item, () => {})).rejects.toThrow();
    expect(useStoreMessageOption.getState()).toMatchObject(replacement);
    expect(deps.setLastSubmittedContext).not.toHaveBeenCalled();
    expect(deps.sendMessage).not.toHaveBeenCalled();
  } finally { unsubscribe(); }
});


it("an image queue without an optional caller fence cannot send after synchronous New", async () => {
  vi.mocked(getActorSceneForChat).mockClear()
  useStoreMessageOption.getState().startNewChatDraft(false)
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  const replacement = [{ id: "new-image-source", isBot: false, name: "You", message: "NEW SOURCE", sources: [] }]
  const deps = baseDeps({
    historyId: null, uploadedFiles: [],
    setLastSubmittedContext: vi.fn(() => {
      useStoreMessageOption.getState().startNewChatDraft(false)
      useStoreMessageOption.setState({ historyId: null, serverChatId: null, messages: replacement })
    })
  })
  const hook = renderHook(() => usePlaygroundQueueManagement(deps))
  act(() => {
    hook.result.current.queueSubmission({
      promptText: "Original image prompt", image: "",
      intent: { message: "Original image prompt", isImageCommand: true, imageBackendOverride: "openai" }
    })
  })
  const enqueued = queueMock.enqueue.mock.calls.at(-1)![0]
  const item = buildQueuedRequest({ ...enqueued, snapshot: queueMock.args.buildQueuedRequestSnapshot() })
  await expect(queueMock.args.sendQueuedRequest(item, () => {})).rejects.toThrow("This chat or setup changed")
  expect(deps.sendMessage).not.toHaveBeenCalled()
  expect(getActorSceneForChat).not.toHaveBeenCalled()
  expect(useStoreMessageOption.getState().messages).toEqual(replacement)
  hook.unmount()
})


it("Full saved Roleplay queue uses saved setup with no current model or Scene read and no settings publication", async () => {
  useStoreMessageOption.getState().startNewChatDraft()
  const conversation_id = "saved-roleplay-queue"
  const owner = {
    kind: "native" as const,
    conversation_id,
    owner_key: "saved-roleplay-owner",
    scope: { type: "global" as const },
    request_scope: requestScope,
    validate_lease: () => true
  }
  useStoreMessageOption.setState({
    serverChatId: conversation_id,
    serverChatAssistantKind: "character",
    temporaryChat: false
  })
  const selected = {
    ...historySelection,
    getCurrent: () => ({ status: "ready", owner })
  } as unknown as HistorySelectionController
  getActorSceneForChat
    .mockReset()
    .mockRejectedValue(new Error("live Scene forbidden"))
  const deps = baseDeps({
    historySelection: selected,
    historyId: null,
    serverChatId: conversation_id,
    selectedModel: "",
    uploadedFiles: [],
    sendMessage: vi.fn(async () => ({ status: "submitted" }))
  })
  renderHook(() => usePlaygroundQueueManagement(deps))
  const item = buildQueuedRequest({
    promptText: "  Exact queued reply\n",
    snapshot: { selectedModel: "", chatMode: "normal", toolChoice: "auto" }
  })
  await queueMock.args.sendQueuedRequest(item, () => {})
  expect(deps.sendMessage).toHaveBeenCalledWith(
    expect.objectContaining({ message: "  Exact queued reply\n" })
  )
  expect(getActorSceneForChat).not.toHaveBeenCalled()
  expect(deps.setSelectedModel).not.toHaveBeenCalled()
  expect(deps.setSelectedSystemPrompt).not.toHaveBeenCalled()
})
