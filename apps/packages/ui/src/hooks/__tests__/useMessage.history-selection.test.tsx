import React from "react"
import { act, render, renderHook, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const setMessages = vi.fn()
  const setHistory = vi.fn()
  const setAbortController = vi.fn()
  const setEmbeddingController = vi.fn()
  const setIsEmbedding = vi.fn()
  const setIsLoading = vi.fn()
  const setIsProcessing = vi.fn()
  const setStreaming = vi.fn()
  const resetChatLoopState = vi.fn()
  const notification = {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  }
  const saveMessageOnSuccess = vi.fn<
    (payload: any) => Promise<string | undefined>
  >(async () => undefined)
  const saveMessageOnError = vi.fn<(payload: any) => Promise<string | false>>(
    async () => false
  )
  const finalModel = {
    stream: vi.fn<(...args: any[]) => Promise<any>>(async () => []),
    invoke: vi.fn()
  }
  const rewriteModel = {
    stream: vi.fn(),
    invoke: vi.fn(async () => ({ content: "standalone retrieval query" }))
  }

  const storeState: Record<string, unknown> = {
    messages: [
      {
        id: "user-old",
        isBot: false,
        name: "You",
        message: "Earlier question",
        images: [],
        sources: [],
        createdAt: 1,
        parentMessageId: null
      },
      {
        id: "assistant-old",
        isBot: true,
        name: "model-1",
        message: "Earlier answer",
        images: [],
        sources: [],
        createdAt: 2,
        parentMessageId: "user-old"
      }
    ],
    setMessages,
    webSearch: false,
    setWebSearch: vi.fn(),
    toolChoice: "required",
    setToolChoice: vi.fn(),
    isSearchingInternet: false,
    setIsSearchingInternet: vi.fn(),
    temporaryChat: false,
    setTemporaryChat: vi.fn(),
    queuedMessages: [],
    addQueuedMessage: vi.fn(),
    setQueuedMessages: vi.fn(),
    clearQueuedMessages: vi.fn(),
    fileRetrievalEnabled: true,
    setActionInfo: vi.fn(),
    replyTarget: null,
    clearReplyTarget: vi.fn(),
    serverChatId: null,
    setServerChatId: vi.fn(),
    serverChatTitle: "",
    setServerChatTitle: vi.fn(),
    serverChatCharacterId: null,
    setServerChatCharacterId: vi.fn(),
    serverChatAssistantKind: null,
    setServerChatAssistantKind: vi.fn(),
    serverChatAssistantId: null,
    setServerChatAssistantId: vi.fn(),
    serverChatPersonaMemoryMode: null,
    setServerChatPersonaMemoryMode: vi.fn(),
    serverChatMetaLoaded: true,
    setServerChatMetaLoaded: vi.fn(),
    serverChatState: null,
    setServerChatState: vi.fn(),
    setServerChatVersion: vi.fn(),
    serverChatTopic: null,
    setServerChatTopic: vi.fn(),
    serverChatClusterId: null,
    setServerChatClusterId: vi.fn(),
    serverChatSource: null,
    setServerChatSource: vi.fn(),
    serverChatExternalRef: null,
    setServerChatExternalRef: vi.fn()
  }

  const history = [
    { role: "user", content: "Earlier question" },
    { role: "assistant", content: "Earlier answer" }
  ]
  const chatBaseState = {
    history,
    setHistory,
    streaming: false,
    setStreaming,
    isFirstMessage: false,
    setIsFirstMessage: vi.fn(),
    historyId: "history-1",
    setHistoryId: vi.fn(),
    isLoading: false,
    setIsLoading,
    isProcessing: false,
    setIsProcessing,
    chatMode: "rag",
    setChatMode: vi.fn(),
    isEmbedding: false,
    setIsEmbedding,
    selectedQuickPrompt: null,
    setSelectedQuickPrompt: vi.fn(),
    selectedSystemPrompt: "Keep this system prompt",
    setSelectedSystemPrompt: vi.fn(),
    useOCR: true,
    setUseOCR: vi.fn()
  }

  const definition = (
    id: "chat.rag.answer" | "chat.rag.question_rewrite",
    requiredVariables: readonly string[]
  ) =>
    Object.freeze({
      id,
      parts: Object.freeze([
        Object.freeze({
          key: "template",
          mode: "template" as const,
          required_variables: Object.freeze([...requiredVariables])
        })
      ])
    })
  const answerDefinition = definition("chat.rag.answer", [
    "context",
    "question"
  ])
  const rewriteDefinition = definition("chat.rag.question_rewrite", [
    "chat_history",
    "question"
  ])
  const makeSnapshot = (
    capability: "supported" | "legacy-404" = "supported",
    answerTemplate = "custom answer {context} :: {question}",
    rewriteTemplate = "custom rewrite {chat_history} :: {question}",
    lifetime?: {
      scopeSignal?: AbortSignal
      scopeInvalidatedSignal?: AbortSignal
      release?: () => void
    }
  ) =>
    Object.freeze({
      scopeKey: "scope:user-1",
      requestScope: Object.freeze({
        config: Object.freeze({
          serverUrl: "https://example.test",
          authMode: "multi-user" as const,
          authSource: "manual" as const,
          orgId: 9
        }),
        userId: 42
      }),
      capability,
      definitions: Object.freeze({
        "chat.rag.answer": Object.freeze({
          definition: answerDefinition,
          parts: Object.freeze({ template: answerTemplate }),
          source: "user" as const,
          revision: capability === "supported" ? "answer-revision" : null
        }),
        "chat.rag.question_rewrite": Object.freeze({
          definition: rewriteDefinition,
          parts: Object.freeze({ template: rewriteTemplate }),
          source: "user" as const,
          revision: capability === "supported" ? "rewrite-revision" : null
        })
      }),
      scopeSignal: lifetime?.scopeSignal ?? new AbortController().signal,
      scopeInvalidatedSignal:
        lifetime?.scopeInvalidatedSignal ?? new AbortController().signal,
      release: lifetime?.release ?? vi.fn()
    })

  return {
    addMedia: vi.fn(),
    answerDefinition,
    chatBaseState,
    finalModel,
    history,
    humanMessageFormatter: vi.fn(),
    loadServicePromptSnapshot: vi.fn(),
    makeSnapshot,
    notification,
    pageAssistModel: vi.fn(),
    promptForRag: vi.fn(),
    ragSearch: vi.fn(),
    renderServicePromptPart: vi.fn(),
    resetChatLoopState,
    rewriteDefinition,
    rewriteModel,
    saveMessageOnError,
    saveMessageOnSuccess,
    setAbortController,
    setEmbeddingController,
    setHistory,
    setIsEmbedding,
    setIsLoading,
    setIsProcessing,
    setMessages,
    setStreaming,
    storeState
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? _key)
  })
}))

vi.mock("~/services/tldw-server", async (original) => ({
  ...(await original<any>()),
  systemPromptForNonRagOption: async () => "system",
  promptForRag: (...args: unknown[]) => mocks.promptForRag(...args),
  systemPromptForNonRag: vi.fn(async () => "system")
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) =>
    mocks.loadServicePromptSnapshot(...args),
  renderServicePromptPart: (...args: unknown[]) =>
    mocks.renderServicePromptPart(...args)
}))

vi.mock("~/store/option", () => ({
  useStoreMessageOption: (
    selector?: (state: Record<string, unknown>) => unknown
  ) => (selector ? selector(mocks.storeState) : mocks.storeState)
}))

vi.mock("~/store", () => ({
  useStoreMessage: () => ({ currentURL: "", setCurrentURL: vi.fn() })
}))

vi.mock("@/context", () => ({
  usePageAssist: () => ({
    controller: null,
    setController: mocks.setAbortController,
    embeddingController: null,
    setEmbeddingController: mocks.setEmbeddingController
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    key === "chatWithWebsiteEmbedding" ? true : fallback,
    vi.fn()
  ]
}))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => mocks.chatBaseState
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "model-1",
    setSelectedModel: vi.fn()
  })
}))

vi.mock("@/store/model", () => {
  const state = { apiProvider: "provider-1", reset: vi.fn() }
  return {
    useStoreChatModelSettings: Object.assign(() => state, {
      getState: () => state
    })
  }
})

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => mocks.notification
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null })
}))

vi.mock("@/hooks/chat/effective-assistant-state", () => ({
  resolveEffectiveAssistantState: () => ({
    mode: "plain",
    kind: null,
    id: null,
    displayName: null,
    avatarUrl: null,
    systemPromptSnapshot: null
  })
}))

vi.mock("@/services/chat-loop/hooks", () => ({
  useChatLoopState: () => ({
    state: {},
    dispatch: vi.fn(),
    reset: mocks.resetChatLoopState
  })
}))

vi.mock("@/services/chat-loop/bridge", () => ({
  subscribeChatLoopEvents: () => vi.fn(),
  publishChatLoopEvent: vi.fn()
}))

vi.mock("@/utils/chat-model-validation", () => ({
  validateSelectedChatModelAvailability: vi.fn(async () => ({
    status: "valid"
  }))
}))

vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: () => []
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({}))
}))

vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return {
    pageAssistModel: async (options: any) =>
      new ChatTldw({ ...options, temperature: 0.23 })
  }
})

vi.mock("@/libs/get-html", () => ({
  getContentFromCurrentTab: vi.fn(async () => ({
    content: "raw page content",
    url: "https://source.example/page",
    type: "html",
    pdf: []
  }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    captureHistorySelection: (...args: any[]) => h1.capture(...args),
    addChatMessage: (...args: any[]) => h1.append(...args),
    streamChatCompletion: (...args: any[]) => h1.wire(...args),
    getConfig: async () => null,
    addMedia: (...args: unknown[]) => mocks.addMedia(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: () => "grounded context"
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) =>
    mocks.humanMessageFormatter(...args)
}))

vi.mock("@/libs/reasoning", () => ({
  isReasoningEnded: () => false,
  isReasoningStarted: () => false,
  mergeReasoningContent: (value: string) => value,
  removeReasoning: (value: string) => value
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => ({
    model_avatar: "avatar.png",
    model_name: "Model One"
  }))
}))

vi.mock("@/db/dexie/helpers", async (original) => {
  const { formatSelectedHistory } = await original<any>()
  let id = 0
  return {
    formatSelectedHistory,
    saveMessage: vi.fn(async () => {}),
    updateLastUsedModel: vi.fn(async () => {}),
    updateLastUsedPrompt: vi.fn(async () => {}),
    updateChatHistoryCreatedAt: vi.fn(async () => {}),
    deleteChatForEdit: vi.fn(),
    deleteChatAfterMessageId: vi.fn(),
    generateID: () => `generated-${++id}`,
    getPromptById: vi.fn(async () => null),
    removeMessageByIndex: vi.fn(),
    removeMessageById: vi.fn(),
    updateMessageByIndex: vi.fn(),
    updateMessageById: vi.fn()
  }
})

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createBranchMessage: () => vi.fn(),
  createRegenerateLastMessage: () => vi.fn()
}))

vi.mock("../chat-modes/tabChatMode", () => ({ tabChatMode: vi.fn() }))
vi.mock("../chat-modes/documentChatMode", () => ({ documentChatMode: vi.fn() }))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

import { useMessage } from "../useMessage"

const h1 = vi.hoisted(() => ({
  controller: null as any,
  capture: vi.fn(),
  append: vi.fn(),
  wire: vi.fn(),
  recover: vi.fn(),
  dismiss: vi.fn()
}))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => h1.controller
}))
vi.mock("@/db/dexie/history-selection", async (original) => ({
  ...(await original<any>()),
  saveHistoryTurnRecovery: h1.recover,
  dismissHistoryTurnRecovery: h1.dismiss
}))
const captureFor = (view: any) => {
  const nodes = [
    {
      id: "u-old",
      revision: "r1",
      parent_id: null,
      role: "user",
      settled: true
    },
    {
      id: "a1",
      revision: "r2",
      parent_id: "u-old",
      role: "assistant",
      settled: true
    },
    {
      id: "a2",
      revision: "r3",
      parent_id: "u-old",
      role: "assistant",
      settled: true
    }
  ]
  const rows =
    view.cursor.kind === "empty"
      ? []
      : nodes.filter(
          (row) => row.id === "u-old" || row.id === view.cursor.message_id
        )
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      fences: { conversation: "1", history: "1", settings: "1" },
      nodes,
      source_digest: "source",
      storage_context_digest: "storage",
      interpretation_status: { kind: "parent_graph_v1" }
    },
    rows,
    selected_content: rows.map((row) => ({
      id: row.id,
      revision: row.revision,
      message: row.id === "u-old" ? "old question" : "same answer",
      images: []
    })),
    view,
    purpose: "send",
    storage_context_digest: "storage"
  }
}
const makeController = () => {
  let current: any = {
    owner: {
      kind: "native",
      conversation_id: "tracked-chat-1",
      request_scope: {
        config: { serverUrl: "https://server.test", authMode: "multi-user" },
        userId: "alice"
      },
      validate_lease: () => true
    },
    view: {
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      view_session_id: "origin",
      selection_revision: 1,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "a1" }
    },
    bookmarkScope: { profile_id: "profile", client_session_id: "client" },
    status: "ready"
  }
  current.capture = captureFor(current.view)
  return {
    getCurrent: () => current,
    fence: () => {
      const origin = current
      return () => current === origin
    },
    followResult: vi.fn(async () => true),
    refreshRecovery: vi.fn(async () => {}),
    navigate: () => {
      current = {
        ...current,
        view: {
          ...current.view,
          conversation_id: "other",
          view_session_id: "other"
        }
      }
    }
  }
}

beforeEach(async () => {
  vi.clearAllMocks()
  h1.controller = makeController()
  mocks.chatBaseState.chatMode = "normal"
  mocks.chatBaseState.useOCR = false
  mocks.chatBaseState.selectedSystemPrompt = ""
  mocks.storeState.fileRetrievalEnabled = false
  mocks.storeState.serverChatId = "tracked-chat-1"
  mocks.chatBaseState.history = [
    { role: "user", content: "old question" },
    { role: "assistant", content: "wrong latest" }
  ]
  mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
  const { HumanMessage } = await import("@/types/messages")
  mocks.humanMessageFormatter.mockImplementation(
    async ({ content }: any) => new HumanMessage({ content })
  )
  h1.capture.mockImplementation(async (_id, request) =>
    captureFor(request.view)
  )
  h1.append.mockImplementation(async (_id: any, body: any) =>
    body.role === "user"
      ? {
          id: body.id,
          tldw_history_admission_v1: {
            version: 1,
            owner_key: "native-key",
            conversation_id: "tracked-chat-1",
            input_message_id: body.id,
            input_message_revision: "accepted-r",
            selection_digest: body.tldw_history_selection_v1.selection_digest,
            messages: body.tldw_history_selection_v1.messages,
            originating_selection_revision: 1
          }
        }
      : { id: body.id }
  )
  h1.wire.mockImplementation(async function* () {
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
})

it("mounted sidepanel normal submit uses canonical A1 selection and pre-admits its input", async () => {
  const { result } = renderHook(() => useMessage())
  h1.wire.mockImplementation(async function* () {
    expect(h1.append.mock.calls.map(([, body]: any) => body.role)).toEqual([
      "user"
    ])
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  const request = h1.wire.mock.calls[0][0]
  expect(request.messages).toContainEqual({
    role: "assistant",
    content: "same answer"
  })
  expect(JSON.stringify(request.messages)).not.toContain("wrong latest")
  expect(request.save_to_db).toBe(false)
  const [user, assistant] = h1.append.mock.calls.map(([, body]: any) => body)
  expect(
    user.tldw_history_selection_v1.messages.map((row: any) => row.id)
  ).toEqual(["u-old", "a1"])
  expect(assistant.tldw_history_admission_v1.input_message_id).toBe(user.id)
  expect(h1.append).toHaveBeenCalledTimes(2)
})

it("before-first on a nonempty sidepanel source admits an empty prior path and sends only current input", async () => {
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  expect(current.capture.snapshot.nodes).toHaveLength(3)
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "fresh question", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(
    h1.wire.mock.calls[0][0].messages.filter(
      (row: any) => row.role !== "system"
    )
  ).toEqual([{ role: "user", content: "fresh question" }])
  const user = h1.append.mock.calls[0][1]
  expect(user.tldw_history_selection_v1.messages).toEqual([])
  expect(user.tldw_history_selection_v1.cursor).toEqual({ kind: "empty" })
  expect(user.parent_message_id ?? null).toBeNull()
})

it("an older sidepanel send cannot clear the newer send's activity or Stop controller", async () => {
  let entered!: () => void,
    releaseFirst!: () => void,
    releaseSecond!: () => void
  const started = new Promise<void>((resolve) => {
    entered = resolve
  })
  const first = new Promise<void>((resolve) => {
    releaseFirst = resolve
  })
  const second = new Promise<void>((resolve) => {
    releaseSecond = resolve
  })
  let secondEntered!: () => void
  const secondStarted = new Promise<void>((resolve) => {
    secondEntered = resolve
  })
  h1.wire.mockImplementationOnce(async function* () {
    entered()
    await first
    yield { choices: [{ delta: { content: "first" } }] }
  })
  h1.wire.mockImplementationOnce(async function* () {
    secondEntered()
    await second
    yield { choices: [{ delta: { content: "second" } }] }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    const old = result.current.onSubmit({ message: "first", image: "" })
    await started
    const newer = result.current.onSubmit({ message: "second", image: "" })
    await secondStarted
    mocks.setStreaming.mockClear()
    mocks.setIsProcessing.mockClear()
    mocks.setAbortController.mockClear()
    releaseFirst()
    await old
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(false)
    expect(mocks.setIsProcessing).not.toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
    releaseSecond()
    await newer
    expect(mocks.setStreaming).toHaveBeenLastCalledWith(false)
    expect(mocks.setAbortController).toHaveBeenLastCalledWith(null)
  })
})
