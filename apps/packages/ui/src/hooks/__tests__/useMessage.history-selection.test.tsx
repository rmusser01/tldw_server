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
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
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

vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: vi.fn(async () => null)
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => mocks.notification
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null })
}))

vi.mock("@/hooks/chat/effective-assistant-state", () => ({
  resolveEffectiveAssistantState: () => ({
    mode: h1.persona
      ? "tracked_persona"
      : h1.native
        ? "tracked_character"
        : "plain",
    kind: h1.persona ? "persona" : h1.native ? "character" : null,
    id: h1.native || h1.persona ? "12" : null,
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
    getCharacter: h1.live,
    createChat: (...args: any[]) => h1.create(...args),
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
  MISSING_FINAL_ANSWER_MESSAGE: "Final answer unavailable",
  isReasoningOnlyResponse: () => false,
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
    formatToChatHistory: (items: any[]) => items,
    formatToMessage: (items: any[]) => items,
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

vi.mock("../chat-modes/tabChatMode", () => ({ tabChatMode: vi.fn() }))
vi.mock("../chat-modes/documentChatMode", () => ({ documentChatMode: vi.fn() }))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

import { useMessage } from "../useMessage"

const h1 = vi.hoisted(() => ({
  native: false,
  persona: false,
  create: vi.fn(),
  live: vi.fn(() => {
    throw new Error("live card deleted")
  }),
  auth: new AbortController(),
  controller: null as any,
  capture: vi.fn(),
  append: vi.fn(),
  wire: vi.fn(),
  recover: vi.fn(),
  dismiss: vi.fn()
}))
vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<any>()),
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
  h1.auth = new AbortController()
  h1.native = false
  h1.persona = false
  mocks.chatBaseState.historyId = "history-1"
  mocks.storeState.temporaryChat = false
  mocks.storeState.serverChatCharacterId = null
  mocks.storeState.serverChatAssistantKind = null
  h1.controller = makeController()
  mocks.chatBaseState.chatMode = "normal"
  mocks.chatBaseState.useOCR = false
  mocks.chatBaseState.selectedSystemPrompt = ""
  mocks.storeState.fileRetrievalEnabled = false
  mocks.storeState.toolChoice = "auto"
  mocks.storeState.serverChatId = "tracked-chat-1"
  mocks.chatBaseState.history = [
    { role: "user", content: "old question" },
    { role: "assistant", content: "wrong latest" }
  ]
  mocks.loadServicePromptSnapshot.mockResolvedValue(
    mocks.makeSnapshot("supported", undefined, undefined, {
      scopeInvalidatedSignal: h1.auth.signal
    })
  )
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

it("edits an ordinary local sidepanel message while history selection is idle", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.editMessage(0, "edited", false, false)
  })
  expect(helpers.updateMessageById).toHaveBeenCalledWith(
    "history-1", "plain-message", "edited"
  )
})

it("deletes an ordinary local sidepanel message while history selection is idle", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.deleteMessage(0)
  })
  expect(helpers.removeMessageById).toHaveBeenCalledWith(
    "history-1", "plain-message"
  )
})

it("routes ordinary sidepanel regeneration while history selection is idle", async () => {
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "again", image: "", isRegenerate: true })
  })
  expect(mocks.setMessages).toHaveBeenCalled()
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

it("mounted sidepanel native character uses owner selection and ACK without legacy persistence", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  h1.wire.mockImplementation(async function* (request) {
    yield {
      tldw_history_admission_v1: {
        version: 1,
        owner_key: "native-key",
        conversation_id: "tracked-chat-1",
        input_message_id: "native-input",
        input_message_revision: "r",
        selection_digest: request.tldw_history_selection_v1.selection_digest,
        messages: request.tldw_history_selection_v1.messages,
        originating_selection_revision: 1
      }
    }
    yield { choices: [{ delta: { content: "native answer" } }] }
    yield {
      tldw_message_id: "native-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.wire.mock.calls[0][0]).toMatchObject({
    save_to_db: true,
    conversation_id: "tracked-chat-1",
    messages: [{ role: "user", content: "next" }]
  })
  expect(
    h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages.map(
      (row: any) => row.id
    )
  ).toEqual(["u-old", "a1"])
  expect(h1.controller.followResult).toHaveBeenCalledWith(
    expect.objectContaining({ view_session_id: "origin" }),
    "native-result"
  )
  expect(h1.append).not.toHaveBeenCalled()
})

const nativeAdmission = (request: any) => ({
  version: 1,
  owner_key: "native-key",
  conversation_id: "tracked-chat-1",
  input_message_id: "native-input",
  input_message_revision: "native-r",
  selection_digest: request.tldw_history_selection_v1.selection_digest,
  messages: request.tldw_history_selection_v1.messages,
  originating_selection_revision:
    request.tldw_history_selection_v1.selection_revision
})

it("native before-first sends no historical content and retains unacknowledged text separately", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      choices: [{ delta: { content: "unsaved text" }, finish_reason: "stop" }]
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][0].messages).toEqual([
    { role: "user", content: "next" }
  ])
  expect(h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages).toEqual(
    []
  )
  expect(h1.recover.mock.calls[0][2]).toMatchObject({
    persistence: "server",
    state: "dispatching"
  })
  expect(h1.recover.mock.calls[0][2].input_id).toBeUndefined()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    persistence: "server",
    state: "generated_unsaved",
    owner_key: "native-key",
    conversation_id: "tracked-chat-1",
    origin_view: {
      conversation_id: "tracked-chat-1",
      view_session_id: "origin"
    },
    admission: {
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      input_message_id: "native-input"
    },
    input_id: "native-input",
    result_text: "unsaved text"
  })
  expect(h1.recover.mock.lastCall?.[0]).toEqual({
    profile_id: "profile",
    client_session_id: "client"
  })
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.controller.refreshRecovery).toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["wrong-selection", "result-before-admission", "wrong-conversation"])(
  "native %s cannot supply an owner result",
  async (failure) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    h1.wire.mockImplementation(async function* (request) {
      if (failure !== "result-before-admission")
        yield {
          tldw_history_admission_v1: {
            ...nativeAdmission(request),
            ...(failure === "wrong-selection"
              ? { selection_digest: "forged" }
              : {})
          }
        }
      yield {
        tldw_message_id: "forged-result",
        tldw_conversation_id:
          failure === "wrong-conversation" ? "other" : "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(h1.dismiss).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      "completed"
    )
    expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("native selection navigation preserves server ownership and never overwrites the new view", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const { result } = renderHook(() => useMessage())
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    h1.controller.navigate()
    mocks.setMessages.mockClear()
    yield { choices: [{ delta: { content: "old result" } }] }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(mocks.setMessages).not.toHaveBeenCalled()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.objectContaining({ conversation_id: "tracked-chat-1" }),
    expect.any(String),
    "completed"
  )
  expect(h1.append).not.toHaveBeenCalled()
})

it("native settings drift during capture prevents dispatch", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const { useMcpToolsStore } = (await import("@/store/mcp-tools")) as any
  const previous = useMcpToolsStore.getState()
  h1.capture.mockImplementationOnce(async (_id, request) => {
    useMcpToolsStore.setState({ ...previous })
    return captureFor(request.view)
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["account change", "disconnect"])(
  "native %s retains accepted evidence under the origin without claiming response settlement",
  async (interruption) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      mocks.setMessages.mockClear()
      if (interruption === "account change") h1.auth.abort()
      else throw new Error("connection lost after admission")
      yield { choices: [{ delta: { content: "new account must not see" } }] }
      yield {
        tldw_message_id: "late-result",
        tldw_conversation_id: "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    if (interruption === "account change")
      expect(mocks.setMessages).not.toHaveBeenCalled()
    else expect(h1.controller.refreshRecovery).toHaveBeenCalledOnce()
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      persistence: "server",
      state: "accepted_unsent",
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      origin_view: {
        conversation_id: "tracked-chat-1",
        view_session_id: "origin"
      },
      admission: {
        owner_key: "native-key",
        conversation_id: "tracked-chat-1",
        input_message_id: "native-input"
      },
      input_id: "native-input",
      result_text: ""
    })
    expect(h1.recover.mock.lastCall?.[0]).toEqual({
      profile_id: "profile",
      client_session_id: "client"
    })
    expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(h1.wire).toHaveBeenCalledOnce()
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("a disconnect after the native owner ACK preserves the completed outcome", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
    throw new Error("late disconnect")
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.anything(),
    expect.any(String),
    "completed"
  )
  expect(h1.controller.followResult).toHaveBeenCalledWith(
    expect.anything(),
    "owner-result"
  )
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["asset", "temporary", "local-owner"])(
  "native %s is visibly gated without dispatch",
  async (unsupported) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    if (unsupported === "local-owner")
      h1.controller.getCurrent().owner = {
        kind: "local",
        owner_key: "local",
        conversation_id: "local",
        profile_id: "profile"
      }
    mocks.storeState.temporaryChat = unsupported === "temporary"
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({
        message: "next",
        image: unsupported === "asset" ? "data:image/png;base64,asset" : ""
      })
    })
    expect(h1.wire).not.toHaveBeenCalled()
    expect(mocks.notification.error).toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it.each([false, true])(
  "native first creation is fenced around its ACK (navigation=%s)",
  async (navigate) => {
    h1.native = true
    mocks.chatBaseState.historyId = null as any
    mocks.storeState.serverChatId = null
    mocks.storeState.serverChatCharacterId = null
    let origin = true
    let current: any = { owner: null, view: null, status: "idle" }
    const ready = makeController().getCurrent()
    ready.view.cursor = { kind: "empty" }
    ready.capture = captureFor(ready.view)
    const load = vi.fn(async (_target, _reference, onLoaded) => {
      current = ready
      onLoaded({ owner: current.owner, view: current.view })
      return true
    })
    h1.controller = {
      getCurrent: () => current,
      fence: () => () => origin,
      loadConversation: load,
      followResult: vi.fn(async () => true),
      refreshRecovery: vi.fn()
    }
    h1.create.mockImplementation(async () => {
      if (navigate) origin = false
      return { id: "tracked-chat-1" }
    })
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      yield {
        tldw_message_id: "native-result",
        tldw_conversation_id: "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(h1.create).toHaveBeenCalledOnce()
    if (navigate) {
      expect(h1.wire).not.toHaveBeenCalled()
      expect(load).not.toHaveBeenCalled()
      expect(mocks.storeState.setServerChatId).not.toHaveBeenCalled()
    } else {
      expect(h1.wire).toHaveBeenCalledOnce()
      expect(
        h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages
      ).toEqual([])
      expect(mocks.storeState.setServerChatId).toHaveBeenCalledWith(
        "tracked-chat-1"
      )
    }
    expect(h1.live).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("native dispatch freezes workspace routing independently of the view owner", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const workspace = { type: "workspace", workspaceId: "original-workspace" }
  h1.controller.getCurrent().owner.scope = workspace
  h1.wire.mockImplementation(async function* (request) {
    workspace.workspaceId = "new-workspace"
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][1].scope).toEqual({
    type: "workspace",
    workspaceId: "original-workspace"
  })
  expect(h1.live).not.toHaveBeenCalled()
})

it("an older native completion cannot release a newer operation's Stop control", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  let enterFirst!: () => void,
    enterSecond!: () => void,
    releaseFirst!: () => void,
    releaseSecond!: () => void
  const firstEntered = new Promise<void>((resolve) => {
    enterFirst = resolve
  })
  const secondEntered = new Promise<void>((resolve) => {
    enterSecond = resolve
  })
  const firstHeld = new Promise<void>((resolve) => {
    releaseFirst = resolve
  })
  const secondHeld = new Promise<void>((resolve) => {
    releaseSecond = resolve
  })
  h1.wire.mockImplementationOnce(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    enterFirst()
    await firstHeld
    yield {
      tldw_message_id: "first-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  h1.wire.mockImplementationOnce(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    enterSecond()
    await secondHeld
    yield {
      tldw_message_id: "second-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    const first = result.current.onSubmit({ message: "first", image: "" })
    await firstEntered
    const second = result.current.onSubmit({ message: "second", image: "" })
    await secondEntered
    mocks.setStreaming.mockClear()
    mocks.setAbortController.mockClear()
    releaseFirst()
    await first
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
    releaseSecond()
    await second
    expect(mocks.setStreaming).toHaveBeenLastCalledWith(false)
    expect(mocks.setAbortController).toHaveBeenLastCalledWith(null)
  })
})

it("native required client tools are explicitly gated rather than silently discarded", async () => {
  h1.native = true
  mocks.storeState.toolChoice = "required"
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mocks.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: "native_history_client_tools_unsupported"
    })
  )
})

it("native persona limitations remain visible before any owner operation", async () => {
  h1.persona = true
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(mocks.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: "native_history_persona_unsupported"
    })
  )
})

const localMutationController = () => {
  const current = h1.controller.getCurrent()
  current.owner = {
    kind: "local",
    profile_id: "profile",
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  current.view = {
    ...current.view,
    owner_key: "local-key",
    conversation_id: "history-1"
  }
}
describe("selected-history mutations", () => {
  beforeEach(localMutationController)
  const visible = [
    {
      id: "greeting",
      isBot: true,
      name: "Assistant",
      message: "welcome",
      messageType: "character:greeting",
      sources: []
    },
    {
      id: "u-old",
      isBot: false,
      name: "You",
      message: "question",
      sources: []
    },
    { id: "a1", isBot: true, name: "Assistant", message: "same", sources: [] }
  ]
  it("editing and deleting A1 use its stable ID, preserving hidden same-text A2 and greeting", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const persisted = new Map([
      ["u-old", "question"],
      ["a1", "same"],
      ["a2", "same"]
    ])
    vi.mocked(helpers.updateMessageById).mockImplementation(
      async (_owner, id, text) => {
        persisted.set(id, text)
      }
    )
    vi.mocked(helpers.removeMessageById).mockImplementation(
      async (_owner, id) => {
        persisted.delete(id)
        return undefined as any
      }
    )
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.editMessage(2, "edited", false, false)
    })
    expect(persisted.get("a1")).toBe("edited")
    expect(persisted.get("a2")).toBe("same")
    expect(persisted.get("u-old")).toBe("question")
    await act(async () => {
      await result.current.deleteMessage(2)
    })
    expect([...persisted.keys()]).toEqual(["u-old", "a2"])
    expect(options.setMessages).toHaveBeenLastCalledWith(visible.slice(0, 2))
    expect(helpers.removeMessageByIndex).not.toHaveBeenCalled()
  })
  it("edit-and-send and regeneration reject without owner writes or display truncation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
    options.setMessages.mockClear()
    options.setHistory.mockClear()
    await expect(
      result.current.editMessage(1, "new question", true, true)
    ).rejects.toThrow("unsupported_history_edit_and_send:u-old")
    await expect(result.current.regenerateLastMessage()).rejects.toThrow(
      "unsupported_history_regeneration"
    )
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("a held successful edit does not replace a newly selected conversation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    let release!: () => void
    vi.mocked(helpers.updateMessageById).mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    let current = true
    h1.controller.fence = () => () => current
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
    options.setMessages.mockClear()
    const editing = result.current.editMessage(2, "edited", false, false)
    await vi.waitFor(() => expect(helpers.updateMessageById).toHaveBeenCalled())
    current = false
    release()
    await editing
    expect(options.setMessages).not.toHaveBeenCalled()
  })
})

it("deleting the selected local leaf follows only its captured parent", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const visible = [
    { id: "u-old", isBot: false, message: "question", sources: [] },
    { id: "a1", isBot: true, message: "same", sources: [] }
  ]
  const choose = vi.fn(async () => true)
  h1.controller.choose = choose
  vi.mocked(helpers.removeMessageById).mockResolvedValue({
    id: "a1",
    history_id: "history-1",
    parent_message_id: "u-old"
  } as any)
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.deleteMessage(1)
  })
  expect(choose).toHaveBeenCalledWith({
    kind: "after_message",
    message_id: "u-old"
  })
})
it("late leaf deletion retains its storage result without moving a new view", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const visible = [{ id: "a1", isBot: true, message: "same", sources: [] }]
  const choose = vi.fn(async () => true)
  h1.controller.choose = choose
  let current = true
  h1.controller.fence = () => () => current
  let release!: (row: any) => void
  vi.mocked(helpers.removeMessageById).mockImplementation(
    () =>
      new Promise((resolve) => {
        release = resolve
      })
  )
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
  options.setMessages.mockClear()
  const pending = result.current.deleteMessage(0)
  await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
  current = false
  release({ id: "a1", parent_message_id: "u-old" })
  await pending
  expect(choose).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("native plain edit/delete reject before any mutation or display change", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const visible = [
    {
      id: "a1",
      serverMessageId: "a1",
      isBot: true,
      message: "same",
      sources: []
    }
  ]
  mocks.storeState.serverChatId = "tracked-chat-1"
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
  options.setMessages.mockClear()
  await expect(
    result.current.editMessage(0, "bad", false, false)
  ).rejects.toThrow("native_history_mutation_unavailable")
  await expect(result.current.deleteMessage(0)).rejects.toThrow(
    "native_history_mutation_unavailable"
  )
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(helpers.updateMessageById).not.toHaveBeenCalled()
  expect(helpers.removeMessageById).not.toHaveBeenCalled()
})

const localForks = vi.hoisted(() => ({
  capture: vi.fn(),
  prepare: vi.fn(),
  commit: vi.fn(),
  digest: (_request: any) => "fork-digest"
}))
vi.mock("@/db/dexie/branch", () => ({
  captureLocalForkSelection: localForks.capture,
  prepareLocalFork: localForks.prepare,
  commitLocalFork: localForks.commit,
  forkRequestDigest: (request: any) => localForks.digest(request)
}))
it("mounted branch action captures a stable boundary and returns the committed owner result", async () => {
  const {historyDigest} = await vi.importActual<any>("@/db/dexie/history-selection")
  localForks.digest = (value: any) => historyDigest({operation_id: value.operation_id, owner_key: value.owner_key, destination_owner_key: value.destination_owner_key, input: value.input})
  const current = h1.controller.getCurrent()
  current.owner = {
    kind: "local",
    profile_id: "profile",
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  current.view = {
    ...current.view,
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  localForks.capture.mockResolvedValue({
    kind: "normal",
    selection: { conversation_id: "history-1" }
  })
  localForks.prepare.mockImplementation(async (request: any) => ({request,
    history: { id: "child" },
    messages: [],
    files: undefined
  }))
  localForks.commit.mockImplementation(async (prepared: any) => ({
    state: "committed",
    owner_key: "local-key",
    operation_id: prepared.request.operation_id,
    child_id: prepared.history.id,
    message_map: { a1: "child-a1" }
  }))
  mocks.storeState.serverChatId = null
  const { result } = renderHook(() => useMessage())
  let outcome: any
  await act(async () => {
    outcome = await result.current.createChatBranch("a1")
  })
  expect(localForks.capture).toHaveBeenCalledWith(
    current.owner,
    expect.objectContaining({
      cursor: { kind: "after_message", message_id: "a1" }
    }),
    expect.any(Object)
  )
  expect(outcome).toMatchObject({
    state: "committed",
    owner_key: "local-key",
    child_id: "child",
    message_map: { a1: "child-a1" }
  })
})

vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory
}))

it.each([false, true])(
  "real controller fork -> immediate send -> reopen keeps source unchanged (comparison=%s)",
  async (comparison) => {
    const { memory, seedLocalFork } = await import(
      "@/hooks/chat/__tests__/local-history-fixture"
    )
    const source = await seedLocalFork(comparison)
    const authority = await vi.importActual<any>("@/db/dexie/history-selection")
    const projector = await vi.importActual<any>("@/db/dexie/branch")
    localForks.capture.mockImplementation(projector.captureLocalForkSelection)
    localForks.prepare.mockImplementation(projector.prepareLocalFork)
    localForks.commit.mockImplementation(projector.commitLocalFork)
    // Bind the actual request digest rather than the older handler-only test stub.
    localForks.digest = projector.forkRequestDigest

    const { useHistorySelection } = await import(
      "@/hooks/chat/useHistorySelection"
    )
    const controller = renderHook(() => useHistorySelection())
    await act(async () => {
      await controller.result.current.loadConversation({
        historyId: "history-1"
      })
    })
    if (!comparison)
      await act(async () => {
        await controller.result.current.choose({
          kind: "after_message",
          message_id: "a1"
        })
      })
    h1.controller = controller.result.current
    mocks.storeState.serverChatId = null
    mocks.storeState.fileRetrievalEnabled = false
    mocks.chatBaseState.historyId = "history-1"
    mocks.chatBaseState.setHistoryId.mockImplementation((id: string) => {
      mocks.chatBaseState.historyId = id
    })
    const hook = renderHook(() => useMessage())
    let outcome: any
    await act(async () => {
      outcome = await hook.result.current.createChatBranch(
        "a1",
        comparison ? { model_id: "A", cluster_id: "c1" } : undefined
      )
    })
    expect(outcome.state, JSON.stringify(outcome)).toBe("committed")
    const childId = outcome.child_id
    expect(controller.result.current.getCurrent().owner).toMatchObject({
      kind: "local",
      conversation_id: childId
    })
    expect(controller.result.current.getReference()).toMatchObject({
      conversation_id: childId
    })
    h1.controller = controller.result.current
    hook.rerender()
    await act(async () => {
      await hook.result.current.onSubmit({ message: "child next", image: "" })
    })
    const childRows = await memory.messages
      .where("history_id")
      .equals(childId)
      .toArray()
    expect(childRows.map((row: any) => row.content)).toEqual([
      "question",
      "selected answer",
      "child next",
      "new answer"
    ])
    expect(
      await memory.messages.where("history_id").equals("history-1").toArray()
    ).toEqual(source)
    const reference = controller.result.current.getReference()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => {
      await reopened.result.current.loadConversation(
        { historyId: childId },
        reference
      )
    })
    expect(reopened.result.current.getCurrent().owner).toMatchObject({
      conversation_id: childId
    })
    const reopenedCapture = reopened.result.current.getCurrent().capture
    expect(reopenedCapture?.status).toBe("captured")
    if (reopenedCapture?.status !== "captured") throw new Error("reopen failed")
    expect(reopenedCapture.rows.map((row) => row.id)).toEqual(
      childRows.map((row: any) => row.id)
    )
    localForks.digest = () => "fork-digest"
  }
)

vi.mock("@/utils/safe-storage", async (original) => {
  const actual = await original<any>()
  const { settings } = await import(
    "@/hooks/chat/__tests__/local-history-fixture"
  )
  return {
    ...actual,
    createSafeStorage: (options: any) => {
      const storage = actual.createSafeStorage(options)
      const area = options?.area ?? "sync"
      return new Proxy(storage, {
        get(target, key) {
          if (key === "hasPersistentBackend") return true
          if (key === "get")
            return async (name: string) =>
              name.startsWith("chatSettings:")
                ? settings.get(area + name)
                : target.get(name)
          if (key === "set")
            return async (name: string, value: unknown) =>
              name.startsWith("chatSettings:")
                ? settings.set(area + name, value)
                : target.set(name, value)
          return Reflect.get(target, key)
        }
      })
    }
  }
})

it.each(["missing", "unavailable", "mismatch"])(
  "%s controller never gains mutation authority from ambient local IDs",
  async (kind) => {
    const helpers = await import("@/db/dexie/helpers")
    localMutationController()
    if (kind === "missing") h1.controller = null
    else if (kind === "unavailable")
      h1.controller.getCurrent().owner = {
        kind: "unavailable",
        code: "owner_unavailable"
      }
    else h1.controller.getCurrent().owner.conversation_id = "other"
    mocks.storeState.serverChatId = null
    const hook = renderHook(() => useMessage())
    mocks.setMessages.mockClear()
    await expect(
      hook.result.current.editMessage(0, "changed", false, false)
    ).rejects.toThrow()
    await expect(hook.result.current.deleteMessage(0)).rejects.toThrow()
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenCalled()
  }
)
it("mounted native fork without local history retains an unknown create and never dispatches the same intent again", async () => {
  const {memory} = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory.forkOperations.rows.clear()
  const projector = await vi.importActual<any>("@/db/dexie/branch")
  localForks.digest = projector.forkRequestDigest
  h1.capture.mockImplementation(async (_id, request) => ({...captureFor(request.view), purpose: request.purpose,
    snapshot: {...captureFor(request.view).snapshot, native_fork_context: {policy: "plain_v1", supported: true, storage_context_digest: "storage"}}}))
  h1.create.mockRejectedValue(new Error("creation response lost"))
  mocks.chatBaseState.historyId = null
  const {result} = renderHook(() => useMessage())
  let first: any, second: any
  await act(async () => {first = await result.current.createChatBranch("a1")})
  await act(async () => {second = await result.current.createChatBranch("a1")})
  expect(first).toMatchObject({state: "unknown", owner_key: "native-key"})
  expect(second.operation_id).toBe(first.operation_id)
  expect(h1.create).toHaveBeenCalledTimes(1)
  expect(localForks.commit).not.toHaveBeenCalled()
  expect([...memory.forkOperations.rows.values()][0].request.operation_id).toBe(first.operation_id)
})
