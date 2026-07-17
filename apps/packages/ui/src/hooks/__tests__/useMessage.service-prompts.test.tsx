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
  const saveMessageOnSuccess = vi.fn(async () => undefined)
  const saveMessageOnError = vi.fn(async () => false)
  const finalModel = {
    stream: vi.fn(async () => []),
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
  ) => Object.freeze({
    id,
    parts: Object.freeze([
      Object.freeze({
        key: "template",
        mode: "template" as const,
        required_variables: Object.freeze([...requiredVariables])
      })
    ])
  })
  const answerDefinition = definition("chat.rag.answer", ["context", "question"])
  const rewriteDefinition = definition("chat.rag.question_rewrite", [
    "chat_history",
    "question"
  ])
  const makeSnapshot = (
    capability: "supported" | "legacy-404" = "supported",
    answerTemplate = "custom answer {context} :: {question}",
    rewriteTemplate = "custom rewrite {chat_history} :: {question}"
  ) => Object.freeze({
    scopeKey: "scope:user-1",
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
    })
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

vi.mock("~/services/tldw-server", () => ({
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
  useStoreMessageOption: (selector?: (state: Record<string, unknown>) => unknown) =>
    selector ? selector(mocks.storeState) : mocks.storeState
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
  useSelectedModel: () => ({ selectedModel: "model-1", setSelectedModel: vi.fn() })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({
    apiProvider: "provider-1",
    reset: vi.fn()
  })
}))

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
  subscribeChatLoopEvents: () => vi.fn()
}))

vi.mock("@/utils/chat-model-validation", () => ({
  validateSelectedChatModelAvailability: vi.fn(async () => ({ status: "valid" }))
}))

vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: () => []
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({}))
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

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
    addMedia: (...args: unknown[]) => mocks.addMedia(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: () => "grounded context"
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) => mocks.humanMessageFormatter(...args)
}))

vi.mock("@/utils/generate-history", () => ({
  generateHistory: () => []
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

vi.mock("@/db/dexie/helpers", () => {
  let id = 0
  return {
    deleteChatForEdit: vi.fn(),
    deleteChatAfterMessageId: vi.fn(),
    generateID: () => `generated-${++id}`,
    getPromptById: vi.fn(),
    removeMessageByIndex: vi.fn(),
    removeMessageById: vi.fn(),
    updateMessageByIndex: vi.fn(),
    updateMessageById: vi.fn()
  }
})

vi.mock("@/hooks/utils/messageHelpers", () => ({
  createSaveMessageOnError: () => mocks.saveMessageOnError,
  createSaveMessageOnSuccess: () => mocks.saveMessageOnSuccess,
  validateBeforeSubmit: () => true
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createBranchMessage: () => vi.fn(),
  createRegenerateLastMessage: () => vi.fn()
}))

vi.mock("../chat-modes/normalChatMode", () => ({ normalChatMode: vi.fn() }))
vi.mock("../chat-modes/tabChatMode", () => ({ tabChatMode: vi.fn() }))
vi.mock("../chat-modes/documentChatMode", () => ({ documentChatMode: vi.fn() }))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

import { useMessage } from "../useMessage"

describe("useMessage legacy Sidepanel Service Prompts", () => {
  const expectNoPreflightMutation = () => {
    expect(mocks.resetChatLoopState).not.toHaveBeenCalled()
    expect(mocks.setAbortController).not.toHaveBeenCalled()
    expect(mocks.setEmbeddingController).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(mocks.setStreaming).not.toHaveBeenCalled()
    expect(mocks.setIsProcessing).not.toHaveBeenCalled()
    expect(mocks.setIsEmbedding).not.toHaveBeenCalled()
    expect(mocks.setIsLoading).not.toHaveBeenCalled()
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: "legacy answer {context} {question}",
      ragQuestionPrompt: "legacy rewrite {chat_history} {question}"
    })
    mocks.pageAssistModel.mockImplementation(async (options: unknown) => {
      const modelOptions = options as { toolChoice?: string }
      return modelOptions.toolChoice === "none"
        ? mocks.rewriteModel
        : mocks.finalModel
    })
    mocks.ragSearch.mockResolvedValue({
      results: [
        {
          content: "retrieved chunk",
          metadata: {
            source: "Source title",
            type: "html",
            url: "https://source.example/page"
          }
        }
      ]
    })
    mocks.addMedia.mockResolvedValue(undefined)
    mocks.humanMessageFormatter.mockImplementation(async (input: unknown) => input)
    mocks.renderServicePromptPart.mockImplementation(
      (_definition: unknown, _partKey: string, authored: string, values: Record<string, string>) =>
        authored.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_match, key: string) =>
          Object.prototype.hasOwnProperty.call(values, key) ? values[key] : _match
        )
    )
  })

  it("loads one immutable RAG snapshot before the first message mutation", async () => {
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })
    })

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledTimes(1)
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      { signal: controller.signal }
    )
    expect(mocks.loadServicePromptSnapshot.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.setMessages.mock.invocationCallOrder[0]
    )
    const snapshot = await mocks.loadServicePromptSnapshot.mock.results[0].value
    expect(Object.isFrozen(snapshot)).toBe(true)
    expect(Object.isFrozen(snapshot.definitions)).toBe(true)
  })

  it("blocks unresolved supported-server migration with a Workflow prompts link", async () => {
    const migrationError = Object.assign(
      new Error("Review workflow prompts before continuing"),
      { code: "service_prompt_migration_required" }
    )
    mocks.loadServicePromptSnapshot.mockRejectedValue(migrationError)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Blocked follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expectNoPreflightMutation()
    expect(mocks.notification.warning).toHaveBeenCalledTimes(1)
    const notice = mocks.notification.warning.mock.calls[0][0]
    render(<>{notice.description}</>)
    expect(screen.getByRole("link", { name: "Review workflow prompts" })).toHaveAttribute(
      "href",
      "/options.html#/settings/prompt"
    )
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("keeps catalog-404 local prompt behavior at the snapshot boundary", async () => {
    mocks.loadServicePromptSnapshot.mockResolvedValue(
      mocks.makeSnapshot(
        "legacy-404",
        "local answer {context} -> {question}",
        "local rewrite {chat_history} -> {question}"
      )
    )
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      1,
      mocks.rewriteDefinition,
      "template",
      "local rewrite {chat_history} -> {question}",
      expect.objectContaining({ question: "Current follow-up" })
    )
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      2,
      mocks.answerDefinition,
      "template",
      "local answer {context} -> {question}",
      {
        context: "grounded context",
        question: "standalone retrieval query"
      }
    )
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("does not fall back to local prompts after a supported-server failure", async () => {
    const secret = "PRIVATE TEMPLATE BODY"
    const supportedFailure = Object.assign(
      new Error(`Supported server detail failed: ${secret}`),
      { detail: { body: secret } }
    )
    mocks.loadServicePromptSnapshot.mockRejectedValue(supportedFailure)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expectNoPreflightMutation()
    expect(mocks.notification.error).toHaveBeenCalledTimes(1)
    const notice = mocks.notification.error.mock.calls[0][0]
    const copy = `${String(notice.message)} ${String(notice.description)}`
    expect(copy).toMatch(/connection/i)
    expect(copy).toMatch(/try again/i)
    expect(copy).not.toContain(secret)
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.renderServicePromptPart).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.ragSearch).not.toHaveBeenCalled()
  })

  it("keeps snapshot preflight mutation-free while resolution is pending", async () => {
    let resolveSnapshot!: (snapshot: ReturnType<typeof mocks.makeSnapshot>) => void
    const pendingSnapshot = new Promise<ReturnType<typeof mocks.makeSnapshot>>(
      (resolve) => {
        resolveSnapshot = resolve
      }
    )
    mocks.loadServicePromptSnapshot.mockReturnValue(pendingSnapshot)
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller
    })
    await vi.waitFor(() => {
      expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledTimes(1)
    })

    expectNoPreflightMutation()

    resolveSnapshot(mocks.makeSnapshot())
    await act(async () => {
      await submission
    })
    expect(mocks.setAbortController.mock.calls).toEqual([
      [controller],
      [null]
    ])
  })

  it("silently stops an aborted snapshot preflight without taking controller ownership", async () => {
    const abortError = new Error("Service Prompt request was aborted")
    abortError.name = "AbortError"
    mocks.loadServicePromptSnapshot.mockRejectedValue(abortError)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expectNoPreflightMutation()
    expect(mocks.notification.error).not.toHaveBeenCalled()
    expect(mocks.notification.warning).not.toHaveBeenCalled()
  })

  it("reuses the snapshot for a tool-disabled rewrite and the final Sidepanel answer", async () => {
    const snapshot = mocks.makeSnapshot()
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })
    })

    expect(mocks.pageAssistModel).toHaveBeenNthCalledWith(1, {
      model: "model-1"
    })
    expect(mocks.pageAssistModel).toHaveBeenNthCalledWith(2, {
      model: "model-1",
      toolChoice: "none",
      tools: [],
      saveToDb: false
    })
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      1,
      snapshot.definitions["chat.rag.question_rewrite"]?.definition,
      "template",
      "custom rewrite {chat_history} :: {question}",
      {
        chat_history: [
          "Human: Earlier question",
          "Assistant: Earlier answer",
          "Human: Current follow-up"
        ].join("\n"),
        question: "Current follow-up"
      }
    )
    expect(mocks.ragSearch).toHaveBeenCalledWith("standalone retrieval query", {
      top_k: 4,
      filters: { url: "https://source.example/page" }
    })
    expect(mocks.addMedia).toHaveBeenCalledWith("https://source.example/page")
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      2,
      snapshot.definitions["chat.rag.answer"]?.definition,
      "template",
      "custom answer {context} :: {question}",
      {
        context: "grounded context",
        question: "standalone retrieval query"
      }
    )
    expect(mocks.humanMessageFormatter).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ model: "model-1", useOCR: true })
    )
    expect(mocks.humanMessageFormatter).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ model: "model-1", useOCR: true })
    )
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedModel: "model-1",
        source: [
          expect.objectContaining({
            name: "Source title",
            type: "html",
            url: "https://source.example/page",
            pageContent: "retrieved chunk"
          })
        ]
      })
    )
    const rewriteMessage = mocks.humanMessageFormatter.mock.results[0].value
    const finalMessage = mocks.humanMessageFormatter.mock.results[1].value
    await expect(rewriteMessage).resolves.toEqual({
      content: [
        {
          text: [
            "custom rewrite Human: Earlier question",
            "Assistant: Earlier answer",
            "Human: Current follow-up :: Current follow-up"
          ].join("\n"),
          type: "text"
        }
      ],
      model: "model-1",
      useOCR: true
    })
    await expect(finalMessage).resolves.toEqual({
      content: [
        {
          text: "custom answer grounded context :: standalone retrieval query",
          type: "text"
        }
      ],
      model: "model-1",
      useOCR: true
    })
    expect(mocks.rewriteModel.invoke).toHaveBeenCalledWith([
      await rewriteMessage
    ])
    expect(mocks.finalModel.stream).toHaveBeenCalledWith(
      [await finalMessage],
      {
        signal: controller.signal,
        callbacks: [
          {
            handleLLMEnd: expect.any(Function)
          }
        ]
      }
    )
  })
})
