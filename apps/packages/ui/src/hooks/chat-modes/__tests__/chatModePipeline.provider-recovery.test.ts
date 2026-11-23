// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  ragSearch: vi.fn(),
  saveMessageOnSuccess: vi.fn(async () => "history-1"),
  saveMessageOnError: vi.fn(async () => "history-1"),
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setAbortController: vi.fn(),
  setHistoryId: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/services/app", () => ({
  getNoOfRetrievedDocs: vi.fn(async () => 8)
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: vi.fn(async (input) => input)
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi
    .fn()
    .mockReturnValueOnce("assistant-1")
    .mockReturnValue("user-1")
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null)
}))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => ({
      setHistory: vi.fn()
    })
  }
}))

import {
  runChatPipeline,
  type ChatModeDefinition,
  type ChatModeParamsBase,
} from "../chatModePipeline"
import { TLDW_ERROR_BUBBLE_PREFIX } from "@/utils/chat-error-message"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { __testing__ as ragTesting } from "../ragMode"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
} from "@/utils/image-generation-chat"

const mode: ChatModeDefinition<ChatModeParamsBase> = {
  id: "normal",
  buildUserMessage: (ctx) => ({
    isBot: false,
    name: "You",
    message: ctx.message,
    sources: [],
    createdAt: ctx.createdAt,
    id: ctx.resolvedUserMessageId
  }),
  buildAssistantMessage: (ctx) => ({
    isBot: true,
    name: "Assistant",
    message: "▋",
    sources: [],
    createdAt: ctx.createdAt,
    id: ctx.resolvedAssistantMessageId
  }),
  preparePrompt: async () => ({
    chatHistory: [{ role: "system", content: "existing system" }],
    humanMessage: { role: "user", content: "Build a dashboard" },
    sources: []
  })
}

const buildParams = (overrides: Record<string, unknown> = {}) => ({
  selectedModel: "test-model",
  useOCR: false,
  setMessages: mocks.setMessages,
  saveMessageOnSuccess: mocks.saveMessageOnSuccess,
  saveMessageOnError: mocks.saveMessageOnError,
  setHistory: mocks.setHistory,
  setIsProcessing: mocks.setIsProcessing,
  setStreaming: mocks.setStreaming,
  setAbortController: mocks.setAbortController,
  historyId: "history-1",
  setHistoryId: mocks.setHistoryId,
  ...overrides
})

describe("runChatPipeline provider recovery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(mocks.pageAssistModel).mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "root = <Card />"
      }
    })
  })

  it.each([
    ["shared-model", "ollama", "shared-model", "ollama"],
    ["openai:shared-model", "ollama", "openai:shared-model", "openai"],
    ["./models/gemma.gguf", "llama.cpp", "./models/gemma.gguf", "llama.cpp"]
  ])("forwards the selected provider for %s", async (selectedModel, settingsProvider, model, apiProvider) => {
    await runChatPipeline(
      mode,
      "Retry the original question",
      "",
      true,
      [],
      [],
      new AbortController().signal,
      {
        ...buildParams(),
        selectedModel,
        currentChatModelSettings: { apiProvider: settingsProvider }
      }
    )

    expect(mocks.pageAssistModel).toHaveBeenCalledWith(
      expect.objectContaining({ model, apiProvider })
    )
  })

  it("preserves default provider resolution when no request provider is selected", async () => {
    await runChatPipeline(
      mode,
      "Use the configured default",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams()
    )

    expect(mocks.pageAssistModel.mock.calls[0][0]).not.toHaveProperty("apiProvider")
  })

  it("does not invoke answer generation during failed selected-source preflight", async () => {
    mocks.ragSearch.mockRejectedValueOnce(new Error("retrieval unavailable"))
    await runChatPipeline(ragTesting.ragModeDefinition, "Question", "", false, [], [], new AbortController().signal, {
      ...buildParams(), selectedModel: "llama:../../../models/gemma:Q4/model.gguf",
      selectedKnowledge: null, ragMediaIds: [7], ragSearchMode: "hybrid",
      ragTopK: null, ragEnableGeneration: true, ragEnableCitations: true, ragSources: [],
      currentChatModelSettings: { apiProvider: "openai" }
    })
    expect(mocks.ragSearch).toHaveBeenCalledWith("Question", expect.objectContaining({
      enable_generation: false, include_media_ids: [7]
    }))
    const retrievalOptions = mocks.ragSearch.mock.calls[0][1]
    for (const key of ["generation_provider", "generation_model", "generation_prompt"]) {
      expect(retrievalOptions).not.toHaveProperty(key)
    }
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
  })

  it("retains the qualified answer provider after retrieval-only selected-source preflight", async () => {
    const retrieval = { results: [{
      content: "The trial starts on 18 November 2026.",
      metadata: { title: "Field memo", media_id: 7 }
    }] }
    let finishRetrieval!: () => void
    mocks.ragSearch.mockReturnValueOnce(new Promise((resolve) => {
      finishRetrieval = () => resolve(retrieval)
    }))
    const signal = new AbortController().signal
    const snapshot: ServicePromptSnapshot = {
      scopeKey: "test-scope",
      requestScope: {
        config: { serverUrl: "https://example.test", authMode: "single-user" },
        userId: 1
      },
      capability: "supported",
      scopeSignal: signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn(),
      definitions: { "chat.rag.answer": {
        definition: { id: "chat.rag.answer", parts: [{
          key: "template", mode: "template", required_variables: ["context", "question"]
        }] },
        parts: { template: "Context: {context}\nQuestion: {question}" },
        source: "packaged", revision: null
      } }
    }
    const pending = runChatPipeline(ragTesting.ragModeDefinition, "Question", "", false, [], [], signal, {
      ...buildParams(), selectedModel: "llama:../../../models/gemma:Q4/model.gguf",
      selectedKnowledge: null, ragMediaIds: [7], ragSearchMode: "hybrid",
      ragTopK: null, ragEnableGeneration: true, ragEnableCitations: true, ragSources: [],
      currentChatModelSettings: { apiProvider: "openai" },
      ragAdvancedOptions: {
        generation_provider: "openai", generation_model: "wrong-model", generation_prompt: "Generate early"
      },
      servicePromptSnapshot: snapshot
    })
    await vi.waitFor(() => expect(mocks.ragSearch).toHaveBeenCalledTimes(1))
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    finishRetrieval()
    const result = await pending
    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    expect(mocks.ragSearch).toHaveBeenCalledWith("Question", expect.objectContaining({
      enable_generation: false, include_media_ids: [7]
    }))
    const retrievalOptions = mocks.ragSearch.mock.calls[0][1]
    for (const key of ["generation_provider", "generation_model", "generation_prompt"]) {
      expect(retrievalOptions).not.toHaveProperty(key)
    }
    expect(mocks.pageAssistModel).toHaveBeenCalledTimes(1)
    expect(mocks.pageAssistModel).toHaveBeenCalledWith(expect.objectContaining({
      apiProvider: "llama.cpp", model: "llama:../../../models/gemma:Q4/model.gguf"
    }))
    expect(mocks.ragSearch.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.pageAssistModel.mock.invocationCallOrder[0]
    )
    expect(result).toMatchObject({ status: "submitted" })
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
  })

  it("keeps a received durable user acknowledgement when the stream fails", async () => {
    const turnId = "12803947-a1f4-4c49-b6eb-bf4218765f6a"
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: true,
      stream: async function* (_messages: unknown, options: { callbacks: { handleLLMEnd: (output: unknown) => void }[] }) {
        try {
          yield "partial"
          throw new Error("connection interrupted")
        } finally {
          options.callbacks[0].handleLLMEnd({ generations: [[{
            generationInfo: { tldw_user_message_id: turnId }
          }]] })
        }
      }
    })
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, {
      ...buildParams(), conversationId: "workspace-chat", tldwTurn: { user_message_id: turnId }
    })
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(expect.objectContaining({
      generationInfo: expect.objectContaining({ tldw_user_message_id: turnId })
    }))
  })

  it("retains the durable turn and raw user content through generation", async () => {
    const turn = { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" }
    const settings = { apiProvider: "ollama", temperature: 0.42, topP: 0.8 }
    await runChatPipeline(mode, "Original question", "", true, [], [], new AbortController().signal, {
      ...buildParams(), conversationId: "workspace-chat", tldwTurn: turn, currentChatModelSettings: settings
    })
    expect(mocks.pageAssistModel).toHaveBeenCalledWith(expect.objectContaining({
      conversationId: "workspace-chat", saveToDb: true,
      tldwTurn: turn, originalUserMessage: "Original question", generationSettings: settings
    }))
  })

  it("leaves ordinary callers on the model factory's live settings", async () => {
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, {
      ...buildParams(), currentChatModelSettings: { temperature: 0.42 }
    })
    expect(mocks.pageAssistModel.mock.calls[0][0]).not.toHaveProperty("generationSettings")
  })

  it("persists the durable receipt and interruption metadata for a partial transport sentinel", async () => {
    const turnId = "12803947-a1f4-4c49-b6eb-bf4218765f6a"
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: true,
      stream: async function* (_messages: unknown, options: { callbacks: { handleLLMEnd: (output: unknown) => void }[] }) {
        yield "partial answer"
        yield { event: "stream_transport_interrupted", detail: "Connection lost" }
        options.callbacks[0].handleLLMEnd({ generations: [[{
          generationInfo: { tldw_user_message_id: turnId }
        }]] })
      }
    })

    const result = await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, {
      ...buildParams(), conversationId: "workspace-chat", tldwTurn: { user_message_id: turnId }
    })

    expect(result).toEqual({ status: "skipped", reason: "Connection lost" })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(expect.objectContaining({
      botMessage: "partial answer",
      generationInfo: {
        tldw_user_message_id: turnId,
        interrupted: true,
        partialResponseSaved: true,
        streamTransportInterrupted: true,
        streamTransportInterruptionReason: "Connection lost"
      }
    }))
  })

  it("turns empty completed provider streams into recoverable assistant errors", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {}
    })

    const result = await runChatPipeline(
      mode,
      "Build a dashboard",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams()
    )

    expect(result).toMatchObject({ status: "failed" })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining(TLDW_ERROR_BUBBLE_PREFIX)
      })
    )
  })

  it("allows empty provider streams for image generation turns", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {}
    })

    const result = await runChatPipeline(
      mode,
      "Create a product mockup",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
        assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
      })
    )

    expect(result).toMatchObject({ status: "submitted" })
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        fullText: "",
        userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
        assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
      })
    )
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
  })

  it.each([0, 503])("does not save a failed selected-source request as an answer (HTTP %s)", async (status) => {
    const error = Object.assign(new Error("Selected-source retrieval failed"), { status })
    mocks.ragSearch.mockRejectedValueOnce(error)
    const history = [{ role: "user" as const, content: "Earlier question" }]

    const result = await runChatPipeline(
      ragTesting.ragModeDefinition,
      "What does the staged memo say?",
      "",
      false,
      [],
      history,
      new AbortController().signal,
      {
        ...buildParams(),
        selectedKnowledge: null,
        ragMediaIds: [7],
        ragSearchMode: "hybrid",
        ragTopK: null,
        ragEnableGeneration: true,
        ragEnableCitations: true,
        ragSources: [],
        currentChatModelSettings: { apiProvider: "llama.cpp" }
      }
    )

    expect(result).toMatchObject({ status: "failed" })
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        e: error,
        history,
        botMessage: expect.stringContaining(TLDW_ERROR_BUBBLE_PREFIX)
      })
    )
    expect(mocks.setHistory).not.toHaveBeenCalled()
  })
})
