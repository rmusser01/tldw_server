import { beforeEach, describe, expect, it, vi } from "vitest"
import { pageAssistModel } from "../index"
import { tldwChat, tldwModels } from "@/services/tldw"
import { useStoreChatModelSettings } from "@/store/model"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { runChatPipeline } from "@/hooks/chat-modes/chatModePipeline"

const transport = vi.hoisted(() => ({ bgRequest: vi.fn() }))

vi.mock("@/services/background-proxy", () => ({ bgRequest: transport.bgRequest }))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => ({}),
  getModelSettings: async () => ({})
}))
vi.mock("@/services/tldw-server", () => ({ getDefaultApiProvider: async () => "ollama" }))
vi.mock("@/services/tldw", () => ({
  tldwModels: { getModel: vi.fn(async () => ({ capabilities: [] })) },
  tldwChat: { sendMessage: vi.fn(), streamMessage: vi.fn() }
}))
vi.mock("@/utils/resolve-api-provider", () => ({
  parseProviderQualifiedModelSelection: (model: string) => ({ modelId: model }),
  resolveApiProviderForModel: async () => "ollama"
}))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "assistant-id" }))
vi.mock("@/db/dexie/nickname", () => ({ getModelNicknameByID: async () => null }))
vi.mock("@/utils/mcp-disclosure", () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))

const supportedSpec = {
  paths: { "/api/v1/chat/completions": { post: {
    requestBody: { content: { "application/json": { schema: { properties: { tldw_turn: {} } } } } }
  } } }
}
const requestScope: ServicePromptRequestScope = {
  config: { serverUrl: "https://pinned-server.test", authMode: "multi-user" },
  userId: 42
}
const durableOptions = {
  model: "gemma", apiProvider: "ollama", saveToDb: true,
  conversationId: "workspace-chat", requestScope,
  tldwTurn: { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" },
  originalUserMessage: "Question"
}

describe("pageAssistModel pinned capability cancellation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    transport.bgRequest.mockReset().mockResolvedValue(supportedSpec)
    useStoreChatModelSettings.getState().reset()
  })

  it("rejects an already cancelled durable turn before probing or model preparation", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(pageAssistModel({ ...durableOptions, signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" })
    expect(transport.bgRequest).not.toHaveBeenCalled()
    expect(tldwModels.getModel).not.toHaveBeenCalled()
  })

  it("cancels pending pinned OpenAPI discovery with the owning signal", async () => {
    const controller = new AbortController()
    let finishProbe!: () => void
    transport.bgRequest.mockImplementationOnce(({ abortSignal }: { abortSignal?: AbortSignal }) =>
      new Promise((resolve, reject) => {
        const onAbort = () => reject(abortSignal?.reason)
        abortSignal?.addEventListener("abort", onAbort, { once: true })
        finishProbe = () => {
          abortSignal?.removeEventListener("abort", onAbort)
          resolve(supportedSpec)
        }
      })
    )
    const preparation = pageAssistModel({ ...durableOptions, signal: controller.signal })
    const cancelled = expect(preparation).rejects.toMatchObject({ name: "AbortError" })
    controller.abort()
    finishProbe()
    await cancelled
    expect(transport.bgRequest).toHaveBeenCalledExactlyOnceWith({
      path: "/openapi.json", method: "GET", abortSignal: controller.signal,
      servicePromptConfig: { ...requestScope.config, expectedUserId: 42 },
      headers: { "X-TLDW-Expected-User-ID": "42" }
    })
    expect(tldwModels.getModel).not.toHaveBeenCalled()
  })

  it("does not construct a model when cancellation races with a supported probe response", async () => {
    const controller = new AbortController()
    transport.bgRequest.mockImplementationOnce(async () => {
      controller.abort()
      return supportedSpec
    })
    await expect(pageAssistModel({ ...durableOptions, signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" })
    expect(tldwModels.getModel).not.toHaveBeenCalled()
  })

  it("does not construct a model if the owning turn is cancelled during later model preparation", async () => {
    const controller = new AbortController()
    vi.mocked(tldwModels.getModel).mockImplementationOnce(async () => {
      controller.abort()
      return { id: "gemma", name: "Gemma", provider: "ollama", type: "chat", capabilities: [] }
    })
    await expect(pageAssistModel({ ...durableOptions, signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" })
  })

  it("cancels the real durable pipeline during pinned discovery without preparing or dispatching a model", async () => {
    const controller = new AbortController()
    const scopeInvalidatedSignal = new AbortController().signal
    let finishProbe!: () => void
    transport.bgRequest.mockImplementationOnce(({ abortSignal }: { abortSignal?: AbortSignal }) =>
      new Promise((resolve, reject) => {
        const onAbort = () => reject(abortSignal?.reason)
        abortSignal?.addEventListener("abort", onAbort, { once: true })
        finishProbe = () => {
          abortSignal?.removeEventListener("abort", onAbort)
          resolve(supportedSpec)
        }
      })
    )
    const saveMessageOnSuccess = vi.fn()
    const saveMessageOnError = vi.fn()
    const pending = runChatPipeline(
      {
        id: "normal",
        buildUserMessage: ctx => ({ id: ctx.resolvedUserMessageId, isBot: false, name: "You", message: ctx.message, sources: [] }),
        buildAssistantMessage: ctx => ({ id: ctx.resolvedAssistantMessageId, isBot: true, name: "Assistant", message: "", sources: [] }),
        preparePrompt: async () => ({ chatHistory: [], sources: [] })
      },
      "Question", "", false, [], [], controller.signal,
      {
        selectedModel: "gemma", useOCR: false,
        conversationId: "workspace-chat", tldwTurn: durableOptions.tldwTurn,
        setMessages: vi.fn(), setHistory: vi.fn(), setIsProcessing: vi.fn(),
        setStreaming: vi.fn(), setAbortController: vi.fn(),
        historyId: null, setHistoryId: vi.fn(),
        saveMessageOnSuccess, saveMessageOnError,
        servicePromptSnapshot: {
          scopeKey: "pinned-owner", requestScope, capability: "supported",
          definitions: {}, scopeSignal: controller.signal, scopeInvalidatedSignal,
          release: vi.fn()
        }
      }
    )
    await vi.waitFor(() => expect(transport.bgRequest).toHaveBeenCalledTimes(1))
    controller.abort()
    finishProbe()
    await expect(pending).resolves.toMatchObject({ status: "skipped" })
    expect(transport.bgRequest).toHaveBeenCalledExactlyOnceWith({
      path: "/openapi.json", method: "GET", abortSignal: controller.signal,
      servicePromptConfig: { ...requestScope.config, expectedUserId: 42 },
      headers: { "X-TLDW-Expected-User-ID": "42" }
    })
    expect(tldwModels.getModel).not.toHaveBeenCalled()
    expect(tldwChat.sendMessage).not.toHaveBeenCalled()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    expect(saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(saveMessageOnError).not.toHaveBeenCalled()
  })

  it.each([null, {}, { paths: {} }])("fails closed without authoritative durable support: %j", async spec => {
    transport.bgRequest.mockResolvedValueOnce(spec)
    await expect(pageAssistModel(durableOptions)).rejects.toThrow(/does not support durable/i)
    expect(tldwModels.getModel).not.toHaveBeenCalled()
  })

  it("propagates an owner-change rejection without constructing a model or repeating discovery", async () => {
    const error = Object.assign(new Error("Request scope changed"), { code: "request_config_scope_changed" })
    transport.bgRequest.mockRejectedValueOnce(error)
    await expect(pageAssistModel(durableOptions)).rejects.toBe(error)
    expect(transport.bgRequest).toHaveBeenCalledTimes(1)
    expect(tldwModels.getModel).not.toHaveBeenCalled()
  })

  it("retains supported durable callers that omit a signal", async () => {
    const model = await pageAssistModel(durableOptions)
    expect(model.requestScope).toBe(requestScope)
    expect(model.tldwTurn).toEqual(durableOptions.tldwTurn)
  })

  it("does not discover durable capabilities for ordinary sibling callers", async () => {
    const model = await pageAssistModel({ model: "gemma", saveToDb: false, requestScope })
    expect(model.saveToDb).toBe(false)
    expect(transport.bgRequest).not.toHaveBeenCalled()
  })
})
