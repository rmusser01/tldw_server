// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  saveMessageOnSuccess: vi.fn<
    (payload: any) => Promise<string | null>
  >(async () => "history-1"),
  saveMessageOnError: vi.fn<
    (payload: any) => Promise<string | null>
  >(async () => "history-1"),
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setAbortController: vi.fn(),
  setHistoryId: vi.fn()
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

// The pipeline only imports a type from this module. Keep this unit test from
// loading the API client's unrelated runtime barrel and its circular imports.
vi.mock("@/services/tldw/TldwApiClient", () => ({}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => "generated-id")
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
  type ChatModeParamsBase
} from "../chatModePipeline"

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
    humanMessage: { role: "user", content: "Tell me a story" },
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

describe("runChatPipeline abort lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "hello world"
      }
    })
  })

  it("discards the empty assistant bubble when aborted before any token", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        // No tokens arrive before the abort.
      }
    })

    const controller = new AbortController()
    controller.abort()

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      controller.signal,
      buildParams()
    )

    expect(result).toMatchObject({ status: "skipped" })
    // Never persisted as a complete answer nor as an empty error bubble.
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
  })

  it("passes the captured prompt request scope to the final model client", async () => {
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const,
        authSource: "manual" as const,
        orgId: 4
      }),
      userId: 19
    })
    const signal = new AbortController().signal

    await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      signal,
      buildParams({
        servicePromptSnapshot: {
          scopeKey: "scope-19",
          requestScope,
          capability: "supported",
          definitions: {},
          scopeSignal: signal,
          release: vi.fn()
        }
      })
    )

    expect(mocks.pageAssistModel).toHaveBeenCalledWith(
      expect.objectContaining({ requestScope })
    )
  })

  it("saves a partially-streamed abort as interrupted, never via the success path", async () => {
    const controller = new AbortController()
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        // User aborts after the first token arrives.
        controller.abort()
      }
    })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      controller.signal,
      buildParams()
    )

    expect(result).toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining("partial answer")
      })
    )
  })

  it("persists a manual-stop partial even when a scoped lease signal also aborts", async () => {
    const userController = new AbortController()
    const scopeController = new AbortController()
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 19
    })
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        userController.abort()
        scopeController.abort()
      }
    })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      scopeController.signal,
      buildParams({
        servicePromptSnapshot: {
          scopeKey: "scope-19",
          requestScope,
          capability: "supported",
          definitions: {},
          scopeSignal: scopeController.signal,
          release: vi.fn()
        },
        discardCurrentTurnOnAbort: () =>
          scopeController.signal.aborted && !userController.signal.aborted
      })
    )

    expect(result).toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining("partial answer"),
        requestScope,
        scopeSignal: scopeController.signal,
        shouldAbortForScopeChange: expect.any(Function)
      })
    )
    const [savePayload] = mocks.saveMessageOnError.mock.calls.at(-1)!
    expect(savePayload.shouldAbortForScopeChange()).toBe(false)
  })

  it("discards a partially-streamed turn when its prompt scope is aborted", async () => {
    const scopeController = new AbortController()
    const previousMessages = [
      {
        id: "previous-user",
        isBot: false,
        name: "You",
        message: "Earlier question",
        sources: [],
        createdAt: 1
      }
    ]
    const previousHistory = [
      { role: "user" as const, content: "Earlier question" }
    ]
    let messagesState = previousMessages
    let historyState = previousHistory
    const setMessages = vi.fn((next: any) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const setHistory = vi.fn((next: any) => {
      historyState = typeof next === "function" ? next(historyState) : next
    })

    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        scopeController.abort()
      }
    })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      previousMessages,
      previousHistory,
      scopeController.signal,
      buildParams({
        setMessages,
        setHistory,
        discardCurrentTurnOnAbort: () => true
      })
    )

    expect(result).toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(messagesState).toEqual(previousMessages)
    expect(historyState).toEqual(previousHistory)
  })

  it("discards a server-rejected scope-changed turn before the scope signal aborts", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    const previousMessages = [
      {
        id: "previous-user",
        isBot: false,
        name: "You",
        message: "Earlier question",
        sources: [],
        createdAt: 1
      }
    ]
    const previousHistory = [
      { role: "user" as const, content: "Earlier question" }
    ]
    let messagesState = previousMessages
    let historyState = previousHistory
    const setMessages = vi.fn((next: any) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const setHistory = vi.fn((next: any) => {
      historyState = typeof next === "function" ? next(historyState) : next
    })

    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        throw scopeChangedError
      }
    })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      previousMessages,
      previousHistory,
      new AbortController().signal,
      buildParams({ setMessages, setHistory })
    )

    expect(result).toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(messagesState).toEqual(previousMessages)
    expect(historyState).toEqual(previousHistory)
  })

  it("discards a turn when its prompt scope changes during final persistence", async () => {
    let resolveSave!: (value: string) => void
    const pendingSave = new Promise<string>((resolve) => {
      resolveSave = resolve
    })
    mocks.saveMessageOnSuccess.mockReturnValueOnce(pendingSave)
    const scopeController = new AbortController()
    const previousMessages = [
      {
        id: "previous-user",
        isBot: false,
        name: "You",
        message: "Earlier question",
        sources: [],
        createdAt: 1
      }
    ]
    const previousHistory = [
      { role: "user" as const, content: "Earlier question" }
    ]
    let messagesState = previousMessages
    let historyState = previousHistory
    const setMessages = vi.fn((next: any) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const setHistory = vi.fn((next: any) => {
      historyState = typeof next === "function" ? next(historyState) : next
    })
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 19
    })

    const submission = runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      previousMessages,
      previousHistory,
      scopeController.signal,
      buildParams({
        setMessages,
        setHistory,
        servicePromptSnapshot: {
          scopeKey: "scope-19",
          requestScope,
          capability: "supported",
          definitions: {},
          scopeSignal: scopeController.signal,
          release: vi.fn()
        },
        discardCurrentTurnOnAbort: () => true
      })
    )

    await vi.waitFor(() => {
      expect(mocks.saveMessageOnSuccess).toHaveBeenCalledTimes(1)
    })
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        requestScope,
        scopeSignal: scopeController.signal
      })
    )
    scopeController.abort()
    resolveSave("history-1")

    await expect(submission).resolves.toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(messagesState).toEqual(previousMessages)
    expect(historyState).toEqual(previousHistory)
  })

  it("discards a turn when its prompt scope changes during error persistence", async () => {
    let resolveSave!: (value: string) => void
    const pendingSave = new Promise<string>((resolve) => {
      resolveSave = resolve
    })
    mocks.saveMessageOnError.mockReturnValueOnce(pendingSave)
    const scopeController = new AbortController()
    const previousMessages = [
      {
        id: "previous-user",
        isBot: false,
        name: "You",
        message: "Earlier question",
        sources: [],
        createdAt: 1
      }
    ]
    const previousHistory = [
      { role: "user" as const, content: "Earlier question" }
    ]
    let messagesState = previousMessages
    let historyState = previousHistory
    const setMessages = vi.fn((next: any) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const setHistory = vi.fn((next: any) => {
      historyState = typeof next === "function" ? next(historyState) : next
    })
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 19
    })
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        throw new Error("provider failed")
      }
    })

    const submission = runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      previousMessages,
      previousHistory,
      scopeController.signal,
      buildParams({
        setMessages,
        setHistory,
        servicePromptSnapshot: {
          scopeKey: "scope-19",
          requestScope,
          capability: "supported",
          definitions: {},
          scopeSignal: scopeController.signal,
          release: vi.fn()
        },
        discardCurrentTurnOnAbort: () => scopeController.signal.aborted
      })
    )

    await vi.waitFor(() => {
      expect(mocks.saveMessageOnError).toHaveBeenCalledTimes(1)
    })
    const [savePayload] = mocks.saveMessageOnError.mock.calls[0]!
    expect(savePayload).toMatchObject({
      requestScope,
      scopeSignal: scopeController.signal
    })
    expect(savePayload.shouldAbortForScopeChange).toEqual(expect.any(Function))
    scopeController.abort()
    expect(savePayload.shouldAbortForScopeChange()).toBe(true)
    resolveSave("history-1")

    await expect(submission).resolves.toMatchObject({ status: "skipped" })
    expect(messagesState).toEqual(previousMessages)
    expect(historyState).toEqual(previousHistory)
  })

  it("marks a stream_transport_interrupted answer as interrupted, never via saveMessageOnSuccess", async () => {
    let messagesState: any[] = []
    const setMessages = vi.fn((updater: any) => {
      messagesState =
        typeof updater === "function" ? updater(messagesState) : updater
    })

    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "partial answer"
        // Extension port dropped after the first byte: the background proxy
        // synthesizes this sentinel, which ChatTldw re-emits as an object chunk.
        yield {
          event: "stream_transport_interrupted",
          detail: "Extension port disconnected",
          partial_response_saved: true
        }
      }
    })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        setMessages,
        userMessageId: "user-1",
        assistantMessageId: "asst-1"
      })
    )

    expect(result).toMatchObject({ status: "skipped" })
    // Never finalized as a complete answer (which would mirror to the server).
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    // Persisted via the interrupted/error path with the partial text preserved.
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining("partial answer")
      })
    )
    const assistant = messagesState.find((m) => m.id === "asst-1")
    expect(assistant?.message).toBe("partial answer")
    expect(assistant?.generationInfo).toMatchObject({
      interrupted: true,
      streamTransportInterrupted: true,
      partialResponseSaved: true
    })
  })

  it("discards the empty variant and restores the prior one when a regenerate is aborted before the first token", async () => {
    const originalAssistant = {
      id: "orig-assistant",
      isBot: true,
      name: "Assistant",
      message: "original answer",
      sources: [],
      createdAt: 2
    }
    let messagesState: any[] = [
      {
        id: "user-1",
        isBot: false,
        name: "You",
        message: "Tell me a story",
        sources: [],
        createdAt: 1
      }
    ]
    const setMessages = vi.fn((updater: any) => {
      messagesState =
        typeof updater === "function" ? updater(messagesState) : updater
    })

    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        // No tokens arrive before the abort.
      }
    })

    const controller = new AbortController()
    controller.abort()

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      true,
      messagesState,
      [],
      controller.signal,
      buildParams({
        setMessages,
        assistantMessageId: "orig-assistant",
        regenerateFromMessage: originalAssistant
      })
    )

    expect(result).toMatchObject({ status: "skipped" })
    // Cleanly discarded — neither a success nor an interrupted variant persisted.
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()

    const assistant = messagesState.find((m) => m.isBot)
    expect(assistant).toBeTruthy()
    // The empty new variant is gone and the prior variant is restored/active.
    expect(assistant.variants?.length ?? 0).toBeLessThanOrEqual(1)
    expect(assistant.activeVariantIndex ?? 0).toBe(0)
    expect(assistant.message).toBe("original answer")
  })

  it("does not reset shared streaming state when the turn no longer owns the controller", async () => {
    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({ releaseAbortControllerIfOwned: () => false })
    )

    expect(result).toMatchObject({ status: "submitted" })
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
  })

  it("resets shared streaming state when the turn still owns the controller", async () => {
    const releaseAbortControllerIfOwned = vi.fn(() => true)

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({ releaseAbortControllerIfOwned })
    )

    expect(result).toMatchObject({ status: "submitted" })
    expect(releaseAbortControllerIfOwned).toHaveBeenCalled()
    expect(mocks.setStreaming).toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).toHaveBeenCalledWith(null)
  })
})
