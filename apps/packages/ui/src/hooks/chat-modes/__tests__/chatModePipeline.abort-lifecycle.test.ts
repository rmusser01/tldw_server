// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  saveMessageOnSuccess: vi.fn(async () => "history-1"),
  saveMessageOnError: vi.fn(async () => "history-1"),
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
