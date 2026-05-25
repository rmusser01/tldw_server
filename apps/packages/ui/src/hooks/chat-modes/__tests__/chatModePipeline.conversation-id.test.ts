// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  getModelNicknameByID: vi.fn(async () => null),
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
  generateID: vi.fn(() => "generated-assistant-id")
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: (...args: unknown[]) =>
    mocks.getModelNicknameByID(...args)
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

import { runChatPipeline, type ChatModeDefinition } from "../chatModePipeline"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"

describe("runChatPipeline conversation id handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.pageAssistModel.mockResolvedValue({
      conversationId: "server-chat-1",
      saveToDb: true,
      stream: async function* () {
        yield "Search-backed answer"
      }
    })
  })

  it("passes explicit conversation ids into pageAssistModel instead of relying on store fallback", async () => {
    const mode: ChatModeDefinition<any> = {
      id: "normal",
      setupMessages: () => ({
        targetMessageId: "generated-assistant-id"
      }),
      preparePrompt: async () => ({
        chatHistory: [{ role: "system", content: "system context" }],
        humanMessage: { role: "user", content: "Hello" },
        sources: []
      })
    }

    await runChatPipeline(
      mode,
      "Hello",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      {
        selectedModel: "openai/gpt-4.1-mini",
        useOCR: false,
        toolChoice: "none",
        setMessages: mocks.setMessages,
        saveMessageOnSuccess: mocks.saveMessageOnSuccess,
        saveMessageOnError: mocks.saveMessageOnError,
        setHistory: mocks.setHistory,
        setIsProcessing: mocks.setIsProcessing,
        setStreaming: mocks.setStreaming,
        setAbortController: mocks.setAbortController,
        historyId: "history-1",
        setHistoryId: mocks.setHistoryId,
        conversationId: "server-chat-1"
      }
    )

    expect(mocks.pageAssistModel).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "openai/gpt-4.1-mini",
        toolChoice: "none",
        conversationId: "server-chat-1"
      })
    )
  })

  it("treats empty model streams as failed recoverable assistant turns", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      conversationId: "server-chat-1",
      saveToDb: true,
      stream: async function* () {}
    })
    let localMessages: any[] = [
      {
        id: "assistant-empty",
        isBot: true,
        name: "Assistant",
        message: "",
        sources: []
      }
    ]
    mocks.setMessages.mockImplementation((next: any[] | ((prev: any[]) => any[])) => {
      localMessages =
        typeof next === "function"
          ? next(localMessages)
          : next
    })

    const mode: ChatModeDefinition<any> = {
      id: "normal",
      setupMessages: () => ({
        targetMessageId: "assistant-empty"
      }),
      preparePrompt: async () => ({
        chatHistory: [{ role: "system", content: "system context" }],
        humanMessage: { role: "user", content: "Hello" },
        sources: []
      })
    }

    const result = await runChatPipeline(
      mode,
      "Hello",
      "",
      false,
      localMessages,
      [],
      new AbortController().signal,
      {
        selectedModel: "openai/gpt-4.1-mini",
        useOCR: false,
        toolChoice: "none",
        setMessages: mocks.setMessages,
        saveMessageOnSuccess: mocks.saveMessageOnSuccess,
        saveMessageOnError: mocks.saveMessageOnError,
        setHistory: mocks.setHistory,
        setIsProcessing: mocks.setIsProcessing,
        setStreaming: mocks.setStreaming,
        setAbortController: mocks.setAbortController,
        historyId: "history-1",
        setHistoryId: mocks.setHistoryId,
        conversationId: "server-chat-1"
      }
    )

    expect(result).toEqual({
      status: "failed",
      errorMessage: "No response text was returned."
    })
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).toHaveBeenCalledTimes(1)
    expect(localMessages[0].message).toContain("__tldw_error__:")
    expect(decodeChatErrorPayload(localMessages[0].message)?.detail).toBe(
      "No response text was returned."
    )
    expect(localMessages[0].generationInfo).toMatchObject({
      interrupted: true,
      interruptionReason: "No response text was returned."
    })
  })
})
