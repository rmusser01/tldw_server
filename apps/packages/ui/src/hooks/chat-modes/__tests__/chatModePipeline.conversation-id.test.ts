// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { SaveMessageData } from "@/types/chat-modes"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  getModelNicknameByID: vi.fn(async (_modelId?: unknown) => null),
  saveMessageOnSuccess: vi.fn<(data: SaveMessageData) => Promise<string | null>>(async () => "history-1"),
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
  getModelNicknameByID: (modelId: unknown) =>
    mocks.getModelNicknameByID(modelId)
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

import { runChatPipeline, type ChatModeDefinition, type ChatModeParamsBase } from "../chatModePipeline"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"
import type { Message } from "@/store/option"

describe("runChatPipeline conversation id handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.setMessages.mockReset()
    mocks.pageAssistModel.mockResolvedValue({
      conversationId: "server-chat-1",
      saveToDb: true,
      stream: async function* () {
        yield "Search-backed answer"
      }
    })
  })

  it.each([
    { acknowledgement: "server-assistant-1", expected: "server-assistant-1" },
    { acknowledgement: undefined, expected: undefined }
  ])("exposes the saved assistant identity only when acknowledged ($acknowledgement)", async ({ acknowledgement, expected }) => {
    mocks.pageAssistModel.mockResolvedValue({
      conversationId: "server-chat-1",
      saveToDb: true,
      serverMessageId: acknowledgement,
      stream: async function* () { yield "Saved reply" }
    })
    let localMessages: Message[] = [{
      id: "generated-assistant-id", isBot: true, name: "Assistant", message: "", sources: []
    }]
    mocks.setMessages.mockImplementation((next: Message[] | ((prev: Message[]) => Message[])) => {
      localMessages = typeof next === "function" ? next(localMessages) : next
    })
    const mode: ChatModeDefinition<ChatModeParamsBase> = {
      id: "normal",
      setupMessages: () => ({ targetMessageId: "generated-assistant-id" }),
      preparePrompt: async () => ({
        chatHistory: [], humanMessage: { role: "user", content: "Hello" }, sources: []
      })
    }

    await runChatPipeline(mode, "Hello", "", false, localMessages, [], new AbortController().signal, {
      selectedModel: "openai/gpt-4.1-mini", useOCR: false,
      setMessages: mocks.setMessages, saveMessageOnSuccess: mocks.saveMessageOnSuccess,
      saveMessageOnError: mocks.saveMessageOnError, setHistory: mocks.setHistory,
      setIsProcessing: mocks.setIsProcessing, setStreaming: mocks.setStreaming,
      setAbortController: mocks.setAbortController, historyId: "history-1",
      setHistoryId: mocks.setHistoryId, conversationId: "server-chat-1"
    })

    expect(localMessages[0].message).toBe("Saved reply")
    expect(localMessages[0].serverMessageId).toBe(expected)
    expect(mocks.saveMessageOnSuccess.mock.calls[0]?.[0]?.assistantServerMessageId).toBe(expected)
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
