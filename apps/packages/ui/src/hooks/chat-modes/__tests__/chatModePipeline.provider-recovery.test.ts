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

import { runChatPipeline, type ChatModeDefinition } from "../chatModePipeline"
import { TLDW_ERROR_BUBBLE_PREFIX } from "@/utils/chat-error-message"

const mode: ChatModeDefinition<any> = {
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
})
