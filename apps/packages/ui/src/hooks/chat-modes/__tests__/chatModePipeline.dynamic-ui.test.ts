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

describe("runChatPipeline dynamic UI request mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(mocks.pageAssistModel).mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "root = <Card />"
      }
    })
  })

  it("injects OpenUI instructions and saves metadata only after source preflight passes", async () => {
    const stream = vi.fn(async function* () {
      yield "root = <Card />"
    })
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream
    })

    await runChatPipeline(
      mode,
      "Build a dashboard",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        dynamicUIRequest: { renderer: "openui" }
      })
    )

    const promptMessages = (stream.mock.calls[0] as unknown[] | undefined)?.[0]
    expect(JSON.stringify(promptMessages)).toContain("OpenUI")
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        assistantMetadataExtra: expect.objectContaining({
          dynamic_ui: expect.objectContaining({
            renderer: "openui",
            source: "root = <Card />"
          })
        })
      })
    )

    const finalMessageUpdate = mocks.setMessages.mock.calls
      .map(([updater]) => updater)
      .filter((updater) => typeof updater === "function")
      .at(-1) as ((messages: any[]) => any[]) | undefined
    expect(finalMessageUpdate?.([
      { isBot: true, name: "Assistant", message: "▋", sources: [], id: "assistant-1" }
    ])).toEqual([
      expect.objectContaining({
        id: "assistant-1",
        metadataExtra: expect.objectContaining({
          dynamic_ui: expect.objectContaining({ renderer: "openui" })
        })
      })
    ])
  })

  it("does not tag plain text responses even when OpenUI was requested", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "I cannot do that."
      }
    })

    await runChatPipeline(
      mode,
      "Build a dashboard",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        dynamicUIRequest: { renderer: "openui" }
      })
    )

    const payload = (
      mocks.saveMessageOnSuccess.mock.calls as unknown[][]
    ).at(-1)?.[0] as { assistantMetadataExtra?: unknown } | undefined
    expect(payload).toBeDefined()
    expect(payload?.assistantMetadataExtra).toBeUndefined()
  })

  it("updates a reserved user message instead of appending a duplicate", async () => {
    await runChatPipeline(
      mode,
      "Summarize this file",
      "",
      false,
      [
        {
          isBot: false,
          name: "You",
          message: "Summarize this file",
          sources: [],
          id: "reserved-user",
          metadataExtra: {
            documentProcessing: {
              status: "processing",
              files: []
            }
          }
        }
      ],
      [],
      new AbortController().signal,
      buildParams({
        userMessageId: "reserved-user",
        assistantMessageId: "assistant-for-reserved",
        userMetadataExtra: {
          documentProcessing: {
            status: "sending_prompt",
            files: [
              {
                id: "file-1",
                filename: "scan.pdf",
                mode: "ocr_pages",
                status: "ready"
              }
            ]
          }
        }
      })
    )

    const setupUpdater = mocks.setMessages.mock.calls.find(
      ([updater]) => typeof updater === "function"
    )?.[0] as ((messages: any[]) => any[]) | undefined
    const nextMessages = setupUpdater?.([
      {
        isBot: false,
        name: "You",
        message: "Summarize this file",
        sources: [],
        id: "reserved-user",
        metadataExtra: {
          documentProcessing: {
            status: "processing",
            files: []
          }
        }
      }
    ])

    expect(nextMessages?.filter((message) => !message.isBot)).toHaveLength(1)
    expect(nextMessages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "reserved-user",
          metadataExtra: expect.objectContaining({
            documentProcessing: expect.objectContaining({
              status: "sending_prompt"
            })
          })
        }),
        expect.objectContaining({
          id: "assistant-for-reserved",
          isBot: true
        })
      ])
    )
  })
})
