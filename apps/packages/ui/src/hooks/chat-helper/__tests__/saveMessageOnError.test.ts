import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatHistory } from "@/store/option"
import { saveMessageOnError } from "../index"

const mocks = vi.hoisted(() => ({
  fallbackSetHistory: vi.fn(),
  saveMessage: vi.fn(async () => undefined),
  getLastChatHistory: vi.fn(async () => ({ id: "last-message-id" })),
  saveHistory: vi.fn(async () => ({ id: "new-history-id" })),
  updateMessage: vi.fn(async () => undefined),
  setLastUsedChatModel: vi.fn(async () => undefined),
  setLastUsedChatSystemPrompt: vi.fn(async () => undefined),
  updateChatHistoryCreatedAt: vi.fn(async () => undefined),
  generateTitle: vi.fn(async () => "Generated title"),
  updatePageTitle: vi.fn(),
  buildAssistantErrorContent: vi.fn((_botMessage: string, error: unknown) =>
    error instanceof Error ? `ERR: ${error.message}` : "ERR"
  )
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => ({
      setHistory: mocks.fallbackSetHistory
    })
  }
}))

vi.mock("@/db/dexie/helpers", () => ({
  getLastChatHistory: mocks.getLastChatHistory,
  saveHistory: mocks.saveHistory,
  saveMessage: mocks.saveMessage,
  updateMessage: mocks.updateMessage,
  updateLastUsedModel: mocks.setLastUsedChatModel,
  updateLastUsedPrompt: mocks.setLastUsedChatSystemPrompt,
  updateChatHistoryCreatedAt: mocks.updateChatHistoryCreatedAt
}))

vi.mock("@/services/title", () => ({
  generateTitle: mocks.generateTitle
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: mocks.updatePageTitle
}))

vi.mock("@/utils/chat-error-message", () => ({
  buildAssistantErrorContent: mocks.buildAssistantErrorContent
}))

describe("saveMessageOnError", () => {
  beforeEach(() => {
    mocks.fallbackSetHistory.mockClear()
    mocks.saveMessage.mockClear()
    mocks.getLastChatHistory.mockClear()
    mocks.saveHistory.mockClear()
    mocks.updateMessage.mockClear()
    mocks.setLastUsedChatModel.mockClear()
    mocks.setLastUsedChatSystemPrompt.mockClear()
    mocks.updateChatHistoryCreatedAt.mockClear()
    mocks.generateTitle.mockClear()
    mocks.updatePageTitle.mockClear()
    mocks.buildAssistantErrorContent.mockClear()
  })

  it("falls back to store setter when setHistory is not callable", async () => {
    const history: ChatHistory = [
      {
        role: "assistant",
        content: "Earlier assistant response"
      }
    ]

    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history,
        setHistory: null as unknown as (history: ChatHistory) => void,
        image: "",
        userMessage: "Hi there",
        botMessage: "",
        historyId: "history-1",
        selectedModel: "kimi-k2",
        setHistoryId: vi.fn(),
        isRegenerating: false
      })
    ).resolves.toBe("history-1")

    expect(mocks.fallbackSetHistory).toHaveBeenCalledTimes(1)
    expect(mocks.fallbackSetHistory).toHaveBeenCalledWith([
      ...history,
      {
        role: "user",
        content: "Hi there",
        image: ""
      },
      {
        role: "assistant",
        content: "ERR: provider failed"
      }
    ])
  })

  it("uses a local fallback title when provider title generation fails during error recovery", async () => {
    const setHistory = vi.fn()
    const setHistoryId = vi.fn()
    mocks.generateTitle.mockRejectedValueOnce(new Error("missing provider key"))

    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history: [],
        setHistory,
        image: "",
        userMessage: "Please summarize this cockpit error recovery path",
        botMessage: "",
        historyId: null,
        selectedModel: "openai/gpt-4o",
        setHistoryId,
        isRegenerating: false
      })
    ).resolves.toBe("new-history-id")

    expect(mocks.saveHistory).toHaveBeenCalledWith(
      "Please summarize this cockpit error recovery path",
      false,
      "web-ui"
    )
    expect(mocks.updatePageTitle).toHaveBeenCalledWith(
      "Please summarize this cockpit error recovery path"
    )
    expect(setHistoryId).toHaveBeenCalledWith("new-history-id")
  })

  it("persists user and assistant metadata extras when saving error records", async () => {
    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history: [],
        setHistory: vi.fn(),
        image: "",
        userMessage: "Submit OpenUI action",
        botMessage: "",
        historyId: "history-1",
        selectedModel: "openai/gpt-4o",
        setHistoryId: vi.fn(),
        isRegenerating: false,
        userMessageId: "user-1",
        assistantMessageId: "assistant-1",
        userMetadataExtra: {
          dynamic_ui_action: { actionId: "survey" }
        },
        assistantMetadataExtra: {
          dynamic_ui: {
            renderer: "openui",
            version: "v1",
            source: "root = <Card />"
          }
        }
      } as any)
    ).resolves.toBe("history-1")

    expect(mocks.saveMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "user-1",
        role: "user",
        metadataExtra: {
          dynamic_ui_action: { actionId: "survey" }
        }
      })
    )
    expect(mocks.saveMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "assistant-1",
        role: "assistant",
        metadataExtra: {
          dynamic_ui: {
            renderer: "openui",
            version: "v1",
            source: "root = <Card />"
          }
        }
      })
    )
  })
})
