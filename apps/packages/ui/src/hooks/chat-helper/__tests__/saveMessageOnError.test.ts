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
  runTransaction: vi.fn(
    async (
      signal: AbortSignal | undefined,
      operation: () => Promise<unknown>,
      shouldAbort?: () => boolean
    ) => {
      if (signal?.aborted && (shouldAbort?.() ?? true)) {
        const error = new Error("Request scope changed")
        error.name = "AbortError"
        throw error
      }
      return operation()
    }
  ),
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

vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: mocks.runTransaction
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
    mocks.runTransaction.mockClear()
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

  it("writes nothing when an error save starts after a scope-only abort", async () => {
    const controller = new AbortController()
    controller.abort()
    const setHistory = vi.fn()

    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history: [],
        setHistory,
        image: "",
        userMessage: "Scoped question",
        botMessage: "partial",
        historyId: "history-1",
        selectedModel: "model-1",
        setHistoryId: vi.fn(),
        isRegenerating: false,
        scopeSignal: controller.signal,
        scopeInvalidatedSignal: controller.signal,
        shouldAbortForScopeChange: () => true
      } as any)
    ).rejects.toMatchObject({ name: "AbortError" })

    expect(mocks.saveMessage).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  })

  it("defers scoped Compare metadata while preserving error messages", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()

    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history: [],
        setHistory: vi.fn(),
        image: "",
        userMessage: "Scoped question",
        botMessage: "",
        historyId: "history-1",
        selectedModel: "model-1",
        setHistoryId: vi.fn(),
        isRegenerating: false,
        scopeSignal: controller.signal,
        scopeInvalidatedSignal: scopeInvalidatedController.signal,
        deferHistoryMetadata: true,
        prompt_id: "prompt-1"
      } as any)
    ).resolves.toBe("history-1")

    expect(mocks.saveMessage).toHaveBeenCalledTimes(2)
    expect(mocks.setLastUsedChatModel).not.toHaveBeenCalled()
    expect(mocks.setLastUsedChatSystemPrompt).not.toHaveBeenCalled()
  })

  it("still persists a manual-stop partial when the scope lease signal is aborted", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()
    controller.abort()

    await expect(
      saveMessageOnError({
        e: Object.assign(new Error("Request cancelled"), { name: "AbortError" }),
        history: [],
        setHistory: vi.fn(),
        image: "",
        userMessage: "Scoped question",
        botMessage: "partial answer",
        historyId: "history-1",
        selectedModel: "model-1",
        setHistoryId: vi.fn(),
        isRegenerating: false,
        scopeSignal: controller.signal,
        scopeInvalidatedSignal: scopeInvalidatedController.signal,
        shouldAbortForScopeChange: () => false
      } as any)
    ).resolves.toBe("history-1")

    expect(mocks.runTransaction).toHaveBeenCalledWith(
      scopeInvalidatedController.signal,
      expect.any(Function),
      expect.any(Function)
    )
    expect(mocks.saveMessage).toHaveBeenCalledWith(
      expect.objectContaining({ role: "assistant", content: "ERR: Request cancelled" })
    )
  })

  it("does not turn a scoped title 412 into an unscoped fallback history", async () => {
    const scopeInvalidatedController = new AbortController()
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: { code: "request_config_scope_changed" }
    })
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 7
    })
    mocks.generateTitle.mockRejectedValueOnce(scopeChangedError)

    await expect(
      saveMessageOnError({
        e: new Error("provider failed"),
        history: [],
        setHistory: vi.fn(),
        image: "",
        userMessage: "Scoped question",
        botMessage: "partial answer",
        historyId: null,
        selectedModel: "model-1",
        setHistoryId: vi.fn(),
        isRegenerating: false,
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: scopeInvalidatedController.signal,
        requestScope,
        shouldAbortForScopeChange: () => false
      } as any)
    ).rejects.toBe(scopeChangedError)

    expect(mocks.generateTitle).toHaveBeenCalledWith(
      "model-1",
      "Scoped question",
      "Scoped question",
      { requestScope, signal: expect.any(AbortSignal) }
    )
    expect(mocks.saveHistory).not.toHaveBeenCalled()
    expect(mocks.saveMessage).not.toHaveBeenCalled()
  })

  it("cancels scoped title generation when the account scope changes", async () => {
    const controller = new AbortController()
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 7
    })
    mocks.generateTitle.mockImplementationOnce(
      async (
        _model: string,
        _message: string,
        _fallback: string,
        options?: { signal?: AbortSignal }
      ) => await new Promise<string>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"))
        }, { once: true })
      })
    )

    const pending = saveMessageOnError({
      e: new Error("provider failed"),
      history: [],
      setHistory: vi.fn(),
      image: "",
      userMessage: "Scoped question",
      botMessage: "partial answer",
      historyId: null,
      selectedModel: "model-1",
      setHistoryId: vi.fn(),
      isRegenerating: false,
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: controller.signal,
      requestScope,
      shouldAbortForScopeChange: () => controller.signal.aborted
    })
    await vi.waitFor(() => expect(mocks.generateTitle).toHaveBeenCalledOnce())

    controller.abort()

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.saveHistory).not.toHaveBeenCalled()
    expect(mocks.saveMessage).not.toHaveBeenCalled()
  })
})
