import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  events: [] as string[],
  settle: vi.fn<(...args: any[]) => Promise<any>>(async () => ({
    id: "assistant-1"
  })),
  generateTitle: vi.fn<(...args: any[]) => Promise<string>>(async () => {
    mocks.events.push("title")
    return "Generated title"
  }),
  getLastChatHistory: vi.fn(async () => ({ id: "last-assistant" })),
  addFileToSession: vi.fn(async (_historyId: string, file: { id: string }) => {
    mocks.events.push(`save:file:${file.id}`)
  }),
  runTransaction: vi.fn(
    async (
      signal: AbortSignal | undefined,
      operation: () => Promise<unknown>
    ) => {
      if (signal?.aborted) {
        const error = new Error("Request scope changed")
        error.name = "AbortError"
        throw error
      }
      mocks.events.push("transaction:start")
      const result = await operation()
      mocks.events.push("transaction:commit")
      return result
    }
  ),
  saveHistory: vi.fn(async () => {
    mocks.events.push("saveHistory")
    return { id: "new-history" }
  }),
  saveMessage: vi.fn(async (message: { role: string }) => {
    mocks.events.push(`save:${message.role}`)
  }),
  setLastUsedModel: vi.fn(async () => {
    mocks.events.push("save:model")
  }),
  setLastUsedPrompt: vi.fn(async () => {
    mocks.events.push("save:prompt")
  }),
  updateCreatedAt: vi.fn(async () => {
    mocks.events.push("save:createdAt")
  }),
  updateMessage: vi.fn(async () => undefined),
  updatePageTitle: vi.fn(() => {
    mocks.events.push("pageTitle")
  })
}))

vi.mock("@/db/dexie/helpers", () => ({
  addFileToSession: mocks.addFileToSession,
  getLastChatHistory: mocks.getLastChatHistory,
  saveHistory: mocks.saveHistory,
  saveMessage: mocks.saveMessage,
  updateMessage: mocks.updateMessage,
  updateLastUsedModel: mocks.setLastUsedModel,
  updateLastUsedPrompt: mocks.setLastUsedPrompt,
  updateChatHistoryCreatedAt: mocks.updateCreatedAt
}))

vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: mocks.runTransaction
}))

vi.mock("@/services/title", () => ({ generateTitle: mocks.generateTitle }))
vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: mocks.updatePageTitle
}))
vi.mock("@/store/option", () => ({
  useStoreMessageOption: { getState: () => ({ setHistory: vi.fn() }) }
}))
vi.mock("@/utils/chat-error-message", () => ({
  buildAssistantErrorContent: vi.fn()
}))

vi.mock("@/services/chat-history-selection", () => ({
  settleAcceptedAssistant: mocks.settle
}))

import { saveMessageOnSuccess, saveMessageOnError } from "../index"

const requestScope = Object.freeze({
  config: Object.freeze({
    serverUrl: "https://scope.example",
    authMode: "multi-user" as const
  }),
  userId: 7
})

const payload = (overrides: Record<string, unknown> = {}) => ({
  historyId: "history-1",
  setHistoryId: vi.fn(),
  isRegenerate: false,
  selectedModel: "model-1",
  message: "question",
  image: "",
  fullText: "answer",
  source: [],
  modelId: "model-1",
  userMessageId: "user-1",
  assistantMessageId: "assistant-1",
  ...overrides
})

describe("saveMessageOnSuccess request scope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.events.length = 0
  })

  it("commits the complete existing-history turn in one scoped transaction", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()

    await expect(saveMessageOnSuccess(payload({
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: scopeInvalidatedController.signal,
      requestScope,
      prompt_id: "prompt-1"
    }) as any)).resolves.toBe("history-1")

    expect(mocks.runTransaction).toHaveBeenCalledWith(
      scopeInvalidatedController.signal,
      expect.any(Function)
    )
    expect(mocks.events).toEqual([
      "transaction:start",
      "save:user",
      "save:assistant",
      "save:model",
      "save:prompt",
      "save:createdAt",
      "transaction:commit"
    ])
  })

  it("commits document session files in the same scoped transaction", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()
    const sessionFile = { id: "file-1", filename: "notes.txt" }

    await expect(saveMessageOnSuccess(payload({
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: scopeInvalidatedController.signal,
      requestScope,
      sessionFilesToAdd: [sessionFile]
    }) as any)).resolves.toBe("history-1")

    expect(mocks.addFileToSession).toHaveBeenCalledWith(
      "history-1",
      sessionFile
    )
    expect(mocks.events).toEqual([
      "transaction:start",
      "save:user",
      "save:assistant",
      "save:model",
      "save:createdAt",
      "save:file:file-1",
      "transaction:commit"
    ])
  })

  it("defers existing-history metadata for a scoped Compare aggregate", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()

    await expect(saveMessageOnSuccess(payload({
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: scopeInvalidatedController.signal,
      requestScope,
      prompt_id: "prompt-1",
      deferHistoryMetadata: true
    }) as any)).resolves.toBe("history-1")

    expect(mocks.events).toEqual([
      "transaction:start",
      "save:user",
      "save:assistant",
      "transaction:commit"
    ])
  })

  it("generates a scoped title before atomically creating a new history", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()
    const setHistoryId = vi.fn(() => mocks.events.push("setHistoryId"))

    await expect(saveMessageOnSuccess(payload({
      historyId: null,
      setHistoryId,
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: scopeInvalidatedController.signal,
      requestScope
    }) as any)).resolves.toBe("new-history")

    expect(mocks.generateTitle).toHaveBeenCalledWith(
      "model-1",
      "question",
      "question",
      { signal: scopeInvalidatedController.signal, requestScope }
    )
    expect(mocks.events).toEqual([
      "title",
      "transaction:start",
      "saveHistory",
      "save:user",
      "save:assistant",
      "save:model",
      "transaction:commit",
      "pageTitle",
      "setHistoryId"
    ])
  })

  it("finishes a new history when Stop is pressed during post-success title generation", async () => {
    const controller = new AbortController()
    const scopeInvalidatedController = new AbortController()
    let titleStarted!: () => void
    let finishTitle!: () => void
    const started = new Promise<void>((resolve) => {
      titleStarted = resolve
    })
    const titleGate = new Promise<void>((resolve) => {
      finishTitle = resolve
    })
    mocks.generateTitle.mockImplementationOnce(async (
      _model: string,
      _message: string,
      _question: string,
      options?: { signal?: AbortSignal }
    ) => {
      titleStarted()
      await new Promise<void>((resolve, reject) => {
        options?.signal?.addEventListener("abort", () => {
          const error = new Error("Request cancelled")
          error.name = "AbortError"
          reject(error)
        }, { once: true })
        void titleGate.then(resolve)
      })
      return "Generated title"
    })

    const persistence = saveMessageOnSuccess(payload({
      historyId: null,
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: scopeInvalidatedController.signal,
      requestScope
    }) as any)

    await started
    controller.abort()
    finishTitle()

    await expect(persistence).resolves.toBe("new-history")
    expect(mocks.generateTitle).toHaveBeenCalledWith(
      "model-1",
      "question",
      "question",
      { signal: scopeInvalidatedController.signal, requestScope }
    )
    expect(mocks.saveHistory).toHaveBeenCalledOnce()
    expect(mocks.saveMessage).toHaveBeenCalledTimes(2)
  })

  it("writes nothing for an already-aborted request scope", async () => {
    const controller = new AbortController()
    controller.abort()

    await expect(saveMessageOnSuccess(payload({
      historyId: null,
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: controller.signal,
      requestScope
    }) as any)).rejects.toMatchObject({ name: "AbortError" })

    expect(mocks.generateTitle).not.toHaveBeenCalled()
    expect(mocks.saveHistory).not.toHaveBeenCalled()
    expect(mocks.saveMessage).not.toHaveBeenCalled()
  })
})


it("settles an accepted native parent without duplicating the user or serializing local placeholders", async () => {
  const turn: any = {
    owner: { kind: "native", conversation_id: "original" },
    admission: { input_message_id: "accepted" },
    createdAt: 123,
    complete: vi.fn(async () => {}),
    followResult: vi.fn(async () => {})
  }
  await saveMessageOnSuccess(
    payload({
      historyTurn: turn,
      documents: [],
      reasoning_time_taken: 0
    }) as any
  )
  expect(mocks.settle).toHaveBeenCalledWith(turn.owner, turn.admission, {
    id: "assistant-1",
    history_id: "original",
    role: "assistant",
    name: "model-1",
    content: "answer",
    createdAt: 123,
    images: [],
    parent_message_id: "accepted"
  })
  expect(mocks.saveMessage).not.toHaveBeenCalled()
  expect(turn.complete).toHaveBeenCalledOnce()
})
it("accepted errors never invoke the legacy duplicate-user persistence path", async () => {
  const turn: any = {
    owner: { kind: "native", conversation_id: "original" },
    admission: { input_message_id: "accepted" },
    recover: vi.fn(async () => {}),
    createdAt: 123
  }
  await saveMessageOnError({
    ...payload(),
    history: [],
    setHistory: vi.fn(),
    userMessage: "question",
    historyTurn: turn,
    assistantMessageId: "assistant-1",
    e: new Error("provider failed"),
    botMessage: "partial"
  } as any)
  expect(mocks.saveMessage).not.toHaveBeenCalled()
  expect(turn.recover).toHaveBeenCalledWith(
    { content: "partial", assistantId: "assistant-1", createdAt: 123 },
    expect.any(Error)
  )
})

it("local tool-only settlement preserves canonical tool metadata for the next owner capture", async () => {
  const calls = [
    {
      id: "call-1",
      type: "function",
      function: { name: "lookup", arguments: "{}" }
    }
  ]
  const turn: any = {
    owner: { kind: "local", conversation_id: "original" },
    admission: { input_message_id: "accepted" },
    createdAt: 123,
    complete: vi.fn(async () => {}),
    followResult: vi.fn(async () => {})
  }
  await saveMessageOnSuccess(
    payload({
      historyTurn: turn,
      fullText: "",
      generationInfo: { tool_calls: calls }
    }) as any
  )
  expect(mocks.settle.mock.lastCall?.[2]).toMatchObject({
    content: "",
    metadataExtra: { tool_calls: calls }
  })
})
