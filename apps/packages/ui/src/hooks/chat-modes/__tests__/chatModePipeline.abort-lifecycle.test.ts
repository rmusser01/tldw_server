// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { HistorySendTurn } from "@/types/chat-modes"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { getLocalHistoryOwner } from "@/db/dexie/history-selection"

// The detached-view control replaces only the physical table engine. Profile
// ownership, capture/bookmark validation and the mounted controller are real.
const tableIO = vi.hoisted(() => {
  type Row = Record<string, unknown>
  type Collection = { toArray: () => Promise<Row[]>; and: (predicate: (row: Row) => boolean) => Collection }
  const tables: Record<string, ReturnType<typeof table>> = {}
  function table(name: string) {
    const rows = new Map<string, Row>()
    const key = (row: Row) => JSON.stringify(name === "historySelections"
      ? [row.profile_id, row.client_session_id, row.owner_key, row.conversation_id] : row.id)
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: predicate => collection(row => matches(row) && predicate(row))
    })
    return { rows,
      get: async (id: unknown) => structuredClone(rows.get(JSON.stringify(id))),
      put: async (row: Row) => { rows.set(key(row), structuredClone(row)) },
      update: async (id: unknown, patch: Row) => { rows.set(JSON.stringify(id), { ...rows.get(JSON.stringify(id)), ...structuredClone(patch) }) },
      toArray: async () => structuredClone([...rows.values()]),
      where: (field: string) => ({ equals: (value: unknown) => collection(row => row[field] === value) })
    }
  }
  for (const name of ["chatHistories", "messages", "userSettings", "sessionFiles", "compareStates", "historySelections", "historyProjections", "forkOperations", "modelNickname", "prompts"]) tables[name] = table(name)
  let tail = Promise.resolve()
  return { tables, db: { ...tables,
    transaction: (_mode: unknown, _tables: unknown, operation: (tx: { abort: () => void }) => Promise<unknown>) => {
      const result = tail.then(async () => {
        const copies = Object.values(tables).map(table => new Map(table.rows))
        try { return await operation({ abort: () => { throw new Error("aborted") } }) }
        catch (error) {
          Object.values(tables).forEach((table, index) => {
            table.rows.clear()
            copies[index].forEach((row, key) => table.rows.set(key, row))
          })
          throw error
        }
      })
      tail = result.then(() => {}, () => {})
      return result
    }
  } }
})
vi.mock("@/db/dexie/schema", () => ({ db: tableIO.db }))

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  nickname: vi.fn(async () => null),
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

// These unit controls use local table ownership and a declared model boundary.
// They never acquire a native API owner; avoid the unrelated runtime barrel.
vi.mock("@/services/tldw/TldwApiClient", () => ({}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => "generated-id")
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: mocks.nickname
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

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const snapshot = (
  requestScope: ServicePromptSnapshot["requestScope"],
  scopeSignal: AbortSignal,
  scopeInvalidatedSignal: AbortSignal = new AbortController().signal
): ServicePromptSnapshot => ({
  scopeKey: "scope-19", requestScope, capability: "supported", definitions: {},
  scopeSignal, scopeInvalidatedSignal, release: vi.fn()
})

const buildParams = (overrides: Partial<ChatModeParamsBase> = {}) => ({
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
    mocks.nickname.mockReset().mockResolvedValue(null)
    mocks.saveMessageOnSuccess.mockReset().mockResolvedValue("history-1")
    mocks.saveMessageOnError.mockReset().mockResolvedValue("history-1")
    Object.values(tableIO.tables).forEach(table => table.rows.clear())
    mocks.pageAssistModel.mockReset().mockResolvedValue({
      saveToDb: false,
      stream: async function* () {
        yield "hello world"
      }
    })
  })

  it("holds Stop activity through preparation and releases it after its response settles", async () => {
    let release!: () => void, enter!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const entered = new Promise<void>((resolve) => {
      enter = resolve
    })
    const heldMode = {
      ...mode,
      preflight: async () => {
        enter()
        await gate
        return null
      }
    }
    const pending = runChatPipeline(
      heldMode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        ownsAbortController: () => true,
        releaseAbortControllerIfOwned: () => true
      })
    )
    try {
      await entered
      expect(mocks.setStreaming).toHaveBeenLastCalledWith(true)
      expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
      release()
      await pending
      expect(mocks.setStreaming).toHaveBeenLastCalledWith(false)
    } finally {
      release()
      await pending
    }
  })

  it("does not acquire or release activity after its controller has been replaced", async () => {
    await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      buildParams({
        ownsAbortController: () => false,
        releaseAbortControllerIfOwned: () => false
      })
    )
    expect(mocks.setStreaming).not.toHaveBeenCalled()
    expect(mocks.setAbortController).not.toHaveBeenCalled()
  })

  it("does not acquire activity for a detached history view", async () => {
    // An unowned persisted local row is qualified by the actual profile owner
    // helper; neither a bare ID nor a fabricated capture supplies H1 authority.
    await tableIO.tables.chatHistories.put({ id: "history-1", title: "Original local history", createdAt: 1, is_rag: false })
    const owner = await getLocalHistoryOwner("history-1")
    const mounted = renderHook(() => useHistorySelection())
    try {
      await act(async () => { expect(await mounted.result.current.open(owner)).toBe(true) })
      const original = mounted.result.current.getCurrent()
      expect(original.status).toBe("ready")
      if (original.capture?.status !== "captured" || original.owner?.kind !== "local") throw new Error("actual_local_capture_required")
      const originalOwner = original.owner
      const viewIsCurrent = mounted.result.current.fence()
      const lease = new AbortController()
      const historyTurn: HistorySendTurn = {
        owner: originalOwner, capture: original.capture,
        currentView: () => mounted.result.current.getCurrent().view,
        validateLease: () => !lease.signal.aborted && originalOwner.validate_lease?.() !== false,
        canUpdateView: viewIsCurrent,
        recover: async () => { throw new Error("Detached preparation must not recover") },
        followResult: async () => { throw new Error("Detached preparation must not follow a result") }
      }
      act(() => mounted.result.current.reset())
      expect(historyTurn.validateLease()).toBe(true)
      expect(historyTurn.canUpdateView()).toBe(false)
      const detachedMode = { ...mode, preflight: async () => ({ handled: true as const, fullText: "prepared result" }) }
      await runChatPipeline(detachedMode, "Tell me a story", "", false, [], [], lease.signal,
        buildParams({ historyTurn, ownsAbortController: () => true, releaseAbortControllerIfOwned: () => false }))
      expect(mocks.setStreaming).not.toHaveBeenCalled()
    } finally { mounted.unmount() }
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
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(true)
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
    const stream = vi.fn(async function* () { yield "hello world" })
    mocks.pageAssistModel.mockResolvedValue({ saveToDb: false, stream })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      signal,
      buildParams({
        servicePromptSnapshot: snapshot(requestScope, signal)
      })
    )

    expect(mocks.pageAssistModel).toHaveBeenCalledWith(
      expect.objectContaining({ requestScope })
    )
    expect(stream).toHaveBeenCalledOnce()
    expect(result).toMatchObject({ status: "submitted" })
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
    const invalidationController = new AbortController()
    const scopeSignal = AbortSignal.any([userController.signal, invalidationController.signal])
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 19
    })
    const stream = vi.fn(async function* () {
      yield "partial answer"
      userController.abort()
    })
    mocks.pageAssistModel.mockResolvedValue({ saveToDb: false, stream })

    const result = await runChatPipeline(
      mode,
      "Tell me a story",
      "",
      false,
      [],
      [],
      scopeSignal,
      buildParams({
        servicePromptSnapshot: snapshot(requestScope, scopeSignal, invalidationController.signal),
        discardCurrentTurnOnAbort: () =>
          scopeSignal.aborted && !userController.signal.aborted
      })
    )

    expect(stream).toHaveBeenCalledOnce()
    expect(invalidationController.signal.aborted).toBe(false)
    expect(result).toMatchObject({ status: "skipped" })
    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining("partial answer"),
        requestScope,
        scopeSignal: scopeSignal,
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
    const saveEntered = deferred<unknown>()
    mocks.saveMessageOnSuccess.mockImplementationOnce(payload => { saveEntered.resolve(payload); return pendingSave })
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
        servicePromptSnapshot: snapshot(requestScope, scopeController.signal, scopeController.signal),
        discardCurrentTurnOnAbort: () => true
      })
    )

    try {
      await Promise.race([saveEntered.promise, submission.then(result => {
        throw new Error("The declared persistence gate was not entered: " + JSON.stringify(result))
      })])
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
    } finally {
      scopeController.abort()
      resolveSave("history-1")
      await submission
    }

  })

  it("discards a turn when its prompt scope changes during error persistence", async () => {
    let resolveSave!: (value: string) => void
    const pendingSave = new Promise<string>((resolve) => {
      resolveSave = resolve
    })
    const saveEntered = deferred<unknown>()
    mocks.saveMessageOnError.mockImplementationOnce(payload => { saveEntered.resolve(payload); return pendingSave })
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
        servicePromptSnapshot: snapshot(requestScope, scopeController.signal, scopeController.signal),
        discardCurrentTurnOnAbort: () => scopeController.signal.aborted
      })
    )

    try {
      await Promise.race([saveEntered.promise, submission.then(result => {
        throw new Error("The declared persistence gate was not entered: " + JSON.stringify(result))
      })])
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
    } finally {
      scopeController.abort()
      resolveSave("history-1")
      await submission
    }

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


import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { captureActorSceneSendForTarget } from "@/hooks/chat/actor-scene-source"
import { createDefaultActorSettings } from "@/types/actor"
const sceneIO = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat: sceneIO }))
it.each(["nickname", "retrieval preflight", "handled retrieval preflight", "prompt", "model factory"])("does not dispatch or save a preparation retired while awaiting %s", async boundary => {
  mocks.pageAssistModel.mockReset()
  mocks.nickname.mockReset().mockResolvedValue(null)
  mocks.setMessages.mockClear()
  mocks.saveMessageOnSuccess.mockClear()
  mocks.saveMessageOnError.mockClear()
  let current = true, release!: () => void, enter!: () => void
  const waiting = new Promise<void>(resolve => { release = resolve })
  const entered = new Promise<void>(resolve => { enter = resolve })
  const requestScope = { config: { serverUrl: "https://pipeline-scene-source.example", authMode: "single-user" as const }, userId: 1 }
  const target = captureActorSettingsTarget({ address: { kind: "fresh", id: "pipeline-original-draft", temporary: false, scope: { type: "global" } }, requestScope, accountIsCurrent: () => true, isCurrent: () => current })!
  sceneIO.mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
  const actorSceneSend = await captureActorSceneSendForTarget(target, target.account)!.read()
  const stream = vi.fn(async function* () { yield "Should not be sent" })
  const hold = async () => { enter(); await waiting }
  mocks.pageAssistModel.mockImplementation(async () => { if (boundary === "model factory") await hold(); return { saveToDb: false, stream } })
  if (boundary === "nickname") mocks.nickname.mockImplementationOnce(async () => { await hold(); return null })
  const heldMode = { ...mode,
    preflight: async () => { if (boundary === "retrieval preflight" || boundary === "handled retrieval preflight") await hold(); return boundary === "handled retrieval preflight" ? { handled: true as const, fullText: "Obsolete selected-source diagnostic" } : null },
    preparePrompt: async () => { if (boundary === "prompt") await hold(); return { chatHistory: [{ role: "user", content: "ORIGINAL INPUT" }] } }
  }
  const pending = runChatPipeline(heldMode, "ORIGINAL INPUT", "", false, [], [], new AbortController().signal, buildParams({ actorSceneSend }))
  await entered
  const publishedBeforeRetirement = mocks.setMessages.mock.calls.length
  current = false
  release()
  expect(await pending).toMatchObject({ status: "skipped" })
  expect(stream).not.toHaveBeenCalled()
  expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
  expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
  expect(mocks.setMessages).toHaveBeenCalledTimes(publishedBeforeRetirement)
})
