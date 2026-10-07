import { beforeEach, describe, expect, it, vi } from "vitest"
import { HumanMessage, SystemMessage } from "@/types/messages"
const mocks = vi.hoisted(() => ({
  append: vi.fn(),
  settle: vi.fn(),
  stream: vi.fn(),
  model: vi.fn(),
  request: vi.fn()
}))
vi.mock("@/models", () => ({ pageAssistModel: mocks.model }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.request(...args)
}))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => crypto.randomUUID() }))
vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: async () => null
}))
vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: () => {}
}))
vi.mock("@/services/chat-history-selection", async (original) => ({
  ...(await original<any>()),
  appendSelectedUser: mocks.append,
  settleAcceptedAssistant: mocks.settle
}))
import { runChatPipeline } from "../chatModePipeline"
import { captureHistorySnapshot, historyAdmissionReference } from "@/services/chat-history-selection"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { inspectHistoryDurableRecovery, validateHistoryDurableAdmission } from "@/services/history-durable-turn"
import { validateHistoryDurableResultReceipt } from "@/utils/history-durable-sources"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import type { HistoryDurableRequestBodyV1 } from "@/types/history-durable-turn"

const makeTurn = () => {
  const view = {
    owner_key: "local-key",
    conversation_id: "chat",
    view_session_id: "view",
    cursor: { kind: "empty" },
    interpretation: { kind: "parent_graph_v1" },
    selection_revision: 1
  }
  const turn: any = {
    owner: {
      kind: "local",
      profile_id: "profile",
      owner_key: "local-key",
      conversation_id: "chat"
    },
    capture: {
      status: "captured",
      snapshot: {
        version: 1,
        owner_key: "local-key",
        conversation_id: "chat",
        fences: { conversation: "1", history: "1", settings: "1" },
        nodes: [],
        source_digest: "source",
        storage_context_digest: "storage",
        interpretation_status: { kind: "parent_graph_v1" }
      },
      rows: [],
      selected_content: [],
      view,
      purpose: "send",
      storage_context_digest: "storage"
    },
    currentView: () => view,
    validateLease: () => true,
    canUpdateView: () => true,
    recover: vi.fn(async () => {}),
    followResult: vi.fn(async () => {})
  }
  return turn
}
const mode: any = {
  id: "normal",
  buildUserMessage: (ctx: any) => ({
    id: ctx.resolvedUserMessageId,
    message: ctx.message,
    isBot: false
  }),
  buildAssistantMessage: (ctx: any) => ({
    id: ctx.resolvedAssistantMessageId,
    message: "",
    isBot: true
  }),
  preparePrompt: async () => ({
    chatHistory: [new SystemMessage("system")],
    humanMessage: new HumanMessage("question")
  })
}
const invoke = (turn: any, overrides = {}, customMode = mode, signal = new AbortController().signal) =>
  runChatPipeline(
    customMode,
    "question",
    "",
    false,
    [],
    [],
    signal,
    {
      selectedModel: "test",
      useOCR: false,
      setMessages: vi.fn(),
      setHistory: vi.fn(),
      setHistoryId: vi.fn(),
      setIsProcessing: vi.fn(),
      setStreaming: vi.fn(),
      setAbortController: vi.fn(),
      historyId: "chat",
      saveMessageOnSuccess: vi.fn(async () => "chat"),
      saveMessageOnError: vi.fn(async () => "chat"),
      historyTurn: turn,
      userMessageId: "user-new",
      assistantMessageId: "assistant-new",
      ...overrides
    }
  )
beforeEach(() => {
  vi.clearAllMocks()
  mocks.append.mockImplementation(async (_owner, selection, input) => ({
    version: 1,
    owner_key: "local-key",
    conversation_id: "chat",
    input_message_id: input.id,
    input_message_revision: "1",
    selection_digest: selection.selection_digest,
    messages: [{ id: input.id, revision: "1" }],
    originating_selection_revision: 1
  }))
  mocks.stream.mockImplementation(async function* () {
    yield "answer"
  })
  mocks.model.mockResolvedValue({
    saveToDb: false,
    prepareClientManagedRequest: () => ({
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "question" }
      ],
      model: "test",
      save_to_db: false,
      stream: true
    }),
    stream: mocks.stream
  })
})
describe("selected normal send admission boundary", () => {
  it("rejects a server-owned durable turn before selected-history admission", async () => {
    const result = await invoke(makeTurn(), {
      tldwTurn: { user_message_id: "12409645-7bce-4cba-b03b-bc4b0b27cc68" }
    })
    expect(result).toMatchObject({ status: "failed", errorMessage: "unsupported_history_durable_turn" })
    expect(mocks.append).not.toHaveBeenCalled()
    expect(mocks.stream).not.toHaveBeenCalled()
  })

  it("admits once before provider dispatch and carries accepted parent to settlement", async () => {
    const turn = makeTurn()
    const save = vi.fn(async () => "chat")
    mocks.stream.mockImplementation(async function* () {
      expect(mocks.append).toHaveBeenCalledTimes(1)
      expect(turn.admission.input_message_id).toBe("user-new")
      yield "answer"
    })
    expect(await invoke(turn, { saveMessageOnSuccess: save })).toEqual({
      status: "submitted"
    })
    expect(save).toHaveBeenCalledWith(
      expect.objectContaining({
        historyTurn: turn,
        assistantParentMessageId: "user-new"
      })
    )
    const request = mocks.stream.mock.lastCall?.[1]?.preparedRequest
    expect(request).toMatchObject({ save_to_db: false, model: "test" })
    expect(Object.isFrozen(request)).toBe(true)
  })
  it("selection changes during asynchronous preparation dispatch no admission or provider request", async () => {
    const turn = makeTurn()
    await invoke(
      turn,
      {},
      {
        ...mode,
        preparePrompt: async () => {
          turn.currentView = () => ({
            ...turn.capture.view,
            selection_revision: 2
          })
          return mode.preparePrompt()
        }
      }
    )
    expect(mocks.append).not.toHaveBeenCalled()
    expect(mocks.stream).not.toHaveBeenCalled()
  })
  it("a swipe after admission dispatch preserves original acceptance and still dispatches frozen request", async () => {
    const turn = makeTurn()
    const append = mocks.append.getMockImplementation()!
    mocks.append.mockImplementation(async (...args) => {
      turn.currentView = () => ({ ...turn.capture.view, selection_revision: 2 })
      turn.canUpdateView = () => false
      return append(...args)
    })
    expect(await invoke(turn)).toEqual({ status: "submitted" })
    expect(mocks.stream).toHaveBeenCalledTimes(1)
  })
  it("retains accepted unsent input when the composer lease changes while admission is in flight", async () => {
    const turn = makeTurn()
    const append = mocks.append.getMockImplementation()!
    mocks.append.mockImplementation(async (...args) => {
      turn.validateLease = () => false
      return append(...args)
    })
    const errorSave = vi.fn()
    await invoke(turn, { saveMessageOnError: errorSave })
    expect(turn.admission?.input_message_id).toBe("user-new")
    expect(mocks.stream).not.toHaveBeenCalled()
    expect(errorSave).not.toHaveBeenCalled()
    expect(turn.recover).toHaveBeenCalledTimes(1)
  })
  it("provider failure retains admission and does not invoke legacy error user persistence", async () => {
    const turn = makeTurn()
    mocks.stream.mockImplementation(async function* () {
      throw new Error("provider down")
    })
    const errorSave = vi.fn()
    await invoke(turn, { saveMessageOnError: errorSave })
    expect(mocks.append).toHaveBeenCalledTimes(1)
    expect(errorSave).not.toHaveBeenCalled()
    expect(turn.recover).toHaveBeenCalledTimes(1)
  })
  it("held provider output after navigation cannot replace the new display", async () => {
    const turn = makeTurn()
    let displayed: any[] = []
    const setter = (next: any) => {
      displayed = typeof next === "function" ? next(displayed) : next
    }
    const setHistory = vi.fn()
    mocks.stream.mockImplementation(async function* () {
      turn.canUpdateView = () => false
      displayed = [{ id: "other", message: "new view" }]
      yield "old answer"
    })
    const save = vi.fn(async () => "chat")
    await invoke(turn, {
      setMessages: setter,
      setHistory,
      saveMessageOnSuccess: save
    })
    expect(displayed).toEqual([{ id: "other", message: "new view" }])
    expect(setHistory).not.toHaveBeenCalled()
    expect(save).toHaveBeenCalledWith(
      expect.objectContaining({ fullText: "old answer", historyTurn: turn })
    )
  })
})

describe("selected durable server settlement", () => {
  const inputId = "12409645-7bce-4cba-b03b-bc4b0b27cc68"
  const resultId = "44209645-7bce-4cba-b03b-bc4b0b27cc68"
  const nativeTurn = async () => {
    const turn = makeTurn()
    turn.serverOwned = true
    turn.owner = { ...turn.owner, kind: "native", validate_lease: () => true }
    turn.beforeDispatch = vi.fn(async () => {})
    turn.afterAdmission = vi.fn(async () => {})
    turn.complete = vi.fn(async () => {})
    vi.spyOn(tldwClient, "captureHistorySelection").mockResolvedValue(turn.capture)
    turn.capture = await captureHistorySnapshot(turn.owner, turn.capture.view, "send")
    return turn
  }
  const serverModel = (turn: any, withResult = true) => {
    const model: any = {
      saveToDb: true,
      conversationId: "chat",
      userServerMessageId: inputId,
      serverMessageId: resultId,
      serverMessagesAlreadyPersisted: true,
      prepareSelectedDurableRequest: () => ({
        model: "test", api_provider: "custom", stream: true, save_to_db: true,
        conversation_id: "chat", tldw_turn: { user_message_id: inputId, result_v1: { version: 1, sources: [] } },
        messages: [{ role: "system", content: "system" }, { role: "user", content: "question" }]
      }),
      stream: vi.fn(async function* (_messages: unknown, options: any) {
        const selection = options.preparedRequest.tldw_turn.history_v1.selection
        model.historyAdmission = {
          version: 1, owner_key: "local-key", conversation_id: "chat",
          input_message_id: inputId, input_message_revision: "1",
          selection_digest: selection.selection_digest,
          messages: [{ id: inputId, revision: "1" }], originating_selection_revision: 1
        }
        yield "answer"
        if (withResult) model.historyResult = {
          version: 1, result_message_id: resultId, result_message_revision: "1",
          admission: historyAdmissionReference(model.historyAdmission),
          request_context_digest: selection.request_context_digest, sources: []
        }
      })
    }
    mocks.model.mockResolvedValue(model)
    return model
  }
  it("dispatches one nested request without client append or client settlement", async () => {
    const turn = await nativeTurn(), save = vi.fn(async () => "chat")
    const model = serverModel(turn)
    expect(await invoke(turn, { tldwTurn: { user_message_id: inputId }, userMessageId: inputId, saveMessageOnSuccess: save }))
      .toEqual({ status: "submitted" })
    expect(mocks.append).not.toHaveBeenCalled()
    expect(save).toHaveBeenCalledWith(expect.objectContaining({
      historyTurn: turn, serverMessagesAlreadyPersisted: true,
      userServerMessageId: inputId, assistantServerMessageId: resultId
    }))
    expect(model.stream.mock.lastCall[1].preparedRequest.tldw_turn.history_v1.kind).toBe("selection")
    expect(turn.resultId).toBe(resultId)
    expect(turn.complete).toHaveBeenCalledOnce()
  })
  it("keeps an accepted answer unresolved without a committed result receipt", async () => {
    const turn = await nativeTurn(), save = vi.fn()
    serverModel(turn, false)
    expect(await invoke(turn, { tldwTurn: { user_message_id: inputId }, userMessageId: inputId, saveMessageOnSuccess: save }))
      .toMatchObject({ status: "failed", errorMessage: "history_result_unverified" })
    expect(save).not.toHaveBeenCalled()
    expect(turn.complete).not.toHaveBeenCalled()
    expect(turn.observedResult).toBeUndefined()
    expect(turn.admission.input_message_id).toBe(inputId)
    expect(turn.recover).toHaveBeenCalledOnce()
  })
  it.each(["unknown", "accepted", "partial"])(
    "presents durable Stop as cancellation while preserving %s recovery",
    async (observation) => {
      const turn = await nativeTurn(), model = serverModel(turn, false)
      const controller = new AbortController(), save = vi.fn(), errorSave = vi.fn()
      model.stream.mockImplementation(async function* (_messages: unknown, options: { preparedRequest: HistoryDurableRequestBodyV1 }) {
        if (observation !== "unknown") {
          const history = options.preparedRequest.tldw_turn.history_v1
          if (history.kind !== "selection") throw new Error("expected_initial_selection")
          const selection = history.selection
          model.historyAdmission = validateHistoryDurableAdmission(
            turn.owner, history, inputId, {
              version: 1, owner_key: "local-key", conversation_id: "chat",
              input_message_id: inputId, input_message_revision: "1",
              selection_digest: selection.selection_digest,
              messages: selection.messages, originating_selection_revision: 1
            }
          )
          yield { tldw_history_admission_v1: model.historyAdmission }
        }
        if (observation === "partial") yield "partial answer"
        controller.abort()
        throw new DOMException("signal is aborted without reason", "AbortError")
      })
      const result = await invoke(turn, {
        tldwTurn: { user_message_id: inputId }, userMessageId: inputId,
        saveMessageOnSuccess: save, saveMessageOnError: errorSave
      }, mode, controller.signal)

      expect(result).toEqual({ status: "skipped", reason: "Request cancelled" })
      expect(turn.recover).toHaveBeenCalledWith(
        expect.objectContaining({ content: observation === "partial" ? "partial answer" : "" }),
        expect.objectContaining({ name: "AbortError" })
      )
      expect(turn.admission?.input_message_id).toBe(observation === "unknown" ? undefined : inputId)
      expect(turn.dispatched).toBe(true)
      expect(turn.observedResult).toBeUndefined()
      expect(turn.complete).not.toHaveBeenCalled()
      expect(save).not.toHaveBeenCalled()
      expect(errorSave).not.toHaveBeenCalled()
      expect(model.stream).toHaveBeenCalledOnce()
    }
  )
  it("preserves already projected RAG source metadata in the durable request", async () => {
    const turn = await nativeTurn(), model = serverModel(turn, false)
    const sources = [{ name: "Document", type: "pdf", mode: "rag" as const,
      url: "", pageContent: "Exact evidence", metadata: { source_type: "vector", chunk_id: "chunk-1" } }]
    model.prepareSelectedDurableRequest = vi.fn((_messages, wireSources) => ({
      model: "test", api_provider: "custom", stream: true, save_to_db: true,
      conversation_id: "chat", tldw_turn: { user_message_id: inputId, result_v1: { version: 1, sources: wireSources } },
      messages: [{ role: "system", content: "system" }, { role: "user", content: "question" }]
    }))
    await invoke(turn, { tldwTurn: { user_message_id: inputId }, userMessageId: inputId }, {
      ...mode, preparePrompt: async () => ({ ...(await mode.preparePrompt()), sources })
    })
    expect(model.prepareSelectedDurableRequest.mock.lastCall?.[1]).toEqual(sources)
    expect(model.stream.mock.lastCall?.[1].preparedRequest.tldw_turn.result_v1.sources).toEqual(sources)
  })
  // Serialized persistence and transport doubles exercise orchestration, not native UAT.
  it.each(["iterator error", "Stop", "catch-only error"])(
    "retains a validated result through %s and reload without resolving unknown siblings",
    async (failure) => {
      const turn = await nativeTurn(), model = serverModel(turn)
      const controller = new AbortController(), save = vi.fn(), errorSave = vi.fn()
      const records = new Map<string, string>()
      const write = (operationId: string, state: HistoryTurnRecovery["state"]) => {
        const record: HistoryTurnRecovery = {
          operation_id: operationId, persistence: "server", state,
          owner_key: turn.owner.owner_key, conversation_id: "chat",
          origin_view: turn.capture.view, selection_digest: turn.selection.selection_digest,
          request_context_digest: turn.requestContextDigest,
          logical_user_message_id: inputId, finalized_selection: turn.selection,
          input_text: "question", input_images: [], result_text: "", created_at: turn.createdAt,
          ...(operationId === "selected-operation" && turn.admission
            ? { admission: historyAdmissionReference(turn.admission), input_id: inputId } : {}),
          ...(operationId === "selected-operation" && turn.observedResult
            ? { assistant_id: turn.observedResult.result_message_id, observed_result: turn.observedResult } : {})
        }
        records.set(operationId, JSON.stringify(record))
      }
      const read = (operationId: string): HistoryTurnRecovery => JSON.parse(records.get(operationId)!)
      turn.beforeDispatch = async () => write("unknown-sibling", "unknown")
      const persistedObservations: HistoryTurnRecovery[] = []
      turn.afterAdmission = async () => {
        write("selected-operation", "accepted_unsent")
        persistedObservations.push(read("selected-operation"))
      }
      turn.recover = async () => write("selected-operation", "accepted_unsent")
      model.stream.mockImplementation(async function* (_messages: unknown, options: { preparedRequest: HistoryDurableRequestBodyV1 }) {
        const history = options.preparedRequest.tldw_turn.history_v1
        if (history.kind !== "selection") throw new Error("expected_initial_selection")
        model.historyAdmission = validateHistoryDurableAdmission(turn.owner, history, inputId, {
          version: 1, owner_key: "local-key", conversation_id: "chat",
          input_message_id: inputId, input_message_revision: "1",
          selection_digest: history.selection.selection_digest,
          messages: history.selection.messages, originating_selection_revision: 1
        })
        if (failure === "iterator error") yield "partial answer"
        model.historyResult = validateHistoryDurableResultReceipt(turn.owner,
          historyAdmissionReference(model.historyAdmission), turn.requestContextDigest, [], {
            version: 1, result_message_id: resultId, result_message_revision: "1",
            admission: historyAdmissionReference(model.historyAdmission),
            request_context_digest: turn.requestContextDigest, sources: []
          })
        if (failure !== "catch-only error") {
          if (failure === "Stop") controller.abort()
          yield { tldw_history_result_v1: model.historyResult }
          // The receipt must be durable before the next iterator pull can fail.
          expect(read("selected-operation").observed_result).toEqual(model.historyResult)
        }
        throw new Error(failure)
      })
      const result = await invoke(turn, {
        tldwTurn: { user_message_id: inputId }, userMessageId: inputId,
        saveMessageOnSuccess: save, saveMessageOnError: errorSave
      }, mode, controller.signal)
      expect(result).toEqual(failure === "Stop"
        ? { status: "skipped", reason: "Request cancelled" }
        : { status: "failed", errorMessage: failure })
      expect(turn.observedResult).toEqual(model.historyResult)
      const reloaded = read("selected-operation")
      expect(reloaded).toMatchObject({ assistant_id: resultId, observed_result: model.historyResult })
      if (failure !== "catch-only error")
        expect(persistedObservations.at(-1)!.observed_result).toEqual(model.historyResult)
      expect(save).not.toHaveBeenCalled()
      expect(errorSave).not.toHaveBeenCalled()
      expect(turn.complete).not.toHaveBeenCalled()

      // A fresh-document read uses only the serialized canonical ID, not the model object.
      const receipt = reloaded.observed_result!
      model.historyAdmission = undefined
      model.historyResult = undefined
      mocks.request.mockResolvedValueOnce({
        id: resultId, conversation_id: "chat", sender: "assistant", content: "answer",
        version: 1, parent_message_id: inputId,
        tldw_history_recovery_v1: { version: 1, status: "result_verified",
          scope: { scope_type: "global", workspace_id: null }, result: receipt }
      })
      expect(await inspectHistoryDurableRecovery(turn.owner, reloaded))
        .toMatchObject({ assistant_id: resultId, observed_result: receipt })
      expect(mocks.request.mock.lastCall?.[0].path).toContain(`/messages/${resultId}?`)
      const sibling = read("unknown-sibling")
      expect(sibling).toMatchObject({ state: "unknown" })
      expect(sibling.observed_result).toBeUndefined()
      mocks.request.mockResolvedValueOnce({
        id: inputId, conversation_id: "chat", sender: "user", content: "question", version: 1,
        tldw_history_recovery_v1: { version: 1, status: "input_verified",
          scope: { scope_type: "global", workspace_id: null },
          admission: { ...receipt.admission, messages: [], originating_selection_revision: 1 } }
      })
      expect((await inspectHistoryDurableRecovery(turn.owner, sibling)).assistant_id).toBeUndefined()
      expect(mocks.request).toHaveBeenCalledTimes(2)
      expect(mocks.request.mock.lastCall?.[0].path).toContain(`/messages/${inputId}?`)
    }
  )
  it("does not dispatch a bounded source projection with duplicate evidence markers", async () => {
    const turn = await nativeTurn(), model = serverModel(turn)
    const source = { name: "Memo", type: "text", mode: "rag", url: "", pageContent: "same", metadata: {} }
    const result = await invoke(turn, { tldwTurn: { user_message_id: inputId }, userMessageId: inputId }, {
      ...mode, preparePrompt: async () => ({ ...(await mode.preparePrompt()), sources: [source, source] })
    })
    expect(result.status).toBe("failed")
    expect(model.stream).not.toHaveBeenCalled()
    expect(turn.beforeDispatch).not.toHaveBeenCalled()
    expect(mocks.append).not.toHaveBeenCalled()
  })
})

it("clears owned streaming state before following a settled result changes the cursor", async () => {
  const turn = makeTurn()
  const streaming = vi.fn()
  const save = vi.fn(async () => {
    turn.resultId = "assistant-new"
    return "chat"
  })
  await invoke(turn, { setStreaming: streaming, saveMessageOnSuccess: save })
  expect(streaming).toHaveBeenCalledWith(false)
})

it("unknown outcomes live only in scoped recovery, not retryable message bubbles", async () => {
  const turn = makeTurn()
  mocks.append.mockRejectedValue(new Error("unknown write"))
  let display: any[] = []
  await invoke(turn, {
    setMessages: (next: any) => {
      display = typeof next === "function" ? next(display) : next
    }
  })
  expect(display).toEqual([])
  expect(turn.recover).toHaveBeenCalledOnce()
})

it("releases the owned activity after a swipe without changing the newly selected transcript", async () => {
  const turn = makeTurn(),
    processing = vi.fn(),
    streaming = vi.fn(),
    controller = vi.fn(),
    history = vi.fn(),
    release = vi.fn(() => true)
  mocks.stream.mockImplementation(async function* () {
    turn.canUpdateView = () => false
    turn.currentView = () => ({ ...turn.capture.view, selection_revision: 2 })
    yield "accepted answer"
  })
  await invoke(turn, {
    setIsProcessing: processing,
    setStreaming: streaming,
    setAbortController: controller,
    setHistory: history,
    releaseAbortControllerIfOwned: release
  })
  expect(release).toHaveBeenCalledOnce()
  expect(processing).toHaveBeenLastCalledWith(false)
  expect(streaming).toHaveBeenLastCalledWith(false)
  expect(controller).toHaveBeenLastCalledWith(null)
  expect(history).not.toHaveBeenCalled()
})
it("a first held token after navigation never marks the destination busy or clears a newer controller", async () => {
  const turn = makeTurn(),
    processing = vi.fn(),
    streaming = vi.fn(),
    controller = vi.fn()
  mocks.stream.mockImplementation(async function* () {
    turn.canUpdateView = () => false
    processing.mockClear()
    streaming.mockClear()
    controller.mockClear()
    yield "old answer"
  })
  await invoke(turn, {
    setIsProcessing: processing,
    setStreaming: streaming,
    setAbortController: controller,
    releaseAbortControllerIfOwned: () => false
  })
  expect(processing).not.toHaveBeenCalled()
  expect(streaming).not.toHaveBeenCalled()
  expect(controller).not.toHaveBeenCalled()
})

it("an old first chunk cannot restart activity after a newer same-view operation completed", async () => {
  const processing = vi.fn(),
    streaming = vi.fn(),
    controller = vi.fn()
  await invoke(makeTurn(), {
    setIsProcessing: processing,
    setStreaming: streaming,
    setAbortController: controller,
    ownsAbortController: () => false,
    releaseAbortControllerIfOwned: () => false
  })
  expect(processing).not.toHaveBeenCalled()
  expect(streaming).not.toHaveBeenCalled()
  expect(controller).not.toHaveBeenCalled()
})
