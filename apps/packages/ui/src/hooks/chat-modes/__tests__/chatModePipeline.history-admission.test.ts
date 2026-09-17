import { beforeEach, describe, expect, it, vi } from "vitest"
import { HumanMessage, SystemMessage } from "@/types/messages"
const mocks = vi.hoisted(() => ({
  append: vi.fn(),
  settle: vi.fn(),
  stream: vi.fn(),
  model: vi.fn()
}))
vi.mock("@/models", () => ({ pageAssistModel: mocks.model }))
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
const invoke = (turn: any, overrides = {}, customMode = mode) =>
  runChatPipeline(
    customMode,
    "question",
    "",
    false,
    [],
    [],
    new AbortController().signal,
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
