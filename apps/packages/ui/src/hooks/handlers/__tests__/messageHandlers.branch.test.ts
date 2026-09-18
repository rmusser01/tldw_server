import { beforeEach, describe, expect, it, vi } from "vitest"
const forks = vi.hoisted(() => ({ prepare: vi.fn(), commit: vi.fn() }))
const nativeForks = vi.hoisted(() => ({prepare: vi.fn(), commit: vi.fn()}))
vi.mock("@/services/chat-history-selection", () => ({prepareNativeFork: nativeForks.prepare, commitNativeFork: nativeForks.commit}))
vi.mock("@/db/dexie/fork-operations", () => ({
  prepareForkOperation: async (request: any, context: any) => ({...request, request, context, state: "prepared"}),
  claimForkOperation: async (record: any) => ({record}),
  recordForkCandidate: vi.fn(async () => {}), finishForkOperation: vi.fn(async () => {}), forkOperationResult: (record: any) => record.result
}))
const edits = vi.hoisted(() => ({ update: vi.fn() }))
vi.mock("@/db/dexie/branch", () => ({
  prepareLocalFork: forks.prepare,
  commitLocalFork: forks.commit
}))
vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: (rows: any[]) =>
    rows.map((row) => ({ content: row.content })),
  formatToMessage: (rows: any[]) =>
    rows.map((row) => ({ id: row.id, message: row.content })),
  updateMessageById: edits.update
}))
import {
  createBranchMessage,
  createEditMessage,
  createRegenerateLastMessage
} from "../messageHandlers"
const request: any = {
  operation_id: "op",
  owner_key: "local",
  destination_owner_key: "local",
  request_digest: "digest",
  input: { kind: "normal", selection: { conversation_id: "source" } }
}
const committed = {
  state: "committed",
  operation_id: "op",
  owner_key: "local",
  child_id: "child",
  message_map: { u: "child-u" }
}
const testController = () => {
  const state: any = { status: "ready" }
  return {
    getCurrent: () => state,
    loadConversation: vi.fn(async ({ historyId }, _ref, loaded) => {
      state.owner = {
        kind: "local",
        owner_key: "local",
        conversation_id: historyId
      }
      state.view = { owner_key: "local", conversation_id: historyId }
      loaded({ owner: state.owner, view: state.view })
      return true
    })
  } as any
}
const setup = (extra = {}) => ({
  historySelection: testController(),
  notification: { error: vi.fn(), warning: vi.fn() } as any,
  historyId: "source",
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setHistoryId: vi.fn(),
  setContext: vi.fn(),
  setSystemPrompt: vi.fn(),
  ...extra
})
beforeEach(() => {
  vi.clearAllMocks()
  forks.prepare.mockResolvedValue({
    history: { id: "child", last_used_prompt: { prompt_content: "literal" } },
    messages: [{ id: "child-u", content: "hello" }],
    files: { files: [] }
  })
  forks.commit.mockResolvedValue(committed)
})
describe("typed fork handler", () => {
  it("returns committed owner binding and child map after one copy", async () => {
    const options = setup()
    expect(await createBranchMessage(options)(request)).toEqual(committed)
    expect(options.setHistoryId).toHaveBeenCalledWith("child")
    expect(options.setSystemPrompt).toHaveBeenCalledWith("literal")
    expect(forks.commit).toHaveBeenCalledTimes(1)
  })
  it("retains a committed result after navigation without replacing the new view", async () => {
    let current = true
    let release!: (result: any) => void
    forks.commit.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve
        })
    )
    const options = setup({ captureViewFence: () => () => current })
    const pending = createBranchMessage(options)(request)
    await vi.waitFor(() => expect(forks.commit).toHaveBeenCalledTimes(1))
    current = false
    release(committed)
    expect(await pending).toEqual(committed)
    expect(options.setHistoryId).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
  })
  it.each(["partial", "unknown", "rejected"])(
    "preserves %s without navigation or fallback",
    async (state) => {
      const result = {
        state,
        operation_id: "op",
        owner_key: "local",
        code: "failed",
        candidate_child_id: "candidate"
      }
      forks.commit.mockResolvedValue(result)
      const options = setup()
      expect(await createBranchMessage(options)(request)).toEqual(result)
      expect(options.setHistoryId).not.toHaveBeenCalled()
      expect(forks.prepare).toHaveBeenCalledTimes(1)
    }
  )
  it("blocks unverified native replay before any copy", async () => {
    const options = setup({ serverChatId: "native" })
    expect(await createBranchMessage(options)(request)).toEqual({
      state: "blocked",
      operation_id: "op",
      owner_key: "local",
      code: "native_fork_projection_unavailable"
    })
    expect(forks.prepare).not.toHaveBeenCalled()
  })
  it("reports preflight rejection without a snapshot fallback", async () => {
    forks.prepare.mockRejectedValue(new Error("unsupported_asset"))
    const options = setup()
    expect(await createBranchMessage(options)(request)).toMatchObject({
      state: "rejected",
      code: "unsupported_asset",
      owner_key: "local",
      operation_id: "op"
    })
    expect(forks.commit).not.toHaveBeenCalled()
  })
})
it("edit uses the stable selected ID despite greeting offset and updates display only on success", async () => {
  const messages: any = [
    { id: "greeting", message: "hi", isBot: true },
    { id: "selected", message: "same", isBot: true }
  ]
  const options: any = {
    ...setup(),
    messages,
    history: [],
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn()
  }
  await createEditMessage(options)(1, "edited", false, false)
  expect(edits.update).toHaveBeenCalledWith("source", "selected", "edited")
  expect(options.setMessages).toHaveBeenCalledWith([
    messages[0],
    { ...messages[1], message: "edited" }
  ])
  edits.update.mockRejectedValue(new Error("message_owner_mismatch"))
  options.setMessages.mockClear()
  await expect(
    createEditMessage(options)(1, "bad", false, false)
  ).rejects.toThrow("message_owner_mismatch")
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("regenerate gate leaves owner inputs and visible user/assistant untouched", async () => {
  const options: any = {
    ...setup(),
    messages: [
      { id: "u", message: "input", isBot: false },
      { id: "a", message: "answer", isBot: true }
    ],
    history: [{ role: "user", content: "input" }],
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn()
  }
  await expect(createRegenerateLastMessage(options)()).rejects.toThrow(
    "unsupported_history_regeneration"
  )
  expect(options.onSubmit).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("view application failure cannot relabel an acknowledged commit as rejected", async () => {
  const options = setup()
  options.setHistory.mockImplementation(() => {
    throw new Error("view unmounted")
  })
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(forks.commit).toHaveBeenCalledTimes(1)
})

it("waits for an exact child load receipt before displaying a committed child", async () => {
  const state: any = {
    status: "ready",
    owner: { kind: "local", owner_key: "local", conversation_id: "source" },
    view: {}
  }
  const controller = {
    getCurrent: () => state,
    fence: () => () => true,
    loadConversation: vi.fn(async (_target, _reference, loaded) => {
      state.owner = {
        kind: "local",
        owner_key: "local",
        conversation_id: "child"
      }
      state.view = { owner_key: "local", conversation_id: "child" }
      loaded({ owner: state.owner, view: state.view })
      return true
    })
  }
  const options = setup({ historySelection: controller })
  await createBranchMessage(options)(request)
  expect(controller.loadConversation).toHaveBeenCalledWith(
    { historyId: "child" },
    undefined,
    expect.any(Function)
  )
  expect(options.setHistoryId).toHaveBeenCalledWith("child")
})
it("a settled load without a receipt never presents the committed child", async () => {
  const options = setup({
    historySelection: { loadConversation: vi.fn(async () => true) }
  })
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.setHistoryId).not.toHaveBeenCalled()
})

it("a throwing child loader reports saved copy without calling it a fork failure", async () => {
  const options = setup()
  options.historySelection.loadConversation.mockRejectedValue(new Error("opening failed"))
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.notification.error).not.toHaveBeenCalled()
  expect(options.notification.warning).toHaveBeenCalledWith(expect.objectContaining({message: "Copy saved; opening failed"}))
})
it.each(["unknown", "partial"])("retains native %s after deferred response without local fallback", async state => {
  const native = {kind: "native", owner_key: "local", conversation_id: "source", scope: {type: "workspace", workspaceId: "original"}}
  let resolve!: (result: any) => void
  nativeForks.prepare.mockResolvedValue({request})
  nativeForks.commit.mockImplementation(() => new Promise(done => { resolve = done }))
  const options = setup({serverChatId: "source", historyId: null, historySelection: {getCurrent: () => ({owner: native}), refreshForkOperations: vi.fn(), loadConversation: vi.fn()}})
  const pending = createBranchMessage(options)(request)
  await vi.waitFor(() => expect(nativeForks.commit).toHaveBeenCalledTimes(1))
  resolve({state, owner_key: "local", operation_id: "op", code: "response_lost", candidate_child_id: state === "partial" ? "child" : undefined})
  expect(await pending).toMatchObject({state, operation_id: "op"})
  expect(forks.prepare).not.toHaveBeenCalled()
  expect(forks.commit).not.toHaveBeenCalled()
  expect(options.historySelection.loadConversation).not.toHaveBeenCalled()
})
