import { beforeEach, expect, it, vi } from "vitest"

const storage = vi.hoisted(() => ({ save: vi.fn(), dismiss: vi.fn(), keep: vi.fn(), pipeline: vi.fn() }))
vi.mock("@/db/dexie/history-selection", async original => ({
  ...await original<typeof import("@/db/dexie/history-selection")>(),
  saveHistoryTurnRecovery: storage.save,
  dismissHistoryTurnRecovery: storage.dismiss
}))
vi.mock("@/services/chat-history-selection", async original => ({
  ...await original<typeof import("@/services/chat-history-selection")>(),
  captureHistorySnapshot: async (_owner: unknown, view: unknown) => ({ status: "captured", view, rows: [], selected_content: [], snapshot: { owner_key: "owner-1", nodes: [] } })
}))
vi.mock("../chatModePipeline", async original => ({
  ...await original<typeof import("../chatModePipeline")>(),
  runChatPipeline: storage.pipeline
}))
vi.mock("@/services/history-turn-keep", async original => ({
  ...await original<typeof import("@/services/history-turn-keep")>(),
  keepHistoryTurnRecovery: storage.keep
}))
vi.mock("@/services/tldw/server-capabilities", async original => ({
  ...await original<typeof import("@/services/tldw/server-capabilities")>(),
  getSelectedDurableTurnSupport: async () => true
}))

import { captureNormalHistoryTurn, normalChatMode } from "../normalChatMode"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { getServerChatSaveStatus, resetServerChatSaveStatus, serverChatSaveStatusStore } from "@/store/server-chat-save-status"

beforeEach(() => {
  vi.clearAllMocks()
  resetServerChatSaveStatus()
})

const captureTurn = async (serverOwned = true) => {
  const owner = { kind: "native", conversation_id: "chat-1", validate_lease: () => true }
  const view = { owner_key: "owner-1", conversation_id: "chat-1", view_session_id: "view-1", selection_revision: 1 }
  const controller = {
    getCurrent: () => ({ status: "ready", owner, view, bookmarkScope: { profile_id: "profile-1", client_session_id: "session-1" }, capture: { status: "captured" } }),
    fence: () => () => true,
    recoveries: []
  } as unknown as HistorySelectionController
  const snapshot = {
    requestScope: { config: { serverUrl: "https://example.test", authMode: "single-user" }, userId: 1 },
    scopeInvalidatedSignal: new AbortController().signal
  } as ServicePromptSnapshot
  const turn = await captureNormalHistoryTurn({
    historySelection: { controller, originIsCurrent: () => true },
    historyId: null, serverChatId: "chat-1", setHistoryId: () => {}, selectedModel: "test", toolChoice: "none",
    ...(serverOwned ? { tldwTurn: { user_message_id: "00000000-0000-4000-8000-000000000001" } } : {})
  }, "Question", snapshot, new AbortController().signal)
  Object.assign(turn, {
    input: { id: "input-1", content: "Question", images: [] },
    selection: { selection_digest: "selection", request_context_digest: "request" },
    admission: { version: 1, owner_key: "owner-1", conversation_id: "chat-1", input_message_id: "input-1", input_message_revision: "1", selection_digest: "selection" },
    createdAt: 1, assistantId: "assistant-1"
  })
  return turn
}

it.each(["failed", "skipped", "kept"])("only reports a complete ordinary reply submitted when Keep is kept (%s)", async (status) => {
  const view = { owner_key: "owner-1", conversation_id: "chat-1", view_session_id: "view-1", selection_revision: 1 }
  const controller = {
    getCurrent: () => ({ status: "ready", owner: { kind: "native", conversation_id: "chat-1", validate_lease: () => true }, view, bookmarkScope: {}, capture: { status: "captured" } }),
    fence: () => () => true, recoveries: [], refreshRecovery: vi.fn()
  } as unknown as HistorySelectionController
  storage.keep.mockResolvedValue(status === "kept" ? { status, retained: false, followed: true, resultId: "reply-1" } : { status, reason: "stale_parent" })
  storage.pipeline.mockImplementation(async (...args: Parameters<typeof import("../chatModePipeline").runChatPipeline>) => {
    const turn = args[7].historyTurn!
    Object.assign(turn, { outcome: "complete", input: { id: "input-1", content: "Question", images: [] }, assistantId: "reply-1", selection: { selection_digest: "selection", request_context_digest: "request" } })
    await turn.recover({ content: "Completed answer", assistantId: "reply-1", createdAt: 1, outcome: "complete" }, new Error("stale_parent"))
    return { status: "submitted" }
  })
  const result = await normalChatMode("Question", "", false, [], [], new AbortController().signal, {
    selectedModel: "test", toolChoice: "none", selectedSystemPrompt: "", useOCR: false,
    historyId: null, serverChatId: "chat-1", setHistoryId: () => {},
    historySelection: { controller, originIsCurrent: () => true },
    servicePromptSnapshot: {
      definitions: {}, requestScope: { config: { serverUrl: "https://example.test", authMode: "single-user" }, userId: 1 },
      scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal
    } as ServicePromptSnapshot
  } as Parameters<typeof normalChatMode>[6])
  expect(storage.keep).toHaveBeenCalledOnce()
  expect(result.status).toBe(status === "kept" ? "submitted" : "failed")
})

it("balances split admission/result notifications with one server-write handle", async () => {
  const turn = await captureTurn()
  try {
    await turn.afterAdmission!()
    await turn.afterAdmission!()
    expect(serverChatSaveStatusStore.getState().entries["chat-1"].inFlight).toBe(1)
    await turn.complete!()
    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  } finally {
    turn.release!()
  }
  expect(serverChatSaveStatusStore.getState().entries["chat-1"].inFlight).toBe(0)
})

it("does not give a server-owned partial checkpoint an ordinary interrupted outcome", async () => {
  const turn = await captureTurn()
  try {
    turn.checkpoint!("Partial reply")
    await vi.waitFor(() => expect(storage.save).toHaveBeenCalled())
    expect(storage.save.mock.calls.at(-1)![2]).toMatchObject({ persistence: "server", result_text: "Partial reply", state: "generated_unsaved" })
    expect(storage.save.mock.calls.at(-1)![2]).not.toHaveProperty("outcome")
  } finally {
    turn.release!()
  }
})

it("keeps the ordinary partial checkpoint available as an interrupted outcome", async () => {
  const turn = await captureTurn(false)
  try {
    turn.checkpoint!("Partial reply")
    await vi.waitFor(() => expect(storage.save).toHaveBeenCalled())
    expect(storage.save.mock.calls.at(-1)![2]).toMatchObject({ result_text: "Partial reply", outcome: "interrupted" })
  } finally {
    turn.release!()
  }
})
