import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { HistoryTurnRecoveryEntry } from "../history-turn-keep"

const mocks = vi.hoisted(() => ({
  load: vi.fn(),
  dismiss: vi.fn(),
  settle: vi.fn()
}))
vi.mock("@/db/dexie/history-selection", async original => ({
  ...(await original<typeof import("@/db/dexie/history-selection")>()),
  loadHistoryTurnRecoveries: mocks.load,
  dismissHistoryTurnRecovery: mocks.dismiss
}))
vi.mock("@/services/chat-history-selection", async original => ({
  ...(await original<typeof import("@/services/chat-history-selection")>()),
  settleAcceptedAssistant: mocks.settle
}))
import { keepHistoryTurnRecovery, resetHistoryTurnRegistry } from "../history-turn-keep"

const fixture = () => {
  const view = {
    owner_key: "owner",
    conversation_id: "chat",
    view_session_id: "view",
    selection_revision: 1,
    cursor: { kind: "empty" as const },
    interpretation: { kind: "parent_graph_v1" as const }
  }
  const scope = { profile_id: "profile", client_session_id: "session" }
  const entry: HistoryTurnRecoveryEntry = {
    scope,
    turn: {
      operation_id: "operation",
      input_id: "question",
      input_text: "Question",
      input_images: [],
      state: "generated_unsaved",
      selection_digest: "selection",
      request_context_digest: "request",
      owner_key: "owner",
      conversation_id: "chat",
      origin_view: view,
      assistant_id: "answer",
      result_text: "Completed reply",
      created_at: 1,
      outcome: "complete",
      admission: {
        version: 1,
        owner_key: "owner",
        conversation_id: "chat",
        input_message_id: "question",
        input_message_revision: "1",
        selection_digest: "selection",
        messages: [{ id: "question", revision: "1" }],
        originating_selection_revision: 1
      }
    }
  }
  const current = {
    status: "ready",
    owner: { kind: "local", profile_id: "profile", owner_key: "owner", conversation_id: "chat" },
    view,
    bookmarkScope: scope
  }
  const controller = {
    getCurrent: () => current,
    fence: () => () => true,
    choose: vi.fn(async () => true),
    refreshRecovery: vi.fn(async () => {})
  } as unknown as HistorySelectionController
  mocks.load.mockResolvedValue([entry])
  return { controller, entry, current }
}

beforeEach(() => {
  vi.resetAllMocks()
  resetHistoryTurnRegistry()
  mocks.settle.mockResolvedValue(undefined)
  mocks.dismiss.mockResolvedValue(undefined)
})

describe("completed retained reply settlement", () => {
  it("reports a persistent settlement failure instead of a harmless skip", async () => {
    const { controller, entry } = fixture()
    mocks.settle.mockRejectedValue(new Error("stale_parent"))
    expect(await keepHistoryTurnRecovery(controller, entry)).toEqual({ status: "failed", reason: "stale_parent" })
    expect(mocks.dismiss).not.toHaveBeenCalled()
    expect(controller.choose).not.toHaveBeenCalled()
  })

  it("does not acknowledge or erase a conflicting assistant message", async () => {
    const { controller, entry } = fixture()
    mocks.settle.mockRejectedValue(new Error("message_id_conflict"))
    expect(await keepHistoryTurnRecovery(controller, entry)).toEqual({ status: "failed", reason: "message_id_conflict" })
    expect(mocks.dismiss).not.toHaveBeenCalled()
    expect(controller.choose).not.toHaveBeenCalled()
  })

  it("reports a failed recovery read without deleting the retained reply", async () => {
    const { controller, entry } = fixture()
    mocks.load.mockRejectedValue(new Error("storage_unavailable"))
    expect(await keepHistoryTurnRecovery(controller, entry)).toEqual({ status: "failed", reason: "storage_unavailable" })
    expect(mocks.dismiss).not.toHaveBeenCalled()
  })

  it("reports kept only after settling and clearing the completed record", async () => {
    const { controller, entry } = fixture()
    expect(await keepHistoryTurnRecovery(controller, entry)).toEqual({ status: "kept", resultId: "answer", followed: true, retained: false })
    expect(mocks.settle).toHaveBeenCalledOnce()
    expect(mocks.dismiss).toHaveBeenCalledOnce()
  })

  it("keeps a readiness refusal as skipped without attempting persistence", async () => {
    const { controller, entry, current } = fixture()
    current.status = "loading"
    expect(await keepHistoryTurnRecovery(controller, entry)).toEqual({ status: "skipped", reason: "not_ready" })
    expect(mocks.settle).not.toHaveBeenCalled()
  })
})
