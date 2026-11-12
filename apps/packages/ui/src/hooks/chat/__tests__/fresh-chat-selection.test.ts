import { describe, expect, it, vi } from "vitest"

import {
  addressesNoConversation,
  isFreshChatSelectionPending,
  resetStaleSelectionForFreshTurn
} from "../fresh-chat-selection"
import type { HistorySelectionController } from "../useHistorySelection"

type Status = HistorySelectionController["status"]

const controller = (status: Status) => {
  const state: { status: Status } = { status }
  return {
    getCurrent: () => state,
    reset: vi.fn(() => {
      state.status = "idle"
    })
  }
}

describe("fresh-chat history selection (CS-01, #3106)", () => {
  it("treats no ids and the temporary-chat sentinel as addressing no conversation", () => {
    expect(addressesNoConversation({ historyId: null, serverChatId: null })).toBe(true)
    expect(addressesNoConversation({ historyId: "temp", serverChatId: null })).toBe(true)
    expect(addressesNoConversation({ historyId: "local-1" })).toBe(false)
    expect(addressesNoConversation({ serverChatId: "chat-1" })).toBe(false)
  })

  it("keeps a fresh chat pending until its controller is idle", () => {
    const fresh = { historyId: null, serverChatId: null }
    expect(isFreshChatSelectionPending({ status: "idle" }, fresh)).toBe(false)
    expect(isFreshChatSelectionPending({ status: "loading" }, fresh)).toBe(true)
    expect(isFreshChatSelectionPending({ status: "ready" }, fresh)).toBe(true)
    expect(isFreshChatSelectionPending({ status: "ready" }, { historyId: "local-1" })).toBe(false)
    expect(isFreshChatSelectionPending(null, fresh)).toBe(false)
  })

  it("resets a settled stale selection before a fresh turn", () => {
    const selection = controller("ready")
    expect(resetStaleSelectionForFreshTurn(selection, { historyId: null, serverChatId: null })).toBe(true)
    expect(selection.reset).toHaveBeenCalledOnce()
    expect(selection.getCurrent().status).toBe("idle")
  })

  it("leaves a loading selection and an addressed conversation alone", () => {
    const loading = controller("loading")
    expect(resetStaleSelectionForFreshTurn(loading, { historyId: null })).toBe(false)
    expect(loading.reset).not.toHaveBeenCalled()

    const addressed = controller("ready")
    expect(resetStaleSelectionForFreshTurn(addressed, { serverChatId: "chat-1" })).toBe(false)
    expect(addressed.reset).not.toHaveBeenCalled()

    expect(resetStaleSelectionForFreshTurn(null, { historyId: null })).toBe(false)
  })
})
