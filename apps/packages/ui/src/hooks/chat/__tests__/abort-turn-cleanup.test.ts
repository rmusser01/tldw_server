import { describe, expect, it, vi } from "vitest"

import type { ChatHistory, Message } from "@/store/option"
import {
  beginChatTurnSettlement,
  discardAbortedTurnIfRequested,
  followStoppedChatTurnSelection,
  markChatTurnStoppedByUser
} from "../abort-turn-cleanup"
import type { HistorySelectionController } from "../useHistorySelection"

const createAbortError = () => {
  const error = new Error("AbortError")
  error.name = "AbortError"
  return error
}

describe("discardAbortedTurnIfRequested", () => {
  it("restores the previous transcript snapshots when discard is requested for an abort", () => {
    const previousMessages: Message[] = [
      {
        id: "user-1",
        isBot: false,
        name: "You",
        message: "Existing turn",
        sources: []
      }
    ]
    const previousHistory: ChatHistory = [
      {
        role: "user",
        content: "Existing turn"
      }
    ]
    const setMessages = vi.fn()
    const setHistory = vi.fn()

    const handled = discardAbortedTurnIfRequested({
      discardRequested: true,
      error: createAbortError(),
      previousMessages,
      previousHistory,
      setMessages,
      setHistory
    })

    expect(handled).toBe(true)
    expect(setMessages).toHaveBeenCalledWith(previousMessages)
    expect(setHistory).toHaveBeenCalledWith(previousHistory)
  })

  it("does not restore snapshots when discard was not requested", () => {
    const setMessages = vi.fn()
    const setHistory = vi.fn()

    const handled = discardAbortedTurnIfRequested({
      discardRequested: false,
      error: createAbortError(),
      previousMessages: [],
      previousHistory: [],
      setMessages,
      setHistory
    })

    expect(handled).toBe(false)
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  })
})

// Only H1 I/O/epoch publication is doubled; settlement and Stop lease are real.
const stoppedSelection = () => {
  const signal = new AbortController().signal
  const settlement = beginChatTurnSettlement(signal)
  markChatTurnStoppedByUser(signal)
  let epoch = 1,
    sourceCurrent = true
  const owner = { kind: "local", validate_lease: () => true }
  let state = {
    owner,
    view: {
      owner_key: "owner",
      conversation_id: "chat",
      view_session_id: "view-1",
      selection_revision: 1,
      cursor: { kind: "end" },
      interpretation: { kind: "parent_graph_v1" }
    },
    capture: {},
    status: "ready"
  }
  let release!: (accepted: boolean) => void
  const capture = new Promise<boolean>((resolve) => {
    release = resolve
  })
  const controller = {
    getCurrent: () => state,
    fence: () => {
      const original = epoch
      return () => epoch === original
    },
    choose: vi.fn(async (cursor) => {
      epoch++
      state = {
        ...state,
        status: "loading",
        view: {
          ...state.view,
          cursor,
          selection_revision: state.view.selection_revision + 1
        }
      }
      const accepted = await capture
      state = { ...state, status: accepted ? "ready" : "error" }
      return accepted
    })
  } as unknown as HistorySelectionController
  let historyCurrent = controller.fence()
  const current = () => sourceCurrent && historyCurrent()
  const focus = vi.fn()
  const acceptHistory = vi.fn((guard: () => boolean) => {
    historyCurrent = guard
  })
  settlement.focusReturn = {
    history: controller,
    isCurrent: current,
    acceptHistory,
    finish: () => {
      if (current()) focus()
    }
  }
  return {
    signal,
    settlement,
    controller,
    current,
    focus,
    acceptHistory,
    release,
    retire: () => {
      epoch++
    },
    replaceOwner: () => {
      state = { ...state, owner: { ...owner } }
    },
    replaceSource: () => {
      sourceCurrent = false
    }
  }
}

describe("stopped turn history installation", () => {
  it("keeps its own pending installation current but returns focus only after full settlement", async () => {
    const probe = stoppedSelection()
    const following = followStoppedChatTurnSelection(
      probe.signal,
      probe.controller,
      { kind: "after_message", message_id: "kept-question" }
    )
    expect(probe.current()).toBe(true)
    expect(probe.settlement.pending).toBe(true)
    expect(probe.focus).not.toHaveBeenCalled()
    probe.release(true)
    await following
    expect(probe.current()).toBe(true)
    expect(probe.focus).not.toHaveBeenCalled()
    probe.settlement.finish()
    expect(probe.focus).toHaveBeenCalledOnce()
    expect(probe.settlement.pending).toBe(false)
    probe.settlement.finish()
    expect(probe.focus).toHaveBeenCalledOnce()
  })

  it.each([
    "H1 same-ID restart",
    "owner",
    "source",
    "refused",
    "replacement response",
    "held refresh"
  ])("does not authorize focus after %s", async (boundary) => {
    const probe = stoppedSelection()
    const following = followStoppedChatTurnSelection(
      probe.signal,
      probe.controller,
      { kind: "after_message", message_id: "kept-question" }
    )
    if (boundary === "H1 same-ID restart") probe.retire()
    else if (boundary === "owner") probe.replaceOwner()
    else if (boundary === "source") probe.replaceSource()
    else if (boundary === "replacement response")
      beginChatTurnSettlement(probe.signal)
    probe.release(boundary !== "refused")
    await following
    if (boundary === "held refresh") probe.retire()
    probe.settlement.finish()
    expect(probe.focus).not.toHaveBeenCalled()
  })

  it("does not adopt a synchronous competing selection published by choose", async () => {
    const probe = stoppedSelection()
    const choose = probe.controller.choose
    probe.controller.choose = async (cursor) => {
      const following = choose(cursor)
      probe.retire()
      const state = probe.controller.getCurrent()
      state.view!.selection_revision++
      return following
    }
    const following = followStoppedChatTurnSelection(
      probe.signal,
      probe.controller,
      { kind: "after_message", message_id: "kept-question" }
    )
    probe.release(true)
    await following
    probe.settlement.finish()
    expect(probe.acceptHistory).not.toHaveBeenCalled()
    expect(probe.focus).not.toHaveBeenCalled()
  })

  it("never adopts a history change after the original Stop lease retired", async () => {
    const probe = stoppedSelection()
    probe.retire()
    const following = followStoppedChatTurnSelection(
      probe.signal,
      probe.controller,
      { kind: "after_message", message_id: "kept-question" }
    )
    probe.release(true)
    await following
    probe.settlement.finish()
    expect(probe.acceptHistory).not.toHaveBeenCalled()
    expect(probe.focus).not.toHaveBeenCalled()
  })
})
