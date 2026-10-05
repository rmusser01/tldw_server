import { describe, expect, it } from "vitest"

import type { Message } from "@/store/option"
import { latestHistoryTipId } from "@/utils/history-selection"
import {
  compareWithServerTurns,
  hasUnsentLocalMessages,
  shownServerMessageIds
} from "../sidepanel-chat-freshness"

/** q1 → a1 → q2 → a2, with a1b as a regenerated variant of a1. */
const nodes = [
  { id: "q1", parent_id: null },
  { id: "a1", parent_id: "q1" },
  { id: "a1b", parent_id: "q1" },
  { id: "q2", parent_id: "a1" },
  { id: "a2", parent_id: "q2" }
]
const message = (id: string, extra: Partial<Message> = {}): Message => ({
  id,
  isBot: false,
  name: "You",
  message: id,
  sources: [],
  ...extra
})

describe("latestHistoryTipId", () => {
  it("is the last message nothing replies to", () => {
    expect(latestHistoryTipId(nodes)).toBe("a2")
    expect(latestHistoryTipId(nodes.slice(0, 3))).toBe("a1b")
    expect(latestHistoryTipId([])).toBeNull()
  })
})

describe("compareWithServerTurns", () => {
  it("counts the newer turns a tab that stopped at a1 doesn't show", () => {
    expect(
      compareWithServerTurns({
        nodes,
        cursorId: "a1",
        shownIds: new Set(["q1", "a1", "a1b"]),
        latestSeenId: "a1b"
      })
    ).toEqual({ status: "behind", latestId: "a2", newCount: 2 })
  })

  it("treats a tab at the latest message as current", () => {
    expect(
      compareWithServerTurns({
        nodes,
        cursorId: "a2",
        shownIds: new Set(["q1", "a1", "q2", "a2"]),
        latestSeenId: null
      })
    ).toEqual({ status: "current", latestId: "a2" })
  })

  it("keeps a position chosen on purpose when nothing arrived since", () => {
    expect(
      compareWithServerTurns({
        nodes,
        cursorId: "q1",
        shownIds: new Set(["q1"]),
        latestSeenId: "a2"
      })
    ).toEqual({ status: "current", latestId: "a2" })
  })

  it("treats the latest message shown as a variant as current", () => {
    expect(
      compareWithServerTurns({
        nodes: nodes.slice(0, 3),
        cursorId: "a1",
        shownIds: new Set(["q1", "a1", "a1b"]),
        latestSeenId: null
      })
    ).toEqual({ status: "current", latestId: "a1b" })
  })
})

describe("shownServerMessageIds", () => {
  it("includes server ids and assistant variants", () => {
    const ids = shownServerMessageIds([
      message("local-q", { serverMessageId: "q1" }),
      message("a1", { isBot: true, variants: [{ id: "a1", message: "" }, { id: "a1b", message: "" }] })
    ])
    expect([...ids].sort()).toEqual(["a1", "a1b", "local-q", "q1"])
  })
})

describe("hasUnsentLocalMessages", () => {
  it("is true only for messages the server doesn't have", () => {
    expect(hasUnsentLocalMessages([message("q1"), message("a1")], nodes)).toBe(false)
    expect(hasUnsentLocalMessages([message("tmp", { serverMessageId: "a1" })], nodes)).toBe(false)
    expect(hasUnsentLocalMessages([message("q1"), message("error-reply")], nodes)).toBe(true)
  })
})
