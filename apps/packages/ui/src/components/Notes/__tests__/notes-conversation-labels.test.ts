import { describe, expect, it } from "vitest"
import { collectConversationIdsToResolve, settleInBatches } from "../notes-conversation-labels"

describe("collectConversationIdsToResolve (NL-04)", () => {
  it("keeps every visible chat id, UUIDs included, once, plus the open note's", () => {
    const uuid = "acd86f3a-492e-4a8b-90fb-ff282b9721fb"
    expect(
      collectConversationIdsToResolve(
        [{ conversation_id: uuid }, { conversation_id: "conv-2" }, { conversation_id: uuid }, {}, null],
        "conv-open"
      )
    ).toEqual([uuid, "conv-2", "conv-open"])
  })
})

describe("settleInBatches", () => {
  it("runs at most batchSize workers at once and keeps the input order", async () => {
    let inFlight = 0
    let peak = 0
    const results = await settleInBatches(
      [1, 2, 3, 4, 5],
      async (value) => {
        inFlight += 1
        peak = Math.max(peak, inFlight)
        await new Promise((resolve) => setTimeout(resolve, 1))
        inFlight -= 1
        if (value === 3) throw new Error("three")
        return value * 10
      },
      2
    )
    expect(peak).toBe(2)
    expect(results.map((result) => (result.status === "fulfilled" ? result.value : "rejected"))).toEqual([
      10,
      20,
      "rejected",
      40,
      50
    ])
  })
})
