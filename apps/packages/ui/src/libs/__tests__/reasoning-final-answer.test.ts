import { describe, expect, it } from "vitest"
import { isReasoningOnlyResponse } from "../reasoning"

describe("settled reasoning without a final answer", () => {
  it.each([
    "<think>Thinking</think>",
    "<think>Thinking",
    "<reason>Thinking</reason><thought>More thinking</thought>",
    "<THINK></THINK>"
  ])("requires recovery for %s", content => {
    expect(isReasoningOnlyResponse(content)).toBe(true)
  })

  it.each([
    "<think>Thinking</think>Final answer",
    "Final answer<think>Thinking</think>",
    "Ordinary prose",
    "<thinker>Ordinary prose</thinker>",
    ""
  ])("leaves normal or unclassified content alone: %s", content => {
    expect(isReasoningOnlyResponse(content)).toBe(false)
  })
})
