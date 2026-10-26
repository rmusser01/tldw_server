import { describe, expect, it } from "vitest"
import {
  isOutputLimitFinishReason,
  isOutputLimitedReply,
  markOutputLimitedReply,
  readChunkFinishReason
} from "@/utils/output-limit-reply"

describe("output-limit replies (CM-03)", () => {
  it("reads the finish reason from a stream chunk", () => {
    expect(readChunkFinishReason({ choices: [{ delta: {}, finish_reason: "length" }] })).toBe("length")
    expect(readChunkFinishReason({ choices: [{ delta: {}, finish_reason: " Stop " }] })).toBe("stop")
    expect(readChunkFinishReason({ choices: [{ delta: { content: "a" }, finish_reason: null }] })).toBeNull()
    expect(readChunkFinishReason({ event: "stream_transport_interrupted" })).toBeNull()
    expect(readChunkFinishReason("token")).toBeNull()
  })

  it("only treats 'length' as the output limit", () => {
    expect(isOutputLimitFinishReason("length")).toBe(true)
    expect(isOutputLimitFinishReason("stop")).toBe(false)
    expect(isOutputLimitFinishReason(undefined)).toBe(false)
  })

  it("marks a reply while keeping its other generation info", () => {
    const marked = markOutputLimitedReply({ tool_calls: [] })
    expect(marked).toEqual({ tool_calls: [], finish_reason: "length" })
    expect(isOutputLimitedReply(marked)).toBe(true)
    expect(isOutputLimitedReply(markOutputLimitedReply(undefined))).toBe(true)
  })

  it("leaves an interrupted reply to the interruption notice", () => {
    expect(isOutputLimitedReply({ finish_reason: "length", interrupted: true })).toBe(false)
    expect(isOutputLimitedReply({ finish_reason: "stop" })).toBe(false)
    expect(isOutputLimitedReply(null)).toBe(false)
  })
})
