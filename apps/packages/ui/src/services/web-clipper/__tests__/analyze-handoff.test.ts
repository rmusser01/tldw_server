import { describe, expect, it } from "vitest"
import {
  collectWebClipAnalyzeMessageIds,
  hasSubmittedWebClipAnalyzeMessage,
  WEB_CLIPPER_ANALYZE_MESSAGE_TYPE
} from "../analyze-handoff"

describe("web clipper analyze handoff helpers", () => {
  it("tracks baseline ids for existing messages", () => {
    expect(
      collectWebClipAnalyzeMessageIds([
        { id: "msg-1", messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE },
        { id: 2, messageType: "other" },
        { id: null, messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE }
      ])
    ).toEqual(new Set(["msg-1", "2"]))
  })

  it("detects a newly submitted analyze message beyond the baseline ids", () => {
    const baselineIds = new Set(["msg-1"])

    expect(
      hasSubmittedWebClipAnalyzeMessage(
        [
          { id: "msg-1", messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE },
          { id: "msg-2", messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE }
        ],
        baselineIds
      )
    ).toBe(true)
  })

  it("ignores unchanged or differently typed messages", () => {
    const baselineIds = new Set(["msg-1", "msg-2"])

    expect(
      hasSubmittedWebClipAnalyzeMessage(
        [
          { id: "msg-1", messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE },
          { id: "msg-2", messageType: "save-to-notes" }
        ],
        baselineIds
      )
    ).toBe(false)
  })
})
