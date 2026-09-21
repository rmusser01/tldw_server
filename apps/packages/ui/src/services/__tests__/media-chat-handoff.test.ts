import { describe, expect, it } from "vitest"

import {
  buildDiscussMediaHint,
  getMediaChatHandoffMode,
  normalizeMediaChatHandoffPayload,
  parseMediaIdAsNumber
} from "@/services/tldw/media-chat-handoff"

describe("media chat handoff helpers", () => {
  it("normalizes payload and keeps supported mode", () => {
    const payload = normalizeMediaChatHandoffPayload({
      mediaId: "42",
      url: "https://example.com/video",
      title: "Demo",
      content: "Summary text",
      mode: "rag_media",
      ownerScope: "server:alice"
    })

    expect(payload).toEqual({
      mediaId: "42",
      url: "https://example.com/video",
      title: "Demo",
      content: "Summary text",
      mode: "rag_media",
      ownerScope: "server:alice"
    })
  })

  it("defaults mode to normal when mode is not provided", () => {
    const payload = normalizeMediaChatHandoffPayload({
      mediaId: "9"
    })
    expect(payload).toEqual({ mediaId: "9" })
    expect(getMediaChatHandoffMode(payload || {})).toBe("normal")
  })

  it("parses numeric media id and rejects invalid values", () => {
    expect(parseMediaIdAsNumber({ mediaId: "123" })).toBe(123)
    expect(parseMediaIdAsNumber({ mediaId: "abc" })).toBeNull()
    expect(parseMediaIdAsNumber({ mediaId: "-5" })).toBeNull()
  })

  it("builds hint text from structured payload content", () => {
    expect(
      buildDiscussMediaHint({
        mediaId: "7",
        title: "Weekly Meeting",
        content: "Transcript excerpt"
      })
    ).toContain("Chat with this media: Weekly Meeting")

    expect(
      buildDiscussMediaHint({
        mediaId: "7"
      })
    ).toBe("Let's talk about media 7.")
  })
})
