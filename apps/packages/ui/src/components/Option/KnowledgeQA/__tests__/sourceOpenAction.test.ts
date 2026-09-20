import { describe, expect, it } from "vitest"
import { getSourceOpenAction } from "../sourceOpenAction"
import type { RagResult } from "../types"

const source = (url: string, sourceType = "media_db", sourceId = "1"): RagResult => ({
  id: "result", content: "Source", score: 1, sourceId, sourceType,
  metadata: { url, source_type: sourceType }
})

describe("evidence source navigation", () => {
  it.each(["study.txt", "/tmp/study.txt", "javascript:alert(1)", "//untrusted.example/path"])(
    "opens uploaded or unsafe reference %s in the supported media inspector", (url) => {
      expect(getSourceOpenAction(source(url))).toEqual({ href: "/media?id=1", label: "Open in Media" })
    }
  )
  it("retains valid original web links", () => {
    expect(getSourceOpenAction(source("https://example.com/article"))).toEqual({ href: "https://example.com/article", label: "Open original" })
  })
  it("opens Character and Chat evidence in their existing owner-aware pages", () => {
    expect(getSourceOpenAction(source("", "characters", "3"))).toEqual({
      href: "/characters?focusCharacterId=3", label: "Open in Characters"
    })
    expect(getSourceOpenAction(source("", "chats", "chat-123"))).toEqual({
      href: "/chat?settingsServerChatId=chat-123", label: "Open in Chat"
    })
  })
  it("rejects malformed Character IDs and safely encodes Chat IDs", () => {
    expect(getSourceOpenAction(source("", "characters", "../other"))).toBeNull()
    expect(getSourceOpenAction(source("", "chats", "chat&other=1"))?.href).toBe(
      "/chat?settingsServerChatId=chat%26other%3D1"
    )
  })
  it("does not invent a media route for another source type or invalid media ID", () => {
    expect(getSourceOpenAction(source("note.md", "notes"))).toBeNull()
    expect(getSourceOpenAction(source("study.txt", "media_db", "../other"))).toBeNull()
  })
})
