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
  it("does not invent a media route for another source type or invalid media ID", () => {
    expect(getSourceOpenAction(source("note.md", "notes"))).toBeNull()
    expect(getSourceOpenAction(source("study.txt", "media_db", "../other"))).toBeNull()
  })
})
