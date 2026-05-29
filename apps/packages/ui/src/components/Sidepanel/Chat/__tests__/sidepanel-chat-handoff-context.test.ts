import { describe, expect, it } from "vitest"
import { buildVisibleDocumentHandoffSnippetText } from "../sidepanel-chat-handoff-context"

describe("sidepanel chat handoff context helpers", () => {
  it("omits missing document title and URL fields from snippet text", () => {
    expect(
      buildVisibleDocumentHandoffSnippetText({
        title: "Research note"
      })
    ).toBe("Title: Research note")

    expect(
      buildVisibleDocumentHandoffSnippetText({
        url: "https://example.test/source"
      })
    ).toBe("URL: https://example.test/source")

    expect(buildVisibleDocumentHandoffSnippetText({})).toBe("")
  })
})
