import { describe, expect, it, vi } from "vitest"
import { buildVisibleDocumentHandoffSnippetText, readVisiblePageContextForHandoff } from "../sidepanel-chat-handoff-context"

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

it("marks surviving document references partial when the requested active-page read fails", async () => {
  const readTabs = vi.fn().mockRejectedValue(new Error("tab unavailable"))
  const context = await readVisiblePageContextForHandoff(
    [{ title: "Research note", url: "https://example.test/source" }], true, readTabs
  )
  expect(context).toEqual({
    snippets: [{ kind: "visible-context", label: "Research note", text: "Title: Research note\nURL: https://example.test/source" }],
    truncated: true
  })
  expect(readTabs).toHaveBeenCalledTimes(1)
})

it("retains an unavailable indication for a context-only failed read", async () => {
  await expect(readVisiblePageContextForHandoff([], true, async () => {
    throw new Error("tab unavailable")
  })).resolves.toEqual({ snippets: [], truncated: true })
})

it("copies available page references without reading document bodies or inventing partial status", async () => {
  const context = await readVisiblePageContextForHandoff([], true, async () => [
    { title: "Active page", url: "https://example.test/active" }
  ])
  expect(context).toEqual({ title: "Active page", url: "https://example.test/active", snippets: [] })
})
