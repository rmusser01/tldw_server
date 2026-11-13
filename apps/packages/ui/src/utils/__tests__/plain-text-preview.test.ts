import { describe, expect, it } from "vitest"
import { markdownLineToPlainText, toPlainPreview } from "../plain-text-preview"

// NL-04 (#3101): list previews show one line of plain text, not raw Markdown.

describe("markdownLineToPlainText", () => {
  it.each([
    ["# Heading", "Heading"],
    ["###   Deep heading ##", "Deep heading"],
    ["**Mood:** calm", "Mood: calm"],
    ["__bold__ and _italic_ and *star*", "bold and italic and star"],
    ["~~gone~~ kept", "gone kept"],
    ["Run `npm test` first", "Run npm test first"],
    ["See [[Project Alpha]] for more", "See Project Alpha for more"],
    ["See [[Project Alpha|the project]]", "See the project"],
    ["Linked [[id:acd86f3a-492e-4a8b-90fb-ff282b9721fb]] note", "Linked note"],
    ["Read [the docs](https://example.test/docs)", "Read the docs"],
    ["![diagram](https://example.test/a.png) caption", "diagram caption"],
    ["> quoted text", "quoted text"],
    ["- [ ] open task", "open task"],
    ["* bullet", "bullet"],
    ["12. numbered", "numbered"],
    ["snake_case_name stays", "snake_case_name stays"],
    ["2 * 3 * 4", "2 * 3 * 4"]
  ])("turns %j into %j", (line, expected) => {
    expect(markdownLineToPlainText(line)).toBe(expected)
  })
})

describe("toPlainPreview", () => {
  it("drops code fence lines and keeps the first meaningful line", () => {
    expect(toPlainPreview("```sql\nSELECT 1;\n```\nAfter")).toBe("SELECT 1;")
  })

  it("skips rules, blank lines and table separators", () => {
    expect(toPlainPreview("---\n\n| --- | --- |\n**Mood:** calm")).toBe("Mood: calm")
  })

  it("skips a line that repeats the title", () => {
    expect(toPlainPreview("# Research plan\n\nFocus on **coverage**.", { skipLine: "Research plan" })).toBe(
      "Focus on coverage."
    )
  })

  it("returns nothing when the only line repeats the title", () => {
    expect(toPlainPreview("# Research plan", { skipLine: "# Research plan" })).toBe("")
  })

  it("collapses whitespace and truncates to one short line", () => {
    const preview = toPlainPreview(`Lots   of\tspace ${"word ".repeat(60)}`, { maxLength: 20 })
    expect(preview).toBe("Lots of space word w...")
  })

  it("handles CRLF content", () => {
    expect(toPlainPreview("# Title\r\nBody line\r\n", { skipLine: "Title" })).toBe("Body line")
  })
})
