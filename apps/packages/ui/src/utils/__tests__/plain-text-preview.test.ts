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
    ["<div><p>Hello <b>world</b></p></div>", "Hello world"],
    ["**bold _and italic_ and `code`**", "bold and italic and code"],
    ["***both***", "both"],
    ["[x](https://en.wikipedia.org/wiki/Foo_(bar)) after", "x after"],
    ["\\*not italic\\* and 1\\_2", "*not italic* and 1_2"],
    ["| a | b |", "a · b"],
    ["> [!NOTE] Heads up", "Heads up"],
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
    expect(preview).toBe("Lots of space word w…")
  })

  it("cuts by character, never through an emoji's surrogate pair", () => {
    const preview = toPlainPreview("😀".repeat(200), { maxLength: 60 })
    expect(Array.from(preview)).toHaveLength(61)
    expect(preview.endsWith("😀…")).toBe(true)
    expect(preview).not.toMatch(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/)
  })

  it("skips YAML front matter", () => {
    expect(toPlainPreview("---\ntitle: Hello\ntags: [a]\n---\nBody")).toBe("Body")
  })

  it("skips a callout marker line and a setext underline", () => {
    expect(toPlainPreview("> [!NOTE]\n> text")).toBe("text")
    expect(toPlainPreview("Title\n=====\nBody", { skipLine: "Title" })).toBe("Body")
  })

  it("reads a table's first row as its cells", () => {
    expect(toPlainPreview("| a | b |\n|---|---|\n| 1 | 2 |")).toBe("a · b")
  })

  it("stays fast on a huge line with unclosed emphasis", () => {
    const started = performance.now()
    toPlainPreview("**a ".repeat(80000))
    toPlainPreview("_a ".repeat(40000))
    toPlainPreview("~~a ".repeat(40000))
    toPlainPreview("x".repeat(5_000_000))
    expect(performance.now() - started).toBeLessThan(500)
  })

  it("handles CRLF content", () => {
    expect(toPlainPreview("# Title\r\nBody line\r\n", { skipLine: "Title" })).toBe("Body line")
  })
})
