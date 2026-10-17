import { describe, expect, it } from "vitest"

import {
  continueMarkdownList,
  formatMarkdownLines
} from "../notes-manager-utils"

describe("Markdown list Enter", () => {
  it.each([
    ["- item", "- item\n- "],
    ["* item", "* item\n* "],
    ["+ item", "+ item\n+ "],
    ["9. item", "9. item\n10. "],
    ["\t- [x] done", "\t- [x] done\n\t- [ ] "],
    ["  - [X] done", "  - [X] done\n  - [ ] "],
    ["- [ ] ", ""],
    ["- [ ]", ""],
    ["  - ", "  "]
  ])("continues or exits %j", (source, expected) => {
    expect(continueMarkdownList(source, source.length, source.length)).toEqual({
      content: expected,
      cursor: expected.length
    })
  })

  it("keeps the text after the caret", () => {
    expect(continueMarkdownList("- hello suffix", 7, 7)).toEqual({
      content: "- hello\n-  suffix",
      cursor: 10
    })
  })

  it("does not exit an item with a suffix", () => {
    expect(continueMarkdownList("- suffix", 2, 2)).toEqual({
      content: "- \n- suffix",
      cursor: 5
    })
  })

  it.each([
    "```\n- sample",
    "~~~js\n- sample",
    "````\n```\n- sample",
    "~~~\n```\n- sample"
  ])("leaves fenced code alone: %j", (source) => {
    expect(
      continueMarkdownList(source, source.length, source.length)
    ).toBeNull()
  })

  it.each(["```\n```\n- item", "~~~js\n~~~~\n- item"])(
    "continues after a matching fence closes: %j",
    (source) => {
      expect(
        continueMarkdownList(source, source.length, source.length)?.content
      ).toBe(`${source}\n- `)
    }
  )

  it("leaves selection replacement and marker-prefix carets native", () => {
    expect(continueMarkdownList("- item", 2, 4)).toBeNull()
    expect(continueMarkdownList("- item", 1, 1)).toBeNull()
  })

  it("does not emit a rounded or exponential ordered marker", () => {
    const source = `${Number.MAX_SAFE_INTEGER}. item`
    expect(
      continueMarkdownList(source, source.length, source.length)
    ).toBeNull()
  })
})

describe("Markdown toolbar lines", () => {
  it.each(["heading1", "heading2", "heading3"] as const)(
    "replaces a heading prefix with %s",
    (action) => {
      const source = "## title suffix\nuntouched"
      expect(formatMarkdownLines(source, 5, 5, action).content).toBe(
        `${"#".repeat(Number(action.slice(-1)))} title suffix\nuntouched`
      )
    }
  )

  it("numbers nonblank selected logical lines and keeps indent", () => {
    const source = "before\n  - first\n\n  - [x] second\nafter"
    const result = formatMarkdownLines(
      source,
      11,
      source.indexOf("\nafter"),
      "numbered"
    )
    expect(result.content).toBe("before\n  1. first\n\n  2. second\nafter")
  })

  it("formats a complete midline item without losing its suffix", () => {
    expect(
      formatMarkdownLines("before\nhello suffix\nafter", 10, 10, "checklist")
        .content
    ).toBe("before\n- [ ] hello suffix\nafter")
  })

  it("selects the empty-line placeholder", () => {
    const next = formatMarkdownLines("", 0, 0, "checklist")
    expect(next.content.slice(next.start, next.end)).toBe("List item")
  })

  it("does not format the next line for a selection ending at its start", () => {
    expect(formatMarkdownLines("one\ntwo", 0, 4, "list").content).toBe(
      "- one\ntwo"
    )
  })
})

