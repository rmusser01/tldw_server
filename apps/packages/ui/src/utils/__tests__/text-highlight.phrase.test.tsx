import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { highlightText, parseHighlightTerms, toPhraseHighlightQuery } from "../text-highlight"

const marksOf = (node: React.ReactNode) =>
  Array.from(render(<p>{node}</p>).container.querySelectorAll("mark")).map((mark) => mark.textContent)

describe("highlightText phrases (CS-11)", () => {
  it("still highlights each word of an unquoted query", () => {
    expect(marksOf(highlightText("Part 2 and part 17", "part 17"))).toEqual(["Part", "part", "17"])
  })

  it("highlights a quoted phrase as a whole and leaves its words alone elsewhere", () => {
    expect(marksOf(highlightText("Part 2 covers testing; Part 17 covers the rollout.", '"Part 17"'))).toEqual([
      "Part 17"
    ])
  })

  it("matches a phrase across line breaks and repeated spaces", () => {
    expect(marksOf(highlightText("see Part\n17  now", '"part 17"'))).toEqual(["Part\n17"])
  })

  it("treats regex characters in a phrase literally", () => {
    expect(marksOf(highlightText("cost (USD) is 5", '"(USD) is"'))).toEqual(["(USD) is"])
  })

  it("quotes a typed phrase for whole-phrase highlighting", () => {
    expect(toPhraseHighlightQuery("  Part   17 ")).toBe('"Part 17"')
    expect(toPhraseHighlightQuery('say "hi"')).toBe('"say hi"')
    expect(toPhraseHighlightQuery("   ")).toBe("")
    expect(parseHighlightTerms('"Part 17" rollout')).toEqual(["Part 17", "rollout"])
  })
})
