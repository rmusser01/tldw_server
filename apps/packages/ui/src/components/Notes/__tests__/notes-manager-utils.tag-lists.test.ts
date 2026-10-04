import { describe, expect, it } from "vitest"
import { mergeTagLists, normalizeTagList, tagSetsMatch } from "../notes-manager-utils"

describe("notes tag list helpers", () => {
  it("normalizes tags: trims, drops blanks, keeps the first spelling", () => {
    expect(normalizeTagList([" Research ", "", "research", "beta", 3, "  "])).toEqual([
      "Research",
      "beta"
    ])
  })

  it("merges additions after the existing tags without duplicates", () => {
    expect(mergeTagLists(["Alpha", "beta"], ["BETA", "gamma", "Gamma"])).toEqual([
      "Alpha",
      "beta",
      "gamma"
    ])
  })

  it("compares tag sets ignoring case, order and duplicates", () => {
    expect(tagSetsMatch(["a", "B"], ["b", "A", "a"])).toBe(true)
    expect(tagSetsMatch(["a"], ["a", "b"])).toBe(false)
    expect(tagSetsMatch([], [])).toBe(true)
  })
})
