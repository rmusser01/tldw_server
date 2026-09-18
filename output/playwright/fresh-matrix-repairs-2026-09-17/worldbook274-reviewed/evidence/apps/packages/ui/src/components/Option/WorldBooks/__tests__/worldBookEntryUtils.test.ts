import { describe, expect, it } from "vitest"
import {
  estimateEntryTokens,
  formatEntryContentStats,
  getPriorityBand,
  getPriorityTagColor,
  normalizeWorldBookEntryIdentifier,
  normalizeKeywordList,
  validateRegexKeywords
} from "../worldBookEntryUtils"

describe("worldBookEntryUtils", () => {
  it("normalizes keyword inputs from string and array values", () => {
    expect(normalizeKeywordList("alpha, beta , gamma")).toEqual(["alpha", "beta", "gamma"])
    expect(normalizeKeywordList(["alpha", " beta ", "", "gamma"])).toEqual([
      "alpha",
      "beta",
      "gamma"
    ])
  })

  it("maps a canonical positive API id to the manager entry_id while preserving legacy and invalid values", () => {
    expect(normalizeWorldBookEntryIdentifier({ id: 41, content: "canonical" })).toEqual({
      id: 41,
      entry_id: 41,
      content: "canonical"
    })
    expect(normalizeWorldBookEntryIdentifier({ id: 41, entry_id: 0 })).toEqual({
      id: 41,
      entry_id: 0
    })
    expect(normalizeWorldBookEntryIdentifier({ id: 0, content: "invalid" })).toEqual({
      id: 0,
      content: "invalid"
    })
    expect(normalizeWorldBookEntryIdentifier({ id: "41", content: "legacy-string" })).toEqual({
      id: "41",
      entry_id: 41,
      content: "legacy-string"
    })
    expect(normalizeWorldBookEntryIdentifier({ id: "not-an-id", content: "invalid" })).toEqual({
      id: "not-an-id",
      content: "invalid"
    })
    expect(normalizeWorldBookEntryIdentifier({ id: true, content: "invalid" })).toEqual({
      id: true,
      content: "invalid"
    })
    expect(normalizeWorldBookEntryIdentifier({ id: [91], content: "invalid" })).toEqual({
      id: [91],
      content: "invalid"
    })
    expect(normalizeWorldBookEntryIdentifier({ content: "missing" })).toEqual({ content: "missing" })
    expect(normalizeWorldBookEntryIdentifier({ id: null, content: "null" })).toEqual({
      id: null,
      content: "null"
    })
  })

  it("estimates tokens and formats content stats", () => {
    expect(estimateEntryTokens("abcd")).toBe(1)
    expect(estimateEntryTokens("abcdefghij")).toBe(3)
    expect(formatEntryContentStats("abcdefghij")).toBe("10 chars / ~3 tokens")
  })

  it("maps priority to visual bands", () => {
    expect(getPriorityBand(10)).toBe("low")
    expect(getPriorityBand(34)).toBe("medium")
    expect(getPriorityBand(67)).toBe("high")
    expect(getPriorityTagColor("low")).toBe("default")
    expect(getPriorityTagColor("medium")).toBe("blue")
    expect(getPriorityTagColor("high")).toBe("green")
  })

  it("validates regex keyword syntax", () => {
    expect(validateRegexKeywords(["valid.*pattern"])).toBeNull()
    expect(validateRegexKeywords(["[broken"])).toContain("Invalid regex pattern")
  })
})
