import { describe, expect, it } from "vitest"
import { sliceCodeUnitsSafely } from "../safe-slice"

// Review fix: a cut must not leave half of a surrogate pair.
describe("sliceCodeUnitsSafely", () => {
  it("steps back when the cut would split a pair", () => {
    expect(sliceCodeUnitsSafely("ab\u{1F600}c", 3)).toBe("ab")
    expect(sliceCodeUnitsSafely("ab\u{1F600}c", 4)).toBe("ab\u{1F600}")
  })
  it("leaves short text and plain cuts alone", () => {
    expect(sliceCodeUnitsSafely("abc", 10)).toBe("abc")
    expect(sliceCodeUnitsSafely("abcdef", 3)).toBe("abc")
  })
})
