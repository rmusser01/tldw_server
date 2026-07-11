import { describe, expect, it } from "vitest"
import {
  normalizeSourceIds,
  shouldConfirmMultiSourceCheck
} from "../check-now-utils"

describe("check-now utils", () => {
  it("normalizes and deduplicates valid positive integer IDs", () => {
    expect(normalizeSourceIds([1, 2, 2, "3", 0, -1, null, undefined, "abc"])).toEqual([1, 2, 3])
  })

  it("requires confirmation only for multi-source checks", () => {
    expect(shouldConfirmMultiSourceCheck([1])).toBe(false)
    expect(shouldConfirmMultiSourceCheck([1, 2])).toBe(true)
  })
})
