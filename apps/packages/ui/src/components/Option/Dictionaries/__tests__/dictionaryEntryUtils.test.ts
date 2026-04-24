import { describe, expect, it } from "vitest"
import {
  buildRestorableDictionaryEntryPayload,
  buildTextDiffSegments,
  buildTimedEffectsPayload,
  extractRegexSafetyMessage,
  formatProbabilityFrequencyHint,
  humanizeRegexError,
  humanizeValidationCode,
  normalizeProbabilityValue,
  toSafeNonNegativeInteger,
  validateRegexPattern,
} from "../components/dictionaryEntryUtils"

describe("dictionaryEntryUtils", () => {
  it("validates regex patterns and reports invalid syntax", () => {
    expect(validateRegexPattern("/foo+/i")).toBeNull()
    expect(validateRegexPattern("hello.*world")).toBeNull()
    expect(validateRegexPattern("[unclosed")).toBe("Opening bracket [ has no closing ].")
  })

  it("normalizes safe integers and timed-effects payload shape", () => {
    expect(toSafeNonNegativeInteger(3.8)).toBe(3)
    expect(toSafeNonNegativeInteger(-1)).toBe(0)
    expect(toSafeNonNegativeInteger("5")).toBe(5)
    expect(toSafeNonNegativeInteger("abc")).toBe(0)

    expect(buildTimedEffectsPayload(undefined)).toBeUndefined()
    expect(buildTimedEffectsPayload({ sticky: "2", cooldown: 5.9, delay: -3 })).toEqual({
      sticky: 2,
      cooldown: 5,
      delay: 0,
    })
    expect(buildTimedEffectsPayload(undefined, { forceObject: true })).toEqual({
      sticky: 0,
      cooldown: 0,
      delay: 0,
    })
  })

  it("clamps probability values and renders frequency hint copy", () => {
    expect(normalizeProbabilityValue(2)).toBe(1)
    expect(normalizeProbabilityValue(-0.3)).toBe(0)
    expect(normalizeProbabilityValue("invalid", 0.42)).toBe(0.42)
    expect(formatProbabilityFrequencyHint(0.56)).toBe("Fires ~6 out of 10 messages (56%).")
  })

  it("builds diff segments for removed and added tokens", () => {
    const segments = buildTextDiffSegments("alpha beta", "alpha gamma")
    expect(segments.some((segment) => segment.type === "removed")).toBe(true)
    expect(segments.some((segment) => segment.type === "added")).toBe(true)
    expect(segments.map((segment) => segment.text).join("")).toContain("alpha")
  })

  it("extracts regex-focused safety message with sensible fallback", () => {
    expect(
      extractRegexSafetyMessage({
        errors: [{ code: "regex_unsafe", field: "entries[0].pattern", message: "Unsafe regex" }],
      })
    ).toBe("Unsafe regex")

    expect(extractRegexSafetyMessage({ errors: [{ message: "Generic validation error" }] })).toBe(
      "Generic validation error"
    )
    expect(extractRegexSafetyMessage({ errors: [] })).toBeNull()
  })

  it("builds restorable payload with normalized defaults", () => {
    const payload = buildRestorableDictionaryEntryPayload({
      pattern: "BP",
      replacement: "blood pressure",
      type: "regex",
      enabled: false,
      case_sensitive: false,
      probability: 2,
      max_replacements: -1,
      group: "Clinical",
      timed_effects: { sticky: 1, cooldown: 2, delay: 3 },
    })

    expect(payload).toEqual({
      pattern: "BP",
      replacement: "blood pressure",
      type: "regex",
      enabled: false,
      case_sensitive: false,
      probability: 1,
      max_replacements: 0,
      group: "Clinical",
      timed_effects: { sticky: 1, cooldown: 2, delay: 3 },
    })
  })
})

describe("humanizeRegexError", () => {
  it("translates unterminated character class", () => {
    expect(humanizeRegexError("Invalid regular expression: /[abc/: Unterminated character class"))
      .toBe("Opening bracket [ has no closing ].")
  })

  it("translates nothing to repeat", () => {
    expect(humanizeRegexError("Invalid regular expression: Nothing to repeat"))
      .toBe("A repeat symbol (*, +, ?) has nothing before it.")
  })

  it("passes through unknown errors unchanged", () => {
    expect(humanizeRegexError("Something weird happened"))
      .toBe("Something weird happened")
  })

  it("does not over-match unrelated words that contain 'lone'", () => {
    expect(humanizeRegexError("alone"))
      .toBe("alone")
  })

  it("uses the provided localizer for known regex copy", () => {
    expect(
      humanizeRegexError(
        "Invalid regular expression: /[abc/: Unterminated character class",
        (key, fallback) => `${key}:${fallback}`
      )
    ).toBe(
      "option:dictionaries.validation.regex.unterminatedCharacterClass:Opening bracket [ has no closing ]."
    )
  })
})

describe("humanizeValidationCode", () => {
  it("translates known codes", () => {
    const result = humanizeValidationCode("regex_catastrophic_backtracking")
    expect(result.label).toBe("Slow pattern")
    expect(result.fix).toContain("Simplify nested")
  })

  it("translates pattern_duplicate", () => {
    const result = humanizeValidationCode("pattern_duplicate")
    expect(result.label).toBe("Duplicate pattern")
  })

  it("falls back to raw code for unknown codes", () => {
    expect(humanizeValidationCode("unknown_code").label).toBe("unknown_code")
    expect(humanizeValidationCode("unknown_code").fix).toBeUndefined()
  })

  it("uses the provided localizer for known validation codes", () => {
    const result = humanizeValidationCode(
      "pattern_duplicate",
      (key, fallback) => `${key}:${fallback}`
    )
    expect(result.label).toBe(
      "option:dictionaries.validation.code.patternDuplicate.label:Duplicate pattern"
    )
    expect(result.fix).toBe(
      "option:dictionaries.validation.code.patternDuplicate.fix:Another entry already uses this pattern. Remove one to avoid conflicts."
    )
  })
})
