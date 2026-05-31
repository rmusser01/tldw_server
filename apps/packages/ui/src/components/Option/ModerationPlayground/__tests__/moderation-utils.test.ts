import { describe, expect, it } from "vitest"

import {
  ACTION_OPTIONS,
  CATEGORY_SUGGESTIONS,
  ONBOARDING_KEY,
  PRESET_PROFILES,
  areRulesEquivalent,
  buildOverridePayload,
  createRuleId,
  formatJson,
  formatRulePhase,
  getErrorStatus,
  isEqualJson,
  normalizeManagedBlocklistRows,
  normalizeCategories,
  normalizeOverrideForCompare,
  normalizeOverrideRules,
  normalizeRuleIsRegex,
  normalizeSettingsDraft,
  sortOverrideRules,
  stableSort,
  type ModerationScope,
  type SettingsDraft
} from "../moderation-utils"

// ---------------------------------------------------------------------------
// normalizeCategories
// ---------------------------------------------------------------------------
describe("normalizeCategories", () => {
  it("returns array values trimmed and filtered", () => {
    expect(normalizeCategories(["pii", " violence ", "", "hate_speech"])).toEqual([
      "pii",
      "violence",
      "hate_speech"
    ])
  })

  it("splits a comma-separated string", () => {
    expect(normalizeCategories("pii, violence, hate_speech")).toEqual([
      "pii",
      "violence",
      "hate_speech"
    ])
  })

  it("filters out empty segments from strings", () => {
    expect(normalizeCategories(",pii,,violence,")).toEqual(["pii", "violence"])
  })

  it("returns empty array for non-array, non-string values", () => {
    expect(normalizeCategories(42)).toEqual([])
    expect(normalizeCategories(null)).toEqual([])
    expect(normalizeCategories(undefined)).toEqual([])
    expect(normalizeCategories({})).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// formatJson
// ---------------------------------------------------------------------------
describe("formatJson", () => {
  it("formats an object as indented JSON", () => {
    expect(formatJson({ a: 1 })).toBe('{\n  "a": 1\n}')
  })

  it("returns empty object string for null/undefined", () => {
    expect(formatJson(null)).toBe("{}")
    expect(formatJson(undefined)).toBe("{}")
  })

  it("handles circular references gracefully", () => {
    const obj: any = {}
    obj.self = obj
    expect(formatJson(obj)).toBe("{}")
  })
})

// ---------------------------------------------------------------------------
// normalizeRuleIsRegex
// ---------------------------------------------------------------------------
describe("normalizeRuleIsRegex", () => {
  it("returns boolean values as-is", () => {
    expect(normalizeRuleIsRegex(true)).toBe(true)
    expect(normalizeRuleIsRegex(false)).toBe(false)
  })

  it("returns false for null/undefined", () => {
    expect(normalizeRuleIsRegex(null)).toBe(false)
    expect(normalizeRuleIsRegex(undefined)).toBe(false)
  })

  it("returns null for other types", () => {
    expect(normalizeRuleIsRegex("true")).toBeNull()
    expect(normalizeRuleIsRegex(1)).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// formatRulePhase
// ---------------------------------------------------------------------------
describe("formatRulePhase", () => {
  it('formats "both" as "Both phases"', () => {
    expect(formatRulePhase("both")).toBe("Both phases")
  })

  it('formats "input" as "Input phase"', () => {
    expect(formatRulePhase("input")).toBe("Input phase")
  })

  it('formats "output" as "Output phase"', () => {
    expect(formatRulePhase("output")).toBe("Output phase")
  })
})

// ---------------------------------------------------------------------------
// normalizeOverrideRules
// ---------------------------------------------------------------------------
describe("normalizeOverrideRules", () => {
  it("normalizes valid rules", () => {
    const input = [
      { id: "r1", pattern: "bad", is_regex: false, action: "block", phase: "input" }
    ]
    const result = normalizeOverrideRules(input)
    expect(result).toHaveLength(1)
    expect(result[0]).toEqual({
      id: "r1",
      pattern: "bad",
      is_regex: false,
      action: "block",
      phase: "input"
    })
  })

  it("filters out rules with missing id or pattern", () => {
    const input = [
      { id: "", pattern: "bad", is_regex: false, action: "block", phase: "input" },
      { id: "r2", pattern: "", is_regex: false, action: "block", phase: "input" }
    ]
    expect(normalizeOverrideRules(input)).toHaveLength(0)
  })

  it("filters out rules with invalid is_regex", () => {
    const input = [
      { id: "r1", pattern: "bad", is_regex: "yes", action: "block", phase: "input" }
    ]
    expect(normalizeOverrideRules(input)).toHaveLength(0)
  })

  it("filters out rules with invalid action", () => {
    const input = [
      { id: "r1", pattern: "bad", is_regex: false, action: "redact", phase: "input" }
    ]
    expect(normalizeOverrideRules(input)).toHaveLength(0)
  })

  it("defaults phase to 'both' for unknown phases", () => {
    const input = [
      { id: "r1", pattern: "bad", is_regex: false, action: "block", phase: "unknown" }
    ]
    const result = normalizeOverrideRules(input)
    expect(result[0].phase).toBe("both")
  })

  it("returns empty array for non-array input", () => {
    expect(normalizeOverrideRules("not an array")).toEqual([])
    expect(normalizeOverrideRules(null)).toEqual([])
  })

  it("skips non-object entries", () => {
    const input = [null, 42, "string", { id: "r1", pattern: "ok", is_regex: false, action: "warn", phase: "both" }]
    expect(normalizeOverrideRules(input)).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// sortOverrideRules
// ---------------------------------------------------------------------------
describe("sortOverrideRules", () => {
  it("sorts rules by id", () => {
    const rules = [
      { id: "c", pattern: "x", is_regex: false, action: "block" as const, phase: "both" as const },
      { id: "a", pattern: "y", is_regex: false, action: "warn" as const, phase: "both" as const },
      { id: "b", pattern: "z", is_regex: true, action: "block" as const, phase: "input" as const }
    ]
    const sorted = sortOverrideRules(rules)
    expect(sorted.map((r) => r.id)).toEqual(["a", "b", "c"])
  })

  it("does not mutate the original array", () => {
    const rules = [
      { id: "b", pattern: "x", is_regex: false, action: "block" as const, phase: "both" as const },
      { id: "a", pattern: "y", is_regex: false, action: "warn" as const, phase: "both" as const }
    ]
    sortOverrideRules(rules)
    expect(rules[0].id).toBe("b")
  })
})

// ---------------------------------------------------------------------------
// areRulesEquivalent
// ---------------------------------------------------------------------------
describe("areRulesEquivalent", () => {
  const base = { id: "r1", pattern: "Bad", is_regex: false, action: "block" as const, phase: "input" as const }

  it("returns true for equivalent rules (case-insensitive pattern)", () => {
    const other = { ...base, id: "r2", pattern: "bad" }
    expect(areRulesEquivalent(base, other)).toBe(true)
  })

  it("returns false when action differs", () => {
    const other = { ...base, action: "warn" as const }
    expect(areRulesEquivalent(base, other)).toBe(false)
  })

  it("returns false when is_regex differs", () => {
    const other = { ...base, is_regex: true }
    expect(areRulesEquivalent(base, other)).toBe(false)
  })

  it("returns false when phase differs", () => {
    const other = { ...base, phase: "output" as const }
    expect(areRulesEquivalent(base, other)).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// createRuleId
// ---------------------------------------------------------------------------
describe("createRuleId", () => {
  it("returns a non-empty string", () => {
    const id = createRuleId()
    expect(typeof id).toBe("string")
    expect(id.length).toBeGreaterThan(0)
  })

  it("returns unique ids on successive calls", () => {
    const ids = new Set(Array.from({ length: 20 }, () => createRuleId()))
    expect(ids.size).toBe(20)
  })
})

// ---------------------------------------------------------------------------
// buildOverridePayload
// ---------------------------------------------------------------------------
describe("buildOverridePayload", () => {
  it("includes only defined fields", () => {
    const payload = buildOverridePayload({ enabled: true })
    expect(payload).toEqual({ enabled: true })
  })

  it("normalizes categories_enabled", () => {
    const payload = buildOverridePayload({ categories_enabled: "pii, violence" })
    expect(payload.categories_enabled).toEqual(["pii", "violence"])
  })

  it("normalizes rules", () => {
    const payload = buildOverridePayload({
      rules: [
        { id: "r1", pattern: "bad", is_regex: false, action: "block", phase: "both" },
        { id: "", pattern: "", is_regex: false, action: "block", phase: "both" }
      ]
    })
    expect(payload.rules).toHaveLength(1)
  })

  it("omits undefined fields entirely", () => {
    const payload = buildOverridePayload({})
    expect(Object.keys(payload)).toHaveLength(0)
  })

  it("includes all fields when all are set", () => {
    const draft = {
      enabled: true,
      input_enabled: true,
      output_enabled: false,
      input_action: "block" as const,
      output_action: "warn" as const,
      redact_replacement: "[REMOVED]",
      categories_enabled: ["pii"],
      rules: [{ id: "r1", pattern: "test", is_regex: false, action: "block" as const, phase: "both" as const }]
    }
    const payload = buildOverridePayload(draft)
    expect(payload.enabled).toBe(true)
    expect(payload.input_enabled).toBe(true)
    expect(payload.output_enabled).toBe(false)
    expect(payload.input_action).toBe("block")
    expect(payload.output_action).toBe("warn")
    expect(payload.redact_replacement).toBe("[REMOVED]")
    expect(payload.categories_enabled).toEqual(["pii"])
    expect(payload.rules).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// stableSort
// ---------------------------------------------------------------------------
describe("stableSort", () => {
  it("sorts strings alphabetically", () => {
    expect(stableSort(["c", "a", "b"])).toEqual(["a", "b", "c"])
  })

  it("does not mutate the original array", () => {
    const arr = ["b", "a"]
    stableSort(arr)
    expect(arr[0]).toBe("b")
  })
})

// ---------------------------------------------------------------------------
// normalizeSettingsDraft
// ---------------------------------------------------------------------------
describe("normalizeSettingsDraft", () => {
  it("normalizes and sorts categories", () => {
    const result = normalizeSettingsDraft({
      piiEnabled: true,
      categoriesEnabled: ["violence", "pii"],
      persist: false
    })
    expect(result.categoriesEnabled).toEqual(["pii", "violence"])
    expect(result.piiEnabled).toBe(true)
    expect(result.persist).toBe(false)
  })

  it("coerces truthy/falsy values to boolean", () => {
    const result = normalizeSettingsDraft({
      piiEnabled: 1 as unknown as boolean,
      categoriesEnabled: [],
      persist: "" as unknown as boolean
    })
    expect(result.piiEnabled).toBe(true)
    expect(result.persist).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// normalizeOverrideForCompare
// ---------------------------------------------------------------------------
describe("normalizeOverrideForCompare", () => {
  it("sorts categories and rules for stable comparison", () => {
    const result = normalizeOverrideForCompare({
      categories_enabled: ["violence", "pii"],
      rules: [
        { id: "b", pattern: "word", is_regex: false, action: "block", phase: "both" },
        { id: "a", pattern: "other", is_regex: false, action: "warn", phase: "input" }
      ]
    })
    expect(result.categories_enabled).toEqual(["pii", "violence"])
    expect(result.rules![0].id).toBe("a")
    expect(result.rules![1].id).toBe("b")
  })
})

// ---------------------------------------------------------------------------
// isEqualJson
// ---------------------------------------------------------------------------
describe("isEqualJson", () => {
  it("returns true for identical objects", () => {
    expect(isEqualJson({ a: 1 }, { a: 1 })).toBe(true)
  })

  it("returns false for different objects", () => {
    expect(isEqualJson({ a: 1 }, { a: 2 })).toBe(false)
  })

  it("returns true for identical primitives", () => {
    expect(isEqualJson("hello", "hello")).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// getErrorStatus
// ---------------------------------------------------------------------------
describe("getErrorStatus", () => {
  it("extracts top-level status", () => {
    expect(getErrorStatus({ status: 404 })).toBe(404)
  })

  it("extracts nested response.status", () => {
    expect(getErrorStatus({ response: { status: 500 } })).toBe(500)
  })

  it("prefers top-level status over nested", () => {
    expect(getErrorStatus({ status: 403, response: { status: 500 } })).toBe(403)
  })

  it("returns null for non-object", () => {
    expect(getErrorStatus(null)).toBeNull()
    expect(getErrorStatus("error")).toBeNull()
    expect(getErrorStatus(42)).toBeNull()
  })

  it("returns null for object without status", () => {
    expect(getErrorStatus({ message: "oops" })).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// normalizeManagedBlocklistRows
// ---------------------------------------------------------------------------
describe("normalizeManagedBlocklistRows", () => {
  it("uses lint metadata to classify active and non-active rows", () => {
    const rows = normalizeManagedBlocklistRows([
      {
        id: 0,
        line: "bad -> block #profanity",
        pattern_type: "literal",
        action: "block",
        categories: ["profanity"],
        sample: "bad",
        ok: true
      },
      {
        id: 1,
        line: "# internal note",
        pattern_type: "comment",
        ok: true,
        warning: "comment (ignored)"
      },
      {
        id: 2,
        line: "   ",
        pattern_type: "empty",
        ok: true,
        warning: "blank line (ignored)"
      },
      {
        id: 3,
        line: "/[bad/ -> block",
        pattern_type: "regex",
        ok: false,
        error: "invalid regex"
      }
    ])

    expect(rows.map((row) => row.rowKind)).toEqual(["literal", "comment", "empty", "regex"])
    expect(rows.filter((row) => row.isActive).map((row) => row.id)).toEqual([0])
    expect(rows[0]).toMatchObject({
      pattern: "bad",
      action: "block",
      categories: ["profanity"],
      statusLabel: "Active"
    })
    expect(rows[1].statusLabel).toBe("Comment")
    expect(rows[2].statusLabel).toBe("Blank")
    expect(rows[3].statusLabel).toBe("Invalid")
  })

  it("falls back to parsing legacy id and line rows", () => {
    const rows = normalizeManagedBlocklistRows([
      { id: 5, line: "/secret/i -> redact:[MASK] #confidential,pii" },
      { id: 6, line: "# note" },
      { id: 7, line: "" }
    ])

    expect(rows[0]).toMatchObject({
      id: 5,
      pattern: "/secret/i",
      rowKind: "regex",
      action: "redact",
      replacement: "[MASK]",
      categories: ["confidential", "pii"],
      isActive: true
    })
    expect(rows[1]).toMatchObject({ rowKind: "comment", isActive: false })
    expect(rows[2]).toMatchObject({ rowKind: "empty", isActive: false })
  })
})

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
describe("constants", () => {
  it("ONBOARDING_KEY is a string", () => {
    expect(typeof ONBOARDING_KEY).toBe("string")
    expect(ONBOARDING_KEY).toBe("moderation-playground-onboarded")
  })

  it("CATEGORY_SUGGESTIONS includes all required backend categories", () => {
    const values = CATEGORY_SUGGESTIONS.map((c) => c.value)
    expect(values).toContain("violence")
    expect(values).toContain("self_harm")
    expect(values).toContain("sexual_content")
    expect(values).toContain("hate_speech")
    expect(values).toContain("pii")
    expect(values).toContain("pii_email")
    expect(values).toContain("pii_phone")
    expect(values).toContain("profanity")
    expect(values).toContain("drugs_alcohol")
    expect(values).toContain("gambling")
    expect(values).toContain("confidential")
    expect(CATEGORY_SUGGESTIONS.length).toBeGreaterThanOrEqual(11)
  })

  it("every CATEGORY_SUGGESTIONS entry has severity", () => {
    for (const cat of CATEGORY_SUGGESTIONS) {
      expect(["critical", "high", "medium", "low"]).toContain(cat.severity)
    }
  })

  it("ACTION_OPTIONS has block, redact, warn", () => {
    const values = ACTION_OPTIONS.map((o) => o.value)
    expect(values).toEqual(["block", "redact", "warn"])
  })

  it("PRESET_PROFILES has strict, balanced, monitor", () => {
    expect(Object.keys(PRESET_PROFILES)).toEqual(
      expect.arrayContaining(["strict", "balanced", "monitor"])
    )
    for (const key of Object.keys(PRESET_PROFILES)) {
      expect(PRESET_PROFILES[key].label).toBeTruthy()
      expect(PRESET_PROFILES[key].description).toBeTruthy()
      expect(PRESET_PROFILES[key].payload.enabled).toBe(true)
    }
  })
})

// ---------------------------------------------------------------------------
// Type exports (compile-time checks)
// ---------------------------------------------------------------------------
describe("type exports", () => {
  it("ModerationScope type is usable", () => {
    const scope: ModerationScope = "server"
    expect(scope).toBe("server")
    const scope2: ModerationScope = "user"
    expect(scope2).toBe("user")
  })

  it("SettingsDraft type is usable", () => {
    const draft: SettingsDraft = {
      piiEnabled: true,
      categoriesEnabled: ["pii"],
      persist: false
    }
    expect(draft.piiEnabled).toBe(true)
  })
})
