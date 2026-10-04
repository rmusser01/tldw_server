import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import { compareRatchet, ratchetEntryProblems, type RatchetEntry, type RatchetObservation } from "../ratchet"

const entry = (overrides: Partial<RatchetEntry> = {}): RatchetEntry => ({
  key: "color-contrast",
  count: 3,
  reviewIds: ["AX-07"],
  issue: 3124,
  note: "Token-level contrast",
  ...overrides,
})

const compare = (observed: Record<string, number>, baseline: RatchetEntry[]) =>
  compareRatchet({
    scope: "/notes (empty library)",
    subject: "axe violation",
    baselineFile: "e2e/ux-regression/baselines/a11y.json",
    observed: new Map<string, RatchetObservation>(
      Object.entries(observed).map(([key, count]) => [key, { count, details: [`${key} sample`] }])
    ),
    baseline,
  })

describe("compareRatchet", () => {
  it("passes when observations match the baseline", () => {
    expect(compare({ "color-contrast": 3 }, [entry()])).toEqual([])
  })

  it("fails on a problem that is not in the baseline", () => {
    const [problem] = compare({ "color-contrast": 3, "aria-allowed-attr": 1 }, [entry()])
    expect(problem).toContain('new axe violation "aria-allowed-attr" (1)')
    expect(problem).toContain("aria-allowed-attr sample")
  })

  it("fails when a baselined problem is no longer present", () => {
    const [problem] = compare({}, [entry()])
    expect(problem).toContain('"color-contrast" [AX-07 (#3124)] is no longer present, remove it from the baseline')
  })

  it("fails on growth beyond the tolerance and accepts growth within it", () => {
    expect(compare({ "color-contrast": 4 }, [entry()])[0]).toContain("regressed: 4, baseline 3 ±0")
    expect(compare({ "color-contrast": 4 }, [entry({ tolerance: 1 })])).toEqual([])
  })

  it("fails on improvement beyond the tolerance so the baseline is lowered", () => {
    expect(compare({ "color-contrast": 1 }, [entry()])[0]).toContain("Lower its count")
    expect(compare({ "color-contrast": 2 }, [entry({ tolerance: 1 })])).toEqual([])
  })

  it("lets a race-dependent entry be absent only when its tolerance covers zero", () => {
    expect(compare({}, [entry({ count: 1, tolerance: 1 })])).toEqual([])
    expect(compare({}, [entry({ count: 2, tolerance: 1 })])[0]).toContain("no longer present")
  })

  it("rejects duplicate baseline keys", () => {
    expect(compare({ "color-contrast": 3 }, [entry(), entry()])[0]).toContain("duplicate baseline entry")
  })
})

describe("committed ux-regression baselines", () => {
  const read = (name: string) =>
    JSON.parse(readFileSync(path.join(__dirname, "../../ux-regression/baselines", name), "utf8"))

  it("tag every a11y entry with a review id, issue and note", () => {
    const { entries } = read("a11y.json") as { entries: Array<RatchetEntry & { rule: string; nodes: number }> }
    expect(entries.length).toBeGreaterThan(0)
    const problems = entries.flatMap((item, index) =>
      ratchetEntryProblems({ ...item, key: item.rule, count: item.nodes }, `a11y.json entries[${index}]`)
    )
    expect(problems).toEqual([])
  })

  it("tag every request-budget entry with a review id, issue and note", () => {
    const { entries } = read("request-budget.json") as { entries: RatchetEntry[] }
    expect(entries.length).toBeGreaterThan(0)
    const problems = entries.flatMap((item, index) => ratchetEntryProblems(item, `request-budget.json entries[${index}]`))
    expect(problems).toEqual([])
  })
})

describe("ratchetEntryProblems", () => {
  it("accepts a complete entry", () => {
    expect(ratchetEntryProblems(entry(), "entries[0]")).toEqual([])
  })

  it("requires review ids, an issue and a note", () => {
    const problems = ratchetEntryProblems(entry({ reviewIds: [], issue: 0, note: " " }), "entries[0]")
    expect(problems).toHaveLength(3)
  })
})
