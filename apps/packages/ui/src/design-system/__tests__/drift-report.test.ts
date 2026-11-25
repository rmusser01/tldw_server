import { describe, expect, it } from "vitest"
import { assembleDriftReport, formatDriftSummary } from "../../../scripts/design-system-drift-report.mjs"

const finding = {
  path: "apps/packages/ui/src/Notice.tsx", surface: "shared-ui", line: 3, column: 12,
  rule: "literal-color", value: "red", owner: "Notice", classification: "token-review",
  reason: "Review the semantic color token"
}
const entry = {
  path: finding.path, rule: finding.rule, value: finding.value, owner: finding.owner,
  category: "data-visualization", reason: "Fixed chart-series color"
}
const options = () => ({
  analyses: [{ findings: [{ ...finding }], definitions: [], diagnostics: [] }],
  exceptions: { schemaVersion: 1, exceptions: [] as (typeof entry)[] },
  coverage: { roots: [{ path: "apps/packages/ui/src/", discovered: 1, analyzed: 1 }], excluded: [], limitations: ["Runtime styles are not analyzed"] },
  provenance: { sourceCommit: "1".repeat(40), dirtySource: false, untrackedSourceFileCount: 0 },
  diagnostics: [] as { path: string; code: string; severity: string; message: string }[]
})

describe("advisory drift report", () => {
  it("retains existing findings without a debt baseline", () => {
    const report = assembleDriftReport(options())
    expect(report.status).toBe("complete")
    expect(report.findings).toEqual([finding])
    expect(report.acceptedExceptions).toEqual([])
  })

  it("matches scoped exceptions without depending on line numbers", () => {
    const input = options()
    input.analyses[0].findings.push({ ...finding, line: 30 })
    input.exceptions.exceptions = [entry]
    const report = assembleDriftReport(input)
    expect(report.findings).toEqual([])
    expect(report.acceptedExceptions[0]).toMatchObject({ exception: entry, matchedCount: 2 })
    expect(report.acceptedExceptions[0].findings.map(f => f.line)).toEqual([3, 30])
  })

  it("keeps stale entries visible while retaining unrelated findings", () => {
    const input = options()
    input.exceptions.exceptions = [{ ...entry, value: "blue" }]
    const report = assembleDriftReport(input)
    expect(report.staleExceptions).toEqual(input.exceptions.exceptions)
    expect(report.findings).toEqual([finding])
  })

  it("leaves ambiguous exceptions unsuppressed", () => {
    const input = options()
    input.exceptions.exceptions = [entry, { ...entry, reason: "Another reason" }]
    const report = assembleDriftReport(input)
    expect(report.findings).toEqual([finding])
    expect(report.acceptedExceptions).toEqual([])
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ code: "exception-ambiguous" }))
  })

  it.each([
    null,
    { schemaVersion: 2, exceptions: [] },
    { schemaVersion: 1, exceptions: [{ ...entry, rule: "unknown" }] },
    { schemaVersion: 1, exceptions: [{ ...entry, reason: " " }] },
    { schemaVersion: 1, exceptions: [{ ...entry, path: "../Notice.tsx" }] },
    { schemaVersion: 1, exceptions: [{ ...entry, category: "everything" }] }
  ])("marks invalid manifest data incomplete", exceptions => {
    const report = assembleDriftReport({ ...options(), exceptions })
    expect(report.status).toBe("incomplete")
    expect(report.findings).toEqual([finding])
  })

  it("does not confuse incomplete scans with zero drift", () => {
    const input = options()
    input.analyses = []
    input.diagnostics = [{ path: "admin-ui", code: "missing-root", severity: "error", message: "Missing root" }]
    const report = assembleDriftReport(input)
    expect(report.status).toBe("incomplete")
    expect(formatDriftSummary(report, { limit: 25, reportPath: "/tmp/report.json" })).toContain("INCOMPLETE")
  })

  it("keeps dynamic coverage limitations advisory", () => {
    const input = options()
    input.diagnostics = [{ path: finding.path, code: "dynamic-style", severity: "coverage", message: "Unresolved expression" }]
    expect(assembleDriftReport(input).status).toBe("complete")
  })

  it("sorts independently of source and diagnostic order", () => {
    const input = options()
    input.analyses.push({ findings: [{ ...finding, path: "admin-ui/app/page.tsx", surface: "admin-ui" }], definitions: [], diagnostics: [] })
    const first = assembleDriftReport(input)
    input.analyses.reverse()
    expect(JSON.stringify(assembleDriftReport(input))).toBe(JSON.stringify(first))
  })

  it("caps only the summary and keeps full JSON findings", () => {
    const input = options()
    input.analyses[0].findings.push({ ...finding, line: 9, rule: "layout-value", classification: "layout-review", value: "44px" })
    const report = assembleDriftReport(input)
    const summary = formatDriftSummary(report, { limit: 1, reportPath: "/tmp/full.json" })
    expect(report.findings).toHaveLength(2)
    expect(summary).toContain("1 finding omitted")
    expect(summary).toContain("Layout review candidates: 1")
    expect(summary).toContain("/tmp/full.json")
    expect(summary).toContain("Runtime styles are not analyzed")
  })

  it("exposes dirty and untracked source provenance", () => {
    const input = options()
    input.provenance.dirtySource = true
    input.provenance.untrackedSourceFileCount = 2
    const report = assembleDriftReport(input)
    const summary = formatDriftSummary(report, { limit: 25, reportPath: "/tmp/report.json" })
    expect(summary).toContain("modified working files")
    expect(summary).toContain("Untracked source files: 2")
  })
})
