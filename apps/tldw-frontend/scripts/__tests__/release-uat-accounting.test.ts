import { describe, expect, it } from "vitest"
import {
  assertProjectAccounting,
  parseListOutput,
  summarizePlaywrightReport,
} from "../live-tier-uat/report.mjs"

const runContext = {
  cell: "pg-multi",
  surface: "web",
  phase: "fresh-state",
  revision: "a".repeat(40),
  sourceSha256: "b".repeat(64),
  artifacts: { backend: "c".repeat(64), web: "d".repeat(64) },
}
const requiredCases = [
  { id: "saved-chat", project: "chromium", file: "e2e/chat.spec.ts", titlePath: ["Chat", "saved"], workflow: "A-03", variant: "saved" },
  { id: "image-retry", project: "chromium", file: "e2e/chat.spec.ts", titlePath: ["Chat", "image retry"], workflow: "A-04", variant: "retry" },
]
const makeReport = () => ({
  config: {
    metadata: { uat: structuredClone(runContext) },
    projects: [{ name: "chromium", retries: 0, repeatEach: 1 }],
  },
  errors: [] as { message: string }[],
  stats: { expected: 2, skipped: 0, unexpected: 0, flaky: 0 },
  suites: [{ title: "Chat", specs: requiredCases.map(({ id, project, file, titlePath }) => ({
    id, file, title: titlePath.at(-1)!,
    tests: [{
      projectName: project,
      expectedStatus: "passed",
      status: "expected",
      results: [{ status: "passed", retry: 0, duration: 1 }],
    }],
  })) }],
})
type Report = ReturnType<typeof makeReport>
function verify(report = makeReport(), required = requiredCases, context = runContext) {
  return assertProjectAccounting({
    projects: ["chromium"],
    listed: { chromium: 2 },
    results: summarizePlaywrightReport(report),
    requiredCases: required,
    report,
    runContext: context,
  })
}

describe("release required-case accounting", () => {
  it("accepts exactly the required first-attempt cases with matching provenance", () => {
    expect(() => verify()).not.toThrow()
  })

  it.each([
    ["substituted case", (r: Report) => { r.suites[0].specs[1].title = "different image check" }],
    ["duplicate case", (r: Report) => { r.suites[0].specs[1].title = "saved" }],
    ["global error", (r: Report) => { r.errors.push({ message: "global teardown failed" }) }],
    ["changed candidate", (r: Report) => { r.config.metadata.uat.artifacts.web = "e".repeat(64) }],
    ["changed cell", (r: Report) => { r.config.metadata.uat.cell = "sqlite-multi" }],
    ["changed phase", (r: Report) => { r.config.metadata.uat.phase = "supported-upgrade" }],
    ["enabled retries", (r: Report) => { r.config.projects[0].retries = 2 }],
    ["enabled repeats", (r: Report) => { r.config.projects[0].repeatEach = 2 }],
    ["recovered failed attempt", (r: Report) => {
      r.suites[0].specs[0].tests[0].results.unshift({ status: "failed", retry: 0, duration: 1 })
      r.suites[0].specs[0].tests[0].results[1].retry = 1
    }],
    ["only retry retained", (r: Report) => { r.suites[0].specs[0].tests[0].results[0].retry = 1 }],
    ["failure expectation", (r: Report) => { r.suites[0].specs[0].tests[0].expectedStatus = "failed" }],
    ["flaky status", (r: Report) => { r.suites[0].specs[0].tests[0].status = "flaky" }],
    ["statistics disagree", (r: Report) => { r.stats.expected = 3 }],
    ["statistics conceal a skip", (r: Report) => { r.stats.skipped = 1 }],
  ] as const)("rejects %s even when project totals still match", (_name, mutate) => {
    const report = makeReport()
    mutate(report)
    expect(() => verify(report)).toThrow()
  })

  it("rejects an empty required manifest", () => {
    expect(() => verify(makeReport(), [])).toThrow()
  })
  it("rejects duplicate required identities", () => {
    expect(() => verify(makeReport(), [requiredCases[0], requiredCases[0]])).toThrow()
  })
  it.each(["workflow", "variant"] as const)("requires each case's %s", (key) => {
    expect(() => verify(makeReport(), requiredCases.map(c => ({ ...c, [key]: "" })))).toThrow()
  })
  it("requires a source hash and artifact hashes in run provenance", () => {
    const context = { ...runContext, sourceSha256: "", artifacts: { backend: "", web: "" } }
    const report = makeReport()
    report.config.metadata.uat = context
    expect(() => verify(report, requiredCases, context)).toThrow()
  })
  it.each(["skipped", "failed", "interrupted"])("rejects a required %s result", (status) => {
    const report = makeReport()
    report.suites[0].specs[0].tests[0].results[0].status = status
    expect(() => verify(report)).toThrow()
  })
  it("rejects missing and extra result rows", () => {
    const missing = makeReport()
    missing.suites[0].specs.pop()
    expect(() => verify(missing)).toThrow()
    const extra = makeReport()
    extra.suites[0].specs.push({ ...extra.suites[0].specs[0], id: "extra" })
    expect(() => verify(extra)).toThrow()
  })
  it("counts root workflows, every tier, journeys and browser variants", () => {
    const projects = ["chromium", "tier-1", "tier-2", "tier-3", "tier-4", "tier-5", "journeys", "standalone-html-firefox", "standalone-html-webkit"]
    expect(parseListOutput(projects.map(p => `[${p}] › example.spec.ts:1:1 › case`).join("\n")))
      .toEqual(Object.fromEntries(projects.map(p => [p, 1])))
  })
})

describe("release accounting entrypoint", () => {
  it("uses exact manifest identities through the public validator", async () => {
    const { assertReleaseUat } = await import("../assert-release-uat.mjs")
    const manifest = { context: runContext, cases: requiredCases }
    expect(() => assertReleaseUat(manifest, makeReport())).not.toThrow()
    const substituted = makeReport()
    substituted.suites[0].specs[0].title = "unplanned control"
    expect(() => assertReleaseUat(manifest, substituted)).toThrow(/Unexpected/)
  })
  it("refuses list-only reports as passing execution evidence", async () => {
    const { assertReleaseUat } = await import("../assert-release-uat.mjs")
    const report = makeReport()
    report.stats = { expected: 0, skipped: 2, unexpected: 0, flaky: 0 }
    report.suites[0].specs.forEach(spec => { spec.tests[0].results = [] })
    expect(() => assertReleaseUat({ context: runContext, cases: requiredCases }, report)).toThrow(/skipped/)
  })
})

// Full Playwright JSON merges cross-project specs using the first project's id.
describe("stable registered case identity", () => {
  it("accepts the same project/file/title path when Playwright spec IDs change", () => {
    const report = makeReport()
    report.suites[0].specs.forEach(spec => { spec.id = "browser-only-" + spec.id })
    expect(() => verify(report)).not.toThrow()
  })
  it("distinguishes identical leaf titles in different describe groups", () => {
    const report = makeReport()
    report.suites[0].title = "Different Chat group"
    expect(() => verify(report)).toThrow()
  })
  it("distinguishes identical titles in different source files", () => {
    const report = makeReport()
    report.suites[0].specs.forEach(spec => { spec.file = "e2e/different.spec.ts" })
    expect(() => verify(report)).toThrow()
  })
})


describe("diagnostic runner registered-case accounting", () => {
  const check = (report: Report, registrations = requiredCases) => assertProjectAccounting({
    projects: ["chromium"], listed: { chromium: 2 },
    results: summarizePlaywrightReport(report), registeredCases: registrations, report,
  })
  it("accepts the exact listed identities without claiming artifact provenance", () => {
    expect(() => check(makeReport())).not.toThrow()
  })
  it.each([
    ["substitution", (r: Report) => { r.suites[0].specs[1].title = "replacement" }],
    ["duplicate", (r: Report) => { r.suites[0].specs[1].title = "saved" }],
    ["concealed retry", (r: Report) => { r.suites[0].specs[0].tests[0].results[0].retry = 1 }],
  ] as const)("rejects %s with unchanged project totals", (_name, mutate) => {
    const report = makeReport()
    mutate(report)
    expect(() => check(report)).toThrow()
  })
  it("rejects duplicate listed identities", () => {
    expect(() => check(makeReport(), [requiredCases[0], requiredCases[0]])).toThrow()
  })
})


it("retains failed first attempts and unrun cases in the readable diagnostic report", async () => {
  const { renderMarkdownReport } = await import("../live-tier-uat/report.mjs")
  const report = makeReport()
  report.suites[0].specs[0].tests[0].results.unshift({ status: "failed", retry: 0, duration: 1 })
  report.suites[0].specs[0].tests[0].results[1].retry = 1
  report.suites[0].specs[1].tests[0].results = []
  const markdown = renderMarkdownReport({ runId: "diagnostic", report })
  expect(markdown).toContain("Certification run: no")
  expect(markdown).toContain("Chat > saved | failed | 0")
  expect(markdown).toContain("Chat > saved | passed | 1")
  expect(markdown).toContain("Chat > image retry | unrun | 0")
})
