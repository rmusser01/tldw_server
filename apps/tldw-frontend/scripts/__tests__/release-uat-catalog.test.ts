import { spawnSync } from "node:child_process"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { RELEASE_UAT_CATALOG, assertReleaseCatalog } from "../live-tier-uat/release-catalog.mjs"

const cells = ["sqlite-single", "sqlite-multi", "pg-single", "pg-multi"]
const identity = { id: "chat-history", project: "chromium", file: "e2e/chat.spec.ts", titlePath: ["Chat", "history"] }
const collection = () => ({
  errors: [],
  suites: [{ title: "Chat", specs: [{ file: identity.file, title: "history", tests: [{ projectName: "chromium", results: [] }] }] }],
})

function plan() {
  const contexts = cells.map(cell => ({ id: cell, cell, browser: "chromium", surface: "web", phase: "fresh-state" }))
  const instances = [
    ["chatbook-import-formats", "control-format"], ["chatbook-source-types", "control-source-type"],
    ["research-workspaces", "control-workspace"], ["research-import-paths", "control-connector"],
    ["watchlist-source-adapters", "control-feed-adapter"], ["backup-storage-adapters", "control-storage-adapter"],
  ].map(([id, instance]) => ({ id, status: "required", instances: [instance], reason: undefined as string | undefined }))
  const entries = contexts.flatMap(context => RELEASE_UAT_CATALOG.workflows.flatMap(workflow =>
    workflow.variants.filter(variant => variant.cells.includes(context.cell) && variant.surfaces.includes(context.surface) && variant.phases.includes(context.phase))
      .flatMap(variant => (variant.instanceSet ? instances.find(group => group.id === variant.instanceSet)!.instances : [undefined]).flatMap(instance => variant.modes.map(mode => ({ workflow: workflow.id, variant: variant.id, contextId: context.id, mode, instance,
        status: "excluded", reason: "Not advertised in this synthetic validator control", mapping: undefined, humanReview: undefined,
      }))))
  ))
  const required = entries.find(entry => entry.workflow === "A-03" && entry.variant === "multi-turn-history" && entry.contextId === "sqlite-single" && entry.mode === "D")!
  const variant = RELEASE_UAT_CATALOG.workflows.find(workflow => workflow.id === "A-03")!.variants.find(v => v.id === "multi-turn-history")!
  Object.assign(required, { status: "required", reason: undefined, mapping: { ...identity, assertions: [...variant.assertions], rationale: "The controlled test checks request history, persisted message order and IDs." } })
  return {
    catalogVersion: RELEASE_UAT_CATALOG.version,
    scope: {
      instances,
      cells: cells.map(id => ({ id, status: "required" })),
      surfaces: ["web", "extension-options", "extension-sidepanel", "cross-surface"].map(id => ({ id, status: id === "web" ? "required" : "excluded", reason: id === "web" ? undefined : "Extension is not part of this profile" })),
      phases: ["fresh-state", "clean-install", "supported-upgrade"].map(id => ({ id, status: id === "fresh-state" ? "required" : "excluded", reason: id === "fresh-state" ? undefined : "This control exercises fresh application state only" })),
      upgradePaths: ["previous-public", "oldest-supported"].map(id => ({ id, status: "excluded", reason: "No upgrade claim in this control" })),
      browsers: [{ id: "chromium", surfaces: ["web"] }],
    },
    contexts, entries,
  }
}
const validate = (manifest = plan(), report = collection()) => assertReleaseCatalog(manifest, report)

// These are validator controls, not real UAT results or mappings of existing suites.
describe("release catalog planning", () => {
  it("accepts a complete explicit plan and counts its single mapped execution once", () => {
    const result = validate()
    expect(result.executions).toHaveLength(1)
    expect(result.requiredCases[0]).toMatchObject({ ...identity, workflow: "A-03", variant: "multi-turn-history", contextId: "sqlite-single", mode: "D" })
    expect(result.exclusions.length).toBeGreaterThan(43)
  })
  it.each([
    ["missing family", (p: ReturnType<typeof plan>) => { p.entries = p.entries.filter(e => e.workflow !== "B-09") }, /Missing.*B-09/s],
    ["missing recovery variant", (p: ReturnType<typeof plan>) => { p.entries = p.entries.filter(e => !(e.workflow === "A-01" && e.variant === "invalid-provider-url")) }, /Missing.*invalid-provider-url/s],
    ["missing mode", (p: ReturnType<typeof plan>) => { p.entries = p.entries.filter(e => !(e.workflow === "A-03" && e.variant === "multi-turn-history" && e.mode === "L")) }, /Missing.*multi-turn-history/s],
    ["missing context", (p: ReturnType<typeof plan>) => { p.contexts = p.contexts.filter(c => c.cell !== "pg-multi"); p.entries = p.entries.filter(e => e.contextId !== "pg-multi") }, /Missing.*context/s],
    ["undeclared cell", (p: ReturnType<typeof plan>) => { p.scope.cells.pop() }, /scope.*cells/i],
    ["implicit phase exclusion", (p: ReturnType<typeof plan>) => { p.scope.phases.pop() }, /scope.*phases/i],
    ["implicit surface exclusion", (p: ReturnType<typeof plan>) => { p.scope.surfaces.pop() }, /scope.*surfaces/i],
    ["implicit upgrade exclusion", (p: ReturnType<typeof plan>) => { p.scope.upgradePaths.pop() }, /upgradePaths/],
    ["blank exclusion reason", (p: ReturnType<typeof plan>) => { p.entries.find(e => e.status === "excluded")!.reason = " " }, /reason/],
    ["unknown workflow", (p: ReturnType<typeof plan>) => { p.entries[0].workflow = "A-99" }, /Unknown/],
    ["unknown variant", (p: ReturnType<typeof plan>) => { p.entries[0].variant = "pretend-complete" }, /Unknown/],
    ["duplicate requirement", (p: ReturnType<typeof plan>) => { p.entries.push({ ...p.entries[0] }) }, /Duplicate/],
    ["wrong catalog version", (p: ReturnType<typeof plan>) => { p.catalogVersion = "old" }, /version/],
  ] as const)("rejects %s", (_name, change, message) => {
    const manifest = plan()
    change(manifest)
    expect(() => validate(manifest)).toThrow(message)
  })
  it.each(["project", "file", "titlePath"] as const)("rejects a nonexistent exact %s mapping", key => {
    const manifest = plan()
    const entry = manifest.entries.find(e => e.status === "required")!
    Object.assign(entry.mapping!, { [key]: key === "titlePath" ? ["Other", "history"] : "other" })
    expect(() => validate(manifest)).toThrow(/not registered/)
  })
  it("rejects a title-only declaration that omits the catalog assertions", () => {
    const manifest = plan()
    Object.assign(manifest.entries.find(e => e.status === "required")!.mapping!, { assertions: [] })
    expect(() => validate(manifest)).toThrow(/assertions/)
  })
  it("rejects duplicate registrations in list-only collection", () => {
    const report = collection()
    report.suites[0].specs.push(report.suites[0].specs[0])
    expect(() => validate(plan(), report)).toThrow(/Duplicate.*registration/)
  })
  it("rejects execution receipts passed off as collection", () => {
    const report = collection()
    Object.assign(report.suites[0].specs[0].tests[0], { results: [{ status: "passed" }] })
    expect(() => validate(plan(), report)).toThrow(/list-only/)
  })
  it("rejects collection errors even when the required test registered", () => {
    const report = collection()
    Object.assign(report, { errors: [{ message: "another file failed to load" }] })
    expect(() => validate(plan(), report)).toThrow(/collection/i)
  })
  it("keeps explicit shared D/L mappings as two requirements but one execution", () => {
    const manifest = plan()
    const d = manifest.entries.find(e => e.status === "required")!
    const l = manifest.entries.find(e => e.workflow === d.workflow && e.variant === d.variant && e.contextId === d.contextId && e.mode === "L")!
    Object.assign(l, { status: "required", reason: undefined, mapping: structuredClone(d.mapping) })
    const result = validate(manifest)
    expect(result.requiredCases).toHaveLength(2)
    expect(result.executions).toHaveLength(1)
    expect(result.executions[0].requirements.map(e => e.mode)).toEqual(["D", "L"])
  })
  it("rejects conflicting case IDs for one mapped execution", () => {
    const manifest = plan()
    const d = manifest.entries.find(e => e.status === "required")!
    const l = manifest.entries.find(e => e.workflow === d.workflow && e.variant === d.variant && e.contextId === d.contextId && e.mode === "L")!
    Object.assign(l, { status: "required", reason: undefined, mapping: { ...d.mapping, id: "second-count-of-same-case" } })
    expect(() => validate(manifest)).toThrow(/case ID/)
  })
  it("cannot satisfy human UX evidence with an automated mapping", () => {
    const manifest = plan()
    const d = manifest.entries.find(e => e.status === "required")!
    const u = manifest.entries.find(e => e.workflow === d.workflow && e.variant === d.variant && e.contextId === d.contextId && e.mode === "U")!
    Object.assign(u, { status: "required", reason: undefined, mapping: structuredClone(d.mapping) })
    expect(() => validate(manifest)).toThrow(/human/i)
  })
  it("retains separately attributed human evidence without creating a Playwright execution", () => {
    const manifest = plan()
    const u = manifest.entries.find(e => e.workflow === "A-03" && e.variant === "multi-turn-history" && e.contextId === "sqlite-single" && e.mode === "U")!
    Object.assign(u, { status: "required", reason: undefined, humanReview: { status: "completed", reviewer: "Test reviewer", evidence: [{ path: "ux/chat.md", sha256: "a".repeat(64) }] } })
    const result = validate(manifest)
    expect(result.humanReviews).toHaveLength(1)
    expect(result.executions).toHaveLength(1)
  })
  it("requires at least one real mapping instead of passing an entirely excluded plan", () => {
    const manifest = plan()
    Object.assign(manifest.entries.find(e => e.status === "required")!, { status: "excluded", reason: "Not exercised", mapping: undefined })
    expect(() => validate(manifest)).toThrow(/No required/)
  })
  it("checks all 43 playbook families and exposes concrete recovery assertions", () => {
    const expected = ["A-01", "A-02", "A-03", "A-04", "A-05", "A-06", "A-07", "A-08", "A-09", "A-10", "A-11", "A-12", "B-01", "B-02", "B-03", "B-04", "B-05", "B-06", "B-07", "B-08", "B-09", "C-01", "C-02", "C-03", "C-04", "C-05", "C-06", "C-07", "C-08", "C-09", "C-10", "C-11", "C-12", "X-01", "X-02", "X-03", "X-04", "S-01", "S-02", "S-03", "S-04", "S-05", "S-06"]
    expect(RELEASE_UAT_CATALOG.workflows.map(w => w.id)).toEqual(expected)
    expect(RELEASE_UAT_CATALOG.workflows.every(w => w.variants.some(v => v.kind === "recovery" && v.assertions.every(a => a.length > 20)))).toBe(true)
  })
})


describe("catalog evidence planning and scope", () => {
  it("keeps required human review pending before execution without inventing evidence or exclusions", () => {
    const manifest = plan()
    const u = manifest.entries.find(e => e.workflow === "A-03" && e.variant === "multi-turn-history" && e.contextId === "sqlite-single" && e.mode === "U")!
    Object.assign(u, { status: "required", reason: undefined, humanReview: { status: "pending" } })
    const result = validate(manifest)
    expect(result.pendingHumanReviews).toHaveLength(1)
    expect(result.humanReviews[0].humanReview.status).toBe("pending")
    expect(result.exclusions).not.toContain(u)
    expect(result.executions).toHaveLength(1)
  })
  it.each([
    { status: "completed" },
    { status: "completed", reviewer: "Reviewer", evidence: [] },
    { status: "completed", reviewer: "Reviewer", evidence: [{ path: "ux.md", sha256: "bad" }] },
    { status: "pending", evidence: [{ path: "ux.md", sha256: "a".repeat(64) }] },
    { status: "automated-pass" },
  ])("rejects invalid human review state %j", humanReview => {
    const manifest = plan()
    const u = manifest.entries.find(e => e.workflow === "A-03" && e.variant === "multi-turn-history" && e.contextId === "sqlite-single" && e.mode === "U")!
    Object.assign(u, { status: "required", reason: undefined, humanReview })
    expect(() => validate(manifest)).toThrow(/human/i)
  })
  it("permits a documented excluded cell without pretending it was covered", () => {
    const manifest = plan()
    Object.assign(manifest.scope.cells.find(c => c.id === "pg-multi")!, { status: "excluded", reason: "PostgreSQL multi-user is outside this declared control scope" })
    manifest.contexts = manifest.contexts.filter(c => c.id !== "pg-multi")
    manifest.entries = manifest.entries.filter(e => e.contextId !== "pg-multi")
    expect(validate(manifest).scopeExclusions).toContainEqual(expect.objectContaining({ axis: "cells", id: "pg-multi", status: "excluded" }))
  })
  it("requires contexts for an additional explicitly declared browser", () => {
    const manifest = plan()
    manifest.scope.browsers.push({ id: "firefox", surfaces: ["web"] })
    expect(() => validate(manifest)).toThrow(/Missing required context/)
  })
  it("requires a supported-upgrade context for each declared starting version", () => {
    const manifest = plan()
    Object.assign(manifest.scope.phases.find(p => p.id === "supported-upgrade")!, { status: "required", reason: undefined })
    Object.assign(manifest.scope.upgradePaths[0], { status: "required", from: "v0.1", reason: undefined })
    expect(() => validate(manifest)).toThrow(/Missing required context/)
  })
  it("rejects specialized mappings without the pinned input, oracle and steps", () => {
    const manifest = plan()
    const entry = manifest.entries.find(e => e.workflow === "S-06" && e.variant === "chunking" && e.contextId === "sqlite-single" && e.mode === "D")!
    const assertions = RELEASE_UAT_CATALOG.workflows.find(w => w.id === "S-06")!.variants.find(v => v.id === "chunking")!.assertions
    Object.assign(entry, { status: "required", reason: undefined, mapping: { ...identity, assertions, rationale: "Explicit synthetic declaration" } })
    expect(() => validate(manifest)).toThrow(/pinned input, oracle and steps/)
  })
  it.each([
    ["A-11", "practice-no-schedule"], ["C-03", "exact-match-two-results"], ["C-09", "separate-target-restore"],
    ["X-01", "published-migration"], ["X-02", "reciprocal-resource-isolation"], ["S-06", "documentation"],
  ])("does not manufacture live-provider evidence requirements for %s %s", (workflow, variant) => {
    const requirement = RELEASE_UAT_CATALOG.workflows.find(w => w.id === workflow)!.variants.find(v => v.id === variant)!
    expect(requirement.modes).toEqual(["D", "U"])
  })
})


describe("catalog CLI", () => {
  function runCatalog(manifest: ReturnType<typeof plan>) {
    const directory = mkdtempSync(join(tmpdir(), "uat392-catalog-cli-"))
    try {
      const manifestFile = join(directory, "plan.json")
      const collectionFile = join(directory, "collection.json")
      writeFileSync(manifestFile, JSON.stringify(manifest))
      writeFileSync(collectionFile, JSON.stringify(collection()))
      return spawnSync(process.execPath, [resolve(dirname(fileURLToPath(import.meta.url)), "..", "assert-release-uat.mjs"), "--catalog", manifestFile, collectionFile], { encoding: "utf8" })
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  }
  it("reports planned mappings and pending human review without certifying a release", () => {
    const manifest = plan()
    const u = manifest.entries.find(e => e.workflow === "A-03" && e.variant === "multi-turn-history" && e.contextId === "sqlite-single" && e.mode === "U")!
    Object.assign(u, { status: "required", reason: undefined, humanReview: { status: "pending" } })
    const result = runCatalog(manifest)
    expect(result.status, result.stderr).toBe(0)
    expect(JSON.parse(result.stdout)).toMatchObject({ kind: "catalog-plan", certifiesRelease: false, executions: 1, humanReviewsPending: 1 })
  })
  it("returns nonzero for an omitted supported context", () => {
    const manifest = plan()
    manifest.contexts = manifest.contexts.filter(c => c.cell !== "pg-multi")
    manifest.entries = manifest.entries.filter(e => e.contextId !== "pg-multi")
    const result = runCatalog(manifest)
    expect(result.status).toBe(1)
    expect(result.stderr).toMatch(/Missing required context/)
  })
})


describe("advertised catalog instances", () => {
  function twoInstances(workflow: string, variant: string, inventory: string) {
    const manifest = plan()
    manifest.scope.instances.find(group => group.id === inventory)!.instances = ["first", "second"]
    // All variants using an inventory (including recovery) need each instance.
    const variants = new Set(RELEASE_UAT_CATALOG.workflows.find(w => w.id === workflow)!.variants
      .filter(v => v.instanceSet === inventory || v.id === variant).map(v => v.id))
    manifest.entries = manifest.entries.flatMap(entry => entry.workflow === workflow && variants.has(entry.variant)
      ? ["first", "second"].map(instance => ({ ...entry, instance })) : [entry])
    const assertions = RELEASE_UAT_CATALOG.workflows.find(w => w.id === workflow)!.variants.find(v => v.id === variant)!.assertions
    for (const entry of manifest.entries.filter(e => e.workflow === workflow && e.variant === variant && e.contextId === "sqlite-single" && e.mode === "D")) {
      Object.assign(entry, { status: "required", reason: undefined, mapping: { ...identity, assertions, rationale: "This synthetic mapping explicitly declares both independently reviewed instance assertions." } })
    }
    return manifest
  }
  it.each([
    ["S-02", "advertised-import-formats", "chatbook-import-formats"],
    ["S-04", "connector-import", "research-import-paths"],
    ["S-02", "chat-note-source-roundtrip", "chatbook-source-types"],
    ["S-04", "research-workspace", "research-workspaces"],
    ["B-07", "external-source-adapter", "watchlist-source-adapters"],
    ["C-09", "external-storage-adapter", "backup-storage-adapters"],
  ])("retains both advertised %s %s instances without inflating execution counts", (workflow, variant, inventory) => {
    const result = validate(twoInstances(workflow, variant, inventory))
    expect(result.requiredCases.filter(e => e.workflow === workflow && e.variant === variant).map(e => e.instance)).toEqual(["first", "second"])
    expect(result.executions).toHaveLength(1)
    expect(result.executions[0].requirements.filter(e => e.workflow === workflow && e.variant === variant).map(e => e.instance)).toEqual(["first", "second"])
  })
  it.each([
    ["S-02", "advertised-import-formats", "chatbook-import-formats"],
    ["S-04", "connector-import", "research-import-paths"],
    ["S-02", "chat-note-source-roundtrip", "chatbook-source-types"],
    ["S-04", "research-workspace", "research-workspaces"],
    ["B-07", "external-source-adapter", "watchlist-source-adapters"],
    ["C-09", "external-storage-adapter", "backup-storage-adapters"],
  ])("rejects an omitted advertised %s %s instance", (workflow, variant, inventory) => {
    const manifest = twoInstances(workflow, variant, inventory)
    manifest.entries = manifest.entries.filter(e => !(e.workflow === workflow && e.instance === "second"))
    expect(() => validate(manifest)).toThrow(/Missing.*second/s)
  })
  it("requires recovery coverage for the second format even when both principal mappings exist", () => {
    const manifest = twoInstances("S-02", "advertised-import-formats", "chatbook-import-formats")
    manifest.entries = manifest.entries.filter(e => !(e.workflow === "S-02" && e.variant === "failed-import-retry" && e.instance === "second"))
    expect(() => validate(manifest)).toThrow(/Missing.*failed-import-retry.*second/s)
  })
  it("rejects an undeclared instance instead of silently accepting its mapping", () => {
    const manifest = plan()
    const entry = manifest.entries.find(e => e.workflow === "S-02" && e.variant === "advertised-import-formats")!
    entry.instance = "undeclared-format"
    expect(() => validate(manifest)).toThrow(/Unknown/)
  })
  it("rejects duplicate rows for the same instance", () => {
    const manifest = plan()
    const entry = manifest.entries.find(e => e.workflow === "S-02" && e.variant === "advertised-import-formats")!
    manifest.entries.push({ ...entry })
    expect(() => validate(manifest)).toThrow(/Duplicate/)
  })
  it.each(["empty", "duplicate", "unknown inventory", "missing inventory"])("rejects %s instance inventory", scenario => {
    const manifest = plan()
    const group = manifest.scope.instances.find(g => g.id === "chatbook-import-formats")!
    if (scenario === "empty") group.instances = []
    if (scenario === "duplicate") group.instances.push(group.instances[0])
    if (scenario === "unknown inventory") group.id = "arbitrary-dimension"
    if (scenario === "missing inventory") manifest.scope.instances = manifest.scope.instances.filter(g => g !== group)
    expect(() => validate(manifest)).toThrow(/instance/i)
  })
  it("requires a reason when no import format is applicable", () => {
    const manifest = plan()
    Object.assign(manifest.scope.instances.find(g => g.id === "chatbook-import-formats")!, { status: "not-applicable", instances: [] })
    expect(() => validate(manifest)).toThrow(/reason/)
  })
  it("keeps an explicit no-applicable-format disposition visible", () => {
    const manifest = plan()
    Object.assign(manifest.scope.instances.find(g => g.id === "chatbook-import-formats")!, { status: "not-applicable", instances: [], reason: "No import format is advertised in this diagnostic profile" })
    const formatVariants = new Set(["advertised-import-formats", "corrupt-import", "failed-import-retry"])
    manifest.entries = manifest.entries.filter(e => e.workflow !== "S-02" || !formatVariants.has(e.variant))
    expect(validate(manifest).scopeExclusions).toContainEqual(expect.objectContaining({ axis: "instances", id: "chatbook-import-formats", status: "not-applicable" }))
  })
  it("requires an explicit disposition for each declared instance even when it is excluded", () => {
    const manifest = twoInstances("S-02", "advertised-import-formats", "chatbook-import-formats")
    const excluded = manifest.entries.find(e => e.workflow === "S-02" && e.variant === "advertised-import-formats" && e.contextId === "sqlite-single" && e.mode === "D" && e.instance === "second")!
    Object.assign(excluded, { status: "excluded", reason: "This specific format is excluded from the declared release profile", mapping: undefined })
    expect(validate(manifest).exclusions).toContainEqual(excluded)
    excluded.reason = " "
    expect(() => validate(manifest)).toThrow(/reason/)
  })
})
