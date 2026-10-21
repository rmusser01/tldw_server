# Design-System Drift Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, or superpowers:subagent-driven-development if the owner selects delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an advisory inventory of existing and new styling drift across all four product UI roots.

**Architecture:** Parse source and token definitions without executing analyzed code. Keep token authority, source analysis, report assembly, and filesystem/CLI orchestration separate. Publish complete JSON and a bounded summary in a nonrequired CI workflow.

**Tech Stack:** Node.js 24, TypeScript compiler API, PostCSS 8, Tailwind 3 default data, jsdom 28 CSS value recognition, Vitest 4, existing GitHub Actions/license-first workflow contracts. All JavaScript dependencies are already declared by `@tldw/ui`; no dependency or lockfile changes.

**Spec:** [Reviewed design](../specs/2026-10-20-design-system-drift-reporting-design.md), including all four review corrections committed in `6e1f93eef3`.

**Tracking:** TASK-45.44.16; [issue #1673](https://github.com/rmusser01/tldw_server/issues/1673).

**ADR check:** ADR required: yes, covered by proposed [ADR-079](../../ADR/079-advisory-ui-design-drift-reporting.md). This plan implements its advisory policy and introduces no further architecture decision. Retain Proposed status until accepted through owner review.

## Global Constraints

- "Scan existing and new code on every run."
- "Publish actionable findings without blocking builds or merges."
- "No automatic rewriting or new dependency is authorized by this design."
- "Do not exempt a whole file because it contains definitions."
- "Do not traverse symlinks or other worktrees."
- "Do not seed an exception for every existing finding."
- "An incomplete scan must never be described as zero drift."
- Preserve the existing product-state guard, its baseline, and build token-sync checks.
- Scan `apps/packages/ui/src/`, `apps/tldw-frontend/`, `apps/extension/`, and `admin-ui/`; handle the admin vocabulary separately.
- Include tracked and nonignored untracked supported source; exclude generated/vendor/test/tooling source using explicit path rules.
- Use backlog-py for task edits. Record red/green evidence, commits, review findings, and known coverage gaps.
- No UI migration, theme change, branch-protection change, or automatic enforcement belongs to this implementation.

## Stages and File Ownership

| Stage | Goal | Success Criteria | Tests | Status |
| --- | --- | --- | --- | --- |
| 1 | Extract token authority and definition scopes | Mixed files retain product analysis; four surfaces use the right definitions | Token authority fixtures and real configs | Not Started |
| 2 | Analyze styling and classify findings | Four rules, named colors, and `@apply` produce located findings without flagging prose | Syntax-aware rule fixtures | Not Started |
| 3 | Assemble reports and exceptions | Deterministic complete/incomplete output, visible debt and stale exceptions | Report/manifest fixtures | Not Started |
| 4 | Deliver CLI, inventory, and provenance | Whole-root scan, honest dirty-source metadata, correct exit codes | Temporary Git repositories and CLI subprocesses | Not Started |
| 5 | Publish advisory CI and migration guidance | License-first ordering, nonrequired job, retained artifacts, documented limits | Existing CI contracts, actionlint, full scan sample | Not Started |

Create these implementation files; do not add a generic parser or plugin framework:

- `apps/packages/ui/scripts/design-system-drift-tokens.mjs`: token authority and exact definition scopes.
- `apps/packages/ui/scripts/design-system-drift-rules.mjs`: syntax analysis and four finding rules.
- `apps/packages/ui/scripts/design-system-drift-report.mjs`: exception matching, sorting, and summary formatting.
- `apps/packages/ui/scripts/verify-design-system-drift.mjs`: Git inventory, file reading, provenance, CLI, and output.
- `apps/packages/ui/scripts/design-system-drift-exceptions.json`: empty initial exception manifest.
- `apps/packages/ui/src/design-system/__tests__/drift-tokens.test.ts`
- `apps/packages/ui/src/design-system/__tests__/drift-rules.test.ts`
- `apps/packages/ui/src/design-system/__tests__/drift-report.test.ts`
- `apps/packages/ui/src/design-system/__tests__/drift-cli.test.ts`
- `.github/workflows/ui-design-system-drift-report.yml`

Modify only `apps/packages/ui/package.json`, the main design-system contract,
the existing CI workflow inventory test named in Stage 5, this plan's progress,
and associated task records as needed. No public application exports change.

## Shared Data Contract

These shapes describe plain objects, not additional runtime classes or schemas.
Define them as JSDoc where used, with the same field names across modules:

```ts
type Surface = "shared-ui" | "webui" | "extension" | "admin-ui"
type Source = { path: string; source: string }
type Diagnostic = {
  path: string; code: string; message: string;
  severity: "error" | "coverage"; line?: number; column?: number
}
type Definition = { path: string; start: number; end: number; owner: string; kind: string }
type Finding = {
  path: string; surface: Surface; line: number; column: number;
  rule: "literal-color" | "palette-color" | "custom-radius" | "layout-value";
  value: string; owner: string | null;
  classification: "token-review" | "radius-review" | "layout-review";
  reason: string; suggestion?: string
}
type Analysis = { findings: Finding[]; definitions: Definition[]; diagnostics: Diagnostic[] }
type Authority = {
  surfaces: Record<Surface, {
    semanticColors: Set<string>; cssVariables: Set<string>;
    radiusClasses: Set<string>; radiusValues: Set<string>;
    definitionPaths: string[]
  }>;
  definitionScopes: Definition[];
  diagnostics: Diagnostic[]
}
type Provenance = {
  sourceCommit: string | null; dirtySource: boolean | null;
  untrackedSourceFileCount: number
}
type Coverage = {
  roots: Array<{path: string; discovered: number; analyzed: number}>;
  excluded: Array<{reason: string; count: number}>;
  limitations: string[]
}
```

Offsets in `Definition` are half-open offsets into the original source string.
Paths are repository-relative with `/` separators. Lines/columns are one-based
and refer to original source, not normalized utility strings. Report failures
through diagnostics; never substitute empty authority for a failed config read.

### Stage 1: Token Authority and Narrow Definition Scopes

**Files:** Create token module and `drift-tokens.test.ts`.

**Interface:** `buildTokenAuthority({sources: Source[]}): Authority`.
It parses only known repository definition inputs; it never imports them.

- [ ] **1. Prepare the existing toolchain and write failing fixtures.** From `apps/`, run `bun install --frozen-lockfile` using the existing lockfile. Use Node 24 for commands. In each new test file use `// @vitest-environment node`. Add token fixtures proving semantic names, nested `DEFAULT` handling, and mixed-file behavior:

```ts
import { expect, it } from "vitest"
import { buildTokenAuthority } from "../../../scripts/design-system-drift-tokens.mjs"

it("keeps product CSS outside token declarations eligible", () => {
  const path = "apps/packages/ui/src/assets/tailwind-shared.css"
  const source = ":root { --color-danger: 224 88 109; } .notice { color: red; }"
  const result = buildTokenAuthority({ sources: [{ path, source }] })
  const productOffset = source.indexOf("color: red")
  expect(result.definitionScopes.some(scope =>
    scope.path === path && scope.start <= productOffset && productOffset < scope.end
  )).toBe(false)
})
```

- [ ] **2. Establish red.** From `apps/packages/ui/`, run `bunx vitest run src/design-system/__tests__/drift-tokens.test.ts --maxWorkers=1`. Confirm the missing module/export failure before production code.
- [ ] **3. Implement static authority extraction.** Use `ts.createSourceFile` and PostCSS; parse diagnostics are errors. For shared/WebUI/extension, read WebUI Tailwind `theme.extend.colors` and `borderRadius`, shared CSS token declarations, and shared `styles/tokens` exports. Pin extension's known static inheritance of `../tldw-frontend/tailwind.config.js`; do not execute `require`. For admin, read its Tailwind config and `app/globals.css`. Flatten nested color objects (`primary.DEFAULT` becomes `primary`; `primary.foreground` becomes `primary-foreground`). A configured raw palette entry is not a semantic token unless backed by an established semantic variable.

```js
import ts from "typescript"
import postcss from "postcss"
import defaultTheme from "tailwindcss/defaultTheme.js"

const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true)
const css = postcss.parse(source, { from: path })
const defaultRadius = defaultTheme.borderRadius
```

Use the relevant parser for the file extension, not both on every file. Build
the shipped radius scale from default radius data plus surface overrides; keep
runtime variable expressions symbolic. Do not assume `1rem` equals `16px` when
the source does not establish that conversion. Unresolved comparisons become
coverage diagnostics instead of definite off-scale claims.

Definition scopes must identify individual recognized custom-property
declarations, Tailwind definition properties, and known token exports. Shared
theme presets under `src/themes/presets.ts` and `primer-preset.ts` may identify
their named theme object declarations as definitions; ordinary components or
unrecognized exports in those files remain eligible. Do not whitelist every
custom property or every object in a theme directory.

- [ ] **4. Add controls and establish green.** Test mixed TypeScript, light/dark definitions, unsupported config expressions, missing required definition inputs, extension inheritance, admin-only semantic names, and allowed radius classes. Load the actual checked-in configs as a smoke fixture and assert no evaluation side effects. Run the Stage 1 command again.
- [ ] **5. Review and commit.** Update Stage 1 status and Backlog verification, inspect the diff, and commit as `feat(design-system): extract static drift token authority`.

### Stage 2: Styling Analysis

**Files:** Create rules module and `drift-rules.test.ts`.

**Interface:** `analyzeDriftSource({path: string, source: string, authority: Authority}): Analysis`.
Consume Stage 1 authority and retain original locations/nearest named owner.

- [ ] **1. Write failing behavior tests.** Include both JSX and CSS using actual supported styling locations:

```ts
const css = ":root { --color-danger: 224 88 109; } .notice { @apply bg-red-500 rounded-[13px]; color: red; }"
const result = analyzeDriftSource({ path: cssPath, source: css, authority })
expect(result.findings.map(f => f.rule)).toEqual([
  "palette-color", "custom-radius", "literal-color"
])
expect(result.findings.every(f => f.line === 1 && f.column > 0)).toBe(true)
```

Define `cssPath` as the shared stylesheet path and build `authority` from the
complete minimal definition fixture used in Stage 1. Add a JSX fixture with
`className="dark:hover:bg-red-500/20 sm:w-[44px]"` and `style={{color: "red"}}`.
Assert named-color coverage and that width is a `layout-review` candidate.

- [ ] **2. Establish red.** Run `bunx vitest run src/design-system/__tests__/drift-rules.test.ts --maxWorkers=1` from the package directory; verify the new expectations fail before implementing them.
- [ ] **3. Implement supported syntax analysis.** Use TypeScript AST nodes for `className`/`class` attributes, inline `style` objects, local immutable class maps and `cn`/`clsx`/`classNames`/`cva` static arguments. Follow local static references conservatively with a visited set; unknown expressions yield coverage diagnostics. Analyze branches independently and do not invent a joined class across interpolation gaps. Do not execute imports or attempt cross-module application evaluation.

Parse CSS declarations and `@apply` through PostCSS. Use one bracket/quote-aware
utility splitter for JSX and `@apply`, retaining offsets for variants, arbitrary
values, opacity modifiers, negative values, and important markers. Classify
color utilities only at color-bearing prefixes (`text`, `bg`, `border`, `ring`,
`outline`, `divide`, `placeholder`, `decoration`, `fill`, `stroke`, shadow color,
and gradient stops); do not mistake `text-[14px]` for a color.

Use the already declared jsdom CSS support for named color keyword recognition,
with one inert document per process and cached keyword decisions:

```js
import { JSDOM } from "jsdom"
const document = new JSDOM("").window.document
const colorProbe = document.createElement("span").style
colorProbe.color = ""
colorProbe.color = "rebeccapurple"
```

Only pass isolated candidate identifiers from color-valued syntax, not arbitrary
application strings or HTML. Allow `transparent`, `currentColor`, inheritance,
and CSS-wide keywords; inspect literal fallbacks inside otherwise semantic
variable expressions. Skip URL/string contents. Normalize property names for
React inline styles; numeric dimensional values follow React's px/unitless
semantics. Evaluate radius shorthands per value without guessing unresolved
variables or units. Do not propose a semantic color based solely on numeric
similarity. Known definitions are emitted separately, not suppressed by filename.

- [ ] **4. Establish green with negative controls.** Add semantic colors and `@apply`, shorthand named colors, unrelated payload strings, CSS URLs, text sizes, arbitrary selectors, escaped/quoted utilities, mixed token/product files, canonical-root components, and dynamic props. Verify TS/JS/JSX/TSX and CSS locations, definitions, diagnostics, and deterministic order. Run token and rule tests together.
- [ ] **5. Review and commit.** Record genuine red/green evidence and commit as `feat(design-system): report syntax-aware styling drift`.

### Stage 3: Reports and Intentional Exceptions

**Files:** Create report module, exception manifest, and `drift-report.test.ts`.

**Interfaces:**

```ts
assembleDriftReport({ analyses: Analysis[], exceptions: unknown,
  coverage: Coverage, provenance: Provenance, diagnostics: Diagnostic[] }): Report
formatDriftSummary(report: Report, options: {limit: number; reportPath: string}): string
```

`Report` contains `schemaVersion: 1`, `status: "complete" | "incomplete"`, all
`Provenance` fields, `coverage`, `findings`, `definitions`, `acceptedExceptions`,
`staleExceptions`, and `diagnostics`. An error diagnostic makes it incomplete;
coverage diagnostics remain visible without changing successful scan status.

- [ ] **1. Write red cases.** Start with no suppression baseline and an empty manifest. Verify findings remain visible and a stale exception does not remove them:

```ts
const report = assembleDriftReport({
  analyses: [analysis], exceptions: { schemaVersion: 1, exceptions: [] },
  coverage, provenance, diagnostics: []
})
expect(report.status).toBe("complete")
expect(report.findings).toEqual(analysis.findings)
expect(report.acceptedExceptions).toEqual([])
```

Use the concrete located finding returned by Stage 2 and clean provenance
`{sourceCommit: "1".repeat(40), dirtySource: false, untrackedSourceFileCount: 0}`.

- [ ] **2. Establish red.** Run `bunx vitest run src/design-system/__tests__/drift-report.test.ts --maxWorkers=1` and verify failure.
- [ ] **3. Implement report assembly and manifest validation.** Seed exactly:

```json
{"schemaVersion":1,"exceptions":[]}
```

Each exception has `path`, `rule`, `value`, `owner` (named owner or null),
`category`, and nonempty `reason`. Validate exact paths, known rules/categories,
and duplicate/overlapping records. Match path/rule/normalized value/owner exactly.
Group repeated matches with locations and `matchedCount`; keep them visible in
accepted exceptions. Unmatched records are stale. Ambiguous records produce
diagnostics and leave the affected findings unsuppressed. Invalid manifest shape
is an error diagnostic. Never interpret values as glob patterns.

Sort every report collection deterministically with codepoint ordering rather
than locale-dependent sorting. Summary sections separate token/radius findings,
layout candidates, accepted exceptions, stale entries, and diagnostics. Show
root counts, limitations, commit, dirty state, and omitted finding count when
the display limit is reached. Full JSON is never truncated.

- [ ] **4. Establish green.** Test duplicate locations, line-shift-stable exceptions, stale/ambiguous/invalid records, reversed input order, summary truncation, complete scans with findings, and incomplete scans that cannot look clean. Run Stage 3 tests and prior stages.
- [ ] **5. Review and commit.** Commit as `feat(design-system): assemble advisory drift reports and exceptions`.

### Stage 4: Whole-Repository CLI and Provenance

**Files:** Create CLI and `drift-cli.test.ts`; add the package script.

**Interface:** `runDriftReport({repoRoot: string, outputPath: string}): Promise<{report: Report, exitCode: number}>`.
Output failures reject with a concise error; command-line entry catches errors,
writes stderr, and exits 1. Imports must not trigger a scan.

CLI accepts only `--repo-root <path>` and `--output <path>`. Defaults are the
repository resolved from `import.meta.url` and the OS temp directory plus
`tldw-design-system-drift.json`. Explicit relative output is relative to repo root.
Resolve existing parent directories with `realpath` before validating output;
this allows platform aliases such as `/tmp`. Reject final-component symlinks and
canonical destinations inside the four scan roots or Git metadata, or naming any
tracked file or reporter input. Existing ordinary report files at valid
destinations may be replaced. Do not let output selection hide a source input or
overwrite repository content.
Invalid or repeated flags exit 1; no enforcement/baseline-update flag.

- [ ] **1. Write temporary-repository red tests.** Use `mkdtemp`, `execFile`, and a private Git fixture containing all four roots and minimal complete definitions. Create commits only inside that fixture, with test-local author configuration. Use staged/unstaged/untracked/ignored files and filenames containing spaces. Call the CLI through Node subprocesses so exit behavior is real.

```ts
const result = await runDriftReport({ repoRoot: fixtureRoot, outputPath })
expect(result.exitCode).toBe(0)
expect(result.report.findings.length).toBeGreaterThan(0)
expect(result.report.dirtySource).toBe(false)
```

- [ ] **2. Establish red.** Run `bunx vitest run src/design-system/__tests__/drift-cli.test.ts --maxWorkers=1` and verify the failure.
- [ ] **3. Implement inventory, provenance, and output.** Use Git subprocess argument arrays, never a shell:

```js
execFile("git", ["-C", repoRoot, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ...roots])
execFile("git", ["-C", repoRoot, "rev-parse", "--verify", "HEAD"])
execFile("git", ["-C", repoRoot, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ...roots, ...supportPaths])
```

Decode NUL-delimited filenames and rename records, deduplicate paths, and verify
each file/ancestor with `lstat` so tracked symlinks cannot lead outside roots.
Do not read missing deleted paths as source; include their deletion in dirty
provenance. Require every expected root and definition input to be present.

Exclude exact dependency/build/cache/coverage/test/fixture/tooling directories,
test/snapshot/config-tooling filenames, and known vendor assets, including
`apps/packages/ui/src/public/pdf.worker.min.mjs` and minified vendor bundles.
Do not exclude all `public` or `assets` code. Classify mixed files after parsing.
Read files in bounded batches of 16 and aggregate read/parse failures instead of
discarding successful findings. Inventory and read errors mean incomplete.

Build dirty provenance from relevant tracked changes and included untracked
files, plus the reporter modules, exception manifest, parser package manifest,
lockfile, and token/config inputs (`supportPaths` above). Filter changed paths
through the same source/exclusion rules; unrelated test/build files do not mark
the scan dirty. Validate the output destination before writing; valid output is
outside the analyzed roots and must not affect provenance. Report failed Git
commands as provenance errors with
nullable unavailable fields. The report output must not mark its own source dirty.

Write complete JSON with a trailing newline and print `formatDriftSummary` to
stdout. Set exit 0 for a completed scan regardless of finding/stale counts and
exit 1 for incomplete scans or output errors. No application startup or build.
Package command:

```json
"report:design-system-drift": "node scripts/verify-design-system-drift.mjs"
```

- [ ] **4. Establish green.** Verify all roots, one missing root, an empty root, unreadable/malformed source, untracked and ignored source, staged/unstaged deletions and renames, symlinks, unrelated changes, modified definition inputs, valid output replacement, rejected source/tracked/symlink output destinations, unknown flags, and output write failure. Run all four focused suites.
- [ ] **5. Inspect the actual repository report and commit.** Run the package command with `--output /tmp/tldw-design-system-drift.json`. Review findings from each rule/surface and inspect diagnostics; do not suppress all debt to obtain clean counts. Commit as `feat(design-system): add repository-wide advisory drift CLI`.

### Stage 5: Advisory CI and Usage Guidance

**Files:** Create `.github/workflows/ui-design-system-drift-report.yml`; modify
`tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`,
`Docs/Design/tldw_web_design_system_contract.md`, and tracking records.

**Consumes:** CLI and report contract from Stage 4. **Produces:** a nonrequired
`UI Design-System Drift Report` job, complete JSON artifact, and job summary.

- [ ] **1. Add workflow contract coverage first.** Extend the existing ordinary-workflow inventory test for the new filename and its exact direct-trigger digest. Add a YAML-structure test asserting the report job is advisory, honors license admission, checks out the PR head with `persist-credentials: false`, and uploads artifacts after scan failure. Verify the new expectations fail before adding the workflow. Do not weaken any existing inventory/security assertions.
- [ ] **2. Add the workflow using current admission conventions.** Trigger `pull_request` for `dev`/`main` on all four UI roots, `apps/bun.lock`, and this workflow; also allow `workflow_dispatch`. Use the same exact inert `admission` and `await_license` job shapes as the existing UI workflows, with the new workflow filename. Keep `workflow_run` absent. The report job needs those admission jobs and uses their existing allow/skip predicate; it has `continue-on-error: true`, a 20-minute timeout, read-only permissions, and no downstream required consumer.

Use the pinned checkout action already enforced by the CI contract, Node 24,
Bun 1.3.2, and `bun install --frozen-lockfile` from `apps/`. Checkout the PR head
for PR events and selected `github.sha` for manual dispatch. Set
`SKIP_WXT_PREPARE: '1'` as existing frontend CI does. Reporting steps:

```yaml
- name: Generate drift report
  id: drift
  working-directory: apps/packages/ui
  continue-on-error: true
  run: bun run report:design-system-drift --output "$RUNNER_TEMP/tldw-design-system-drift.json"
- name: Upload drift report
  if: always()
  uses: actions/upload-artifact@v7
  with:
    name: ui-design-system-drift-${{ github.event.pull_request.head.sha || github.sha }}
    path: ${{ runner.temp }}/tldw-design-system-drift.json
    if-no-files-found: warn
```

Publish a separate `if: always()` summary step. Read JSON through Node and append
`formatDriftSummary` to `GITHUB_STEP_SUMMARY`; when JSON is missing, state that the
scan did not produce a report and direct readers to failed setup/scan steps.
After artifact/summary steps, fail the advisory job on a scan step with outcome
`failure`, retaining job-level nonblocking behavior. Do not turn setup failures
into a clean report. No PR comments, write tokens, queue changes, or new required
aggregate dependencies.

The ordinary-workflow test freezes all direct-trigger digests. Compute only the
new digest with its existing `_trigger` and sorted compact JSON serialization;
leave existing digests unchanged. No license-first path-registry entry is needed
for this direct PR/manual-only workflow because it does not use `workflow_run`.

- [ ] **3. Establish green and document use.** Run the focused reporter tests, the two existing license-first test modules, and actionlint on the new workflow. Document the package command, temp output, four roots, metadata, limitations, exceptions, zero blocking, and migration review procedure in the main contract. Link the spec/ADR; describe the reporter as implemented only after the command and tests pass.
- [ ] **4. Run the final checks and record results.** Commands from the package directory:

```sh
bunx vitest run src/design-system/__tests__/drift-tokens.test.ts src/design-system/__tests__/drift-rules.test.ts src/design-system/__tests__/drift-report.test.ts src/design-system/__tests__/drift-cli.test.ts src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
bun run report:design-system-drift --output /tmp/tldw-design-system-drift.json
```

From repository root, activate the project environment before Python:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/CI/test_license_first_admission.py -q
python -m bandit tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -f json -o /tmp/bandit_design_system_drift_reporting.json
git diff --check
```

Bandit covers the touched Python CI test, not JavaScript. Run scoped existing
JavaScript lint/syntax checks; do not add a new linter. If the real scan is
incomplete, record and fix scanner defects before publishing it as usable;
legitimate syntax limits must appear honestly in coverage diagnostics. Review
one finding per nonempty rule/surface group and record samples, not arbitrary
repo-wide expected counts. Runtime/browser builds are unnecessary for an
advisory development script with no rendered UI changes.

- [ ] **5. Review, commit, and publish the implementation PR.** Verify the required workflow files, product-state rules/baseline, tokens, and product components were not changed. Record the ADR assessment, source SHA, report status, sample review, tests, Bandit, known gaps, and PR URL in Backlog. Commit as `ci(design-system): publish nonblocking UI drift reports`. Open a draft PR against `dev` with a concise change rationale and verification. The owner must supply their own human-written Change summary before merge; do not fabricate it. Update issue #1673/epic #1655 with the PR and actual stage progress, keeping delivery open until review/merge criteria are met.

## Plan Verification and Handoff

This plan covers the reviewed spec's source boundary, narrow definition scopes,
surface-specific tokens, all four rules, named CSS colors, `@apply`, visible
exceptions/debt, local provenance, exit behavior, advisory CI, and usage docs.
No UI migration or enforcement rollout is included.

Before execution, create or reuse one associated implementation Backlog task
using backlog-py and the repository's cross-branch ID allocation rule. Link this
plan and the design task rather than duplicating their text. Do not mark the
implementation complete merely because the planning documents are committed.

At planning time only declared parser/tool versions and CI contracts were read;
the fresh worktree has no installed JS dependencies. The stage commands above
are execution requirements, not claims of passing tests.
