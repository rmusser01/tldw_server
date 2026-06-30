# WebUI Dependency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first reviewable artifact for issue #1346: a WebUI dependency audit that ranks safe package-trimming follow-up work without changing manifests or runtime code.

**Architecture:** This is a docs-first audit slice. It uses the approved dependency trimming design as the policy source, reads the WebUI/shared UI manifests and lockfile as source data, checks extension impact for shared `@tldw/ui` dependencies, and records decisions in one durable audit document. Package removal and `axios` replacement are explicitly later tasks.

**Tech Stack:** Markdown, Backlog.md, Bun workspace manifests, `rg`, Node.js one-off inventory scripts, `git diff --check`.

---

## Source Context

- Spec: `Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md`
- Backlog task: `TASK-104`
- GitHub issue: `https://github.com/rmusser01/tldw_server/issues/1346`
- Primary manifests:
  - `apps/tldw-frontend/package.json`
  - `apps/packages/ui/package.json`
  - `apps/bun.lock`
- Extension impact-check manifest:
  - `apps/extension/package.json`

## Scope Boundaries

Do not edit package manifests, lockfiles, runtime TypeScript, React components, or tests in this audit slice. The only intended repository changes are:

- create `Docs/Design/WebUI_Dependency_Audit.md`
- update `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`

If the audit reveals that a package can be removed immediately, record it as `remove-now` and create or note the follow-up task. Do not remove it in this slice.

## File Structure

- Create `Docs/Design/WebUI_Dependency_Audit.md`
  - Owns the dependency table, methodology, decision rules, ranked follow-up queue, verification log, and known skips.
- Modify `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`
  - Records plan link, progress notes, verification, acceptance-criteria checks, and final summary.

No helper script should be committed unless the audit becomes too large to reproduce with documented commands. Prefer temporary `/tmp` inventory files and paste the final summarized results into the audit document.

## Decision Rules

Use the decision values from the spec:

- `keep`: justified current dependency.
- `remove-now`: no source/test/script/config usage, or trivially replaceable usage.
- `replace-later`: useful replacement candidate that needs a dedicated PR.
- `defer-design`: complex or risky enough to need its own design.
- `investigate-lockfile`: no obvious usage, but ownership or transitive/direct status needs confirmation.

Guardrail defaults:

- Mark `dompurify` and other security-sensitive packages `keep` or `defer-design`.
- Mark document readers, rich text editors, graph/rendering tools, OCR, Monaco, Mermaid, KaTeX, tokenizers, and schema validation `keep` or `defer-design`.
- Treat icon-stack consolidation as future dependency reduction, not platform-native replacement.
- Treat shared `@tldw/ui` packages as affecting the extension until proven otherwise.

## Task 1: Create The Audit Scaffold

**Files:**
- Create: `Docs/Design/WebUI_Dependency_Audit.md`
- Modify: `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`

- [ ] **Step 1: Add the audit document scaffold**

Create `Docs/Design/WebUI_Dependency_Audit.md` with this initial structure:

```markdown
# WebUI Dependency Audit

Date: 2026-05-07
Status: Draft audit for issue #1346

## References

- GitHub issue: https://github.com/rmusser01/tldw_server/issues/1346
- Design spec: ../superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
- Backlog task: TASK-104

## Scope

This audit covers direct package declarations and usage signals for:

- `apps/tldw-frontend/package.json`
- `apps/packages/ui/package.json`
- `apps/bun.lock`
- `apps/extension/package.json` as an impact-check surface for shared UI candidates

This audit does not remove packages or rewrite runtime code.

## Methodology

1. Read direct dependency declarations from the WebUI, shared UI, and extension manifests.
2. Scan source, test, script, and config files for import/config usage.
3. Classify direct dependencies with the approved decision values.
4. Rank follow-up work into quick cleanup, replacement, deferred design, and keep groups.
5. Record verification commands and known skips.

## Decision Legend

| Decision | Meaning |
| --- | --- |
| `keep` | Current dependency is justified. |
| `remove-now` | Candidate for a narrow package-removal PR. |
| `replace-later` | Replacement is plausible but needs its own PR. |
| `defer-design` | Needs a separate design before replacement. |
| `investigate-lockfile` | Needs lockfile or ownership confirmation before action. |

## Dependency Inventory

| Package | Declared locations | Import count | Representative sites | Consumer surface | Category | Decision | Risk | Expected impact | Follow-up slice |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |

## Ranked Follow-Up Queue

### Quick Cleanup Candidates

### Replacement Candidates

### Deferred Design Candidates

### Explicit Keeps

## Verification

## Known Skips And Blockers
```

- [ ] **Step 2: Record the plan link in Backlog**

Run:

```bash
backlog task edit TASK-104 --doc Docs/superpowers/plans/2026-05-07-webui-dependency-audit-implementation-plan.md --append-notes "Started dependency audit implementation plan. First slice is docs-only audit artifact; package removals and axios replacement are follow-up work." --plain
```

If the command replaces the existing spec documentation link, immediately restore
both docs with:

```bash
backlog task edit TASK-104 --doc Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md --doc Docs/superpowers/plans/2026-05-07-webui-dependency-audit-implementation-plan.md --plain
```

Expected: `TASK-104` shows both the spec and plan paths in Documentation and has
the note appended.

- [ ] **Step 3: Verify scaffold sections exist**

Run:

```bash
rg -n "Dependency Inventory|Ranked Follow-Up Queue|Verification|Known Skips" Docs/Design/WebUI_Dependency_Audit.md
```

Expected: all four section names are printed.

- [ ] **Step 4: Commit the scaffold**

Run:

```bash
git add Docs/Design/WebUI_Dependency_Audit.md "backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md"
git commit -m "docs: scaffold webui dependency audit"
```

Expected: commit succeeds with only the audit doc and `TASK-104` changes.

## Task 2: Generate The Dependency Declaration Inventory

**Files:**
- Modify: `Docs/Design/WebUI_Dependency_Audit.md`
- Modify: `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`

- [ ] **Step 1: Generate a direct declaration list**

Run from the repository root:

```bash
node <<'NODE'
const fs = require("node:fs")
const manifests = [
  ["web", "apps/tldw-frontend/package.json"],
  ["shared-ui", "apps/packages/ui/package.json"],
  ["extension", "apps/extension/package.json"],
]
const sections = ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]
const rows = new Map()

for (const [surface, file] of manifests) {
  const pkg = JSON.parse(fs.readFileSync(file, "utf8"))
  for (const section of sections) {
    for (const name of Object.keys(pkg[section] || {})) {
      const row = rows.get(name) || { name, declarations: [] }
      row.declarations.push(`${surface}:${section}`)
      rows.set(name, row)
    }
  }
}

const sorted = [...rows.values()].sort((a, b) => a.name.localeCompare(b.name))
fs.writeFileSync("/tmp/tldw-webui-dependency-declarations.json", JSON.stringify(sorted, null, 2))
console.log(`Wrote ${sorted.length} dependency declarations to /tmp/tldw-webui-dependency-declarations.json`)
NODE
```

Expected: command writes `/tmp/tldw-webui-dependency-declarations.json` and prints a non-zero dependency count.

- [ ] **Step 2: Inspect the declaration list**

Run:

```bash
node -e 'const rows=require("/tmp/tldw-webui-dependency-declarations.json"); console.log(rows.slice(0, 20).map((row) => `${row.name}\t${row.declarations.join(", ")}`).join("\n"))'
```

Expected: first 20 package rows print with declaration locations.

- [ ] **Step 3: Populate declared locations in the audit table**

Use `/tmp/tldw-webui-dependency-declarations.json` to populate the `Package` and `Declared locations` columns in `Docs/Design/WebUI_Dependency_Audit.md`.

For packages declared only by `apps/extension/package.json`, include them only when they also relate to `@tldw/ui` impact or overlap with WebUI/shared UI packages. The extension is an impact-check surface, not the primary audit target.

- [ ] **Step 4: Add a methodology note with the exact command**

Add the declaration-generation command summary under `## Verification` so reviewers can reproduce how the declaration list was built.

- [ ] **Step 5: Commit the manifest inventory**

Run:

```bash
git add Docs/Design/WebUI_Dependency_Audit.md "backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md"
git commit -m "docs: inventory webui dependency declarations"
```

Expected: commit succeeds with audit and Backlog task changes only.

## Task 3: Scan Usage And Classify Decisions

**Files:**
- Modify: `Docs/Design/WebUI_Dependency_Audit.md`
- Modify: `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`

- [ ] **Step 1: Generate usage signals**

Run from the repository root:

```bash
node <<'NODE'
const fs = require("node:fs")
const path = require("node:path")
const deps = JSON.parse(fs.readFileSync("/tmp/tldw-webui-dependency-declarations.json", "utf8"))
const roots = ["apps/tldw-frontend", "apps/packages/ui", "apps/extension"]
const allowedExts = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".css"])
const ignoredParts = new Set(["node_modules", ".next", ".output", "build", "dist", "coverage", "test-results"])
const ignoredFiles = new Set([
  "apps/bun.lock",
  "apps/extension/package.json",
  "apps/packages/ui/package.json",
  "apps/packages/ui/src/public/pdf.worker.min.mjs",
  "apps/tldw-frontend/package.json",
])

function collect(dir) {
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    const rel = full.replaceAll(path.sep, "/")
    if (ignoredParts.has(entry.name)) continue
    if (ignoredFiles.has(rel)) continue
    if (entry.isDirectory()) {
      out.push(...collect(full))
      continue
    }
    if (allowedExts.has(path.extname(entry.name))) out.push(rel)
  }
  return out
}

const files = roots.flatMap(collect)
const rows = deps.map((dep) => {
  const sites = []
  for (const file of files) {
    const text = fs.readFileSync(file, "utf8")
    if (text.includes(dep.name)) sites.push(file)
    if (sites.length >= 8) break
  }
  return { ...dep, importCount: sites.length, representativeSites: sites }
})

fs.writeFileSync("/tmp/tldw-webui-dependency-usage.json", JSON.stringify(rows, null, 2))
console.log(`Scanned ${files.length} files and wrote /tmp/tldw-webui-dependency-usage.json`)
NODE
```

Expected: command writes `/tmp/tldw-webui-dependency-usage.json`.

This signal intentionally excludes manifests, lockfiles, generated PDF worker
output, markdown docs, and JSON locale/data files. Those files are useful audit
context but should not be counted as runtime/test/script/config usage.

- [ ] **Step 2: Inspect likely quick cleanup candidates**

Run:

```bash
node -e 'const rows=require("/tmp/tldw-webui-dependency-usage.json"); for (const name of ["pubsub-js","buffer","stream-browserify","clsx","axios"]) { const row=rows.find((item)=>item.name===name); console.log(name, row?.declarations.join(", "), row?.representativeSites || []); }'
```

Expected: the output shows whether each candidate has source/test/script/config usage and where.

- [ ] **Step 3: Fill usage columns**

For each audited package, fill:

- `Import count`
- `Representative sites`
- `Consumer surface`
- `Category`

Use `0` only when the package has no source/test/script/config usage under the scanned roots after excluding generated/vendor artifacts.

- [ ] **Step 4: Fill decisions and rationale**

Apply the decision rules:

- Mark unused direct declarations as `remove-now` or `investigate-lockfile`.
- Mark `axios` as `replace-later`.
- Mark `clsx` as `remove-now` only if compatibility is mechanical; otherwise `replace-later`.
- Mark document/rendering/editor/security/parser packages as `keep` or `defer-design`.
- Mark icon-stack work as `defer-design`.

Add rationale in `Risk`, `Expected impact`, and `Follow-up slice`.

- [ ] **Step 5: Commit usage classification**

Run:

```bash
git add Docs/Design/WebUI_Dependency_Audit.md "backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md"
git commit -m "docs: classify webui dependency usage"
```

Expected: commit succeeds with audit and Backlog task changes only.

## Task 4: Rank Follow-Up Work And Verify The Audit

**Files:**
- Modify: `Docs/Design/WebUI_Dependency_Audit.md`
- Modify: `backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md`

- [ ] **Step 1: Rank quick cleanup candidates**

Under `### Quick Cleanup Candidates`, list packages in the order they should be attempted. Include why each is safe enough for a small PR and what verification should run.

Expected starting candidates:

- `pubsub-js`
- `buffer`
- `stream-browserify`
- `clsx` only if compatibility stays mechanical

- [ ] **Step 2: Rank replacement candidates**

Under `### Replacement Candidates`, include:

- `axios`: first substantial replacement after quick cleanup.
- Any simple `dayjs` call-site subset only if the audit finds clear native `Intl`/`Date` replacements.

Do not promote complex rendering/parser packages into this queue.

- [ ] **Step 3: Record deferred and kept packages**

Under `### Deferred Design Candidates` and `### Explicit Keeps`, record complex or security-sensitive packages with short rationale. Include at least:

- `dompurify`
- PDF/ePub/document packages
- rich text editor packages
- Mermaid/KaTeX/markdown rendering stack
- graph/layout packages
- OCR/tokenizer/schema packages
- icon-stack consolidation

- [ ] **Step 4: Run final verification**

Run:

```bash
rg -n "pubsub-js|buffer|stream-browserify|clsx|axios|dompurify|defer-design|remove-now|replace-later" Docs/Design/WebUI_Dependency_Audit.md
git diff --check
git status --short
```

Expected:

- `rg` prints all major candidate names and decision values from the audit.
- `git diff --check` exits 0.
- `git status --short` shows only the audit doc and `TASK-104` before the final commit.

- [ ] **Step 5: Update TASK-104**

Run:

```bash
backlog task edit TASK-104 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --status Done --append-notes "Created WebUI dependency audit artifact. Verification: documented inventory commands, rg audit check, git diff --check. Bandit skipped because this audit slice is docs/backlog only and touches no Python code." --final-summary "Created Docs/Design/WebUI_Dependency_Audit.md for issue #1346. The audit classifies direct WebUI/shared UI dependencies, includes extension impact checks for shared UI candidates, ranks quick cleanup and axios replacement follow-ups, and records docs-only verification." --plain
```

Expected: all acceptance criteria and Definition of Done items are checked, task
status is `Done`, and final summary is populated.

- [ ] **Step 6: Commit the completed audit**

Run:

```bash
git add Docs/Design/WebUI_Dependency_Audit.md "backlog/tasks/task-104 - Create-WebUI-dependency-audit-for-issue-1346.md"
git commit -m "docs: audit webui dependencies"
```

Expected: commit succeeds with the final audit artifact and Backlog task update.

## Final Verification

Before reporting completion, run:

```bash
git diff --check HEAD~1..HEAD
git status --short
```

Expected:

- `git diff --check HEAD~1..HEAD` exits 0.
- `git status --short` is clean.

Bandit is intentionally skipped for this audit task because it is docs/backlog only and does not touch Python code.

## Handoff Notes

After this audit lands, create follow-up Backlog tasks from the ranked queue:

1. Quick cleanup package-removal task.
2. `axios` fetch replacement task.
3. Optional icon-stack consolidation design task if the audit shows enough payoff.
4. Optional native-date/time replacement task if simple `dayjs` call sites are clearly separable.
