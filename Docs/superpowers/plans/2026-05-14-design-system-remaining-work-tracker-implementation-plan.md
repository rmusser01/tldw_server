# Design-System Remaining Work Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the durable GitHub epic, linked GitHub sub-issues, and mirrored Backlog.md task tree for the remaining tldw WebUI and extension design-system migration program.

**Architecture:** Use a draft-first workflow. Generate local issue body drafts from the approved spec and current `origin/dev` baseline, get human approval, then create public GitHub issues and Backlog mirror tasks, and finally cross-link both systems. GitHub owns current tracker state; Backlog owns implementation notes and PR evidence.

**Tech Stack:** Markdown, GitHub CLI (`gh`), Backlog.md CLI, Node.js for JSON baseline grouping, Git.

---

## Required Inputs

- Spec: `Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md`
- Contract: `Docs/Design/tldw_web_design_system_contract.md`
- Inventory: `Docs/Design/tldw_web_design_system_inventory.md`
- Baseline: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
- Repository: `rmusser01/tldw_server`

## Files and Artifacts

- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/README.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/github-epic.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-chat-playground.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-ingestion-library-media.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-jobs-scheduler-watchlists.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-mcp-acp.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-evaluations.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-settings-account-security.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-admin-health-expansion.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-prompt-prompt-studio.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-flashcards-quiz-study.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-document-workspace.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-character-persona-presentation.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-writing-review.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-long-tail.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-baseline-reporting.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-ci-gate-tightening.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-token-color-radius-layout-guards.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-component-ownership.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-component-docs-examples.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/governance-visual-qa-checklist.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-pr-body.md`
- Modify: Backlog tasks created by the Backlog CLI during tracker creation.
- Modify: GitHub issues created and edited by `gh issue create` and `gh issue edit`.

## Issue Titles

Use these exact titles when creating GitHub issues and Backlog mirror tasks.

| Slug | Title |
| --- | --- |
| `epic` | Epic: Complete tldw WebUI and extension design-system migration |
| `migration-chat-playground` | Migrate design-system product state: Chat and Playground |
| `migration-ingestion-library-media` | Migrate design-system product state: Ingestion, Library, and media |
| `migration-jobs-scheduler-watchlists` | Migrate design-system product state: Jobs, Scheduler, and Watchlists |
| `migration-mcp-acp` | Migrate design-system product state: MCP and ACP |
| `migration-evaluations` | Migrate design-system product state: Evaluations |
| `migration-settings-account-security` | Migrate design-system product state: Settings and account/security |
| `migration-admin-health-expansion` | Migrate design-system product state: Admin and health expansion |
| `migration-prompt-prompt-studio` | Migrate design-system product state: Prompt and Prompt Studio |
| `migration-flashcards-quiz-study` | Migrate design-system product state: Flashcards, Quiz, and study flows |
| `migration-document-workspace` | Migrate design-system product state: Document and Workspace surfaces |
| `migration-character-persona-presentation` | Migrate design-system product state: Character, Persona, and presentation surfaces |
| `migration-writing-review` | Migrate design-system product state: Writing and Review surfaces |
| `migration-long-tail` | Migrate design-system product state: Other shared surfaces and long-tail triage |
| `governance-baseline-reporting` | Harden design-system baseline reporting and stale-entry cleanup |
| `governance-ci-gate-tightening` | Define design-system CI gate tightening path |
| `governance-token-color-radius-layout-guards` | Design token, color, radius, and layout drift guards |
| `governance-component-ownership` | Define shared design-system component ownership plan |
| `governance-component-docs-examples` | Add shared design-system component documentation and examples |
| `governance-visual-qa-checklist` | Add Browser/WebUI/extension visual QA checklist |

## Task 1: Refresh Baseline Snapshot and Generate Draft Issue Bodies

**Files:**
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/*.md`

- [ ] **Step 1: Confirm clean worktree and current base**

Run:

```bash
git status --short --branch
git fetch origin dev
git rev-parse --short origin/dev
```

Expected: worktree is clean except planned edits. `origin/dev` is reachable.

- [ ] **Step 2: Generate the current grouped baseline summary**

Run:

```bash
node - <<'NODE'
const fs = require("fs")
const baselinePath = "apps/packages/ui/scripts/design-system-product-state-baseline.json"
const entries = JSON.parse(fs.readFileSync(baselinePath, "utf8"))
const categories = [
  ["Chat and Playground", /src\/components\/(Option\/Playground|Common\/Playground|Sidepanel\/Chat)|src\/routes\/sidepanel-chat/],
  ["Ingestion, Library, and media", /src\/components\/(Option\/(Ingestion|Library|Media|Sources|DataTables|AudiobookStudio|ChunkingPlayground)|Common\/QuickIngest|Timeline)/],
  ["Jobs, Scheduler, and Watchlists", /src\/components\/Option\/(Watchlists|AgentTasks)|src\/components\/Common\/Workflow/],
  ["MCP and ACP", /src\/components\/Option\/(MCPHub|ACPPlayground|WorkspacePlayground)/],
  ["Evaluations", /src\/components\/Option\/Evaluations/],
  ["Settings and account\/security", /src\/components\/Option\/(Settings|Setup|Integrations|TTS)/],
  ["Admin and health expansion", /src\/components\/Option\/Admin/],
  ["Prompt and Prompt Studio", /src\/components\/Option\/(Prompt|PromptStudio)/],
  ["Flashcards, Quiz, and study flows", /src\/components\/(Flashcards|Quiz|StudySuggestions)/],
  ["Document and Workspace surfaces", /src\/components\/DocumentWorkspace|src\/components\/Option\/Workspace/],
  ["Character, Persona, and presentation surfaces", /src\/components\/(Option\/(Characters|PresentationStudio)|PersonaGarden)/],
  ["Writing and Review surfaces", /src\/components\/(Option\/WritingPlayground|Review)/],
  ["Other shared surfaces and long-tail triage", /.*/]
]
const out = new Map(categories.map(([name]) => [name, { total: 0, rules: {}, paths: {} }]))
for (const entry of entries) {
  const path = entry.path || entry.file || ""
  const category = categories.find(([, regex]) => regex.test(path))[0]
  const record = out.get(category)
  record.total += 1
  record.rules[entry.rule] = (record.rules[entry.rule] || 0) + 1
  const group = path.split("/").slice(0, 4).join("/")
  record.paths[group] = (record.paths[group] || 0) + 1
}
for (const [name, record] of out) {
  record.topPaths = Object.entries(record.paths).sort((a, b) => b[1] - a[1]).slice(0, 8)
  delete record.paths
}
console.log(JSON.stringify({
  generatedAt: new Date().toISOString(),
  total: entries.length,
  byRule: entries.reduce((acc, entry) => {
    acc[entry.rule] = (acc[entry.rule] || 0) + 1
    return acc
  }, {}),
  categories: Object.fromEntries(out)
}, null, 2))
NODE
```

Expected: total matches the current baseline. If it differs from the spec snapshot, use the fresh count in issue bodies and note the drift in `README.md`.

- [ ] **Step 3: Create the draft issue body directory**

Run:

```bash
mkdir -p Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies
```

Expected: directory exists.

- [ ] **Step 4: Write `README.md` for the draft bodies**

Create `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/README.md` with:

```markdown
# Design-System Remaining Work Tracker Issue Bodies

These are draft GitHub issue bodies generated from:

- `Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md`
- `apps/packages/ui/scripts/design-system-product-state-baseline.json`

Human review is required before creating or updating public GitHub issues.

## Creation Order

1. Create labels if missing.
2. Create the epic from `github-epic.md`.
3. Create the Backlog parent.
4. Create migration and governance issues.
5. Create Backlog child tasks.
6. Update all GitHub issue bodies with Backlog links.
7. Update the epic dashboard with issue links and Backlog links.
```

- [ ] **Step 5: Write `github-epic.md`**

Create `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/github-epic.md` using the epic template from the spec. Include:

- Purpose
- Fresh baseline snapshot
- Migration dashboard table with all product areas
- Governance dashboard table
- Operating rules
- References

Use `TBD` for issue URLs and Backlog IDs during the draft step.

- [ ] **Step 6: Write product-area draft issue bodies**

Create one `migration-*.md` file per product area listed in the Files section. Each file must include:

```markdown
## Scope

Owned paths and product surfaces from the ordered path ownership map.

## Current Baseline Debt

Baseline source: apps/packages/ui/scripts/design-system-product-state-baseline.json
Snapshot date: YYYY-MM-DD

- Total:
- antd-product-state-import:
- canonical-state-label:

## Done Criteria

- This area has zero current product-state baseline exceptions.
- Focused tests cover migrated behavior.
- `bun run verify:design-system-state` passes from `apps/packages/ui`.
- `git diff --check` passes.
- Touched-file TypeScript filtering reports no diagnostics, or unrelated baseline diagnostics are documented.
- Bandit is run for Python touches or explicitly skipped for UI-only work.

## Tracking

- Parent epic: TBD
- Backlog task: TBD
- PRs:

## Notes

- Keep AntD where it is only mechanics.
- Migrate product state language to shared primitives or the state registry.
- Split implementation into reviewable PRs when the area is too broad.
```

- [ ] **Step 7: Write governance draft issue bodies**

Create one `governance-*.md` file per governance track listed in the Files section. Each file must include:

```markdown
## Purpose

What governance risk this track reduces.

## Scope

Included guard, doc, CI, or ownership decision.

## Non-Goals

What this track explicitly does not migrate.

## Done Criteria

- Durable artifact exists and is linked from the epic.
- Verification or review path is documented.
- Follow-up migration tasks know how to use the artifact.

## Tracking

- Parent epic: TBD
- Backlog task: TBD
- PRs:
```

- [ ] **Step 8: Verify draft body files exist**

Run:

```bash
find Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies -type f | sort
git diff --check
```

Expected: all draft body files are present and whitespace check passes.

- [ ] **Step 9: Commit draft issue bodies**

Run:

```bash
git add Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies
git commit -m "Draft design-system tracker issue bodies"
```

Expected: one commit containing only local draft issue body artifacts.

## Task 2: Human Review Gate and Label Preparation

**Files:**
- Read: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/*.md`

- [ ] **Step 1: Ask for human approval before public issue creation**

Report the draft body directory and ask:

```text
Issue body drafts are ready at Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies. Please approve before I create public GitHub issues.
```

Expected: user explicitly approves public GitHub issue creation.

- [ ] **Step 2: Search for existing tracker issues**

Run:

```bash
gh issue list \
  --repo rmusser01/tldw_server \
  --state all \
  --search '"design-system" OR "product-state"' \
  --json number,state,title,url \
  --limit 100
```

Expected: no existing open issue already owns the same epic or product-area tracker purpose. If a duplicate exists, stop and ask whether to update it instead of creating a new tracker.

- [ ] **Step 3: Check existing labels**

Run:

```bash
gh label list --repo rmusser01/tldw_server --limit 200 --json name --jq '.[].name'
```

Expected: label list prints. Existing labels include `WebUI` and `enhancement`.

- [ ] **Step 4: Create missing tracker labels without overwriting existing labels**

Run each command only if the label is missing:

```bash
gh label create design-system --repo rmusser01/tldw_server --description "Shared UI design-system migration and governance" --color 5319E7
gh label create product-state --repo rmusser01/tldw_server --description "Product-state design-system migration debt" --color 0E8A16
gh label create governance --repo rmusser01/tldw_server --description "Design-system guard, CI, documentation, and ownership governance" --color FBCA04
```

Expected: missing labels are created, existing labels are not modified.

## Task 3: Create GitHub Epic and Initial Issue Map

**Files:**
- Read: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/github-epic.md`
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md`

- [ ] **Step 1: Create the epic issue**

Run:

```bash
EPIC_URL=$(gh issue create \
  --repo rmusser01/tldw_server \
  --title "Epic: Complete tldw WebUI and extension design-system migration" \
  --label WebUI \
  --label enhancement \
  --label design-system \
  --body-file Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/github-epic.md)
printf 'Epic: %s\n' "$EPIC_URL"
```

Expected: command prints a GitHub issue URL.

- [ ] **Step 2: Record the initial GitHub issue map**

Create `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md` with:

```markdown
# Design-System Remaining Work Tracker Issue Map

## GitHub Issues

| Slug | URL |
| --- | --- |
| epic | <url> |

## Backlog Tasks

| Slug | Task ID | File |
| --- | --- | --- |
```

Fill in the epic issue URL from the create command. Leave child issue and Backlog rows absent until Task 4 creates them.

- [ ] **Step 3: Commit the initial issue map**

Run:

```bash
git add Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md
git commit -m "Record design-system tracker issue map"
```

Expected: one commit containing the issue map.

## Task 4: Create Backlog Parent and Child Tracker Records

**Files:**
- Read: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/*.md`
- Modify: Backlog task files created by Backlog CLI.
- Modify: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md`

Backlog note: prefer the Backlog MCP `task_create` tool when available. The CLI commands below are the exact fallback-compatible arguments to mirror through MCP fields.

- [ ] **Step 1: Create the Backlog parent task**

Run:

```bash
backlog task create "Track remaining tldw design-system migration and governance" \
  --status "To Do" \
  --priority medium \
  --labels design-system,webui,extension \
  --parent TASK-45 \
  --ref "<epic-url>" \
  --ref Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md \
  --ref apps/packages/ui/scripts/design-system-product-state-baseline.json \
  --description "Mirror the GitHub epic for the remaining tldw WebUI and extension design-system migration and governance program. GitHub owns current counts and public status; Backlog owns execution notes, verification evidence, and PR history." \
  --ac "All product-area and governance GitHub issues are linked from this parent." \
  --ac "Backlog child tasks exist for every GitHub sub-issue." \
  --ac "The parent records the baseline snapshot and source-of-truth rules from the approved spec." \
  --plain
```

Expected: Backlog prints the parent task ID. Record it in the issue map.

- [ ] **Step 2: Create product-area GitHub issues and Backlog child tasks**

For each `migration-*.md` file, create the GitHub issue first:

```bash
ISSUE_URL=$(gh issue create \
  --repo rmusser01/tldw_server \
  --title "<exact title from the Issue Titles table>" \
  --label WebUI \
  --label enhancement \
  --label design-system \
  --label product-state \
  --body-file "<path-to-migration-body>")
printf '<slug>: %s\n' "$ISSUE_URL"
```

Then create the matching Backlog child task:

```bash
backlog task create "<same title as GitHub issue>" \
  --status "To Do" \
  --priority medium \
  --labels design-system,webui,extension,product-state \
  --parent "<tracker-parent-task-id>" \
  --ref "<product-area-github-issue-url>" \
  --ref Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md \
  --ref apps/packages/ui/scripts/design-system-product-state-baseline.json \
  --description "Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue." \
  --ac "The linked GitHub issue owns current count and public status." \
  --ac "Implementation PR tasks are created under this child when the area is too broad for one PR." \
  --ac "Backlog notes record PR links and before/after count evidence." \
  --plain
```

Expected: one GitHub issue URL and one Backlog child task ID per product-area slug.

- [ ] **Step 3: Create governance GitHub issues and Backlog child tasks**

For each `governance-*.md` file, create the GitHub issue first:

```bash
ISSUE_URL=$(gh issue create \
  --repo rmusser01/tldw_server \
  --title "<exact title from the Issue Titles table>" \
  --label WebUI \
  --label enhancement \
  --label design-system \
  --label governance \
  --body-file "<path-to-governance-body>")
printf '<slug>: %s\n' "$ISSUE_URL"
```

Then create the matching Backlog child task:

```bash
backlog task create "<same title as GitHub issue>" \
  --status "To Do" \
  --priority medium \
  --labels design-system,webui,extension,governance \
  --parent "<tracker-parent-task-id>" \
  --ref "<governance-github-issue-url>" \
  --ref Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md \
  --description "Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue." \
  --ac "The linked GitHub issue owns public status." \
  --ac "Backlog notes record PR links and verification evidence." \
  --plain
```

Expected: one GitHub issue URL and one Backlog child task ID per governance slug.

- [ ] **Step 4: Update issue map with child issue URLs and Backlog task IDs**

Edit `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md` and fill in every child GitHub issue URL, Backlog task ID, and Backlog task file.

- [ ] **Step 5: Commit Backlog mirror and issue-map updates**

Run:

```bash
git add backlog/tasks Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md
git commit -m "Mirror design-system tracker in Backlog"
```

Expected: one commit containing Backlog parent/child task files and the updated issue map.

## Task 5: Cross-Link GitHub Issues and Backlog Tasks

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/*.md`
- Modify: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md`

- [ ] **Step 1: Update local issue body drafts with final links**

Edit every issue body file:

- Replace `Parent epic: TBD` with the epic URL.
- Replace `Backlog task: TBD` with the task ID and task path.
- Add PR table placeholders where needed.
- Update `github-epic.md` dashboard rows with issue URLs, current counts, and Backlog task IDs.

- [ ] **Step 2: Push updated bodies to GitHub**

Run for the epic:

```bash
gh issue edit "<epic-url-or-number>" \
  --repo rmusser01/tldw_server \
  --body-file Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/github-epic.md
```

Run for each child issue:

```bash
gh issue edit "<issue-url-or-number>" \
  --repo rmusser01/tldw_server \
  --body-file "<local-body-file>"
```

Expected: all issue bodies contain two-way links.

- [ ] **Step 3: Verify GitHub cross-links**

Run:

```bash
gh issue view "<epic-url-or-number>" --repo rmusser01/tldw_server --json title,body,url
gh issue list --repo rmusser01/tldw_server --state open --limit 100 --search '"design-system" OR "product-state"' --json number,title,url
```

Expected: epic body includes every child issue; child issues include parent epic and Backlog references.

- [ ] **Step 4: Verify Backlog cross-links**

Run:

```bash
backlog task "<tracker-parent-task-id>" --plain
```

Expected: parent task references the epic and child tasks exist with GitHub issue URLs in references.

- [ ] **Step 5: Commit final body updates**

Run:

```bash
git add Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md
git commit -m "Cross-link design-system tracker artifacts"
```

Expected: final local artifacts match public GitHub issue bodies.

## Task 6: Final Verification and PR Packaging

**Files:**
- Create: `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-pr-body.md`
- Read: all tracker artifact files.

- [ ] **Step 1: Run final local checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors. Worktree is clean after committed tracker artifacts.

- [ ] **Step 2: Create the PR body file**

Create `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-pr-body.md` with:

```markdown
## Change summary

This PR creates the durable design-system remaining-work tracker artifacts from the approved design. It keeps public tracker state in GitHub, mirrors execution notes in Backlog, and uses local issue-body drafts so public GitHub mutation is reviewable before issue creation.

## Verification

- `git diff --check`
- GitHub epic and child issues verified with `gh issue view` / `gh issue list`
- Backlog parent and child tasks verified with `backlog task <id> --plain`
- Bandit skipped: Markdown, Backlog metadata, and GitHub issue state only
```

Expected: PR body explains both what changed and why these choices were made.

- [ ] **Step 3: Commit final PR body**

Run:

```bash
git add Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-pr-body.md
git commit -m "Add design-system tracker PR body"
```

Expected: PR body is committed.

- [ ] **Step 4: Push branch and open PR**

Run:

```bash
git push origin codex/design-system-tracker-spec
gh pr create \
  --repo rmusser01/tldw_server \
  --base dev \
  --head codex/design-system-tracker-spec \
  --title "Design and plan remaining design-system tracker" \
  --body-file Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-pr-body.md
```

Expected: PR is created against `dev`. Include a human-written change summary in the PR body before merge, per repo policy.
