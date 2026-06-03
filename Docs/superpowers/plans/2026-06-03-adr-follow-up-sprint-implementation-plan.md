# ADR Follow-Up Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the next ADR workflow sprint by auditing existing decision sources, preparing bounded ADR backfill work, and deciding whether global Superpowers workflow changes are warranted.

**Architecture:** Treat the ADR follow-up as three dependent work packages. First produce a decision inventory without accepting ambiguous decisions. Then convert only owner-reviewed current decisions into bounded backfill slices. Finally evaluate global Superpowers edits only after the repo-local workflow has evidence from the inventory/backfill process.

**Tech Stack:** Markdown documentation, Backlog.md MCP/CLI workflow, `rg`, `find`, `git`, repository ADR template in `Docs/ADR/000-template.md`, ADR index in `Docs/ADR/README.md`.

---

## Scope And Sequencing

This plan covers:

- `TASK-509`: Audit docs for ADR decision inventory.
- `TASK-510`: Backfill authoritative ADRs from decision inventory.
- `TASK-511`: Evaluate global Superpowers ADR workflow updates.

Do the tasks in this order: `TASK-509` first, `TASK-510` second, `TASK-511` third. `TASK-510` depends on an owner-reviewed inventory. `TASK-511` depends on repo-local ADR workflow outcomes, so it should not propose global skill edits before the inventory/backfill process has been exercised.

ADR needed? No new ADR is needed for this plan itself. It operationalizes the accepted ADR workflow from `Docs/ADR/001-adr-workflow-and-governance.md`; it does not create a new durable architecture rule.

## File Structure

- Create: `Docs/ADR/inventory/2026-06-03-decision-inventory.md`
  - Decision inventory table and reviewer notes for `TASK-509`.
- Modify: `backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md`
  - Status, plan link, verification notes, final summary.
- Modify: `backlog/tasks/task-510 - Backfill-authoritative-ADRs-from-decision-inventory.md`
  - Status, plan link, created child-task links, verification notes, final summary.
- Modify: `backlog/tasks/task-511 - Evaluate-global-Superpowers-ADR-workflow-updates.md`
  - Status, plan link, evaluation outcome, verification notes, final summary.
- Potentially create: `backlog/tasks/task-510.x - Backfill-<domain>-ADRs.md`
  - One child task per bounded non-trivial ADR backfill slice.
- Potentially create: `Docs/ADR/00N-<decision-title>.md`
  - Only in later child tasks after owner-reviewed inventory entries identify current governing decisions.
- Potentially modify: high-value source docs under `Docs/Design/**`, `Docs/Plans/**`, `Docs/superpowers/specs/**`, `Docs/superpowers/plans/**`, or module docs.
  - Add links to covering or superseding ADRs where practical within each bounded child slice.
- Potentially create: `Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md`
  - Separate design/spec for `TASK-511` before any global Superpowers skill files are edited.

## Task 1: Establish Execution Branch And Tracking

**Files:**
- Modify: `backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md`
- Modify: `backlog/tasks/task-510 - Backfill-authoritative-ADRs-from-decision-inventory.md`
- Modify: `backlog/tasks/task-511 - Evaluate-global-Superpowers-ADR-workflow-updates.md`

- [ ] **Step 1: Create or reuse a clean worktree**

If already working in the clean worktree that contains this plan, reuse it. Otherwise run from the main repository:

```bash
git worktree add -b codex/adr-follow-up-sprint .worktrees/adr-follow-up-sprint codex/adr-follow-up-plan
```

Expected: Worktree is created from the branch containing this plan, `Docs/ADR/`, and `TASK-509` through `TASK-511`.

- [ ] **Step 2: Confirm task records are present**

Run:

```bash
backlog task TASK-509 --plain
backlog task TASK-510 --plain
backlog task TASK-511 --plain
```

Expected: All three tasks are visible with status `To Do`.

- [ ] **Step 3: Mark `TASK-509` In Progress and link this plan**

Use the Backlog.md MCP task edit flow when available. Add an implementation note:

```text
Plan: Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md
Execution order: TASK-509 -> TASK-510 -> TASK-511.
```

Expected: `TASK-509` status is `In Progress`; `TASK-510` and `TASK-511` remain `To Do`.

- [ ] **Step 4: Commit tracking-only setup**

Run:

```bash
git status --short
git add "backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md"
git commit -m "docs: start ADR follow-up sprint tracking"
```

Expected: Commit contains only the `TASK-509` tracking update.

## Task 2: Produce ADR Decision Inventory (`TASK-509`)

**Files:**
- Create: `Docs/ADR/inventory/2026-06-03-decision-inventory.md`
- Modify: `backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md`

- [ ] **Step 1: Create the inventory directory**

Run:

```bash
mkdir -p Docs/ADR/inventory
```

Expected: `Docs/ADR/inventory/` exists.

- [ ] **Step 2: Search required documentation scopes**

Run:

```bash
rg -n -i --glob '*.md' --glob '*.rst' "decision|decided|choose|chosen|adopt|standardize|default|supersede|deprecated|rejected|alternative|tradeoff|architecture|boundary|public API|persistence|security|worker|provider|dependency|workflow gate" Docs/Design Docs/Plans Docs/superpowers/specs Docs/superpowers/plans Docs/ADR tldw_Server_API/app README.md Project_Guidelines.md
```

Expected: Search results identify candidate decision language across the required documentation and module-doc scopes. Record the command and output summary in `TASK-509` verification notes.

- [ ] **Step 3: Create inventory header and classification rules**

Write `Docs/ADR/inventory/2026-06-03-decision-inventory.md` with this structure:

```markdown
# ADR Decision Inventory - 2026-06-03

**Related task:** TASK-509
**Inventory status:** Draft for owner review
**ADR creation policy:** This inventory does not create accepted ADRs. Ambiguous, contradicted, stale, or owner-sensitive decisions require owner review before backfill.

## Classification Rules

| Candidate status | Meaning | Allowed next action |
| --- | --- | --- |
| Current governing | Appears to describe a durable rule still consistent with current docs/code. | Owner review, then backfill slice planning. |
| Superseded | A newer source or ADR appears to replace it. | Keep classified; do not backfill as accepted. |
| Stale | Source appears outdated or inconsistent with current repo state. | Keep classified; request owner review or doc cleanup. |
| Duplicate | Same decision appears in multiple sources. | Pick a canonical source candidate; link duplicates. |
| Needs owner review | Decision is ambiguous, contradicted, or policy-sensitive. | Do not accept/backfill until owner confirms. |

## Inventory

| ID | Source path | Decision summary | Candidate status | Recommended action | Owner-review need | Notes |
| --- | --- | --- | --- | --- | --- | --- |
```

Expected: Inventory file exists with the required columns from `TASK-509`.

- [ ] **Step 4: Fill the inventory from existing ADRs first**

Add rows for `Docs/ADR/001-*.md` through `Docs/ADR/006-*.md`.

Expected: Existing ADRs are listed as already-covered governing decisions or superseded decisions, including `ADR-005` superseded by `ADR-006`.

- [ ] **Step 5: Fill the inventory from non-ADR decision sources**

Review search results manually and add rows for high-confidence candidates. Include source path, decision summary, candidate status, recommended action, and owner-review need for each row.

Expected: Inventory covers `Docs/Design/**`, `Docs/Plans/**`, `Docs/superpowers/specs/**`, `Docs/superpowers/plans/**`, embedded ADRs, and module docs with decision language.

- [ ] **Step 6: Apply the ambiguity gate**

Review every row marked `Current governing`. If a decision is contradicted, stale, owner-sensitive, or not clearly durable, change it to `Needs owner review`, `Stale`, `Superseded`, or `Duplicate`.

Expected: No ambiguous or contradicted decision is promoted to an accepted ADR candidate without owner review.

- [ ] **Step 7: Verify inventory completeness**

Run:

```bash
rg -n "\| .* \| .* \| .* \| .* \| .* \| .* \|" Docs/ADR/inventory/2026-06-03-decision-inventory.md
rg -n "Needs owner review|Current governing|Superseded|Stale|Duplicate" Docs/ADR/inventory/2026-06-03-decision-inventory.md
```

Expected: Inventory has populated rows and uses the required classification vocabulary.

- [ ] **Step 8: Record non-code verification and Bandit skip**

Update `TASK-509` implementation notes:

```text
Verification:
- Created Docs/ADR/inventory/2026-06-03-decision-inventory.md.
- Ran rg decision-language search across required documentation and module-doc scopes.
- Ran inventory structure/classification checks.
- Bandit skipped: documentation-only task; no Python/code paths touched.
```

Expected: `TASK-509` DoD records verification and the Bandit non-code skip.

- [ ] **Step 9: Commit TASK-509 output**

Run:

```bash
git status --short
git add Docs/ADR/inventory/2026-06-03-decision-inventory.md "backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md"
git commit -m "docs: inventory ADR decision candidates"
```

Expected: Commit contains the inventory and `TASK-509` updates.

## Task 3: Owner Review Gate For Inventory

**Files:**
- Modify: `Docs/ADR/inventory/2026-06-03-decision-inventory.md`
- Modify: `backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md`

- [ ] **Step 1: Send concise owner-review request**

Ask the human owner to review only rows whose recommended action is `Owner review`, `Backfill candidate`, or `Doc cleanup candidate`.

Expected: Owner can approve, reject, or reclassify inventory rows before accepted ADR backfill starts.

- [ ] **Step 2: Apply owner-review decisions**

Update the inventory row notes with review outcome:

```text
Owner review: approved for backfill
Owner review: keep stale, no ADR
Owner review: needs follow-up design
```

Expected: Every backfill candidate has explicit owner-reviewed scope.

- [ ] **Step 3: Complete `TASK-509`**

Update `TASK-509` final summary and acceptance criteria.

Expected: `TASK-509` is `Done`; no accepted ADRs have been created from ambiguous or contradicted decisions.

- [ ] **Step 4: Commit owner-review updates**

Run:

```bash
git status --short
git add Docs/ADR/inventory/2026-06-03-decision-inventory.md "backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md"
git commit -m "docs: record ADR inventory owner review"
```

Expected: Commit records reviewed inventory state.

## Task 4: Plan Bounded ADR Backfill Slices (`TASK-510`)

**Files:**
- Modify: `Docs/ADR/inventory/2026-06-03-decision-inventory.md`
- Modify: `backlog/tasks/task-510 - Backfill-authoritative-ADRs-from-decision-inventory.md`
- Potentially create: `backlog/tasks/task-510.x - Backfill-<domain>-ADRs.md`

- [ ] **Step 1: Mark `TASK-510` In Progress and link this plan**

Use the Backlog.md MCP task edit flow when available. Add:

```text
Plan: Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md
Prerequisite: owner-reviewed Docs/ADR/inventory/2026-06-03-decision-inventory.md.
```

Expected: `TASK-510` is `In Progress`.

- [ ] **Step 2: Group approved current governing decisions by domain**

Add a section to the inventory:

```markdown
## Backfill Slices

| Slice | Inventory IDs | Domain | Expected ADR outputs | Owner-review prerequisite | Backlog task |
| --- | --- | --- | --- | --- | --- |
```

Candidate domains should follow the inventory, not a fixed taxonomy. Likely examples include repo workflow, security gates, Jobs/Scheduler, AuthNZ, RAG, provider integrations, WebUI/extension conventions, or persistence.

Expected: Each approved current governing decision is either assigned to a bounded slice or explicitly marked as too small/no backfill needed.

- [ ] **Step 3: Create child tasks for non-trivial slices**

For each non-trivial slice, create a Backlog task with:

```text
Title: Backfill <domain> ADRs from reviewed inventory
Description: Backfill only the owner-approved inventory rows listed below.
Scope: <inventory IDs and source paths>
Expected outputs: <ADR filenames or count>
Prerequisites: Owner-reviewed inventory rows are approved for backfill.
Do not convert stale, superseded, duplicate, or ambiguous rows.
```

Expected: Every non-trivial module/domain backfill slice has a child task with clear scope and prerequisites.

- [ ] **Step 4: Decide whether any small single-domain conversion can happen directly**

If the inventory has only one small owner-approved domain, document why direct conversion remains reviewable. Otherwise do not write ADRs in `TASK-510`; leave ADR creation to child tasks.

Expected: `TASK-510` either creates child tasks only or records a narrow direct-conversion rationale.

- [ ] **Step 5: Define child-task ADR output rules**

Add this note to every backfill child task:

```text
Backfilled ADR output rules:
- Use Docs/ADR/000-template.md.
- Use Status: Accepted only for still-governing owner-approved decisions.
- Set Backfilled from: <source path>.
- Set Related task: <child task ID>.
- Keep stale, superseded, duplicate, and ambiguous decisions classified in the inventory; do not silently convert them.
```

Expected: Child tasks preserve the Stage 1 ADR rules and `TASK-510` acceptance criteria.

- [ ] **Step 6: Record source-doc link policy for child tasks**

Add this instruction to each child task:

```text
Where practical, update high-value source docs to link to the covering or superseding ADR. Do not churn low-value historical docs solely to add links.
```

Expected: Source-doc link work is bounded and reviewable.

- [ ] **Step 7: Verify `TASK-510` outputs**

Run:

```bash
rg -n "Backfill Slices|Backfilled ADR output rules|Backfilled from:|Owner-reviewed" Docs/ADR/inventory/2026-06-03-decision-inventory.md backlog/tasks
```

Expected: Inventory and child tasks show slice assignments, owner-review prerequisites, and backfilled ADR output rules.

- [ ] **Step 8: Complete or pause `TASK-510` based on created child tasks**

If child tasks were created and no direct conversion was done, complete `TASK-510` as planning/coordination work. If direct conversion happened, verify ADR README index updates and source-doc links before completion.

Expected: `TASK-510` final summary lists child tasks, direct conversions if any, verification, and Bandit non-code skip.

- [ ] **Step 9: Commit TASK-510 coordination output**

Run:

```bash
git status --short
git add Docs/ADR/inventory/2026-06-03-decision-inventory.md backlog/tasks
git commit -m "docs: plan bounded ADR backfill slices"
```

Expected: Commit contains inventory slice updates, child tasks, and `TASK-510` updates.

## Task 5: Evaluate Global Superpowers ADR Workflow Updates (`TASK-511`)

**Files:**
- Create: `Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md`
- Modify: `backlog/tasks/task-511 - Evaluate-global-Superpowers-ADR-workflow-updates.md`
- Potentially modify outside this repo later: `$CODEX_HOME/superpowers/skills/brainstorming/SKILL.md`
- Potentially modify outside this repo later: `$CODEX_HOME/superpowers/skills/writing-plans/SKILL.md`
- Potentially modify outside this repo later: `$CODEX_HOME/superpowers/skills/verification-before-completion/SKILL.md`

- [ ] **Step 1: Mark `TASK-511` In Progress and link this plan**

Use the Backlog.md MCP task edit flow when available. Add:

```text
Plan: Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md
Prerequisite: review outcomes from TASK-509 and TASK-510.
```

Expected: `TASK-511` is `In Progress`.

- [ ] **Step 2: Review repo-local ADR workflow evidence**

Summarize evidence from:

```text
Docs/ADR/README.md
Docs/ADR/001-adr-workflow-and-governance.md
Docs/ADR/inventory/2026-06-03-decision-inventory.md
TASK-509 final summary
TASK-510 final summary
```

Expected: The design/spec is based on observed repo workflow outcomes, not just the original idea.

- [ ] **Step 3: Draft the global Superpowers ADR workflow design/spec**

Create `Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md` with:

```markdown
# Global Superpowers ADR Workflow Design

**Related task:** TASK-511
**Status:** Draft for owner review

## Problem

Repo-local ADR workflow works for tldw_server, but global Superpowers may need repository-agnostic ADR prompts.

## Evidence From Repo-Local Workflow

## Candidate Skill Changes

| Skill | Current gap | Proposed trigger wording | Risk | Owner decision |
| --- | --- | --- | --- | --- |

## Recommendation

## Non-Goals

## Rollout Plan

## Verification
```

Expected: Separate design/spec exists before any global Superpowers skill file is modified.

- [ ] **Step 4: Identify candidate global skill edits**

Review these files:

```bash
sed -n '1,220p' "$CODEX_HOME/superpowers/skills/brainstorming/SKILL.md"
sed -n '1,220p' "$CODEX_HOME/superpowers/skills/writing-plans/SKILL.md"
sed -n '1,220p' "$CODEX_HOME/superpowers/skills/verification-before-completion/SKILL.md"
```

Expected: Spec identifies which skill files would change and what trigger wording they need.

- [ ] **Step 5: Keep global skill edits out of the repo PR unless explicitly approved**

Do not edit `$CODEX_HOME/superpowers/**` inside this repo PR. If owner approves global skill work, create a separate work item and apply the `superpowers:writing-skills` workflow.

Expected: `TASK-511` satisfies the design/spec requirement without mixing global tool edits into repo documentation work.

- [ ] **Step 6: Verify TASK-511 outputs**

Run:

```bash
rg -n "Candidate Skill Changes|brainstorming|writing-plans|verification-before-completion|Recommendation" Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md
```

Expected: Spec includes the required candidate skill files and trigger wording.

- [ ] **Step 7: Complete `TASK-511`**

Update `TASK-511` final summary:

```text
Verification:
- Reviewed repo-local ADR workflow outcomes from TASK-509/TASK-510.
- Identified candidate global Superpowers skill files and trigger wording.
- Produced separate design/spec before modifying global Superpowers files.
- Bandit skipped: documentation-only task; no Python/code paths touched.
```

Expected: `TASK-511` is `Done`, or remains `In Progress` only if waiting on owner review.

- [ ] **Step 8: Commit TASK-511 output**

Run:

```bash
git status --short
git add Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md "backlog/tasks/task-511 - Evaluate-global-Superpowers-ADR-workflow-updates.md"
git commit -m "docs: evaluate global Superpowers ADR workflow"
```

Expected: Commit contains the global workflow design/spec and `TASK-511` updates.

## Task 6: Final Verification And PR Prep

**Files:**
- Modify: `backlog/tasks/task-509 - Audit-docs-for-ADR-decision-inventory.md`
- Modify: `backlog/tasks/task-510 - Backfill-authoritative-ADRs-from-decision-inventory.md`
- Modify: `backlog/tasks/task-511 - Evaluate-global-Superpowers-ADR-workflow-updates.md`

- [ ] **Step 1: Verify task statuses and final summaries**

Run:

```bash
backlog task TASK-509 --plain
backlog task TASK-510 --plain
backlog task TASK-511 --plain
```

Expected: Each task has status, verification notes, Bandit skip or result, known skips/blockers, and final summary.

- [ ] **Step 2: Verify Markdown references**

Run:

```bash
rg -n "TASK-509|TASK-510|TASK-511|Docs/ADR/inventory/2026-06-03-decision-inventory.md" Docs/ADR Docs/superpowers backlog/tasks
```

Expected: Plan, inventory, task records, and any specs consistently link to each other.

- [ ] **Step 3: Run security gate decision**

Because this plan is documentation/backlog only unless child tasks later touch code, record:

```text
Bandit skipped: documentation-only ADR workflow planning/inventory tasks; no Python/code paths touched.
```

If any Python/code files are touched later, run:

```bash
source .venv/bin/activate && python -m bandit -r <touched_paths> -f json -o bandit_<task>.json
```

Expected: Either non-code skip is documented, or Bandit report is generated and `bandit_*.json` remains ignored by `.gitignore`.

- [ ] **Step 4: Self-review changed files**

Run:

```bash
git diff --stat
git diff -- Docs/ADR Docs/superpowers backlog/tasks
```

Expected: Diff is scoped to ADR inventory/backfill coordination/global workflow evaluation docs and Backlog task records.

- [ ] **Step 5: Commit any final task-record updates**

Run:

```bash
git status --short
git add backlog/tasks
git commit -m "docs: close ADR follow-up sprint tasks"
```

Expected: Only final Backlog task updates are committed.

- [ ] **Step 6: Create PR with human-owned change-summary reminder**

Draft PR body with:

```markdown
## Change summary

<Human requester must explain what changed and why these implementation choices were made.>

## Verification

- <commands run>
- Bandit skipped: documentation-only work; no Python/code paths touched.
```

Expected: PR is ready for owner review and satisfies the AI-generated PR change-summary gate only after the human-authored summary is supplied.
