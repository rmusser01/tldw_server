# ADR Workflow Adoption Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 1 ADR workflow framework for `tldw_server`: canonical ADR docs, root `AGENTS.md` policy, seed governance ADRs, and follow-up Backlog tasks for later migration work.

**Architecture:** This is a docs/process change only. `Docs/ADR/` becomes the canonical record set, root `AGENTS.md` tells future agents when ADRs are required, and Backlog tasks track future inventory/backfill and Superpowers-global follow-up work. Stage 1 deliberately does not audit the whole repository or change global Superpowers skills.

**Tech Stack:** Markdown, Backlog.md MCP tools, git, shell verification (`git diff --check`, `rg`, `test`).

---

## Source Artifacts

- Approved design spec: `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`
- Design Backlog task: `TASK-506`
- Planning Backlog task: `TASK-507`
- Stage 1 implementation Backlog task: `TASK-508`
- Root workflow file: `AGENTS.md`

## Scope Boundary

Implement only Stage 1 from the approved design:

- Create `Docs/ADR/` framework files.
- Create the five required seed ADRs.
- Update root `AGENTS.md` with the repo-local ADR policy.
- Create/link follow-up Backlog tasks for decision inventory/backfill and global Superpowers review.
- Verify docs consistency and whitespace.

Do not:

- Audit all historical docs in this plan.
- Modify global files under `/Users/macbook-dev/.codex/superpowers/`.
- Add an automation script for ADR checks.
- Rewrite existing design docs to link ADRs except where root `AGENTS.md` needs the new policy.

## File Structure

Create:

- `Docs/ADR/README.md` - ADR index, status rules, workflow, trigger summary, and links to seed ADRs.
- `Docs/ADR/000-template.md` - reusable ADR template copied from the approved spec.
- `Docs/ADR/001-adr-workflow-and-governance.md` - accepted ADR for adopting this ADR workflow.
- `Docs/ADR/002-backlog-md-task-tracking.md` - backfilled accepted ADR for the existing Backlog.md requirement.
- `Docs/ADR/003-jobs-vs-scheduler-default.md` - backfilled accepted ADR for the Jobs vs Scheduler default.
- `Docs/ADR/004-ai-generated-pr-change-summary-gate.md` - backfilled accepted ADR for the AI-generated PR human summary gate.
- `Docs/ADR/005-bandit-touched-scope-security-gate.md` - backfilled accepted ADR for the Bandit touched-scope security gate.

Modify:

- `AGENTS.md` - insert `### 0.1 Architecture Decision Records (ADRs)` after the existing `### 0. Backlog.md Task Tracking` section and before `### 1. Planning & Staging`.
- `backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md` - update implementation status, verification, and final summary through Backlog MCP.

Created by Backlog MCP during implementation:

- Follow-up task: `Audit docs for ADR decision inventory`
- Follow-up task: `Backfill authoritative ADRs from decision inventory`
- Follow-up task: `Evaluate global Superpowers ADR workflow updates`

## Task 0: Confirm Implementation Tracking

**Files:**
- Modify: `backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md`

- [ ] **Step 1: View the Stage 1 implementation task**

Use Backlog MCP `task_view` for `TASK-508`.

Expected: `TASK-508` exists and describes Stage 1 implementation: ADR framework, seed ADRs, root `AGENTS.md` policy, and follow-up tasks.

- [ ] **Step 2: Move `TASK-508` to In Progress**

Use Backlog MCP `task_edit` to set `TASK-508` status to `In Progress` and add the implementation plan path:

```text
Docs/superpowers/plans/2026-06-02-adr-workflow-adoption-stage-1-implementation-plan.md
```

- [ ] **Step 3: Commit or stage only when paired with related work**

Do not stage unrelated Backlog task files. When task updates are committed, use the exact path returned by Backlog MCP:

```bash
git add 'backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md'
```

## ADR Assessment

Required: yes

Reason: Stage 1 creates durable repository governance for how architectural decisions are recorded and enforced.

Target ADR: `Docs/ADR/001-adr-workflow-and-governance.md`

Existing governing ADRs: none; `Docs/ADR/` does not exist yet.

## Task 1: Create ADR Framework Files

**Files:**
- Create: `Docs/ADR/README.md`
- Create: `Docs/ADR/000-template.md`
- Modify: `backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md`

- [ ] **Step 1: Confirm the ADR directory is absent or inspect existing files**

Run:

```bash
test -d Docs/ADR && find Docs/ADR -maxdepth 2 -type f -print || true
```

Expected: either no directory exists, or any existing files are listed for review. If files exist, stop and adapt numbering without overwriting user work.

- [ ] **Step 2: Create `Docs/ADR/000-template.md`**

Create:

```markdown
# ADR-{N}: {Short title}

**Status:** Proposed | Accepted | Superseded by ADR-{N}
**Date:** YYYY-MM-DD
**Backfilled from:** {source path, or "not backfilled"}
**Decision owner:** {human/session/reviewer}
**Related task:** {Backlog task ID/link}
**Related spec/plan:** {paths}

## Decision

One sentence stating what was decided.

## Context

Why this decision was needed.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| {Alternative A} | {Reason} |

## Consequences

What this means going forward, including accepted tradeoffs.

## Follow-up

Optional implementation, audit, or documentation follow-up links.
```

- [ ] **Step 3: Create `Docs/ADR/README.md`**

Use this structure:

```markdown
# Architecture Decision Records

Architecture Decision Records (ADRs) capture durable architecture decisions for `tldw_server`: what was decided, why, what alternatives were considered, and what tradeoffs were accepted.

Module docs, design specs, and plans describe how things work. ADRs explain why important architecture rules exist.

## Workflow

1. Search existing ADRs before creating a new one.
2. Create a Backlog.md task or use the task already associated with the work.
3. Use `000-template.md`.
4. Use the next sequential number.
5. Record one decision per ADR.
6. Write ADRs at decision time whenever possible.
7. If backfilling, keep `Status: Accepted` for still-governing decisions and set `Backfilled from:` to the source path.
8. Do not rewrite accepted ADR rationale. To change a decision, create a new ADR and mark the old one `Superseded by ADR-{N}`.

## Status Rules

- `Proposed`: drafted for review but not yet accepted.
- `Accepted`: current governing decision.
- `Superseded by ADR-{N}`: no longer governing because a newer ADR replaced it.
- Backfill is metadata, not status. Backfilled still-governing decisions use `Status: Accepted` plus `Backfilled from: <source>`.

## ADR Required When

An ADR is required when a decision creates or changes a durable rule for module boundaries, public API shape, persistence, security, worker ownership, provider integration, WebUI/extension conventions, major dependencies, or repository workflow gates.

Small bug fixes, local implementation details, product copy, temporary experiments, and test-only changes usually do not need ADRs unless they create durable policy.

## Index

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR-001](001-adr-workflow-and-governance.md) | Accepted | Adopt `Docs/ADR/` as the canonical ADR workflow. |
| [ADR-002](002-backlog-md-task-tracking.md) | Accepted | Require Backlog.md tasks for repo-changing work. |
| [ADR-003](003-jobs-vs-scheduler-default.md) | Accepted | Use Jobs by default for new user-visible work and Scheduler for internal dependency orchestration. |
| [ADR-004](004-ai-generated-pr-change-summary-gate.md) | Accepted | Require human-written change summaries for materially AI-authored PRs. |
| [ADR-005](005-bandit-touched-scope-security-gate.md) | Accepted | Run Bandit on touched Python/code scope before completion. |
```

- [ ] **Step 4: Run framework verification**

Run:

```bash
test -f Docs/ADR/000-template.md
test -f Docs/ADR/README.md
rg -n "Backfilled from|Superseded by|ADR Required When|ADR-001" Docs/ADR
git diff --check -- Docs/ADR
```

Expected: all commands pass; `rg` shows the expected template/workflow/index references.

- [ ] **Step 5: Commit framework files**

```bash
git add Docs/ADR/000-template.md Docs/ADR/README.md
git commit -m "docs: add ADR framework"
```

## Task 2: Create Required Seed ADRs

**Files:**
- Create: `Docs/ADR/001-adr-workflow-and-governance.md`
- Create: `Docs/ADR/002-backlog-md-task-tracking.md`
- Create: `Docs/ADR/003-jobs-vs-scheduler-default.md`
- Create: `Docs/ADR/004-ai-generated-pr-change-summary-gate.md`
- Create: `Docs/ADR/005-bandit-touched-scope-security-gate.md`
- Modify: `Docs/ADR/README.md` if numbering changes are required after preflight

- [ ] **Step 1: Confirm no numbered ADRs already exist**

Run:

```bash
find Docs/ADR -maxdepth 1 -type f -name '[0-9][0-9][0-9]-*.md' -print
```

Expected: only `Docs/ADR/000-template.md` exists before this task. If any numbered ADR already exists, stop and renumber the seed ADRs to the next available numbers, then update the README index.

- [ ] **Step 2: Create `Docs/ADR/001-adr-workflow-and-governance.md`**

Use:

```markdown
# ADR-001: ADR Workflow And Governance

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** not backfilled
**Decision owner:** User + Codex collaboration session
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`, `Docs/superpowers/plans/2026-06-02-adr-workflow-adoption-stage-1-implementation-plan.md`

## Decision

Use `Docs/ADR/` as the canonical home for Architecture Decision Records and require ADR assessment for substantial specs, implementation plans, and PRs.

## Context

Architecture decisions existed in scattered design docs, plans, review packets, and embedded ADR-like sections. The project needs a lightweight durable record that explains why architectural rules exist without replacing Backlog.md, Superpowers specs, implementation plans, or module documentation.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Big-bang migration | Too much churn and too high a risk of converting stale decisions into accepted policy. |
| Decision index before ADRs | Safer for audit, but delays the actual ADR workflow. |
| Module-by-module only | Too passive; it would not establish a repo-wide standard. |

## Consequences

Significant durable architecture decisions need ADRs. Substantial specs, plans, and PRs need an explicit ADR assessment. Accepted ADRs are immutable except for supersession metadata. Backfilled decisions use source metadata rather than pretending they were written at decision time.

## Follow-up

Create follow-up Backlog tasks for the decision inventory, module-by-module backfill, and possible global Superpowers updates.
```

- [ ] **Step 3: Create `Docs/ADR/002-backlog-md-task-tracking.md`**

Base this ADR on `AGENTS.md` Backlog.md policy:

```markdown
# ADR-002: Backlog.md Task Tracking

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`, `Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md`
**Decision owner:** User + prior Codex collaboration session
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md`

## Decision

Require an associated Backlog.md task before work changes repository files.

## Context

The repository needs a durable task and history layer that records why work exists, how it was planned, what files changed, what verification ran, and what was skipped or blocked.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Git commits only | Commits do not capture task state, verification history, blockers, or reviewable unit boundaries. |
| GitHub issues only | Not every local agent task maps cleanly to a remote issue, and local work needs MCP/CLI-first task tracking. |
| Manual markdown notes | Too easy to duplicate or bypass; Backlog.md provides a consistent task workflow. |

## Consequences

Repo-changing work must search for or create a Backlog task before edits begin. Read-only investigation can proceed without a task. Backlog tasks link to specs, plans, PRs, verification, and final summaries; they do not replace those artifacts.

## Follow-up

None for Stage 1.
```

- [ ] **Step 4: Create `Docs/ADR/003-jobs-vs-scheduler-default.md`**

Base this ADR on `AGENTS.md` lines 142-160:

```markdown
# ADR-003: Jobs Vs Scheduler Default

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`

## Decision

Use Jobs by default for new user-visible work that needs admin or ops visibility, and use Scheduler for internal orchestration where dependency handling is central.

## Context

The project has both Jobs and Scheduler systems. Future contributors need a durable default to avoid ad hoc queue/orchestration choices.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Jobs for all async work | Internal dependency orchestration fits Scheduler better and does not always need user/admin controls. |
| Scheduler for all async work | User-facing work often needs pause, resume, drain, retries, quotas, RLS, status endpoints, and worker processes. |
| Decide per feature with no default | Repeated debates and inconsistent ownership would slow implementation and increase maintenance cost. |

## Consequences

New user-visible features or work needing admin controls should use Jobs. Internal orchestration with task dependencies, idempotency keys, and registered handlers should use Scheduler. Recurring schedules use APScheduler to enqueue into whichever backend the feature chooses.

## Follow-up

Later ADR inventory work should identify any module-specific exceptions.
```

- [ ] **Step 5: Create `Docs/ADR/004-ai-generated-pr-change-summary-gate.md`**

Base this ADR on `AGENTS.md` lines 544-550 and `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`:

```markdown
# ADR-004: AI-Generated PR Change Summary Gate

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`, `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`

## Decision

Materially AI-authored PRs are not merge-ready until the human requester writes a `Change summary` explaining what changed and why those implementation choices were made.

## Context

The project allows AI-assisted development but needs human ownership of architectural and implementation rationale before merge.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Allow AI-generated summaries | A diff recap or AI-authored rationale does not prove human understanding or ownership. |
| Require no summary | Reviewers lose a concise human explanation of why the implementation is the right one. |
| Ban AI-authored PRs | Too restrictive for the project workflow. |

## Consequences

AI-generated PRs need a human-written summary. If the requester cannot explain the rationale in their own words, the PR is not merge-ready. Agents may prepare context, but the merge gate requires human ownership.

## Follow-up

None for Stage 1.
```

- [ ] **Step 6: Create `Docs/ADR/005-bandit-touched-scope-security-gate.md`**

Base this ADR on `AGENTS.md` lines 552-558:

```markdown
# ADR-005: Bandit Touched-Scope Security Gate

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`

## Decision

Run Bandit on touched Python/code scope before considering work complete; for docs-only changes, document why Bandit is not applicable.

## Context

The project handles authentication, media ingestion, sandboxing, providers, and local/self-hosted data. Security-sensitive Python changes need an explicit security scan gate that scales to the touched scope.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Full-repo Bandit every time | Can be expensive and noisy for narrow changes. |
| No routine Bandit gate | Security regressions in touched code could be missed. |
| Run only in CI | Local completion should catch new findings before review. |

## Consequences

Agents should activate the project virtual environment and run `python -m bandit -r <touched_paths> -f json -o /tmp/bandit_<task>.json` for touched Python/code paths. New findings in changed code should be fixed before finishing. Docs-only work records Bandit as not applicable.

## Follow-up

None for Stage 1.
```

- [ ] **Step 7: Verify seed ADRs**

Run:

```bash
rg -n "Status: Accepted|Backfilled from|## Alternatives considered|TASK-506|TASK-507|TASK-508" Docs/ADR
git diff --check -- Docs/ADR
```

Expected: all ADRs include status, backfill metadata, alternatives, and task links; diff check is clean.

- [ ] **Step 8: Commit seed ADRs**

```bash
git add Docs/ADR/README.md Docs/ADR/001-adr-workflow-and-governance.md Docs/ADR/002-backlog-md-task-tracking.md Docs/ADR/003-jobs-vs-scheduler-default.md Docs/ADR/004-ai-generated-pr-change-summary-gate.md Docs/ADR/005-bandit-touched-scope-security-gate.md
git commit -m "docs: add seed ADRs"
```

## Task 3: Add ADR Policy To Root AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Inspect the insertion point**

Run:

```bash
nl -ba AGENTS.md | sed -n '395,450p'
```

Expected: `### 0. Backlog.md Task Tracking` appears before `### 1. Planning & Staging`.

- [ ] **Step 2: Insert `### 0.1 Architecture Decision Records (ADRs)` after Backlog.md policy**

Add this section after the Backlog.md section and before `### 1. Planning & Staging`:

```markdown
### 0.1 Architecture Decision Records (ADRs)

This repository uses Architecture Decision Records in `Docs/ADR/` for significant, durable architecture decisions. ADRs explain why an architecture rule exists; design docs, module docs, and plans describe how work is shaped or implemented.

Every substantial Superpowers spec, implementation plan, or PR must make an explicit `ADR needed?` call. Use the trigger list in `Docs/ADR/README.md`: module boundaries, public API shape, persistence, security, worker ownership, provider integration, WebUI/extension conventions, major dependencies, and repository workflow gates usually require ADR consideration.

If an ADR is required, create or supersede it in the same reviewable unit of work and link it from the Backlog task, spec, plan, or PR. If no ADR is required, record a brief rationale in the task, spec, plan, or PR notes.

Accepted ADRs are immutable except for metadata needed to mark supersession. If a decision changes, create a new ADR and mark the old one `Superseded by ADR-{N}`. Backfilled still-governing ADRs use `Status: Accepted` plus `Backfilled from: <source>`; do not pretend they were written at original decision time.

For workflow details, numbering, template, and the current ADR index, start with `Docs/ADR/README.md`.
```

- [ ] **Step 3: Verify root policy references**

Run:

```bash
rg -n "Architecture Decision Records|ADR needed|Docs/ADR/README.md|Superseded by ADR" AGENTS.md
git diff --check -- AGENTS.md
```

Expected: new ADR section appears and diff check is clean.

- [ ] **Step 4: Commit AGENTS.md policy**

```bash
git add AGENTS.md
git commit -m "docs: document ADR workflow policy"
```

## Task 4: Create Follow-Up Backlog Tasks

**Files:**
- Created by MCP: `backlog/tasks/task-<id> - Audit-docs-for-ADR-decision-inventory.md`
- Created by MCP: `backlog/tasks/task-<id> - Backfill-authoritative-ADRs-from-decision-inventory.md`
- Created by MCP: `backlog/tasks/task-<id> - Evaluate-global-Superpowers-ADR-workflow-updates.md`
- Modify: `backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md`

- [ ] **Step 1: Search before creating follow-up tasks**

Use Backlog MCP `task_search` for:

- `ADR decision inventory`
- `backfill authoritative ADRs`
- `global Superpowers ADR workflow`

Expected: no existing matching tasks. If a matching task exists, use and link it rather than creating a duplicate.

- [ ] **Step 2: Create or reuse `Audit docs for ADR decision inventory`**

Create via Backlog MCP with:

```text
Title: Audit docs for ADR decision inventory
Description: Audit existing decision sources and produce Docs/ADR/inventory/YYYY-MM-DD-decision-inventory.md with current, superseded, stale, duplicate, and needs-owner-review classifications.
Acceptance criteria:
- Inventory covers Docs/Design/**, Docs/Plans/**, Docs/superpowers/specs/**, Docs/superpowers/plans/**, embedded ADRs, and module docs with decision language.
- Inventory records source path, decision summary, candidate status, recommended action, and owner-review need.
- No accepted ADR is created for ambiguous or contradicted decisions without owner review.
```

- [ ] **Step 3: Create or reuse `Backfill authoritative ADRs from decision inventory`**

Create via Backlog MCP with:

```text
Title: Backfill authoritative ADRs from decision inventory
Description: Convert owner-reviewed current governing decisions from the ADR decision inventory into numbered ADRs and add short source-doc references where practical.
Acceptance criteria:
- Current governing decisions from the reviewed inventory have ADRs or explicit owner-reviewed exclusions.
- Backfilled ADRs use Status: Accepted plus Backfilled from metadata.
- Stale, superseded, duplicate, and ambiguous decisions remain classified in the inventory.
- High-value source docs link to covering or superseding ADRs where practical.
```

- [ ] **Step 4: Create or reuse `Evaluate global Superpowers ADR workflow updates`**

Create via Backlog MCP with:

```text
Title: Evaluate global Superpowers ADR workflow updates
Description: After repo-local ADR workflow validation, decide whether to update global Superpowers skills so brainstorming, writing-plans, and verification workflows consider ADR assessment across repositories.
Acceptance criteria:
- Review repo-local ADR workflow outcomes before proposing global skill edits.
- Identify which skill files would change and what trigger wording they need.
- Produce a separate design/spec before modifying global Superpowers files.
```

- [ ] **Step 5: Update TASK-508 with linked follow-up task IDs**

Use Backlog MCP `task_edit` to add the implementation plan path, follow-up task IDs, and verification notes to `TASK-508`.

- [ ] **Step 6: Verify follow-up task files are present**

Run:

```bash
rg -n "ADR decision inventory|Backfill authoritative ADRs|global Superpowers ADR workflow" backlog/tasks
git diff --check -- \
  'backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md' \
  'backlog/tasks/task-<actual-id> - Audit-docs-for-ADR-decision-inventory.md' \
  'backlog/tasks/task-<actual-id> - Backfill-authoritative-ADRs-from-decision-inventory.md' \
  'backlog/tasks/task-<actual-id> - Evaluate-global-Superpowers-ADR-workflow-updates.md'
```

Expected: matching task files exist and diff check is clean for touched Backlog files.

- [ ] **Step 7: Commit follow-up Backlog tasks**

Stage only the exact ADR-related task files returned by Backlog MCP. Do not run `git add backlog/tasks` in this repository because many unrelated task files may be dirty.

```bash
git add 'backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md'
git add 'backlog/tasks/task-<actual-id> - Audit-docs-for-ADR-decision-inventory.md'
git add 'backlog/tasks/task-<actual-id> - Backfill-authoritative-ADRs-from-decision-inventory.md'
git add 'backlog/tasks/task-<actual-id> - Evaluate-global-Superpowers-ADR-workflow-updates.md'
git commit -m "docs: track ADR workflow follow-up tasks"
```

## Task 5: Final Verification And Task Closure

**Files:**
- Modify: `backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md`

- [ ] **Step 1: Run final documentation verification**

Run:

```bash
test -f Docs/ADR/000-template.md
test -f Docs/ADR/README.md
test -f Docs/ADR/001-adr-workflow-and-governance.md
test -f Docs/ADR/002-backlog-md-task-tracking.md
test -f Docs/ADR/003-jobs-vs-scheduler-default.md
test -f Docs/ADR/004-ai-generated-pr-change-summary-gate.md
test -f Docs/ADR/005-bandit-touched-scope-security-gate.md
rg -n "ADR needed\\?|Docs/ADR/README.md" AGENTS.md
rg -n "ADR-00[1-5]" Docs/ADR/README.md
git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Document Bandit applicability**

Because this Stage 1 implementation is docs/process-only, Bandit is not applicable unless the executor touches Python or executable code. Record this in the Backlog final summary.

- [ ] **Step 3: Inspect staged/uncommitted work**

Run:

```bash
git status --short
```

Expected: only intended Stage 1 files are modified/untracked before final commit. Existing unrelated dirty worktree entries may remain; do not stage or revert them.

- [ ] **Step 4: Update Backlog final summary**

Use Backlog MCP `task_edit` to add:

```text
Final summary: Created Docs/ADR framework, seed ADRs, root AGENTS.md policy, and follow-up Backlog tasks for inventory/backfill and global Superpowers review. Verification: final documentation checks passed; git diff --check passed; Bandit not applicable for docs-only changes.
```

If the implementation task is complete, mark the relevant `TASK-508` acceptance criteria and Definition of Done items complete, then move `TASK-508` to Done using the normal Backlog finalization path.

- [ ] **Step 5: Commit final Backlog updates**

```bash
git add 'backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md'
git commit -m "docs: record ADR workflow verification"
```

If Backlog follow-up task files are touched in this final update, include only the ADR-related task files in `git add`.

## Plan Review Checklist

Before executing:

- [ ] Plan review loop has approved this plan.
- [ ] User has selected subagent-driven or inline execution.
- [ ] Executor has read `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`.
- [ ] Executor has read root `AGENTS.md` Backlog and ADR-related workflow instructions.
- [ ] Executor has checked for existing `Docs/ADR/` files immediately before editing.
