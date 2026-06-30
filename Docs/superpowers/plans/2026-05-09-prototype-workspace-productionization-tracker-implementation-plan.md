# Prototype Workspace Productionization Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved prototype workspace productionization issue-tree spec into reviewed tracker artifacts and GitHub sub-issues under #1440.

**Architecture:** Keep GitHub issue creation separate from design review. First create repo-local artifacts: a contract matrix shell and a reviewed issue-body source document. Then create the eight prefixed GitHub sub-issues and update #1440 with a dependency summary. This plan is documentation/tracker work only unless a later implementation task explicitly changes backend or frontend code.

**Tech Stack:** Markdown, Backlog.md MCP, GitHub CLI, existing `Docs/superpowers` and `Docs/API-related` documentation conventions.

---

## Scope Check

The approved spec covers a tracker structure, not product implementation. Do not implement prototype workspace backend or frontend behavior from this plan. This plan creates the coordination artifacts needed before the two implementers start Risk Gate work.

Source spec:

- `Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md`

Parent GitHub tracker:

- https://github.com/rmusser01/tldw_server/issues/1440

Backlog task for this plan:

- `TASK-193`

## File Map

- Create: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
  - Responsibility: standalone contract matrix shell referenced by Risk Gates 1 and 4.
- Create: `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`
  - Responsibility: reviewable source of the eight GitHub sub-issue bodies before creating them.
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`
  - Responsibility: track plan progress, verification, and any blockers.
- Reference only: `Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md`
  - Responsibility: source of truth for the risk gate design.

## Task 1: Prepare A Clean Tracker Work Surface

**Files:**
- Read: `Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md`
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`

- [ ] **Step 1: Confirm current git state**

Run:

```bash
git status --short
```

Expected: identify whether the current checkout has unrelated dirty or unmerged files. If it does, do not stage unrelated paths.

- [ ] **Step 2: Keep tracker work in this checkout unless source artifacts are copied**

The approved spec and this implementation plan may be untracked in the current checkout. A new worktree from `origin/dev` will not contain them unless they have already been committed or copied.

Default: continue in the current checkout and only touch the files named in this plan.

If a clean worktree is required, copy these source artifacts into it before continuing:

- `Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md`
- `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-tracker-implementation-plan.md`
- `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`

- [ ] **Step 3: Optional clean-worktree setup only after preserving source artifacts**

Run:

```bash
git fetch origin dev
git worktree add /private/tmp/tldw-prototype-productionization-tracker -b codex/prototype-workspace-productionization-tracker origin/dev
```

Expected: a clean worktree exists for tracker artifacts. If the branch already exists, choose a unique `codex/` branch suffix. Copy the source artifacts listed in Step 2 into the new worktree before running later steps.

- [ ] **Step 4: Read the approved spec**

Run:

```bash
sed -n '1,260p' Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
sed -n '261,560p' Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
```

Expected: the spec includes the eight risk gates, Jobs/Scheduler decisions, contract matrix path, title-prefix decisions, and release evidence template.

- [ ] **Step 5: Update the Backlog task note**

Record the selected worktree/branch and whether GitHub issue creation is still gated behind review.

## Task 2: Create The Contract Matrix Shell

**Files:**
- Create: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`

- [ ] **Step 1: Create the contract matrix document**

Create `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md` with this structure:

```markdown
# Prototype Workspaces Contract Matrix

## Purpose

This document is the frontend/backend contract artifact for prototype workspace collaboration productionization. Risk Gate 1 creates the draft. Risk Gate 4 freezes it.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

## Status

- Draft owner: Backend/Core
- Frontend reviewer: Frontend/Product
- Current gate: Risk Gate 1 draft
- Frozen by: Risk Gate 4

## Error And State Matrix

| State | Backend condition | HTTP status | Stable error category | Frontend state bucket | Retryable | User-facing handling | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_link | Token cannot be verified without confirming existence | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| expired_link | Token/link is expired | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| revoked_link | Token/link or shared actor is revoked | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| exhausted_link | Link has no remaining collaborator uses | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| archived_workspace | Workspace is archived | 404 or configured non-enumerating status | workspace_unavailable | Workspace unavailable | No | Show unavailable workspace state | Draft |
| bootstrap_failed | Branch session bootstrap failed | 409 or 500-class mapped safe response | bootstrap_failed | Setup failed | Yes, if backend marks retryable | Offer retry when allowed | Draft |
| preview_unavailable | Preview handle missing, revoked, or unhealthy | 409 or 503 | preview_unavailable | Preview unavailable | Yes, if backend marks retryable | Show preview retry/status state | Draft |
| stale_promotion | Candidate is stale versus canonical snapshot | 409 | stale_promotion | Promotion stale | No | Ask user to resubmit from current branch | Draft |
| promotion_conflict | Promotion validation detects conflict | 409 | promotion_conflict | Promotion conflict | No | Show conflict/review state | Draft |
| promotion_validation_failed | Validation failed without promoting | 409 | promotion_validation_failed | Promotion failed | Yes, if backend marks retryable | Show validation failure details | Draft |

## Token And Session Security Dispositions

| Requirement | Disposition | Gate | Notes |
| --- | --- | --- | --- |
| Token storage/hash rules | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| TTLs | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Replay handling | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Cookie flags | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Referrer leakage controls | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Password-protected link behavior | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Signing secret rotation | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Revocation propagation | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |

## Frontend Fixture Notes

- Fixture schema owner:
- Mock state owner:
- Contract feedback deadline:
- Open frontend questions:

## Gate 4 Freeze Checklist

- [ ] All stable error categories are final.
- [ ] HTTP statuses are final or explicitly documented as non-enumerating policy choices.
- [ ] Retryability is final.
- [ ] Frontend state buckets are final.
- [ ] Backend/Core reviewer recorded.
- [ ] Frontend/Product reviewer recorded.
```

- [ ] **Step 2: Verify the contract matrix is referenced by the spec**

Run:

```bash
rg -n "Prototype_Workspaces_Contract_Matrix" Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
```

Expected: at least one match.

- [ ] **Step 3: Check markdown formatting**

Run:

```bash
git diff --check -- Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
```

Expected: no whitespace errors.

## Task 3: Draft The Eight GitHub Sub-Issue Bodies

**Files:**
- Create: `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`
- Read: `Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md`

- [ ] **Step 1: Create the issue-body source document**

Create `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`.

The file must start with:

```markdown
# Prototype Workspace Productionization GitHub Issue Bodies

Source spec: Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

These issue bodies are drafts. Do not create GitHub issues until this file is reviewed.
```

- [ ] **Step 2: Add the eight titles exactly**

Use these titles:

```text
[Risk Gate 1][Split] Prototype collaboration threat model and authorization invariants
[Risk Gate 2][Backend/Core] Prototype workspace persistence and transaction hardening
[Risk Gate 3][Backend/Core] Runtime jobs and preview lifecycle durability
[Risk Gate 4][Backend/Core] Backend API contract and error semantics freeze
[Risk Gate 5][Frontend/Product] Collaborator entry and route-state safety
[Risk Gate 6][Frontend/Product] Owner review and promotion UX hardening
[Risk Gate 7][Split] Operational visibility and documentation
[Risk Gate 8][Split] End-to-end release gate and production readiness review
```

- [ ] **Step 3: For each title, draft a GitHub-ready body**

Each body must include:

```markdown
## Parent

Tracks part of #1440.

## Risk Being Burned Down

...

## Owner Lane

...

## Dependencies

- Depends on:
- Blocks:

## Scope

- [ ] ...

## Non-Goals

- ...

## Acceptance Criteria

- [ ] ...

## Verification

- Backend tests:
- Frontend tests:
- Security checks:
- Manual/browser checks:
```

Use the corresponding Risk Gate section from the spec as the source of truth. Do not add product implementation scope that is not already in the spec.

- [ ] **Step 4: Add a creation checklist**

At the end of the file, add:

```markdown
## Creation Checklist

- [ ] Reviewed issue-body draft file.
- [ ] Created Risk Gate 1 issue.
- [ ] Created Risk Gate 2 issue.
- [ ] Created Risk Gate 3 issue.
- [ ] Created Risk Gate 4 issue.
- [ ] Created Risk Gate 5 issue.
- [ ] Created Risk Gate 6 issue.
- [ ] Created Risk Gate 7 issue.
- [ ] Created Risk Gate 8 issue.
- [ ] Posted summary comment on #1440 with all child issue URLs.
```

- [ ] **Step 5: Verify issue-body draft formatting**

Run:

```bash
git diff --check -- Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md
```

Expected: no whitespace errors.

## Task 4: Review Tracker Artifacts Before GitHub Creation

**Files:**
- Review: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- Review: `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`

- [ ] **Step 1: Run text checks**

Run:

```bash
rg -n "TBD|TODO|Open question|unresolved" Docs/API-related/Prototype_Workspaces_Contract_Matrix.md Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md
```

Expected: only intentional draft placeholders remain in the contract matrix; issue bodies should have no unresolved process questions.

- [ ] **Step 2: Confirm every Risk Gate appears once**

Run:

```bash
rg -n "\\[Risk Gate [1-8]\\]" Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md
```

Expected: eight title matches.

- [ ] **Step 3: Ask for human review**

Stop and ask the user to review:

- `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`

Do not create GitHub sub-issues until the user approves.

- [ ] **Step 4: Commit or otherwise persist reviewed draft artifacts before GitHub mutation**

Preferred: commit the reviewed draft artifacts before creating GitHub issues.

Run:

```bash
git add Docs/API-related/Prototype_Workspaces_Contract_Matrix.md Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md "backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md"
git commit -m "docs: draft prototype workspace productionization tracker artifacts"
```

Expected: the reviewed source docs are durable before `gh issue create` mutates GitHub. If unrelated unmerged files prevent a safe commit, record the blocker in `TASK-193` and ask the user whether to continue with GitHub issue creation from uncommitted reviewed files.

## Task 5: Create And Link GitHub Sub-Issues

**Files:**
- Read: `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`

- [ ] **Step 1: Create a temporary issue body file per Risk Gate**

For each Risk Gate, copy the reviewed body into a temp file:

```bash
/private/tmp/prototype-risk-gate-1.md
/private/tmp/prototype-risk-gate-2.md
/private/tmp/prototype-risk-gate-3.md
/private/tmp/prototype-risk-gate-4.md
/private/tmp/prototype-risk-gate-5.md
/private/tmp/prototype-risk-gate-6.md
/private/tmp/prototype-risk-gate-7.md
/private/tmp/prototype-risk-gate-8.md
```

- [ ] **Step 2: Create each GitHub issue**

Run one command per issue:

```bash
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 1][Split] Prototype collaboration threat model and authorization invariants" --body-file /private/tmp/prototype-risk-gate-1.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 2][Backend/Core] Prototype workspace persistence and transaction hardening" --body-file /private/tmp/prototype-risk-gate-2.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 3][Backend/Core] Runtime jobs and preview lifecycle durability" --body-file /private/tmp/prototype-risk-gate-3.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 4][Backend/Core] Backend API contract and error semantics freeze" --body-file /private/tmp/prototype-risk-gate-4.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 5][Frontend/Product] Collaborator entry and route-state safety" --body-file /private/tmp/prototype-risk-gate-5.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 6][Frontend/Product] Owner review and promotion UX hardening" --body-file /private/tmp/prototype-risk-gate-6.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 7][Split] Operational visibility and documentation" --body-file /private/tmp/prototype-risk-gate-7.md
gh issue create --repo rmusser01/tldw_server --title "[Risk Gate 8][Split] End-to-end release gate and production readiness review" --body-file /private/tmp/prototype-risk-gate-8.md
```

Expected: eight issue URLs.

- [ ] **Step 3: Post the child issue summary on #1440**

Create `/private/tmp/prototype-workspace-child-issues-summary.md`:

```markdown
Created the risk-gated productionization sub-issues:

- [Risk Gate 1][Split]: <url>
- [Risk Gate 2][Backend/Core]: <url>
- [Risk Gate 3][Backend/Core]: <url>
- [Risk Gate 4][Backend/Core]: <url>
- [Risk Gate 5][Frontend/Product]: <url>
- [Risk Gate 6][Frontend/Product]: <url>
- [Risk Gate 7][Split]: <url>
- [Risk Gate 8][Split]: <url>

Implementation should proceed in risk-gate order. Frontend/Product can begin fixture/mock preparation after the Risk Gate 1 draft contract matrix exists.
```

Run:

```bash
gh issue comment 1440 --repo rmusser01/tldw_server --body-file /private/tmp/prototype-workspace-child-issues-summary.md
```

Expected: #1440 has a summary comment with all child issue URLs.

- [ ] **Step 4: Update Backlog task with URLs**

Record the eight child issue URLs and the #1440 summary comment URL in `TASK-193`.

## Task 6: Verify And Close Out Tracker Work

**Files:**
- Modify: `backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md`
- Commit if safe: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- Commit if safe: `Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md`

- [ ] **Step 1: Verify git diff hygiene**

Run:

```bash
git diff --check -- Docs/API-related/Prototype_Workspaces_Contract_Matrix.md Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md
```

Expected: no whitespace errors.

- [ ] **Step 2: Verify no backend/frontend code was changed**

Run:

```bash
git diff --name-only
```

Expected: only tracker docs and Backlog task files changed. If backend code changed, stop and run appropriate tests plus Bandit.

- [ ] **Step 3: Record Bandit disposition**

Because this tracker plan should only change Markdown and Backlog task files, record in `TASK-193`:

```text
Bandit not run: documentation/tracker-only changes, no Python code changed.
```

If Python code changed, instead run:

```bash
source .venv/bin/activate && python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_prototype_tracker.json
```

- [ ] **Step 4: Final Backlog update before commit**

Mark `TASK-193` acceptance criteria complete, add final summary, and record any skipped GitHub issue creation or commit blocker.

- [ ] **Step 5: Commit tracker artifacts if the worktree is clean enough**

Run:

```bash
git add Docs/API-related/Prototype_Workspaces_Contract_Matrix.md Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md "backlog/tasks/task-193 - Write-implementation-plan-for-prototype-workspace-productionization-tracker.md"
git commit -m "docs: add prototype workspace productionization tracker artifacts"
```

Expected: commit succeeds and includes the final Backlog closeout. If a draft-artifact commit already exists from Task 4, use a second closeout commit message such as `docs: finalize prototype workspace productionization tracker`.

- [ ] **Step 6: Report blockers if commit is unsafe**

If unrelated unmerged files exist, do not force the commit from that checkout. Report the blocker and keep the artifacts available for review.
