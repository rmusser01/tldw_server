---
id: TASK-45.42
title: Create design-system tracker issue-body drafts
status: Done
assignee: []
created_date: '2026-05-14 02:26'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - >-
    Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-implementation-plan.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 1 of the design-system remaining-work tracker implementation plan by refreshing the current product-state baseline snapshot and creating local draft GitHub issue bodies for human review. Public GitHub issue creation remains out of scope until the drafts are approved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Issue-body draft directory exists with README, epic, product-area, and governance draft files.
- [x] #2 Draft bodies use the current baseline snapshot and approved issue titles/scope from the plan/spec.
- [x] #3 Drafts preserve the human approval gate before any public GitHub issue creation.
- [x] #4 Verification is recorded before the task is closed.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Confirmed worktree `codex/design-system-tracker-spec` and refreshed `origin/dev` at `7c652ce2d`.
- Regenerated the grouped baseline summary from `apps/packages/ui/scripts/design-system-product-state-baseline.json` using the implementation plan's ordered product-area categories.
- Fresh baseline snapshot matched the spec: 500 total entries, with 481 `antd-product-state-import` and 19 `canonical-state-label`.
- Created local-only draft GitHub issue bodies under `Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/`.
- Draft set includes README, epic body, 13 product-area migration bodies, and 6 governance bodies.
- All draft issue bodies preserve the human approval gate before public GitHub issue, label, PR, or other GitHub-state mutation.
- Bandit skipped: non-code Markdown-only tracker draft work; no Python files touched.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `find Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies -type f | sort` listed the expected 21 draft files.
- `git diff --check` passed.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the local draft issue-body package for the design-system remaining-work tracker. The drafts use the live 2026-05-14 baseline snapshot, exact implementation-plan issue titles, product-area rule splits, governance scopes, and the approved source-of-truth model where GitHub owns mutable tracker state and Backlog.md owns execution evidence.
<!-- SECTION:FINAL_SUMMARY:END -->
