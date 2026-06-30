---
id: TASK-45.9
title: Create shared UI product-state guard implementation plan
status: Done
assignee: []
created_date: '2026-05-06 00:28'
updated_date: '2026-05-06 00:39'
labels:
  - design-system
  - frontend
  - docs
  - planning
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-06-design-system-product-state-guard-design.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for the approved design-system product-state
guard spec. The plan should be executable by future agents and cover the
standalone guard script, rules, baseline schema, tests, package script,
documentation updates, and verification gates without implementing runtime code
in this planning slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A uniquely named implementation plan is added under Docs/superpowers/plans.
- [x] #2 The plan follows the writing-plans header and task checklist format.
- [x] #3 The plan decomposes the guard into small TDD tasks with exact files,
  commands, expected failures, and expected passes.
- [x] #4 The plan is reviewed and updated before handoff.
- [x] #5 The plan and Backlog task are committed on the clean spec branch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan file: Docs/superpowers/plans/2026-05-06-design-system-product-state-guard-implementation-plan.md

Planned structure:
1. Add pure rule-engine tests under
   apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts and
   implement apps/packages/ui/scripts/design-system-product-state-rules.mjs using
   TypeScript AST detection for imports/JSX plus explicit literal/context
   signals.
2. Add baseline validation, baseline matching, stale-baseline reporting, and
   readable report formatting in the pure rules module with focused Vitest
   coverage.
3. Add apps/packages/ui/scripts/verify-design-system-product-state.mjs,
   apps/packages/ui/scripts/design-system-product-state-baseline.json, and
   package script verify:design-system-state; run the real scan and baseline
   existing shared UI product-state debt without weakening rules.
4. Document the guard workflow in
   Docs/Design/tldw_web_design_system_inventory.md and verify focused
   design-system tests, the new guard command, git diff --check, and Bandit
   non-applicability for non-Python changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote
Docs/superpowers/plans/2026-05-06-design-system-product-state-guard-implementation-plan.md
in the clean design-system-product-state-guard-spec worktree. The plan
decomposes the guard into TDD tasks for the pure rule engine, baseline/report
semantics, CLI plus initial baseline, and documentation/verification closeout.

Plan review loop completed with the plan-document reviewer. First review found
missing fixture coverage for local empty/loading/status rules and a too-weak
suggested pattern; the plan was updated with explicit fixtures, corrected
patterns, same-file mixed-context AntD coverage, report totals by rule and
migration queue, and same-path/same-rule/different-subject baseline coverage.
Final review approved with no recommendations.

Verification for this planning slice: plan review approved; git diff --check
passed in the spec worktree. Runtime tests were not run because this task only
adds a future implementation plan and Backlog metadata. Bandit skipped because
no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the implementation plan for the shared UI product-state guard. The plan
specifies a TypeScript AST-backed rules module, Vitest fixture coverage,
baseline validation and stale-entry reporting, a thin verify:design-system-state
CLI, initial real-scan baseline workflow, inventory documentation updates, and
final verification gates. Plan review approved after addressing fixture coverage
and report-format issues. Verification: git diff --check passed; runtime tests
and Bandit are not applicable for this docs/planning-only slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
