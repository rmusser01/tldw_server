---
id: TASK-45.10
title: Implement shared UI product-state guard
status: In Progress
assignee: []
created_date: '2026-05-06 01:02'
updated_date: '2026-05-06 01:02'
labels:
  - design-system
  - frontend
  - guardrails
  - implementation
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-06-design-system-product-state-guard-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-06-design-system-product-state-guard-design.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved shared UI product-state guard for apps/packages/ui/src.
The guard should prevent new duplicated product-state UI patterns by scanning
shared UI source, applying explicit canonical-root handling, matching existing
debt against a checked-in baseline, reporting stale baseline entries, and
exposing a package script for verification. Work from the plan in
Docs/superpowers/plans/2026-05-06-design-system-product-state-guard-implementation-plan.md
and keep the implementation conservative and test-driven.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pure product-state guard rules are implemented with Vitest coverage for
  local recovery, empty, loading, status, AntD product-state context, allowed
  mechanics, canonical labels, canonical roots, stable finding IDs, baseline
  validation, stale baseline reporting, and report formatting.
- [ ] #2 A CLI script scans apps/packages/ui/src, reads a checked-in baseline,
  prints actionable grouped output, exits nonzero for blocked findings or
  invalid baselines, and exits zero for baselined legacy debt and stale-entry
  warnings.
- [ ] #3 apps/packages/ui/package.json exposes bun run verify:design-system-state
  and the real shared UI scan passes with an explicit baseline for current
  legacy product-state debt.
- [ ] #4 Docs/Design/tldw_web_design_system_inventory.md documents the guard
  workflow and baseline rules for future shared UI product-state work.
- [ ] #5 Focused design-system Vitest tests, bun run verify:design-system-state,
  git diff --check, and applicable security checks or documented skips are
  recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute
Docs/superpowers/plans/2026-05-06-design-system-product-state-guard-implementation-plan.md
using subagent-driven development.

Task sequence:
1. Implement pure rule-engine fixture tests and
   apps/packages/ui/scripts/design-system-product-state-rules.mjs using
   TypeScript AST import/JSX detection.
2. Add baseline validation, stable-id matching, stale-baseline reporting, and
   readable report formatting with focused tests.
3. Add apps/packages/ui/scripts/verify-design-system-product-state.mjs,
   apps/packages/ui/scripts/design-system-product-state-baseline.json, and the
   verify:design-system-state package script; run the real shared UI scan and
   baseline existing debt without weakening rules.
4. Document the guard workflow in
   Docs/Design/tldw_web_design_system_inventory.md and run final focused
   verification.

Execution notes:
- Follow TDD red/green for each implementation slice.
- Use explicit canonical-root handling rather than broad namespace exemptions.
- Do not store blocked in the baseline; blocked is computed for unbaselined
  findings.
- Stale baseline entries are warnings in v1 and should appear in reports.
- Runtime code changes are TypeScript/Node/UI only; Bandit should be skipped
  unless Python files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 accepted. Commits: 5800111ce, 1f65113f, 7204b7d. Focused Vitest
passed from apps/packages/ui with 13/13 tests. Spec review approved.
Code-quality review approved after scoped AntD owner and helper-function
false-positive fixes.

Task 2 accepted. Commits: 81848299c and 9d9d625db. Focused Vitest passed from
apps/packages/ui with 25/25 tests. Spec review approved. Code-quality review
approved after baseline identity validation was added and Ready canonical-label
coverage was restored.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
