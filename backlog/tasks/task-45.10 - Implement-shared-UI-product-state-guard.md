---
id: TASK-45.10
title: Implement shared UI product-state guard
status: Done
assignee: []
created_date: '2026-05-06 01:02'
updated_date: '2026-05-06 04:21'
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
- [x] #1 Pure product-state guard rules are implemented with Vitest coverage for
  local recovery, empty, loading, status, AntD product-state context, allowed
  mechanics, canonical labels, canonical roots, stable finding IDs, baseline
  validation, stale baseline reporting, and report formatting.
- [x] #2 A CLI script scans apps/packages/ui/src, reads a checked-in baseline,
  prints actionable grouped output, exits nonzero for blocked findings or
  invalid baselines, and exits zero for baselined legacy debt and stale-entry
  warnings.
- [x] #3 apps/packages/ui/package.json exposes bun run verify:design-system-state
  and the real shared UI scan passes with an explicit baseline for current
  legacy product-state debt.
- [x] #4 Docs/Design/tldw_web_design_system_inventory.md documents the guard
  workflow and baseline rules for future shared UI product-state work.
- [x] #5 Focused design-system Vitest tests, bun run verify:design-system-state,
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

Task 3 accepted. Commits: 987cbc35d and 0bfc74590. Focused Vitest passed from
apps/packages/ui with 31/31 tests. The real guard command exited 0 with 260
allowed legacy exceptions, no blocked findings, no baseline errors, and no stale
entries. Spec review approved after JSX handler-source false positives were
fixed and the baseline was refreshed. Code-quality review approved.

Task 4 accepted. Docs/Design/tldw_web_design_system_inventory.md now documents
when to run bun run verify:design-system-state from apps/packages/ui, directs new
shared UI product-state surfaces to src/components/ui and
src/design-system/states.ts, and records baseline entry requirements plus
stale-entry cleanup expectations. Final verification from apps/packages/ui:
focused Vitest guard suite exited 0 with 4 test files passing and 39/39 tests
passing; bun run verify:design-system-state exited 0 with 260 allowed legacy
exceptions, no Blocked product-state findings section, no Stale baseline entries
section, and no baseline errors. From repo root: git diff --check exited 0.
Bandit skipped: no Python files touched.

Final review fix accepted. The guard now preserves duplicate same-rule/path/
subject findings instead of letting one baseline entry cover later same-subject
occurrences. Every occurrence in a duplicate group receives a deterministic
duplicate-suffixed id; singleton findings keep their base ids. The existing
baseline was refreshed from 260 to 523 entries to represent the newly exposed
current duplicate debt, and the inventory documents preserving exact
duplicate-suffixed ids for exceptions. Fresh verification from apps/packages/ui:
product-state guard unit suite exited 0 with 35/35 tests passing; focused
design-system suite exited 0 with 4 test files passing and 43/43 tests passing;
bun run verify:design-system-state exited 0 with 523 allowed legacy exceptions
and no blocked, stale, or baseline-error sections. From repo root: git diff
--check exited 0. Final review re-check passed with no findings.

PR #1338 review-fix pass started. Actionable comments to address: make duplicate finding IDs stable without positional unsuffixed first occurrence; bound verify-design-system-state file read concurrency; short-circuit JSX-return traversal after a JSX return is found.

PR #1338 review fixes completed: duplicate product-state finding groups now suffix every occurrence so legacy base IDs cannot be inherited by newly inserted earlier duplicates; baseline refreshed with stable duplicate IDs while keeping 523 live exceptions; source reads are bounded and order-preserving; JSX-return traversal short-circuits after finding JSX.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and closed out the shared UI product-state guard workflow. Tasks 1-3
added the pure product-state rule engine, baseline/report semantics, CLI/package
script, and the checked-in baseline for existing shared UI product-state
exceptions. Final review found that same-rule/path/subject findings could share
one baseline entry, so the guard now emits duplicate-suffixed ids for every
occurrence in duplicate groups while singleton findings preserve compact base
ids. The refreshed baseline tracks 523 existing shared UI product-state
exceptions. Task 4 documented the contributor workflow in
Docs/Design/tldw_web_design_system_inventory.md, including when to run the guard,
expected design-system primitives for new shared product-state UI, and
owner/reason/replacement/queue requirements for any new baseline exception.
Final verification passed: focused Vitest guard coverage from apps/packages/ui
exited 0 with 2 files passing and 40/40 tests passing; bun run
verify:design-system-state exited 0 with 523 allowed legacy exceptions, no
blocked product-state findings, no stale baseline entries, and no baseline
errors; git diff --check exited 0 from the repo root. Bandit skipped: no Python
files touched.

Addressed PR #1338 review comments for the product-state design-system guard. Added regression coverage for duplicate ID stability and bounded source reading, refreshed the baseline to stable duplicate IDs, and verified with focused Vitest coverage, verify:design-system-state, and git diff --check. Bandit was not applicable because the touched scope contains no Python.
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
