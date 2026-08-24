---
id: TASK-12989
title: Design Claims Jobs Stage 2A analytics exports
status: Done
created_date: 2026-08-08 17:35
labels:
- claims
- jobs
- design
priority: high
references:
- TASK-9935
- TASK-9937
- Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md
documentation:
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
modified_files:
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- backlog/tasks/task-12989 - Design-Claims-Jobs-Stage-2A-analytics-exports.md
updated_date: 2026-08-08 17:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved Stage 2A design for moving Claims analytics export execution onto the shared Jobs control plane while preserving a synchronous feature-flag fallback. Claims owns export artifacts and domain handlers; Jobs owns durable lifecycle, retries, leases, status, and admin controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The specification defines asynchronous HTTP 202 behavior when Claims analytics export Jobs are enabled and preserves synchronous HTTP 200 behavior when disabled.
- [x] #2 The specification keeps Jobs payloads ID-only and defines owner-scoped SQLite and PostgreSQL behavior.
- [x] #3 The specification defines deterministic snapshots, output-size bounds, CSV safety, retry-safe artifact transitions, reconciliation, retention, and rollout behavior.
- [x] #4 The specification does not add Claims-owned queue lifecycle or administrative controls.
- [x] #5 The written specification passes placeholder, consistency, ambiguity, scope, and testing-coverage review and is committed for user review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write and self-review the approved Stage 2A design specification. After user approval, create a separate detailed implementation plan using the writing-plans workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design completed in isolated worktree on branch `codex/claims-jobs-stage2a-analytics-exports-design`.
Verification: `test_claims_dashboard_analytics.py` and `test_claims_analytics_exports_cleanup.py` passed (8 passed, 26 warnings in 11.27s).
Specification placeholder scan, trailing-whitespace scan, and `git diff --cached --check` completed with no findings.
Bandit was not run because this task changes only documentation and Backlog metadata; no executable code was added or modified.
Two unrelated untracked watchlist template files were intentionally excluded from staging and commits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Produced and self-reviewed the Claims Jobs Stage 2A analytics exports design. The specification defines the Claims/Jobs ownership boundary, synchronous fallback and asynchronous 202 behavior, owner-scoped artifact model, deterministic snapshots, resource and CSV safeguards, retry-safe state transitions, reconciliation and retention behavior, rollout sequence, and verification strategy. No implementation work was started.
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
