---
id: TASK-13361
title: Validate local email startup and bounded synthetic search performance
status: Done
assignee: []
created_date: '2026-09-25 18:28'
updated_date: '2026-09-25 18:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue core email release validation with offline synthetic data. Exercise a real local startup path if feasible, collect bounded search performance evidence, and document exact limits against the 1M-message and intended-deployment gates. Keep Gmail disabled and do not use personal mail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Synthetic search benchmark records dataset, query mix, latency, and environment without personal mail.
- [x] #2 Local startup and core email routes are checked or the exact blocker is recorded.
- [x] #3 Release checklist distinguishes local evidence from target deployment and 1M-message approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Executed three stages: synthetic benchmark baseline; real main-app lifespan and scoped route integration; evidence review and release-limit documentation. The temporary implementation-plan file was removed after completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Synthetic SQLite benchmark: 10,000 messages, 150 warm default-mix queries p50 15.39 ms/p95 30.85 ms; 90 warm NFR-operator queries p50 9.91 ms/p95 21.19 ms. Main app lifespan + search/detail/media integration passed; full affected module 11 passed. Ruff check/format pass; Bandit zero findings. Reviewer found the initial /tmp-only benchmark evidence gap; both synthetic JSON reports are now committed under Docs/Operations with two-step fixture provenance. No other actionable review finding. Remaining release gates: chosen deployment and backend, real socket readiness, PostgreSQL/RLS if chosen, 1M-message scale, archive ingestion throughput, and owner sign-off. Optional Gmail/OAuth remains deferred.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a main-app lifespan integration regression for scoped synthetic email and media search, preserved 10k SQLite benchmark reports with exact query mixes, and updated the release checklist without claiming production or 1M readiness. Full affected module: 11 passed. Ruff check/format and git diff --check pass; Bandit reports zero findings.
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
