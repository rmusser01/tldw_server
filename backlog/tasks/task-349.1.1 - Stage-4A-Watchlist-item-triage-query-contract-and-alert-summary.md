---
id: TASK-349.1.1
title: Stage 4A Watchlist item triage query contract and alert summary
status: Done
assignee: []
created_date: '2026-05-15 18:18'
updated_date: '2026-05-15 18:31'
labels:
  - watchlists
  - stage4
  - backend
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend/API query contract for server-authoritative Watchlist item review queues. Scope is limited to item list sorting/filtering and optional content-alert summary enrichment for the existing /api/v1/watchlists/items list; do not add batch updates, saved views, or frontend UI in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/watchlists/items supports validated server-side sort modes while preserving the current default ordering for existing clients.
- [x] #2 Item list filtering supports content-alert context such as alert presence, alert status, alert severity, and alert rule ID without duplicating item rows.
- [x] #3 Optional item alert summary returns compact content-alert context for each item only when requested.
- [x] #4 Focused DB and API tests cover sort stability, Watchlist/user scoping, alert-aware filters, static route ordering, and no fake confidence/novelty filters.
- [x] #5 Backward compatibility tests for existing first-class Watchlist and Stage 3 content-alert APIs continue to pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 4A after planning commit 3e29318c1. Scope: backend/API item triage query contract, alert-aware filters, optional alert_summary, tests first.

Implemented validated item sort modes, alert-aware item filters, optional per-item alert summaries, and matching alert-aware smart-count filters. Verification: focused Stage 4A DB/API tests passed (4 passed); broader nearby Watchlists API tests passed (11 passed); Bandit touched scope passed with zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4A adds the server-authoritative Watchlist item triage query contract. /api/v1/watchlists/items now supports validated sort modes, alert presence/status/severity/rule filters, and opt-in compact alert summaries; /api/v1/watchlists/items/smart-counts accepts the same alert filters so filtered result counts stay aligned. Focused and nearby Watchlists backend/API tests pass, Bandit reports zero findings for touched code, and git diff --check is clean.
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
