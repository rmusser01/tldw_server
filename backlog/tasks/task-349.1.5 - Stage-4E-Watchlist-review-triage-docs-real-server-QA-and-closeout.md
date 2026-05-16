---
id: TASK-349.1.5
title: Stage 4E Watchlist review triage docs real-server QA and closeout
status: Done
assignee: []
created_date: '2026-05-15 18:19'
updated_date: '2026-05-15 20:04'
labels:
  - watchlists
  - stage4
  - qa
dependencies:
  - TASK-349.1.1
  - TASK-349.1.2
  - TASK-349.1.3
  - TASK-349.1.4
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
Close Stage 4 by documenting the item triage API/UI contract, running focused backend/frontend verification, running Bandit for touched backend code, and performing a real-server CDP smoke of the refreshed Updates triage flow in desktop and extension-sized viewports. Do not mock the server for browser evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists API docs describe Stage 4 item sort/filter additions, alert summaries, batch triage, saved views, and the Stage 5 report-builder boundary.
- [x] #2 Focused backend and frontend Stage 4 test suites pass or any scoped failures are fixed/documented with concrete evidence.
- [x] #3 Bandit runs against touched backend files and results are recorded.
- [x] #4 Real FastAPI plus real WebUI CDP smoke covers desktop and extension-sized Updates triage flows without API mocking.
- [x] #5 Stage 4 Backlog tasks are closed or left with explicit blockers and final summaries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after Stage 4D commit 2cc6494fb. Scope: API docs, focused Stage 4 backend/frontend verification, Bandit for backend touches, and real-server CDP smoke in desktop plus extension-sized viewports. Browser evidence must use the real FastAPI/WebUI stack without API mocks.

Implemented Stage 4E closeout updates: docs, selected Watchlist Updates copy, help anchor, locale parity, copy-contract tests, accessibility labels, and real-server QA evidence.

Verification: backend focused Watchlists tests 16 passed; frontend focused suite 14 files and 117 tests passed; Bandit JSON had 0 errors and 0 results; git diff --check passed; API docs mirror matched with cmp.

Real-server CDP smoke used real FastAPI 127.0.0.1:18001 and real Next WebUI localhost:18002 with no API mocking. Verified seeded Watchlist id 4, alert match, saved view, reviewed state, briefing queue state, persisted API state, desktop screenshot, and 420x760 constrained screenshot in /private/tmp/tldw-watchlists-stage4e.

Known observation: repeated reload/resize cycles can trigger Watchlists rate_limit responses in the dev console; the loaded queue remained usable. Logged as a follow-up resilience issue, not a Stage 4E blocker.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4E is closed. Docs now cover Updates triage filters/sorts, alert summaries, batch updates, saved item views, briefing queue handoff, and the Stage 5 report-builder boundary. The selected Watchlist review surface now consistently says Updates while preserving internal items/API compatibility. Real-server CDP smoke and focused backend/frontend/security verification passed.
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
