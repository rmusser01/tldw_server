---
id: TASK-349.2.2
title: Stage 5B Watchlist output evidence snapshots API
status: Done
assignee: []
created_date: '2026-05-15 21:40'
updated_date: '2026-05-16 01:54'
labels:
  - watchlists
  - stage5
  - backend
  - api
dependencies:
  - TASK-349.2.1
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349.2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the Stage 5 report evidence contract into Watchlists output creation. New reports should persist immutable evidence snapshot sidecars referenced from output metadata and expose output-scoped evidence/readiness APIs while preserving existing Markdown, HTML, Chatbook, TTS, audio, download, and legacy output behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/watchlists/outputs accepts backwards-compatible report preset/readiness options and still supports existing clients that omit them.
- [x] #2 New Watchlists outputs persist an immutable evidence snapshot sidecar and output metadata includes report preset, snapshot path, readiness, included/excluded counts, source count, alert count, and weak-evidence warning count.
- [x] #3 GET /api/v1/watchlists/outputs/{output_id}/evidence returns immutable snapshot evidence for Stage 5 outputs and a clear legacy live-only response for older outputs.
- [x] #4 GET /api/v1/watchlists/outputs/{output_id}/readiness returns readiness without requiring artifact download.
- [x] #5 Focused API tests cover snapshot persistence, endpoint scoping, legacy fallback, missing sidecar handling, and compatibility with existing output download/delivery behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 5B in worktree .worktrees/watchlists-stage1a. Scope is output creation snapshot persistence plus evidence/readiness APIs, using real Watchlists API tests and existing output artifact storage.

Verification recorded for Stage 5B. Red check: test_watchlist_reports_api.py initially failed for missing report metadata/endpoints. Green checks: Stage 5B API tests pass; combined Watchlists report/output regression set reports 15 passed and 5 warnings. git diff --check passes. Bandit on touched backend files reports 0 errors and 0 findings. API reference docs are deferred to Stage 5E per plan; the Stage 5 plan checklist was updated for this slice. No frontend/browser QA was run because this is a backend API slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired Watchlists report evidence snapshots into output creation. New outputs accept report preset/options, build immutable evidence/readiness snapshots from real run items, source rows, excluded same-run items, and content alerts, persist sidecar JSON in the user outputs directory, merge report metadata with existing output/delivery metadata, and expose output-scoped evidence and readiness endpoints with legacy live-only fallback and missing-sidecar handling.
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
