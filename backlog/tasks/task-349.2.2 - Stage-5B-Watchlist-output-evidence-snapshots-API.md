---
id: TASK-349.2.2
title: Stage 5B Watchlist output evidence snapshots API
status: To Do
assignee: []
created_date: '2026-05-15 21:40'
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
- [ ] #1 POST /api/v1/watchlists/outputs accepts backwards-compatible report preset/readiness options and still supports existing clients that omit them.
- [ ] #2 New Watchlists outputs persist an immutable evidence snapshot sidecar and output metadata includes report preset, snapshot path, readiness, included/excluded counts, source count, alert count, and weak-evidence warning count.
- [ ] #3 GET /api/v1/watchlists/outputs/{output_id}/evidence returns immutable snapshot evidence for Stage 5 outputs and a clear legacy live-only response for older outputs.
- [ ] #4 GET /api/v1/watchlists/outputs/{output_id}/readiness returns readiness without requiring artifact download.
- [ ] #5 Focused API tests cover snapshot persistence, endpoint scoping, legacy fallback, missing sidecar handling, and compatibility with existing output download/delivery behavior.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
