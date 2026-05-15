---
id: TASK-349.2.1
title: Stage 5A Watchlist report evidence readiness contract
status: To Do
assignee: []
created_date: '2026-05-15 21:39'
labels:
  - watchlists
  - stage5
  - backend
dependencies: []
references:
  - Docs/API-related/Watchlists_API.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
parent_task_id: TASK-349.2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend-only report evidence/readiness contract for Watchlists reports. This slice should add deterministic schemas and helper logic for immutable report evidence snapshots, source diversity, included/excluded item trails, alert evidence summaries, and readiness warnings without wiring it into output creation yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Report evidence/readiness schemas cover presets, readiness states, warnings, evidence items, alert evidence, excluded items, source summary, and output evidence responses.
- [ ] #2 A focused backend helper builds deterministic JSON-serializable report evidence snapshots from supplied job, run, item, source, and alert rows.
- [ ] #3 Readiness evaluation returns ready, warning, blocked, and legacy/live-only-compatible states with stable warning codes for empty reports, single-source evidence, missing provenance, no CTI alert evidence, stale news updates, and unreviewed queued items.
- [ ] #4 Focused pytest coverage validates happy path, warning path, blocked path, and JSON-serializable output.
- [ ] #5 No network, LLM, or output persistence behavior is added in this slice.
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
