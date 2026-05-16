---
id: TASK-349.2.1
title: Stage 5A Watchlist report evidence readiness contract
status: Done
assignee: []
created_date: '2026-05-15 21:39'
updated_date: '2026-05-16 01:41'
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
- [x] #1 Report evidence/readiness schemas cover presets, readiness states, warnings, evidence items, alert evidence, excluded items, source summary, and output evidence responses.
- [x] #2 A focused backend helper builds deterministic JSON-serializable report evidence snapshots from supplied job, run, item, source, and alert rows.
- [x] #3 Readiness evaluation returns ready, warning, blocked, and legacy/live-only-compatible states with stable warning codes for empty reports, single-source evidence, missing provenance, no CTI alert evidence, stale news updates, and unreviewed queued items.
- [x] #4 Focused pytest coverage validates happy path, warning path, blocked path, and JSON-serializable output.
- [x] #5 No network, LLM, or output persistence behavior is added in this slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 5A implementation in worktree .worktrees/watchlists-stage1a. Scope is backend schemas, deterministic report evidence/readiness helper, and focused pytest coverage only.

Verification recorded for Stage 5A. Red check: importing WatchlistOutputEvidenceResponse failed before implementation. Green checks: focused Stage 5A report evidence tests pass; regression Watchlists selectors pass; git diff --check passes; Bandit reports 0 errors and 0 findings for touched backend files. Frontend/browser QA intentionally not run because this slice adds backend schemas and deterministic helper logic only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added first-class Watchlist report evidence/readiness contract schemas and a deterministic backend helper for immutable report evidence snapshots. The helper normalizes supplied job/run/item/source/alert rows into JSON-safe evidence payloads, computes source and alert summaries, and returns readiness states/warnings for empty reports, weak provenance, single-source evidence, CTI reports without alert evidence, stale news briefings, unreviewed queued items, and legacy live-only output compatibility.
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
