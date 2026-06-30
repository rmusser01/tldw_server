---
id: TASK-481
title: Design Watchlists durable audio artifact projection
status: Done
labels:
- watchlists
- design
- audio
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-watchlists-durable-audio-artifact-projection-design.md
modified_files:
- Docs/superpowers/specs/2026-05-22-watchlists-durable-audio-artifact-projection-design.md
- backlog/tasks/task-481 - Design-Watchlists-durable-audio-artifact-projection.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for durable Watchlists audio artifact projections. Workflows artifacts remain canonical; Watchlists mirrors a compact artifact graph into run stats and canonical output metadata with proactive best-effort sync plus lazy read-repair.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Durable audio artifact projection spec exists and records Workflows-as-canonical projection requirements.
- [x] #2 Implementation planning proceeded through `TASK-482`.
- [x] #3 Implementation proceeded through `TASK-483`.
- [x] #4 This reconciliation changed task metadata only and did not modify design/spec content.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design/spec only. Captures durable Watchlists audio artifact projection with Workflows as canonical, Watchlists compact mirror, audio_request_id, lazy read-repair, best-effort proactive projection, retry stale-state handling, and test scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Second design review found and addressed four implementation-critical gaps: real Workflows run correlation metadata is not currently persisted despite endpoint tests using metadata_json; retry idempotency must include audio_request_id; proactive projection must poll Workflow run state rather than Scheduler task terminal state and must not use an unensured queue; admin target_user_id download links need target-aware handling. The spec now documents these as required constraints before implementation planning.

Metadata reconciliation note: no design/spec content was changed in this cleanup. The implementation plan and implementation records are complete in `TASK-482` and `TASK-483`, so this stale `In Progress` status is closed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and refined Docs/superpowers/specs/2026-05-22-watchlists-durable-audio-artifact-projection-design.md. The spec defines Workflows as canonical artifact storage, a compact Watchlists projection in run stats and canonical output metadata, real correlation metadata requirements, audio_request_id retry disambiguation including idempotency-key changes, lazy read-repair, best-effort Watchlists-owned proactive projection with queue safeguards, fallback artifact handling, stale-state handling, status normalization, raw URI boundaries, metadata merge requirements, admin target-user link constraints, frontend scope, and backend/frontend/API test coverage. Verification: git diff --check passed. Bandit skipped because this task only changes documentation and Backlog metadata.
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
