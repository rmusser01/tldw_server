---
id: TASK-2324
title: Design Scheduled Tasks Phase 2B Watch/Ingest product contract
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 01:13'
labels:
  - scheduled-tasks
  - ux
  - prd
  - phase-2b
  - watchlists
dependencies: []
references:
  - TASK-2320
  - TASK-2322
  - >-
    Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md
  - >-
    Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
modified_files:
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
  - >-
    backlog/tasks/task-2324 -
    Design-Scheduled-Tasks-Phase-2B-Watch-Ingest-product-contract.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the product and UX contract for moving Scheduled Tasks Watch for new items and Ingest new content templates from handoff-only to safely actionable, while preserving Watchlists as the deep source/monitor/ingest workspace. Scope stays product/UX-first with backend/API work listed only as dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines when Watch and Ingest templates can move from handoff-only to available without duplicating or limiting Watchlists.
- [x] #2 Spec treats GitHub, YouTube, RSS, sites, advisories, publications, and other sources as examples inside source-agnostic Watch/Ingest intents.
- [x] #3 Spec covers capability health, preview, duplicate detection, created entity responses, task/detail links, failure reasons, result destinations, and recovery paths.
- [x] #4 Spec defines first-time-user and power-user flows for Watch and Ingest from /scheduled-tasks plus handoff/return behavior with Watchlists.
- [x] #5 Spec records backend/API dependencies only as contracts and does not prescribe implementation details.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the Phase 2B Watch/Ingest product contract spec. The spec defines source-agnostic Watch and Ingest intents, preserves Watchlists as the deep workspace, and lists capability health, preview, duplicate detection, creation response, deep-link, failure, result-destination, handoff, extension, and accessibility contracts required before the templates can become available from /scheduled-tasks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Phase 2B Watch/Ingest product contract spec and linked it from this Backlog task. The spec keeps GitHub/YouTube as examples only, defines when Watch/Ingest can move out of handoff-only, preserves Watchlists as the full source/monitor/ingest workspace, and records backend/API work only as product-facing dependencies. Verification: ran git diff --check after fixing an EOF whitespace issue; scanned the new spec and task record for unresolved planning markers with none found. Bandit is not applicable because this is documentation/backlog-only work.
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
