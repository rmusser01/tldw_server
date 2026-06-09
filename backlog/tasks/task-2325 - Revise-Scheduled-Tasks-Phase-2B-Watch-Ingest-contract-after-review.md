---
id: TASK-2325
title: Revise Scheduled Tasks Phase 2B Watch/Ingest contract after review
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 01:39'
labels:
  - scheduled-tasks
  - ux
  - prd
  - phase-2b
  - review
dependencies: []
references:
  - TASK-2324
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
modified_files:
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
  - >-
    backlog/tasks/task-2325 -
    Revise-Scheduled-Tasks-Phase-2B-Watch-Ingest-contract-after-review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revise the Phase 2B Watch/Ingest product contract after UX review. Tighten preview availability, split state models, add notification and source-intent capability contracts, clarify ingest destinations, duplicate policy, Home copy, extension redaction, and delivery slice gating before implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec makes preview a hard gate for Available and defines Limited availability without overclaiming first-time creation.
- [x] #2 Spec splits template capability, task lifecycle, run state, and result outcome models.
- [x] #3 Spec adds notification, source-intent capability, ingest destination, duplicate policy, result-destination, and redaction contracts.
- [x] #4 Spec clarifies delivery slices so frontend capability states cannot promote Watch/Ingest to Available before all gates pass.
- [x] #5 Verification is recorded and Bandit is documented as not applicable for docs-only work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Revised the Phase 2B Watch/Ingest product contract after UX review. The spec now makes preview a hard gate for Available, defines Limited availability, splits template capability/task lifecycle/run/result state models, adds source-intent capability, notification, ingest destination, duplicate policy, result-destination, and redaction contracts, and clarifies that the 2B.2 frontend shell cannot promote Watch/Ingest to Available before all gates pass.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the Phase 2B Watch/Ingest product contract after review. The revision hardens availability gates, prevents search/RAG/Home/notification overpromising, separates state models, adds missing product-facing contracts, and clarifies delivery sequencing before implementation planning. Verification: git diff --check passed; unresolved planning marker scan passed. Bandit is not applicable because this is documentation/backlog-only work.
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
