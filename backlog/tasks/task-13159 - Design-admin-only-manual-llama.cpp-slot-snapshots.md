---
id: TASK-13159
title: Design admin-only manual llama.cpp slot snapshots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 02:14'
updated_date: '2026-09-05 02:17'
labels: []
dependencies: []
documentation:
  - Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define safe manual prompt-cache save and restore for server-managed llama.cpp runtimes, with automatic behavior deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Written design covers first-time and power-user workflows, API boundaries, storage, compatibility, failure recovery, and a wireframe.
- [x] #2 Proposed ADR records ownership and lifecycle decisions and alternatives.
- [x] #3 Design is self-reviewed and ready for requester review; no runtime implementation is claimed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Inspect current dev and upstream contract; write linked design and proposed ADR; check consistency and references; request written-spec review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Written design and proposed ADR-043 cover manual admin-only managed-runtime snapshots. Source review baseline: origin/dev c5dfe0ff73; upstream server API documentation checked. Self-review covered authorization, storage ownership, uncertain outcomes, lifecycle conflicts and explicit non-goals. Runtime tests and Bandit skipped: documentation-only. Written-spec approval remains pending; no production functionality implemented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design and wireframe ready for requester review. Implementation planning follows written-spec approval.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
