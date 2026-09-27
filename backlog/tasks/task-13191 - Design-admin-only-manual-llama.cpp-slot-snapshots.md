---
id: TASK-13191
title: Design admin-only manual llama.cpp slot snapshots
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 02:14'
updated_date: '2026-09-05 18:17'
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

Requester approved the written spec and chose subagent-driven implementation. Linked implementation plan and tasks13186-13188 created. Documentation-only verification recorded; runtime tests and Bandit not applicable to this design task.

Requester approved this second collision migration after PR2884 landed on dev: historical snapshot design ID13184 moved to13191. Buddy task13184 remains unchanged; design status and acceptance history preserved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Written design approved; implementation tracked separately.
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
