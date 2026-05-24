---
id: TASK-463.5
title: Add Research Workspace source status drilldown
status: To Do
labels:
- research-workspace
- workspace
- source-status
- frontend
- phase-d
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Phase D trust/transparency slice by giving each Research Workspace source a focused status drilldown that explains lifecycle/readiness, source-of-truth, retry/stale state, timestamps, and next action without adding /workspace-playground aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SourcesPane exposes a keyboard-accessible status details action for sources with processing, failed, stale, retryable, or diagnostic status fields.
- [ ] #2 Status details show user-facing lifecycle/readiness summary, status code/message, source of truth, last refresh, retry eligibility, stale state, media/source identifiers, and practical next action copy.
- [ ] #3 The drilldown remains compact, does not duplicate the preview/annotation workflow, and uses existing Research Workspace visual patterns.
- [ ] #4 Focused Vitest coverage proves status details render and no /workspace-playground alias is introduced.
- [ ] #5 Implementation is frontend-only unless verification finds a backend/API gap.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use TDD. Add a focused SourcesPane test that fails because no status-details action/dialog exists. Implement minimal SourcesPane state/action/dialog using existing WorkspaceSource status fields and source-status formatting helpers. Run focused Vitest suite, route guard tests, CDP validation against live backend, and git diff hygiene. Bandit is not applicable unless Python backend files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
