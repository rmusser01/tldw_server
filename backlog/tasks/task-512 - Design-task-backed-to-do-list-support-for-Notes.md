---
id: TASK-512
title: Design task-backed to-do list support for Notes
status: Done
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
modified_files:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a PRD/design spec for first-class task-backed to-do list support in Notes, including /notes, Notes Dock, MCP Unified tools, permissions, reconciliation, audit notices, and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec captures approved product model, architecture, data flow, MCP permissions, error handling, and test strategy.
- [ ] #2 Spec preserves markdown portability with no hidden IDs while defining durable first-class task records.
- [ ] #3 Spec is reviewed and committed with the Backlog task linked/updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Refined the approved PRD/spec based on human design review findings before implementation planning. Addressed dirty-note checkbox semantics, markdown identity limits without hidden IDs, rollout phasing, task deletion, reconciliation-aware discovery, autonomous activity notices, parser normalization, MCP delete/create behavior, and metadata version checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec re-review completed after revisions. Reviewer reported no remaining must-fix contradictions and approved the revised spec. Advisory clarifications were applied for transactional delete behavior, metadata projection version checks, and activity-notice/non-notification wording.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-512 spec refined after design review. The updated PRD now resolves dirty-note checkbox behavior, identity without hidden IDs, reconciliation-aware task discovery for existing notes, canonical task soft-delete/projection removal semantics, autonomous activity retention/read state, task creation conflicts, completed_at reopen behavior, and parser normalization. Spec re-review approved with no remaining must-fix issues. Bandit skipped because this is documentation/task-tracking only; no Python code was changed.
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
