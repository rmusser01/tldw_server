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
Addressed follow-up design review findings before implementation planning. Patched the PRD for dirty-save conflict semantics, unlinked/ambiguous task mutations, bounded reconciliation-aware discovery, task text update projection/version rules, and nested markdown delete semantics.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up design fixes applied after human review. Added hard rules for stale dirty saves, unlinked/ambiguous projection-changing mutations, bounded discovery work limits and incomplete-result metadata, text-update version checks, and nested checklist delete conflicts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-512 spec refined again after follow-up design review. The PRD now explicitly requires stale dirty saves to conflict or merge rather than overwrite remote/autonomous projections, projection-changing writes against unlinked/ambiguous tasks to conflict unless repaired/relinked, broad discovery to be bounded with incomplete reconciliation metadata, task text updates to use expected task/note versions, and nested child content to be preserved by default delete conflicts. Verification: reviewed spec diff and ran git diff --check. Bandit skipped because this is documentation/task-tracking only; no Python code was changed.
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
