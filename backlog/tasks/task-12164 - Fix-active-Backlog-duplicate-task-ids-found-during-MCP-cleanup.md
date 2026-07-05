---
id: TASK-12164
title: Fix active Backlog duplicate task ids found during MCP cleanup
status: Done
labels:
- backlog
- cleanup
- mcp
modified_files:
- backlog/tasks/task-12147 - Harden-audio-briefing-generation-before-Research-Workspace-reuse.md
- backlog/tasks/task-12164 - Fix-active-Backlog-duplicate-task-ids-found-during-MCP-cleanup.md
- backlog/tasks/task-12165 - Narrow-SQLite-memory-URI-filesystem-assertion.md
- backlog/tasks/task-12166 - Apply-shared-validation-gate-across-Research-Workspace-generated-artifact-pipelines.md
- backlog/tasks/task-12167 - Fix-PostgreSQL-setup-self-verify-timestamp-timezone-mismatch.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the active Backlog task-id collisions discovered while closing the first-run MCP setup stream. Preserve the canonical MCP first-run TASK-12148 record and renumber the unrelated duplicate records so id-based Backlog MCP/CLI lookups stop resolving the wrong task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active TASK-12148 collision is resolved without changing the MCP first-run task id.
- [x] #2 The active TASK-12160 collision is resolved without breaking direct Research Workspace WP2 documentation references.
- [x] #3 Focused duplicate-id checks for TASK-12148 and TASK-12160 return one active file each after the fix.
- [x] #4 The diff is limited to Backlog/Docs metadata and no runtime code changes are made.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered the unrelated active duplicate records while preserving ids with known downstream references. Kept TASK-12148 on the MCP first-run tool packs task and TASK-12160 on Research Workspace NotebookLM media outputs WP2. Moved SQLite memory URI assertion to TASK-12165, shared Research Workspace generated-artifact validation to TASK-12166, and PostgreSQL setup self-verify timestamp fix to TASK-12167. Updated TASK-12147's direct reference from TASK-12148 to TASK-12166. Verification: focused id scan returned one active file each for TASK-12148, TASK-12160, TASK-12165, TASK-12166, and TASK-12167; Backlog MCP task_view resolves TASK-12148 to the MCP first-run task and TASK-12160 to WP2; stale old-filename search returned no matches; git diff --check passed. Bandit not applicable because this changed only Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the active Backlog id collisions found during MCP cleanup. The canonical MCP first-run setup task remains TASK-12148, Research Workspace WP2 remains TASK-12160, and the unrelated duplicate records were renumbered to TASK-12165 through TASK-12167 with the direct Research Workspace dependency reference updated. Verification confirmed the focused ids resolve uniquely and Backlog MCP lookup now returns the intended TASK-12148/TASK-12160 records. No runtime code changed.
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
