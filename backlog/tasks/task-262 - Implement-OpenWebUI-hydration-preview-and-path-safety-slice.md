---
id: TASK-262
title: Implement OpenWebUI hydration preview and path-safety slice
status: Done
assignee: []
created_date: '2026-05-11 15:52'
updated_date: '2026-05-11 16:01'
labels:
  - chatbooks
  - openwebui
  - implementation
dependencies:
  - TASK-261
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the OpenWebUI attachment hydration implementation plan: preview-safe data-root validation, file path resolution, reference extraction, and DB chat_file fallback without writing images or Media DB rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Data-root validation enforces ingestion-source allowed roots and requires webui.db plus uploads only when bytes are needed
- [x] #2 OpenWebUI file path resolution rejects traversal and symlink escapes and supports safe uploads/{file.id}_{file.filename} fallback
- [x] #3 Reference extraction reads preserved OpenWebUI import metadata and reports unsupported reference shapes without raw path leakage
- [x] #4 DB chat_file fallback uses preserved OpenWebUI source chat row ids and skips fallback when absent
- [x] #5 Focused pytest, diff checks, and Bandit for touched backend code are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 2 of the implementation plan locally because subagent execution is blocked by account quota. Use TDD: add path and preview/reference tests first, verify red, implement openwebui_hydration.py with dataclasses and pure service helpers, run focused pytest/regression/diff/Bandit checks, then update plan/task and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-11: Completed Stage 2 locally because subagent execution is blocked by account quota. Red run failed because openwebui_hydration.py did not exist; implementation added preview dataclasses, allowed-root validation, safe file-path resolution, basic byte classification, metadata reference extraction, and chat_file fallback using metadata.row_id only.

Verification: combined pytest for hydration path/service/db-helper/OpenWebUI DB adapter returned 29 passed; git diff --check clean; Bandit on openwebui_hydration.py wrote /tmp/bandit_openwebui_hydration_preview_service.json with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 complete: added OpenWebUI hydration preview/path service helpers and focused tests. Data roots are constrained to ingestion-source allowed roots, webui.db is required, uploads is required only when bytes are needed, traversal and symlink escapes are rejected, fallback upload paths resolve safely, basic file kind classification is byte-sniffed, message metadata refs are extracted, unsupported ref shapes are reported, and DB chat_file fallback uses preserved row_id without guessing from external_ref. No known skips or blockers for this slice.
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
