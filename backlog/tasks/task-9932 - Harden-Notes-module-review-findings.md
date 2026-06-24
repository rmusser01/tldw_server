---
id: TASK-9932
title: Harden Notes module review findings
status: Done
assignee: []
created_date: 2026-06-23 18:55
updated_date: 2026-06-24 03:36
labels:
- notes
- review
- security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the Notes module review findings under tldw_Server_API/app/core/: reject invalid Studio diagram sections, require optimistic concurrency for Studio regenerate, avoid diagram sidecar clobbering, bound Studio and keyword-search inputs, and close workflow NotesInteropService connections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Invalid Notes Studio diagram source_section_ids are rejected instead of broadening to all sections.
- [x] #2 Notes Studio regenerate requires caller expected_version and rejects stale versions.
- [x] #3 Diagram manifest updates only mutate diagram_manifest_json and detect concurrent sidecar changes.
- [x] #4 Notes Studio request fields and keyword token search inputs are bounded.
- [x] #5 Workflow notes adapter closes NotesInteropService user DB connections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_notes_module_review_hardening_9932.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Notes module review hardening: Studio regenerate now requires expected_version and rejects stale editor writes; diagram source_section_ids are validated; diagram manifest persistence updates only diagram_manifest_json with sidecar compare-and-swap; Studio request fields and keyword token search are bounded; workflow notes adapter closes NotesInteropService connections; NotesStudioService imports workflow adapters lazily; focused Studio API tests use deterministic adapters to avoid pulling the workflow/RAG stack.

Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py -q -> 14 passed, 41 warnings. source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py -q -> 11 passed, 30 warnings. source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_search_notes_rejects_excessive_keyword_tokens -q -> 1 passed, 10 warnings. source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_knowledge_adapters.py::test_notes_adapter_create_production_mode -q -> 1 passed, 10 warnings. source .venv/bin/activate && python -m compileall -q <touched app files> -> passed. Bandit report: /tmp/bandit_notes_module_review_hardening_9932.json -> 0 results, 0 errors, 0 skipped. No external documentation update was needed for this internal hardening.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Notes module review findings. Notes Studio regenerate now requires expected_version and rejects stale editor writes; diagram source section IDs are validated; diagram manifest writes use a narrow compare-and-swap update that only mutates diagram_manifest_json; Studio request schemas and keyword-token search inputs have bounds; workflow notes CRUD closes NotesInteropService connections. Focused regression tests pass and Bandit reported no findings.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/2478
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
