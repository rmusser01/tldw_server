---
id: TASK-96.3
title: Implement deterministic Auto Chunking planner
status: Done
assignee:
  - codex
created_date: '2026-05-06 17:02'
updated_date: '2026-05-06 17:07'
labels:
  - backend
  - chunking
  - quick-ingest
  - auto-chunking
dependencies:
  - TASK-96.2
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second execution slice from the approved Auto Chunking plan: add a pure deterministic planner that maps media type, source hints, content profiles, template matches, requested Auto goal, and AI-assist availability into existing chunking options plus serializable chunking_plan metadata. This task must not wire runtime media ingestion yet and must not make LLM calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pure backend planner module exists without FastAPI request, database, network, or LLM dependencies.
- [x] #2 Planner returns normalized chunking options and JSON-serializable chunking_plan metadata for Auto requests.
- [x] #3 Planner preserves legacy/manual/no-chunking markers by returning no Auto plan when chunking is disabled, chunking_mode is missing, or chunking_mode is manual.
- [x] #4 Planner handles document/PDF, audio/video, ebook, email, and web/article profile cases across balanced, qa_search, and navigation_summary goals.
- [x] #5 Planner records deterministic fallback reasons for missing AI adapter, template no-match/failure, and unavailable semantic capability where relevant.
- [x] #6 Focused unit tests cover planner behavior without external services.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pure planner module tldw_Server_API/app/core/Chunking/auto_planner.py with AutoChunkingProfile, AutoChunkingRequest, AutoChunkingPlan, AutoChunkingDecision, source/text profile builders, deterministic goal/media rules, JSON-safe metadata, and deterministic fallback reasons. Verification: RED import failure for missing request/plan types; GREEN python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py -v -> 8 passed; broader python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_request_contract.py -v -> 21 passed; Bandit on auto_planner.py -> zero findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the deterministic Auto Chunking planner slice without runtime ingestion wiring or LLM calls. The planner now returns existing chunker-compatible options plus serializable chunking_plan metadata for Auto mode, and preserves legacy/manual/no-chunking behavior by returning no Auto plan.
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
