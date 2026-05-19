---
id: TASK-96.4
title: Wire Auto Chunking planner into backend ingestion paths
status: Done
assignee:
  - codex
created_date: '2026-05-06 17:09'
updated_date: '2026-05-06 17:30'
labels:
  - backend
  - chunking
  - quick-ingest
  - auto-chunking
dependencies:
  - TASK-96.3
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the third execution slice from the approved Auto Chunking plan: add a resolver that combines existing manual chunk option behavior with deterministic Auto plans, wire backend process/web/job paths to return or persist chunking_plan metadata, and preserve legacy/manual behavior. Keep this slice deterministic and do not add real LLM boundary calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resolver returns legacy/manual chunk options with no plan and Auto planner options with chunking_plan.
- [x] #2 Direct process endpoints attach chunking_plan metadata for Auto requests without changing legacy/manual responses.
- [x] #3 Web scraping and ingest-web-content JSON paths accept Auto fields and use the same resolver behavior.
- [x] #4 Async media ingest job results include chunking_plan when Auto planning is used.
- [x] #5 Persistence preserves safe_metadata.chunking_plan as JSON-safe nested metadata.
- [x] #6 Focused backend tests cover Auto wiring and legacy/manual preservation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented first backend wiring slice: added resolve_chunking_options_and_plan() in chunking_options.py, preserving legacy/manual prepare_chunking_options_dict behavior and routing chunking_mode=auto through the deterministic planner while ignoring stale manual fields. Updated media_ingest_jobs_worker to use the resolver and include chunking_plan in job results when Auto planning is active. Verification: RED resolver import failure before implementation; RED worker test showed stale manual method was still used; GREEN focused suite python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_resolver.py tldw_Server_API/tests/Chunking/test_auto_chunking_planner.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_request_contract.py -v -> 35 passed; Bandit on chunking_options.py and media_ingest_jobs_worker.py -> zero findings; git diff --check clean. Remaining for this task: direct process endpoint metadata, web/article JSON paths, and safe_metadata persistence.

Completed backend wiring slice: direct process endpoints finalize Auto plans from extracted content and attach metadata.chunking_plan; web scraping and ingest-web-content JSON paths forward Auto fields and record plan metadata; persistence now preserves JSON-safe safe_metadata.chunking_plan for AV and document-like writes. Verification: python -m pytest focused backend suite including resolver/planner/request/job/direct/web/persistence and media ingest integration -> 63 passed; Bandit touched backend scope -> zero findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backend Auto Chunking wiring is complete for TASK-96.4. The resolver is used by direct process endpoints, async ingest jobs, web scraping JSON paths, and persistence. Auto requests return or persist chunking_plan metadata while legacy/manual requests keep existing chunk option behavior. Focused pytest suite passed with 63 tests, Bandit reported zero findings, and diff whitespace checks were clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Acceptance criteria completed
- [x] #8 Tests or verification recorded
- [x] #9 Bandit run for touched backend code
- [x] #10 Plan documentation updated
- [x] #11 Final summary added
- [x] #12 Known skips or blockers documented
<!-- DOD:END -->
