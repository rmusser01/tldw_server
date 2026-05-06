---
id: TASK-96.2
title: Implement Auto Chunking backend request contract parsing
status: Done
assignee:
  - codex
created_date: '2026-05-06 16:55'
updated_date: '2026-05-06 17:01'
labels:
  - backend
  - chunking
  - quick-ingest
  - auto-chunking
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first execution slice from the approved Auto Chunking plan: add backend request contract support for chunking_mode, auto_chunking_goal, and auto_chunking_use_llm across media forms and web/article JSON request models while preserving legacy behavior for requests that omit chunking_mode. This task is limited to schema/dependency parsing and focused contract tests; planner behavior and runtime media processing wiring are later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend schemas accept chunking_mode, auto_chunking_goal, and auto_chunking_use_llm for media chunking options and web/article request models.
- [x] #2 Media add and direct process form dependencies parse the Auto fields consistently and preserve legacy defaults when chunking_mode is omitted.
- [x] #3 Existing template/hierarchical chunking fields are parsed consistently across media add and direct process forms where schemas already expose them.
- [x] #4 Invalid chunking_mode and invalid auto_chunking_goal fail through normal validation, and perform_chunking=false with Auto fields remains valid for later no-op planning.
- [x] #5 Focused backend tests cover the new contract and parsing parity.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented backend request contract parsing for Auto Chunking fields across AddMediaForm, direct media process form dependencies, WebScrapingRequest, and IngestWebContentRequest. Added focused unit coverage in tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_request_contract.py.

Verification: RED run of the new contract test file failed on missing chunking_mode/auto fields as expected. GREEN run passed 13 tests. Broadened suite passed 24 tests: test_auto_chunking_request_contract.py, test_media_add_deps_error_mapping.py, test_process_endpoints_contract_parity.py, test_process_web_scraping_strategy_validation.py, and test_ingest_web_content_endpoint_sanitization.py. Bandit touched-scope check reported zero findings in /tmp/bandit_auto_chunking_contract.json. git diff --check passed.

Known skips: did not run the large legacy test_add_media_endpoint.py integration file because this slice is covered by focused direct dependency and nearby endpoint contract tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Auto Chunking request contract fields for backend schemas and form dependencies: chunking_mode, auto_chunking_goal, and auto_chunking_use_llm. Media add and process dependencies now parse Auto fields plus template/hierarchical parity fields, including JSON validation for hierarchical_template. Web/article request models now accept the same Auto contract. Added focused unit tests for valid Auto parsing, legacy missing-mode behavior, invalid value validation, disabled chunking with Auto fields, malformed hierarchical_template JSON, and web/article request models.
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
