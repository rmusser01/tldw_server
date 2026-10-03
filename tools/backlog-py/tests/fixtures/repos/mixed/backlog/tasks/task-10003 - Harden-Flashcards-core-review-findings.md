---
id: TASK-10003
title: Harden Flashcards core review findings
status: Done
assignee: []
created_date: 2026-06-23 22:15
updated_date: 2026-06-24 03:52
labels:
- flashcards
- review-hardening
dependencies: []
references:
- tldw_Server_API/app/core/Flashcards
priority: high
modified_files:
- tldw_Server_API/app/core/Flashcards/apkg_exporter.py
- tldw_Server_API/app/core/Flashcards/apkg_importer.py
- tldw_Server_API/app/core/Flashcards/study_assistant.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/flashcards_module.py
- tldw_Server_API/tests/Flashcards/test_apkg_exporter.py
- tldw_Server_API/tests/Flashcards/test_apkg_importer.py
- tldw_Server_API/tests/Flashcards/test_study_assistant_service.py
- tldw_Server_API/tests/MCP_unified/test_flashcards_module_sanitization.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the validated current-code review findings in `tldw_Server_API/app/core/Flashcards`: APKG import resource caps, APKG export media cap propagation, assistant prompt grounding, import side-effect cleanup, empty APKG export handling, and exporter dead-code removal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 APKG import rejects oversized or excessive archive contents before unbounded reads.
- [x] #2 APKG export rejects oversized data URI media instead of silently preserving it.
- [x] #3 Study assistant prompts include bounded card context beyond the front text.
- [x] #4 APKG import does not leave imported assets attached to skipped or failed rows.
- [x] #5 Empty APKG exports return a clear controlled error.
- [x] #6 Focused Flashcards tests and touched-scope Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing-first regression tests for APKG import caps, data URI export caps, empty APKG export, import asset cleanup, and assistant prompt grounding.
2. Harden APKG importer archive scanning and media loading with explicit size and entry caps while keeping the parser side-effect callback compatible.
3. Harden APKG exporter cap propagation, empty export behavior, and remove the unused media extraction block.
4. Include bounded card/question context in study assistant prompts.
5. Run focused tests, py_compile, and Bandit on touched Flashcards scope; update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Touched files:
- IMPLEMENTATION_PLAN_flashcards_core_review_fixes_10003.md
- tldw_Server_API/app/core/Flashcards/apkg_exporter.py
- tldw_Server_API/app/core/Flashcards/apkg_importer.py
- tldw_Server_API/app/core/Flashcards/study_assistant.py
- tldw_Server_API/tests/Flashcards/test_apkg_exporter.py
- tldw_Server_API/tests/Flashcards/test_apkg_importer.py
- tldw_Server_API/tests/Flashcards/test_study_assistant_service.py

Verification:
- Red run: the five new focused regression tests failed before production changes.
- `source .venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/Flashcards tldw_Server_API/tests/Flashcards/test_apkg_exporter.py tldw_Server_API/tests/Flashcards/test_apkg_importer.py tldw_Server_API/tests/Flashcards/test_study_assistant_service.py -q` -> 27 passed.
- `source .venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/Flashcards tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_export_apkg_rejects_oversized_total_media tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_import_apkg_rejects_oversized_total_media -q` -> 2 passed.
- `source .venv/bin/activate && python -m py_compile tldw_Server_API/app/core/Flashcards/apkg_exporter.py tldw_Server_API/app/core/Flashcards/apkg_importer.py tldw_Server_API/app/core/Flashcards/study_assistant.py` -> passed.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Flashcards/apkg_exporter.py tldw_Server_API/app/core/Flashcards/apkg_importer.py tldw_Server_API/app/core/Flashcards/study_assistant.py -f json -o /tmp/bandit_flashcards_task_10003.json` -> 0 findings.

Known skips/blockers: full repository pytest was not run because this workspace contains many unrelated in-progress changes; focused Flashcards helper/service and APKG endpoint cap tests were run. The official Backlog CLI was unavailable for mutation because its index references a missing unrelated task file, so this task was created via the approved manual fallback.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Flashcards core APKG handling and assistant prompt grounding. APKG import now validates archive entry count, collection size, media mapping size, and mapped media bytes before reading large content; it also pre-validates note fields before invoking asset import callbacks. APKG export now raises clear errors for empty exports and oversized data URI media, and the dead duplicate media extraction block was removed. Study assistant prompts now include bounded serialized card/question context and recent history so fact-check/explain actions are grounded beyond the front text.
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
PR #2459 rebase/review follow-up:
- Rebased branch `codex/flashcards-core-review-fixes-10003` onto latest `origin/dev`.
- Addressed review findings by adding default APKG total media caps, preflight data-URI decoded-size checks, whitespace-tolerant data URI decoding, valid JSON assistant context compaction, helper docstrings, MCP empty APKG export handling, and markers/return annotations for the new focused tests.
- Reviewed the raw-SQL comment and kept the APKG collection SQL scoped inside the importer because it parses an uploaded Anki SQLite file rather than accessing application DB state through DB_Management.
- Verification after review fixes: Flashcards focused suite -> 30 passed; APKG endpoint cap tests -> 2 passed; MCP Flashcards sanitization suite -> 14 passed; py_compile -> passed; Bandit touched Python scope -> 0 findings (`/tmp/bandit_flashcards_pr_2459_rebase.json`).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
