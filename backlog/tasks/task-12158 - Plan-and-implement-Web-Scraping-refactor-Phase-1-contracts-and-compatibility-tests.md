---
id: TASK-12158
title: Plan and implement Web_Scraping refactor Phase 1 contracts and compatibility
  tests
status: Done
created_date: 2026-07-04 22:23
labels:
- web-scraping
- refactor
- phase-1
- contracts
priority: High
references:
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
- Docs/Design/web_scraping_refactor_import_inventory.json
- backlog/tasks/task-12027 - Implement-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md
modified_files:
- Docs/Design/WebScraping.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
- Docs/Design/web_scraping_refactor_import_inventory.json
- Docs/superpowers/plans/2026-07-04-web-scraping-phase-1-contracts-compatibility.md
- tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py
- tldw_Server_API/app/core/Web_Scraping/contracts/statuses.py
- tldw_Server_API/app/core/Web_Scraping/contracts/errors.py
- tldw_Server_API/app/core/Web_Scraping/contracts/requests.py
- tldw_Server_API/app/core/Web_Scraping/contracts/results.py
- tldw_Server_API/app/core/Web_Scraping/contracts/conversion.py
- tldw_Server_API/app/core/DB_Management/sqlite_schema_helpers.py
- tldw_Server_API/app/core/AuthNZ/repos/media_ingest_dedupe_repo.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py
- tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
- tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py
- tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py
- backlog/tasks/task-12158 - Plan-and-implement-Web-Scraping-refactor-Phase-1-contracts-and-compatibility-tests.md
updated_date: 2026-07-04 18:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create and execute the Phase 1 implementation plan for the Web_Scraping modular refactor. Scope is limited to internal contracts, status/error/conversion helpers, public compatibility contract tests, and verification; no runtime behavior should move in this phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phase 1 implementation plan is written under Docs/superpowers/plans with a critical self-review and bite-sized TDD tasks.
- [x] #2 Internal contracts package defines request/result/status/error/conversion boundaries needed by later phases without importing legacy wrapper files.
- [x] #3 Public compatibility contract tests capture current dict-shaped return contracts and importable entry points for Article_Extractor_Lib, enhanced_web_scraping, and WebSearch_APIs.
- [x] #4 Phase 1 does not move runtime behavior or change public API shapes.
- [x] #5 Focused Web_Scraping/WebScraping/WebSearch verification, import guardrail tests, git diff hygiene, and Bandit on touched production code are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the approved refactor design and Phase 0 import inventory. 2. Write a Phase 1 implementation plan with test-first tasks for contracts, conversion helpers, import boundaries, and compatibility contract tests. 3. Execute the plan in a bounded behavior-preserving slice. 4. Verify and update Backlog with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Phase 1 contracts and compatibility guardrails.

Plan and review:
- Wrote `Docs/superpowers/plans/2026-07-04-web-scraping-phase-1-contracts-compatibility.md`.
- Performed design self-review and subagent review before implementation.
- Addressed review findings before and after implementation: direct article preflight coverage, WebSearch processed/initialized shape split, preflight conversion public shape, contract import boundary, failure converter public-shape leakage, and enhanced robots-denial compatibility.

Implementation:
- Added stdlib-only `tldw_Server_API.app.core.Web_Scraping.contracts` package with status, failure, request, result, and conversion contracts.
- Added contract tests for immutability, conversion shapes, policy failure shapes, WebSearch shapes, and import boundary.
- Added no-network compatibility tests for inventory imports, direct article and enhanced preflight payloads, policy-denial dictionaries, WebSearch initialized and processed result shapes, extraction pipeline shape, and scraping job shape.
- Updated the legacy JS-required metric test to use pytest-asyncio instead of bare `asyncio.Future()`/default event loop assumptions under Python 3.14.
- Regenerated Web_Scraping import inventory artifacts after adding compatibility imports.
- Added Phase 1 note to `Docs/Design/WebScraping.md`.

Verification:
- `python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py` -> 14 passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py` -> 12 passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py` -> 10 passed.
- `python -m pytest -q -x --tb=short tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/WebScraping tldw_Server_API/tests/WebSearch` -> 371 passed, 13 skipped.
- `python -m py_compile` on all contracts modules -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Web_Scraping/contracts -f json -o /tmp/bandit_web_scraping_phase1_contracts.json` -> 0 findings.
- `git diff --check` -> passed.
Post-PR code review requested via `superpowers:requesting-code-review` on PR #2636. Reviewer found no Critical issues and three Important contract risks to fix before merge:
- `extraction_result_to_public_dict()` must preserve legacy extraction diagnostics such as `extraction_trace`, `extraction_strategy`, and `extraction_strategy_order`.
- `search_results_to_public_dict()` must not let `extra_fields` overwrite canonical public keys like `results`, `error`, or `warnings`.
- Search domain allow/deny normalization must avoid string-to-character-list corruption and provide a legacy-compatible representation for later adapters.

Next step: add failing tests first, then update contracts/conversion helpers and rerun focused verification.
Post-review fixes implemented:
- Added `ExtractionResult.extra_fields` and guarded public conversion so legacy diagnostics (`extraction_trace`, `extraction_strategy`, `extraction_strategy_order`) are preserved without allowing extras to overwrite canonical article fields.
- Guarded `SearchResultsPayload.extra_fields` so provider extras cannot replace canonical initialized WebSearch keys such as `results`, `error`, or `warnings`.
- Normalized string/list domain filters in `SearchResultsPayload` and added `search_request_to_legacy_kwargs()` so future adapters can pass list-compatible `site_whitelist` / `site_blacklist` values to the legacy WebSearch surface.
- Added regression tests for all three review findings.

Post-review verification:
- `python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py` -> 19 passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py` -> 12 passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py` -> 10 passed.
- `python -m py_compile tldw_Server_API/app/core/Web_Scraping/contracts/*.py` -> passed.
- `python -m pytest -q -x --tb=short tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/WebScraping tldw_Server_API/tests/WebSearch` -> 376 passed, 13 skipped.
- `python -m bandit -r tldw_Server_API/app/core/Web_Scraping/contracts -f json -o /tmp/bandit_web_scraping_phase1_contracts_review_fixes.json` -> 0 findings.
- `git diff --check` -> passed.
- Follow-up code-review subagent confirmed the three Important findings are closed and found no new Critical, Important, or Minor issues.
PR #2636 remediation pass requested: rebase branch on latest `dev`, inspect and address all PR issues/comments/check findings, then rerun verification and update the PR branch.

PR #2636 latest-dev remediation:
- Rebased `codex/web-scraping-phase-0-inventory` onto `origin/dev` at `6b727b221e55646eba663a03571e38302f7fafc2`.
- Addressed Qodo review comments by adding explicit docstrings across `Web_Scraping/contracts`, replacing mixed per-test markers in the Phase 1 compatibility tests with one module-level `unit` marker, moving SQLite column-existence logic to `DB_Management.sqlite_schema_helpers`, and removing the broad `contextlib.suppress(Exception)` fallback in the media ingest dedupe repo.
- Fixed stale `source_db_path` transcript reuse by falling back to the current default source user's media DB path when the stored path is stale, preserving cross-user transcript reuse without reprocessing.
- Added a regression test for stale source DB path fallback and regenerated Web_Scraping import inventory artifacts after line-number changes.

Post-latest-dev verification:
- `python -m py_compile tldw_Server_API/app/core/Web_Scraping/contracts/*.py tldw_Server_API/app/core/DB_Management/sqlite_schema_helpers.py` -> passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` -> 7 passed.
- `python -m pytest -q --tb=short tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPRunnerClientPermissions` -> 4 passed.
- `python -m pytest -q -x --tb=short tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/WebScraping tldw_Server_API/tests/WebSearch` -> 376 passed, 13 skipped.
- `python -m bandit -r tldw_Server_API/app/core/Web_Scraping/contracts tldw_Server_API/app/core/DB_Management/sqlite_schema_helpers.py tldw_Server_API/app/core/AuthNZ/repos/media_ingest_dedupe_repo.py tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py -f json -o /tmp/bandit_pr2636_rebase_comments_after_latest_dev.json` -> 0 findings.
- Contract docstring AST check -> passed.
- `git diff --check` -> passed.
- Review-only subagent caught the moved `origin/dev` ancestry and untracked helper risks; both are addressed by the second rebase and explicit staging of the new helper file.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 of the Web_Scraping refactor is implemented as an additive contract and compatibility-test slice. Runtime behavior was not moved. The branch is rebased onto latest `dev` at `6b727b221e55646eba663a03571e38302f7fafc2`. Post-PR comments are addressed: contract docstrings are explicit, Phase 1 compatibility tests use a single approved marker, SQLite schema migration logic lives in DB_Management, stale media ingest transcript source DB paths fall back to the current default source DB path, and import inventory artifacts are regenerated. Focused media ingest and ACP checks, broad Web_Scraping/WebScraping/WebSearch verification, compile, Bandit, contract docstring scan, and diff hygiene all pass.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched production code or documented skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
