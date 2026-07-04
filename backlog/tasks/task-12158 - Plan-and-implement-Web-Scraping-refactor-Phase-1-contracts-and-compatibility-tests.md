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
- tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py
- tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
- tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py
- backlog/tasks/task-12158 - Plan-and-implement-Web-Scraping-refactor-Phase-1-contracts-and-compatibility-tests.md
updated_date: 2026-07-04 23:08
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 of the Web_Scraping refactor is implemented as an additive contract and compatibility-test slice. Runtime behavior was not moved. The new contracts package is stdlib-only and protected by import-boundary tests; compatibility tests now lock current legacy imports, pre-scrape analyzer attachment for direct and enhanced scrapers, policy-denial dict shapes, and WebSearch initialized/processed result shapes. Focused Web_Scraping/WebScraping/WebSearch verification, compile, Bandit, import guardrail, and diff hygiene all pass.
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
