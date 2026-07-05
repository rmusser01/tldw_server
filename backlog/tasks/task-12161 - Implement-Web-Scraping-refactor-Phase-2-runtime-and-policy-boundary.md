---
id: TASK-12161
title: Implement Web_Scraping refactor Phase 2 runtime and policy boundary
status: In Progress
created_date: 2026-07-05 05:10
labels:
- web-scraping
- implementation
- refactor
references:
- Docs/superpowers/plans/2026-07-05-web-scraping-phase-2-runtime-policy-boundary.md
- Docs/superpowers/specs/2026-07-04-web-scraping-phase-2-runtime-policy-boundary-design.md
- backlog/tasks/task-12160 - Plan-Web-Scraping-refactor-Phase-2-runtime-and-policy-boundary-implementation.md
modified_files:
- tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py
- tldw_Server_API/app/core/Web_Scraping/runtime/requests.py
- tldw_Server_API/app/core/Web_Scraping/runtime/responses.py
- tldw_Server_API/app/core/Web_Scraping/runtime/policy.py
- tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py
- tldw_Server_API/app/core/Web_Scraping/runtime/browser.py
- tldw_Server_API/app/core/Web_Scraping/runtime/sessions.py
- tldw_Server_API/app/core/Web_Scraping/runtime/timeouts.py
- tldw_Server_API/app/core/Web_Scraping/runtime/cancellation.py
- tldw_Server_API/app/core/Web_Scraping/policy/__init__.py
- tldw_Server_API/app/core/Web_Scraping/policy/adapters.py
- tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
- tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py
updated_date: 2026-07-05 06:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Phase 2 runtime and policy boundary implementation plan. Add runtime contracts, policy and fetch adapters, contract-only runtime modules, and wire only the Article_Extractor_Lib.scrape_article lightweight policy/fetch path while preserving preflight analyzer behavior and public compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime contracts and import-boundary tests are implemented.
- [ ] #2 Concrete policy adapter lives outside runtime and delegates to existing outbound policy.
- [x] #3 Default fetch adapter preserves central http_client simplified GET mode, curl backend support, and response normalization.
- [ ] #4 Article scrape path uses runtime policy/fetch adapters while preserving policy-before-preflight order, curl-to-httpx fallback, preflight payloads, public return dicts, and public function signature.
- [ ] #5 Focused Phase 2 tests and existing compatibility/hardening tests pass.
- [ ] #6 Bandit is run on touched Python scope and new findings are fixed or documented if pre-existing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-07-05-web-scraping-phase-2-runtime-policy-boundary.md using subagent-driven development. Rebase on latest origin/dev before Python edits, then execute Tasks 1-6 with review checkpoints.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-05: Started Task 1 (runtime request/response contracts only). Scope limited to runtime __init__/requests/responses plus test_phase2_runtime_contracts.py; Task 2+ files intentionally left untouched.
2026-07-05: Task 1 TDD verification completed. Red run: focused runtime contract test failed during collection with ModuleNotFoundError for tldw_Server_API.app.core.Web_Scraping.runtime. Green run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py -> 7 passed, 20 warnings. Bandit: full touched scope produced only pytest B101 assert_used findings in test_phase2_runtime_contracts.py; production runtime-only scan exited 0 with no findings. git diff --check exited 0.
2026-07-05: Started Task 1 review-fix pass for explicit boolean normalization, stronger runtime import-boundary checks, and immutability assertions. Scope remains limited to requests.py, responses.py, and test_phase2_runtime_contracts.py.
2026-07-05: Started Task 3 (default runtime fetch adapter). Scope limited to runtime/fetch.py, runtime/__init__.py, and test_phase2_runtime_adapters.py; Task 4 policy adapter and Task 5 Article_Extractor_Lib wiring intentionally left untouched.
2026-07-05: Task 3 TDD verification completed. Red run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py failed during collection with ModuleNotFoundError for runtime.fetch. Green run: same focused adapter test -> 3 passed, 12 warnings. Import-boundary run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py::test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules -> 1 passed, 8 warnings. Bandit full touched scope wrote /tmp/bandit_task3_fetch_adapter.json and reported only LOW B101 pytest assert findings in test_phase2_runtime_adapters.py; production runtime-only scan wrote /tmp/bandit_task3_fetch_adapter_runtime.json and had no results. git diff --check exited 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
