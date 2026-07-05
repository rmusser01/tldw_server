---
id: TASK-12161
title: Implement Web_Scraping refactor Phase 2 runtime and policy boundary
status: Done
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
updated_date: 2026-07-05 23:48
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Phase 2 runtime and policy boundary implementation plan. Add runtime contracts, policy and fetch adapters, contract-only runtime modules, and wire only the Article_Extractor_Lib.scrape_article lightweight policy/fetch path while preserving preflight analyzer behavior and public compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime contracts and import-boundary tests are implemented.
- [x] #2 Concrete policy adapter lives outside runtime and delegates to existing outbound policy.
- [x] #3 Default fetch adapter preserves central http_client simplified GET mode, curl backend support, and response normalization.
- [x] #4 Article scrape path uses runtime policy/fetch adapters while preserving policy-before-preflight order, curl-to-httpx fallback, preflight payloads, public return dicts, and public function signature.
- [x] #5 Focused Phase 2 tests and existing compatibility/hardening tests pass.
- [x] #6 Bandit is run on touched Python scope and new findings are fixed or documented if pre-existing.
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
2026-07-05: Started Task 3 review hardening follow-up. Scope limited to adding timeout forwarding assertion in test_phase2_runtime_adapters.py and switching runtime/fetch.py elapsed duration measurement from time.time() to time.monotonic().
2026-07-05: Task 3 review hardening verification completed. Red run after adding monotonic elapsed test failed as expected: test_default_fetch_client_measures_elapsed_with_monotonic_clock expected 2.5 but saw a near-zero time.time() delta. Green run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py -> 4 passed, 14 warnings. Import-boundary run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py::test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules -> 1 passed, 8 warnings. Bandit full touched scope wrote /tmp/bandit_task3_fetch_adapter_hardening.json and reported only LOW B101 pytest assert findings in test_phase2_runtime_adapters.py; production runtime-only scan wrote /tmp/bandit_task3_fetch_adapter_hardening_runtime.json and had no results. git diff --check exited 0.
2026-07-05: Task 4 concrete outbound policy adapter completed. Scope limited to tldw_Server_API/app/core/Web_Scraping/policy/__init__.py, tldw_Server_API/app/core/Web_Scraping/policy/adapters.py, and policy-adapter tests in test_phase2_runtime_adapters.py; runtime package and Article_Extractor_Lib intentionally left untouched. Red run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py -> expected ModuleNotFoundError for tldw_Server_API.app.core.Web_Scraping.policy after new tests. Green run: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py -> 58 passed, 122 warnings. Bandit production scan: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Web_Scraping/policy -f json -o /tmp/bandit_task4_policy_adapter.json -> exit 0, no findings.
2026-07-05: Task 5 article runtime boundary wiring completed. Added focused article boundary tests for runtime policy-before-preflight blocking, runtime httpx fetch, curl-to-httpx fallback, and TLS preflight curl advice. Updated scrape_article to use DefaultWebOutboundPolicyChecker and DefaultFetchClient via article-local adapters while preserving preflight ordering and downstream extraction/Playwright behavior. Verification: red run of test_phase2_article_runtime_boundary.py failed on missing _ARTICLE_POLICY_CHECKER as expected. Green runs: test_phase2_article_runtime_boundary.py -> 4 passed, 14 warnings; test_router_backend_selection.py -> 3 passed, 13 warnings; test_phase2_runtime_adapters.py plus test_phase2_runtime_contracts.py -> 58 passed, 122 warnings. Production Bandit scan of Article_Extractor_Lib.py wrote /tmp/bandit_task5_article_runtime.json and reported no findings. Full touched-file Bandit scan of Article_Extractor_Lib.py plus the two Task 5 pytest files produced expected LOW B101 assert_used findings in pytest files only. Review follow-up: rg found no production callers for _fetch_with_curl, but existing WebScraping tests still monkeypatch it, so the helper was retained per review instruction. git diff --check exited 0.
2026-07-05: Task 5 legacy WebScraping compatibility follow-up completed. Updated affected legacy article tests to patch _ARTICLE_POLICY_CHECKER and _ARTICLE_FETCH_CLIENT instead of the removed async decide_web_outbound_policy/_fetch_with_curl seams, preserving policy-block-before-network, TLS curl selection, JS Playwright selection, and curl backend assertions without live network. rg confirmed no remaining _fetch_with_curl references, so the private dead helper was removed from Article_Extractor_Lib.py. Remaining decide_web_outbound_policy references are sync helper or EnhancedWebScraper seams. Verification: legacy failing subset -> 6 passed, 18 warnings; test_phase2_article_runtime_boundary.py -> 4 passed, 14 warnings; test_router_backend_selection.py -> 3 passed, 13 warnings; test_phase2_runtime_adapters.py plus test_phase2_runtime_contracts.py -> 58 passed, 122 warnings; production Bandit /tmp/bandit_task5_article_legacy_compat.json -> 0 findings; git diff --check exited 0.
2026-07-05 final verification after clean rebase onto latest origin/dev completed. Verification base: HEAD e6bfdabe1fd2d3d67ef85eab59fc7da77f434108, merge-base with origin/dev 242297a2b8e5defeb5b1d5d74253ea75e787c4b0, branch initially ahead 16/not behind. Commands and results: focused Phase 2 tests (`test_phase2_runtime_contracts.py`, `test_phase2_runtime_adapters.py`, `test_phase2_article_runtime_boundary.py`) passed 62 tests; compatibility/hardening tests (`test_phase1_contracts.py`, `test_router_backend_selection.py`, `test_enhanced_web_scraping_guards.py`, `test_outbound_policy.py`, `test_http_client_fetch.py`) passed 49 tests; legacy compatibility subset passed 6 tests; `git diff --check` exited 0; Bandit production scan over runtime, policy, and Article_Extractor_Lib.py wrote `/tmp/bandit_web_scraping_phase2_runtime_policy.json`, exited 0, and reported zero results. Only unrelated untracked Config_Files files were present before finalization and were not staged.
2026-07-05 final code-review fix completed. Whole-branch review found the runtime DefaultFetchClient sent default article httpx fetches through the simplified no-method http_client path, weakening the old response-mode DNS pinning semantics. Fixed DefaultFetchClient so curl still uses the simplified backend path, while httpx/default fetches call http_client.fetch with method="GET" and url=... to retain response-mode egress/DNS-pin behavior. Updated adapter tests with a red/green regression for httpx response-mode call shape while preserving the curl no-method test. Verification after the fix: red test failed on positional simplified httpx call as expected; test_phase2_runtime_adapters.py -> 6 passed; focused Phase 2 group -> 62 passed; compatibility/hardening group -> 49 passed; legacy compatibility subset -> 6 passed; git diff --check exited 0; Bandit production scan wrote /tmp/bandit_web_scraping_phase2_runtime_policy_final_review_fix.json, exited 0, and reported zero results.
2026-07-05 PR #2665 review follow-up completed after rebasing onto latest origin/dev (origin/dev 903a139a0c). Addressed Gemini inline comments by explicitly rejecting boolean RuntimeTimeouts values and removing redundant Mapping dict() copies in runtime helper modules. TDD red run for boolean timeout rejection failed 6 cases as expected before the fix; focused boolean test then passed 6 cases. Final verification: focused Phase 2 group passed 68 tests; compatibility/hardening group passed 49 tests; legacy article subset passed 6 tests; git diff --check exited 0; Bandit production scan wrote /tmp/bandit_web_scraping_phase2_pr2665_review_comments.json, exited 0, and reported zero results.
2026-07-05 PR #2665 Qodo review follow-up completed after rebasing cleanly onto latest origin/dev. Addressed runtime helper docstrings, converted the article curl fallback message to Loguru formatting, added type annotations to new review helper fakes, removed noncompliant asyncio markers from new async unit tests, and added validated FetchRequest.timeout handling for negative, non-finite, and boolean values. TDD red run for timeout validation failed 6 cases as expected before the fix. Final verification: focused Phase 2 group passed 74 tests; compatibility/hardening group passed 49 tests; legacy article subset passed 6 tests; adapter cleanup rerun passed 6 tests; git diff --check exited 0; Bandit production scan wrote /tmp/bandit_web_scraping_phase2_qodo_followup.json, exited 0, and reported zero results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and finalized the Web_Scraping Phase 2 runtime and policy boundary. The work added runtime request/response contracts, default fetch and policy adapters, placeholder runtime modules, and article scrape wiring that preserves policy-before-network checks, preflight guidance, curl/httpx behavior, and compatibility contracts. After clean rebases onto latest origin/dev, final security review fixes, and PR review follow-ups for Gemini and Qodo, the required focused Phase 2, compatibility, legacy subset, whitespace, and Bandit checks all passed. No open verification blockers remain.
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
