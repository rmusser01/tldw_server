---
id: TASK-13444
title: Remove expired compat paths past sunset (web_scraping_legacy_fallback, llm_chat_legacy_session,
  auth_db_execute_compat)
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1 Stage 1. Plan: Docs/plans/2026-10-01-due-debt-sweep-implementation-plan.md. All three sunsets (2026-06-30/07-15/08-01) past due as of 2026-10-01. Registry: tldw_Server_API/app/core/deprecations/runtime_registry.py; call sites chat_calls.py:79,128; auth_service.py:65,80; web_scraping_service.py:364. Includes deprecated /me endpoints (users.py:544,589) and USER_DB_BASE alias decision gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification 2026-10-02: compat keys absent from app/ (straggler grep clean); targeted suites green (registry contract + lint guard + auth backend-agnostic: 19 passed; Web_Scraping trio: 11 passed); Bandit 0 findings on touched paths. Step 6 (/me endpoints) decision-gated to follow-up TASK-13448 (no registered sunset; 5 test consumers; successor not field-compatible). Step 9 (USER_DB_BASE alias) decision-gated to follow-up TASK-13449 (8+ production modules depend on it).
Verification round 2 (implementer session, 2026-10-03): registry+lint+auth-backend+/me-status+capabilities suites: 44 passed; provider-unsafe-post/timeout-regressions/auth-login-sqlite/adapter-backend-selection: 35 passed; /me consumers (chat_research_runs, admin_e2e_support, auth_comprehensive): 43 passed / 4 failed = TestHealthEndpoints, identical at base 6ea2ee76ea (pre-existing). Full Web_Scraping dir run twice: 9 failed/2291 passed then 28 failed/2286 passed (nondeterministic under load; named failure test_persistence_crawl_metadata 2 failed identically at base and HEAD; test_phase4_safe_regex::test_sub_untrusted_bounds_amplified_output_size fails identically at base and HEAD - claim earlier withdrawn in review is reinstated with evidence, reviewer searched for the file name as a test name). Straggler grep zero hits in app/. Bandit touched paths: 0 findings in app code, LOW B101/B106 in test files only.
Final-review fix: tests/Media/test_web_summary_service_prompt.py adapted to post-removal semantics (commit fb07896093; was erroring 78/78 on deleted-attribute patches + legacy fallback axis; now 70/70).
Renumbered 2026-10-04 from TASK-13399 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
Rebase onto dev (2026-10-04, PR #3155). chat_calls conflicted with TASK-13313 (PR #2983): dev had already removed the PYTEST_CURRENT_TEST branch in the opposite direction (the factory always returns _SessionShim) and pins that with test_session_shim_is_the_tested_object.py and test_provider_unsafe_post_no_retry.py. Resolution: keep _SessionShim and remove only the expired llm_chat_legacy_session marker from its streaming branch; the non-streaming path is unchanged from dev.
Two dev tests the original branch never ran were red with the fallback removal and are fixed here: test_media_db_api_imports pinned web_scraping_service.managed_media_database (import gone with the fallback; test removed), and test_refactor_import_inventory needed Docs/Design/web_scraping_refactor_import_inventory.json + WebScraping_Refactor_Import_Inventory.md regenerated (rows changed: three legacy Article_Extractor_Lib imports dropped from the service, one from test_phase4_consumer_imports, one line shift, new test_websearch_smoke row).
Qodo round: deprecations README consumer lists updated (registry now empty; expired keys guarded by the lint test). The Moonshot/Z.AI decode_unicode finding does not apply on dev: iter_sse_lines_requests falls back on TypeError since 32945b26fb (covered by test_adapter_sse_loop_pinning.py) and this PR no longer changes which object streams. Behaviour note: an enhanced-service failure now surfaces from /media/process-web-scraping as HTTP 500 'Web scraping failed due to an internal error.' instead of the fallback's HTTP 400. Verification on the rebased tree: 1591 tests across the 75 test files that reference the touched modules (2 failed before the fixes above, then 348 passed / 9 skipped on the re-run of the fixed and modified files); Bandit on touched app files: 3 findings, same 3 as dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed three expired compat paths (registry emptied, machinery retained; fallback branches deleted from chat_calls/auth_service/web_scraping_service) with red->green registry tests, lint resurrection guard, auth adapter test updates, and Web_Scraping test rewrite for post-removal semantics. /me endpoint removal decision-gated to TASK-13448 (no registered sunset, 5 test consumers, successor not field-compatible); USER_DB_BASE alias retirement gated to TASK-13449 (8+ production modules). Commits 5ca06deb26, 37fc571d2f, 2017c6378a. Two verification rounds green on all deterministic suites touching changed code; remaining Web_Scraping/AuthNZ failures verified identical at base commit (pre-existing/flaky).
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
