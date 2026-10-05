---
id: TASK-13446
title: 'WebSearch_APIs hygiene: evict inline smoke tests, fix Google args, remove
  Bing remnants'
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1 Stage 3. Plan: Docs/plans/2026-10-01-due-debt-sweep-implementation-plan.md. tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py lines ~1547-1820 inline test_perform_websearch_* functions; Google arg-formatting FIXME at :1747; Bing deprecated remnants. TDD per fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Executed 2026-10-03. 29 zero-assert inline functions evicted (scope expanded from the 8 surveyed - same class); Bing trio removed (independent copies live in Web_Search.py, untouched); Google CSE cr-normalization + sort-drop fixes with red->green unit tests; sanitizer tests adapted with lazy-redaction coverage moved to the live google parse path. 53 passed / 13 deselected; Bandit 3 LOW pre-existing.
Review fix round 1: external smokes self-gated via RUN_EXTERNAL_API_TESTS (C1, CI-safe now); sort comment corrected (C1/M3); brave test renamed (M4); count corrected to 27 test functions + 4 helper defs (M2).
Renumbered 2026-10-04 from TASK-13401 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
Qodo round (2026-10-04, PR #3155): search_web_google forwards date sort expressions (date, date:r:..., date:d:s) as the documented CSE sort parameter and drops other values such as the configured default relevance; the previous revision dropped every value, which silently lost date sorting for API and Chatbook research clients. The ignored-value debug log now uses a loguru {} placeholder (the %r form never rendered). yandex removed from the opt-in live smoke list because search_web_yandex is an unimplemented stub. Declined: mocking the RUN_EXTERNAL_API_TESTS smoke cases -- they are the opt-in live replacement for the inline smoke functions and stay skipped in CI; mocked coverage is the unit tests beside them. Tests: date sort forwarded x3, unsupported value dropped x3.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Production module free of inline smoke tests; two real Google CSE request-formatting bugs fixed with tests.
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
