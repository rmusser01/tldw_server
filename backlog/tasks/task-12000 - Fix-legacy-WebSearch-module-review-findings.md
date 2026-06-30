---
id: TASK-12000
title: Fix legacy WebSearch module review findings
status: Done
assignee: []
created_date: 2026-06-23 11:20
updated_date: 2026-06-24 19:48
labels:
- websearch
- legacy
- review-fix
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated review findings in `tldw_Server_API/app/core/WebSearch/Web_Search.py`, which is a legacy/parallel WebSearch implementation. Scope is secret-safe logging, non-interactive server behavior, provider failure propagation, bounded evidence payloads, stale docs cleanup, provider stub clarity, and runtime validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Google WebSearch logging does not include provider API keys.
- [x] #2 Legacy `generate_and_search` returns a structured `processing_error` when every provider call fails.
- [x] #3 Legacy `user_review` cannot block server execution on `input()`.
- [x] #4 Legacy aggregate evidence does not return full scraped article content by default.
- [x] #5 Legacy provider stubs are explicitly non-production or removed from advertised support.
- [x] #6 Focused tests, compile check, and Bandit touched-scope verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing regression tests for the accepted review findings.
2. Patch legacy `Web_Search.py` with minimal behavior changes.
3. Refresh `core/WebSearch/README.md` to mark the module as legacy/non-routable.
4. Run focused tests, compile check, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task created manually because Backlog MCP was unavailable and the Backlog CLI hung on search/list/create operations in this workspace. User approved the temporary manual fallback. The task was moved to TASK-12000 after unrelated untracked task files appeared with overlapping TASK-10000 IDs.

Red verification before production changes:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py -q` reported 10 failed and 4 passed for the new regressions.

Final verification:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py tldw_Server_API/tests/WebSearch/unit/test_deprecated_session_shims_removed.py tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py -q` passed with 42 passed and 97 warnings.
- `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/WebSearch tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py` passed.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/WebSearch -f json -o /tmp/bandit_legacy_websearch_review_fixes.json` exited 0; JSON reported `results: 0`, `errors: []`.
- `git diff --check -- <touched files>` passed.

Pull request:
- https://github.com/rmusser01/tldw_server/pull/2492
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the legacy `core/WebSearch` module by redacting sensitive Google request parameters in logs, surfacing all-provider failures as structured `processing_error` responses with warnings, replacing terminal-based `user_review` with a server-side fail-fast error, removing raw `original_content` from aggregate evidence payloads, making legacy provider stubs explicit, replacing DuckDuckGo `assert` validation with `ValueError`, and switching random delay jitter to `secrets.SystemRandom` so Bandit reports no findings. Updated the README to mark this package as legacy/non-routable and added focused regression tests.
Follow-up PR review pass rebased the branch onto latest `dev`, addressed all actionable review threads, and reran focused tests, compile, Bandit, and diff checks successfully.
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
2026-06-24 follow-up: Reopened to rebase PR #2492 on latest `dev` and address review comments/check failures before re-finalizing.
2026-06-24 follow-up results: Rebased PR #2492 onto latest `dev` and addressed the four actionable review comments: direct `logger.info` for the touched Google parameter log line, docstrings on new helper functions, removal of the extra `pytest.mark.asyncio` marker while retaining the module-level `unit` marker, and bounded `relevant_results` response/debug-log projection that omits `original_content` by default with an explicit `include_original_content` opt-in.

Follow-up verification:
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py tldw_Server_API/tests/WebSearch/unit/test_deprecated_session_shims_removed.py tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py -q` passed with 44 passed and 96 warnings.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/WebSearch tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py` passed.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/WebSearch -f json -o /tmp/bandit_legacy_websearch_review_fixes_rebased.json` exited 0; JSON reported `results: 0`, `errors: 0`.
- `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
