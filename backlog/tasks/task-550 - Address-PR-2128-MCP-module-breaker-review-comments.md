---
id: TASK-550
title: Address PR 2128 MCP module breaker review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 20:38'
labels:
  - mcp
  - mcp-unified
  - review-fix
  - stage3
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2128'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated Qodo, CodeRabbit, and Gemini review feedback on PR 2128 after rebasing onto latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto latest origin/dev.
- [x] #2 Validated review findings for the fallback module circuit breaker are addressed with minimal scoped changes.
- [x] #3 Regression tests cover recovery_at wall-clock compatibility and half-open probe-slot limits.
- [x] #4 Focused pytest, Ruff, Bandit, and diff whitespace verification are recorded before push.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebase check completed: branch is up to date with origin/dev at 02a017e655.
- Validated review findings: Qodo call_async typing, recovery_at epoch compatibility, helper docstring, and half-open probe limit are applicable. Gemini and CodeRabbit overlap on the half-open probe-limit finding.
- Added regression coverage for fallback recovery_at using epoch time and half-open probe-slot rejection. The new tests failed before implementation as expected: recovery_at used monotonic time, and the second half-open probe was admitted until module timeout.
- Fixed fallback breaker behavior by using epoch time for _opened_at/recovery_at compatibility and tracking _half_open_in_flight against half_open_max_calls.
- Added call_async type hints and documented _is_circuit_breaker_open_error compatibility behavior.
- Cleaned touched test-file typing/broad-exception issues reported by Ruff.
- Focused pytest: .venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::TestBaseModule tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py -q -> 40 passed, 3 warnings.
- Ruff touched scope: All checks passed.
- Bandit touched implementation scope: 0 findings in /tmp/bandit_mcp_stage3i_review_fixes.json.
- Whitespace: git diff --check -> passed.
- Known skips/blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the PR 2128 fallback circuit-breaker review findings: typed call_async, epoch-compatible recovery_at, documented open-error compatibility detection, and enforced half-open probe concurrency limits with regression coverage.
<!-- SECTION:FINAL_SUMMARY:END -->

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
