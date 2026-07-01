---
id: TASK-12083
title: Address PR 2564 review comments after dev rebase
status: Done
assignee: []
created_date: '2026-07-01 02:14'
updated_date: '2026-07-01 02:15'
labels:
  - codeql
  - pr-review
  - security
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2564'
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2564 is rebased onto latest origin/dev.
- [x] #2 All open CodeQL/review-thread issues are either fixed or explicitly documented.
- [x] #3 Focused frontend/Python regression tests pass.
- [x] #4 Bandit is run on touched Python source and new findings are fixed or documented as pre-existing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created because TASK-12076 has duplicate task files and the Backlog CLI resolves the numeric id to the wrong legacy task. This task tracks the PR #2564 rebase/review follow-up work.

Rebased PR #2564 onto latest origin/dev, then addressed review threads covering notification redaction whitespace, persisted frontend history redaction migration, task metadata casefold indexing, navigation title cleanup, MCP path normalization failure handling, generated-file symlink outputs roots, RAG exemplar tenant/user roots, Local LLM wildcard probe handling, WebSearch log formatting, and dead safe_join branches.

Verification: bun run test:run lib/__tests__/history.test.ts; python -m pytest -q tldw_Server_API/tests/Monitoring/test_notification_service.py; python -m pytest -q tldw_Server_API/tests/Notes_Tasks/unit/test_service.py tldw_Server_API/tests/Media/test_media_navigation.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/tests/Storage/test_generated_file_helpers.py tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py tldw_Server_API/tests/RAG_NEW/unit/test_payload_exemplars.py; python -m pytest -q tldw_Server_API/tests/Personalization/test_companion_activity_adapters.py::test_persona_summary_and_tool_adapters_capture_compact_metadata; py_compile for touched Python app files; git diff --check.

Bandit: touched app-source run wrote /tmp/bandit_codeql_review.json and reports only 3 pre-existing low findings in WebSearch_APIs.py outside changed lines (B311 at lines 576 and 2803, B101 at line 2144). Touched monitoring/personalization tests run with B101 skipped wrote /tmp/bandit_codeql_review_tests.json and reported 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2564 review comments after rebasing onto latest dev. Added focused regressions for credential redaction migration, notification colon-whitespace redaction, Unicode casefold metadata validation, navigation title preservation, MCP path normalization fail-closed behavior, generated-file symlink outputs roots, and full-form IPv6 wildcard probes. Removed reviewed dead safe_join branches and obsolete nosec/noqa suppressions without weakening runtime sanitizer coverage.
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
