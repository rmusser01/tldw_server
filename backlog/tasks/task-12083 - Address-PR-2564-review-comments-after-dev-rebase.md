---
id: TASK-12083
title: Address PR 2564 review comments after dev rebase
status: Done
assignee: []
created_date: '2026-07-01 02:14'
updated_date: '2026-07-01 02:28'
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

Follow-up after push: refreshed PR review threads showed additional CodeRabbit comments on the pre-fix commit. Reopening this task to address the remaining unresolved PR threads before finalizing.

Second follow-up addressed the remaining CodeRabbit threads: runtime/config credential scrubbing now removes legacy apiBearer and refreshToken, directory creation rejects existing symlinked directories, metric and companion log hashes use deployment-provided env secrets with process-local fallback warnings, RAG exemplar sinks reject dot-only tenant/user ids, skill import preview preserves public File messages while filtering traceback frames, and wildcard port probe docs now spell out loopback proxy semantics. Verification after follow-up: bun run test:run hooks/__tests__/useConfig.networking.test.tsx lib/__tests__/history.test.ts; bun install --frozen-lockfile followed by bun run test:run __tests__/extension/runtime-bootstrap.test.ts; python -m pytest -q tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/Metrics/test_sensitive_label_hashing.py tldw_Server_API/tests/Personalization/test_companion_activity_adapters.py tldw_Server_API/tests/RAG_NEW/unit/test_payload_exemplars.py tldw_Server_API/tests/Skills/unit/test_skills_service.py; previous focused backend suite rerun with 110 passed; py_compile passed for touched app files; git diff --check passed; Bandit app-source wrote /tmp/bandit_pr2564_app.json and now reports only the pre-existing WebSearch_APIs.py baseline findings (B311 lines 576/2803, B101 line 2144). Test Bandit wrote /tmp/bandit_pr2564_tests.json; findings are pre-existing fixture strings in test code, not new touched lines.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2564 CodeQL/review comments after rebasing onto latest dev. The follow-up fixes close the remaining CodeRabbit threads around frontend credential persistence, symlinked storage directories, configurable HMAC keys for metrics/companion log refs, dot-only RAG exemplar ids, skill import preview sanitization, and wildcard port probe documentation. Focused frontend and backend regressions now cover the changed behavior.
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
