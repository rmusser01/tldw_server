---
id: TASK-12005
title: Harden Logging module review findings
status: Done
assignee: []
created_date: 2026-06-23 20:42
updated_date: 2026-06-24 20:13
labels:
- logging
- observability
- review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated current-code review findings in `tldw_Server_API/app/core/Logging`: non-blocking system log persistence, redaction before buffer/file storage, UTC JSON timestamps, traceparent validation, safe env parsing, synchronized sink setup, and duplicate JSON logging helper cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 System log capture no longer blocks log callers on file lock or compaction work.
- [x] #2 System log buffer/file entries redact common secrets before persistence or admin exposure.
- [x] #3 JSON log timestamps are emitted as true UTC values.
- [x] #4 Invalid traceparent headers are not propagated into logging context.
- [x] #5 Malformed logging environment variables fall back safely.
- [x] #6 System log sink installation is concurrency-safe.
- [x] #7 Unused duplicate JSON logging helper is removed or reconciled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-logging-module-review-fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Touched files:
- `Docs/superpowers/plans/2026-06-23-logging-module-review-fixes.md`
- `backlog/tasks/task-12005 - Harden-Logging-module-review-findings.md`
- `tldw_Server_API/app/core/Logging/json_log_formatter.py`
- `tldw_Server_API/app/core/Logging/log_context.py`
- `tldw_Server_API/app/core/Logging/system_log_buffer.py`
- `tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py`
- `tldw_Server_API/tests/Logging/test_system_log_buffer.py`
- `tldw_Server_API/tests/Logging/test_trace_context.py`

Verification:
- Red phase confirmed 7 focused tests failed against the original behavior: redaction, non-blocking sink append, malformed env parsing, invalid log level fallback, concurrent sink setup, invalid traceparent handling, and UTC timestamp conversion.
- `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Logging tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Logging -f json -o /tmp/bandit_logging_task_12005.json`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py -q` passed with 43 tests.
- Rebased PR branch on latest `origin/dev` and addressed the new runtime lock-timeout coverage by making `_log_file_lock` use monotonic elapsed time, sleep only up to the remaining timeout, and avoid treating low timeout values as near-immediate stale-lock expiry.
- Post-rebase verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py -q` passed with 43 tests.
- Post-rebase verification: `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Logging tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py`
- Post-rebase verification: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Logging -f json -o /tmp/bandit_logging_task_12005_rebase.json`
- Post-rebase verification: `git diff --check`
- Addressed Qodo review comments: added docstrings and type hints for new helpers/tests, wrapped the traceparent regex, added pytest markers for touched tests, made `_log_sink` resilient to enqueue failures, persisted the file queue/worker across reloads, and expanded dedupe keys with tenant/correlation fields.

Known skips/blockers: full repository pytest was not run; the focused Logging and formatter tests cover the reviewed module scope. Backlog CLI/MCP was unavailable for task creation because the CLI index referenced a missing task file, so this task was created manually with user approval.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Logging module and addressed the latest PR review pass: system-log redaction now treats sensitive structured keys as secrets, internal diagnostics avoid raw exception text, dedupe keys include `event`, UTC formatter coverage uses exact output, sink tests cover structured secret extras, and traceparent tests cover mixed-case normalization. Focused Logging/formatter tests, Ruff, compileall, Bandit, and diff whitespace checks passed.
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
Reopened to rebase PR #2487 on latest `dev` and address the new CodeRabbit review comments on commit `824bc1d5a748295cf311fbdf99b25b4d65df4cf5`.
Follow-up PR #2487 pass after rebasing on latest `origin/dev`:
- Addressed CodeRabbit comments by redacting values for sensitive structured extra keys, omitting raw exception text from internal diagnostics, adding `event` to system-log dedupe keys, asserting exact UTC JSON formatter output, covering structured-secret redaction through the Loguru sink helper, and covering mixed-case traceparent normalization.
- Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py -q` passed with 47 tests.
- Verification: `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Logging/log_context.py tldw_Server_API/app/core/Logging/system_log_buffer.py tldw_Server_API/tests/Logging/test_system_log_buffer.py tldw_Server_API/tests/Logging/test_trace_context.py tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py` passed.
- Verification: `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Logging tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py` passed.
- Verification: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Logging -f json -o /tmp/bandit_logging_task_12005_rebase_comments.json` passed with 0 results and 0 errors.
- Verification: `git diff --check` passed.
Follow-up review pass completed locally after rebase; staged into the amended PR commit before pushing.
Final review-thread cleanup: added short docstrings to touched private system-log helper functions and a return type for `_log_file_lock()` to satisfy the remaining Qodo helper-docstring/type thread before resolving stale bot conversations.
- Verification after docstring cleanup: `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Logging/log_context.py tldw_Server_API/app/core/Logging/system_log_buffer.py tldw_Server_API/tests/Logging/test_system_log_buffer.py tldw_Server_API/tests/Logging/test_trace_context.py tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py` passed.
- Verification after docstring cleanup: `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Logging tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py` passed.
- Verification after docstring cleanup: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging tldw_Server_API/tests/Infrastructure/test_json_log_formatter.py -q` passed with 47 tests.
- Verification after docstring cleanup: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Logging -f json -o /tmp/bandit_logging_task_12005_rebase_comments_docstrings.json` passed with 0 results and 0 errors.
- Verification after docstring cleanup: `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
