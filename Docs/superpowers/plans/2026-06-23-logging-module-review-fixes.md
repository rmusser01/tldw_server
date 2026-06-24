## Stage 1: Regression Coverage
**Goal**: Add focused failing tests for the accepted Logging review findings.
**Success Criteria**: Tests demonstrate the current failures for redaction, UTC timestamps, traceparent validation, safe env parsing, synchronized sink installation, and non-blocking log-file persistence.
**Tests**: Targeted pytest cases in `tests/Logging` and `tests/Infrastructure/test_json_log_formatter.py`.
**Status**: Complete

## Stage 2: Logging Hot-Path Hardening
**Goal**: Keep system log capture bounded and non-blocking for ordinary log callers.
**Success Criteria**: `_log_sink` stores redacted entries in memory and enqueues file writes without waiting for file locks or compaction.
**Tests**: New system log buffer tests plus existing `tests/Logging/test_system_log_buffer.py`.
**Status**: Complete

## Stage 3: Context and Formatter Safety
**Goal**: Tighten structured log formatter and inbound context behavior.
**Success Criteria**: JSON formatter emits real UTC timestamps; invalid traceparent headers are discarded; malformed logging env values fall back safely.
**Tests**: JSON formatter, trace context, and env reload tests.
**Status**: Complete

## Stage 4: Cleanup and Verification
**Goal**: Remove or reconcile unused duplicate JSON logging helper behavior, update task notes, and run focused verification.
**Success Criteria**: Relevant tests pass, touched Python compiles, Bandit scan completes for touched Logging code, and Backlog task records verification.
**Tests**: Focused pytest suite, py_compile, Bandit.
**Status**: Complete

## Stage 5: PR Rebase Follow-Up
**Goal**: Rebase the PR branch on latest `dev` and address new base-branch Logging test coverage.
**Success Criteria**: Branch rebases cleanly; `_log_file_lock` honors runtime lock timeout settings without oversleeping or treating very low timeout values as near-immediate stale-lock expiry.
**Tests**: Focused pytest suite, compileall, Bandit, `git diff --check`.
**Status**: Complete

## Stage 6: PR Review Remediation
**Goal**: Address actionable Qodo comments on style, test metadata, and Logging reliability/correctness.
**Success Criteria**: New helpers/tests have docstrings/type hints/markers where required; traceparent regex is wrapped; `_log_sink` cannot leak enqueue failures; reloads reuse file-writer worker state; dedupe preserves tenant-distinct entries.
**Tests**: Focused pytest suite, Ruff on touched files, compileall, Bandit, `git diff --check`.
**Status**: Complete
