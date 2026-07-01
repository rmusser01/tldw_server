# Main CodeQL Alerts Against Dev Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the open CodeQL findings reported on `refs/heads/main` in a branch based on `origin/dev`, then open a PR targeting `dev`.

**Architecture:** Treat the 100 alerts as a small set of repeated security classes: path containment, exception response sanitization, clear-text secret storage/logging, regex hardening, SQL identifier validation, and hash algorithm choice. Prefer shared helpers and existing local validation patterns over one-off suppressions. Add focused tests per helper or affected endpoint cluster so regressions are pinned to behavior.

**Tech Stack:** FastAPI, Python pathlib/regex/sqlite helpers, pytest, Bandit, TypeScript/React, Bun/Vitest where frontend tests are required.

---

## Stage 1: Alert Inventory and Applicability
**Goal**: Produce a compact inventory of open `main` CodeQL alerts and confirm every affected path exists on `origin/dev`.
**Success Criteria**: The task record names the alert classes, affected files, and any alerts deemed test-only or false-positive with a rationale.
**Tests**: Read-only GitHub API query plus local `git diff origin/main..origin/dev -- <affected paths>`.
**Status**: Complete

- [x] Query GitHub code-scanning alerts for `tool_name=CodeQL`, `state=open`, `ref=refs/heads/main`.
- [x] Query GitHub code-scanning alerts for `refs/heads/dev`.
- [x] Create branch `codex/main-codeql-alerts-dev` from `origin/dev`.
- [x] Record the final alert-class summary in Backlog task `TASK-12076`.

## Stage 2: Python Path and Exception Hardening
**Goal**: Fix Python `py/path-injection` and `py/stack-trace-exposure` alerts by validating path inputs at trust boundaries and returning sanitized errors.
**Success Criteria**: User-controlled paths are constrained to configured roots or explicit file allowlists; external HTTP errors do not include exception objects or traceback text.
**Tests**: Add or update pytest coverage for representative endpoints/helpers in each changed cluster.
**Status**: Complete

- [x] Add failing tests for at least one path traversal case per shared path helper or endpoint cluster.
- [x] Add failing tests for stack-trace exposure paths in audio voices, skills, RAG, and sidecar handlers where applicable.
- [x] Implement minimal validation/sanitization changes.
- [x] Run the focused pytest targets.

## Stage 3: Secret Handling and Logging
**Goal**: Fix clear-text secret storage/logging alerts by removing persisted frontend API keys where possible, using session-only storage for dev/test shims where persistence is required, and redacting logged sensitive values.
**Success Criteria**: Production frontend config no longer persists API keys to durable browser storage; test/e2e helpers use ephemeral fake credentials or local CodeQL suppressions only when the value is intentionally non-secret; Python logs redact API key-like values.
**Tests**: Add/update frontend unit tests for API-key persistence behavior and Python tests for log redaction.
**Status**: Complete

- [x] Add failing frontend tests around `useConfig` API-key persistence.
- [x] Add failing Python tests for WebSearch API credential logging redaction.
- [x] Implement minimal storage/logging changes.
- [x] Run the focused frontend and Python tests.

## Stage 4: Regex, SQL, and Hashing Hardening
**Goal**: Fix polynomial ReDoS, SQL injection, weak sensitive-data hashing, and bind-all-interface alerts.
**Success Criteria**: Regexes are bounded or replaced with simple parsers; dynamic SQL identifiers are allowlisted; sensitive hashes use keyed HMAC or a strong digest where appropriate; default bind host is loopback unless explicitly overridden.
**Tests**: Add regression tests for worst-case regex inputs, invalid SQL sort/filter identifiers, hash stability, and default bind host behavior.
**Status**: Complete

- [x] Add failing tests for each changed utility/cluster.
- [x] Implement minimal changes.
- [x] Run focused tests.

## Stage 5: Verification, PR, and Tracking
**Goal**: Verify the touched scope and publish the PR against `dev`.
**Success Criteria**: Focused tests pass, touched Python files pass Bandit, frontend checks for touched modules pass or documented skips are recorded, branch is pushed, and PR is opened against `dev`.
**Tests**: `python -m pytest ...`, `python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_main_codeql_alerts.json`, and relevant Bun/Vitest commands.
**Status**: Complete

- [x] Run focused pytest targets.
- [x] Run focused frontend tests or type checks for touched TypeScript modules.
- [x] Run Bandit on touched Python scope.
- [x] Update `TASK-12076` with touched files, verification results, and final summary.
- [x] Commit, push, and open PR against `dev`: https://github.com/rmusser01/tldw_server/pull/2564
