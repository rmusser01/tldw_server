# Monitoring Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and superpowers:verification-before-completion before finalizing. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and fix validated Monitoring module review findings from TASK-2417.

**Architecture:** Keep changes scoped to Monitoring services, their immediate API schemas/endpoints, and Guardian/Topic monitoring DB helpers where atomic behavior is required. Prefer existing project patterns: FastAPI dependency authorization, DB_Management wrappers, Loguru sanitized logging, and focused pytest coverage.

**Tech Stack:** FastAPI, Pydantic, SQLite DB wrappers, pytest, Bandit.

---

## Stage 1: Validate Findings
**Goal:** Re-check each review finding against current code and decide whether it remains valid.
**Success Criteria:** Each finding is classified as valid or not applicable with concrete code evidence.
**Tests:** Read-only inspection only.
**Status:** Complete

- [x] Confirm notification settings routes are protected only by `SYSTEM_LOGS` and can change/read local notification paths.
- [x] Confirm SMTP STARTTLS failure is suppressed before login/send.
- [x] Confirm notification dispatch and digest buffering are unbounded.
- [x] Confirm self-monitoring partner approval uses the current user's Guardian DB and cannot resolve the owner's rule in per-user DB mode.
- [x] Confirm dedupe/escalation check-then-act behavior and decide a minimal atomicity fix.
- [x] Confirm topic regex compilation uses local heuristics instead of the shared regex safety validator.
- [x] Confirm notification module docs still describe webhook/email as simulated future behavior.

## Stage 2: Red Tests
**Goal:** Add failing regression tests for every validated behavior change.
**Success Criteria:** Focused tests fail for the expected reason before production code changes.
**Tests:** Focused pytest invocations for new/updated tests.
**Status:** Complete

- [x] Add endpoint tests for requiring stronger permission on notification settings updates/test sends while preserving read access.
- [x] Add notification-service tests for path bounding, STARTTLS fail-closed, queue/digest caps, and docs-relevant behavior.
- [x] Add self-monitoring endpoint/service test proving partner approval resolves the owner's DB rather than the approver's DB.
- [x] Add TopicMonitoringDB/service tests for atomic duplicate suppression and shared regex safety.

## Stage 3: Implement Fixes
**Goal:** Make the minimal production changes needed to satisfy the red tests.
**Success Criteria:** Focused tests from Stage 2 pass without broad refactors.
**Tests:** Same focused pytest commands as Stage 2.
**Status:** Complete

- [x] Split notification mutation authorization from log-read authorization.
- [x] Restrict notification sink paths to project-owned allowed roots and make recent-notification tail use the validated path.
- [x] Fail SMTP sends when `smtp_starttls` is enabled and STARTTLS cannot be established.
- [x] Replace per-notification daemon threads with a bounded queue/worker and cap digest buffers.
- [x] Resolve partner approval against the owner rule's Guardian DB and keep partner authorization explicit.
- [x] Add atomic TopicMonitoringDB duplicate insert support and use it from topic evaluation.
- [x] Reuse shared regex safety validation in topic monitoring compilation.
- [x] Refresh stale Monitoring module comments.

## Stage 4: Verification
**Goal:** Run focused and security verification for touched scope.
**Success Criteria:** Tests and Bandit complete with no new failures/findings in touched production files.
**Tests:** Focused Monitoring/Guardian pytest files, `git diff --check`, and Bandit on touched production paths.
**Status:** Complete

- [x] Run focused tests for Monitoring notification/API/topic changes.
- [x] Run focused tests for Guardian self-monitoring partner approval changes.
- [x] Run `git diff --check`.
- [x] Run Bandit on touched production files.
- [x] Update TASK-2417 with verification results and final summary.
