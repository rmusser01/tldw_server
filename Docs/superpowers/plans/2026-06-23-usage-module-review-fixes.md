# Usage Module Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix validated review findings in `tldw_Server_API/app/core/Usage` without changing unrelated Usage behavior.

**Architecture:** Keep existing module boundaries for this pass, but add small helpers where they reduce risk: explicit audio minute operation ids, a single daily-minute consume helper, safe pricing fallback behavior, and bounded metrics labels. Preserve legacy endpoint compatibility with wrappers around the new consume API.

**Tech Stack:** FastAPI service code, async Python, AuthNZ `DatabasePool`, `ResourceDailyLedger`, pytest, Bandit.

---

## Stage 1: Audio Minute Ledger Correctness
**Goal**: Make repeated same-duration audio minute events count separately.
**Success Criteria**: Two calls with the same user/day/minute value produce two ledger entries unless the same explicit operation id is reused.
**Tests**: `tests/Usage/test_usage_review_fixes.py::test_add_daily_minutes_counts_repeated_same_duration_events`.
**Status**: Complete

- [x] Write the failing duplicate-duration ledger test.
- [x] Run the test and confirm it fails because only one event is counted.
- [x] Add an explicit operation-id path to `add_daily_minutes`, generating a unique id by default.
- [x] Run the focused test and confirm it passes.

## Stage 2: Daily Minute Consume Semantics
**Goal**: Replace separate check/add endpoint usage with a single consume helper and safe fallback behavior.
**Success Criteria**: Callers can consume daily minutes through one helper; ledger unavailability surfaces a quota-store failure instead of silently reading stale legacy usage.
**Tests**: Add tests for `consume_daily_minutes`, denial without mutation, ledger unavailable failure, and endpoint wrappers using the new helper.
**Status**: Complete

- [x] Write failing tests for `consume_daily_minutes` success, denial, and ledger-unavailable failure.
- [x] Run those tests and confirm expected failures.
- [x] Implement `consume_daily_minutes` and route wrappers/callers through it.
- [x] Run Usage and touched Audio endpoint tests.

## Stage 3: Cancellation And Metrics Hardening
**Goal**: Stop swallowing `asyncio.CancelledError` and remove per-user Prometheus labels.
**Success Criteria**: Cancellation propagates from audio quota helpers; LLM usage metrics no longer emit `*_by_user` metrics or `user_id` labels.
**Tests**: Add cancellation and metrics-regression tests.
**Status**: Complete

- [x] Write failing tests for cancellation propagation and no per-user metric labels.
- [x] Run tests and confirm failures.
- [x] Remove `CancelledError` from noncritical exception handling and delete per-user metric emission.
- [x] Run focused tests.

## Stage 4: Pricing Budget Safety
**Goal**: Prevent placeholder billable models from recording zero USD budget usage.
**Success Criteria**: Placeholder pricing entries resolve to a conservative estimated fallback instead of zero unless explicitly marked free.
**Tests**: Add pricing tests for Qwen placeholder behavior and true free placeholders.
**Status**: Complete

- [x] Write failing pricing tests.
- [x] Run tests and confirm failures.
- [x] Update pricing entry handling with conservative placeholder fallback while preserving documented free non-placeholder entries.
- [x] Run pricing and usage tracker tests.

## Stage 5: Documentation And Verification
**Goal**: Update docs and verify touched scope.
**Success Criteria**: README reflects current AuthNZ/RG/ledger behavior; focused tests and Bandit results are recorded in TASK-12004.
**Tests**: Existing focused test commands plus Bandit.
**Status**: Complete

- [x] Update `tldw_Server_API/app/core/Usage/README.md`.
- [x] Run focused pytest for `tests/Usage` and relevant Audio tests.
- [x] Run Bandit on touched Usage files.
- [x] Update TASK-12004 with notes, checked acceptance criteria, and final summary.
