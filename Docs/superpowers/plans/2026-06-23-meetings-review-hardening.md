# Meetings Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the validated Meetings module review findings for SSE framing, session transition races, finalize include validation, and finalization idempotency.

**Architecture:** Keep API behavior in the existing Meetings endpoint and service boundaries. Add persistence support in `Meetings_DB.py` only where service-level fixes need atomic database behavior.

**Tech Stack:** FastAPI, Pydantic, SQLite, pytest, Bandit.

---

## Stage 1: Regression Tests
**Goal:** Capture the reviewed failure modes before production code changes.
**Success Criteria:** New focused tests fail for the current implementation for the expected reasons.
**Tests:** Targeted pytest selections for stream framing, session transitions, and finalize behavior.
**Status:** Complete

- [x] Add an SSE framing test in `tldw_Server_API/tests/Meetings/test_meetings_events_sse.py` showing newline-bearing `id` and `type` values cannot create extra SSE control lines.
- [x] Add a session service race test in `tldw_Server_API/tests/Meetings/test_meetings_session_service.py` showing a stale transition is rejected when the stored status changes between read and update.
- [x] Add finalize API tests in `tldw_Server_API/tests/Meetings/test_meetings_ingest_finalize_api.py` showing unsupported final artifact kinds return `400`, do not write partial artifacts, and repeated commits do not create duplicate final artifacts.
- [x] Run the new tests and record the red failures in Backlog task `TASK-9924`.

## Stage 2: SSE and Transition Fixes
**Goal:** Make event framing safe and status transitions atomic.
**Success Criteria:** SSE field injection test and stale transition test pass.
**Tests:** `test_meetings_events_sse.py` and `test_meetings_session_service.py`.
**Status:** Complete

- [x] Update `tldw_Server_API/app/core/Meetings/stream_adapter.py` to sanitize SSE `id` and `event` control fields while keeping the JSON payload unchanged.
- [x] Update `tldw_Server_API/app/core/DB_Management/Meetings_DB.py` so `update_session_status` can require an expected current status.
- [x] Update `tldw_Server_API/app/core/Meetings/session_service.py` to pass the expected status and reject stale transitions.
- [x] Run the focused tests for this stage.

## Stage 3: Finalization Contract Fixes
**Goal:** Validate finalizable artifact kinds and make final artifact writes atomic and idempotent.
**Success Criteria:** Unsupported kinds are rejected before writes, explicit empty include lists do not fall back to defaults, and repeated commits leave one current artifact per kind/version.
**Tests:** `test_meetings_ingest_finalize_api.py`.
**Status:** Complete

- [x] Update `tldw_Server_API/app/core/Meetings/artifact_service.py` to distinguish `include is None` from an explicit list, reject unsupported final artifact kinds, and preserve requested order without duplicates.
- [x] Add a bulk replacement helper in `tldw_Server_API/app/core/DB_Management/Meetings_DB.py` that validates all artifacts before replacing `(session_id, kind, version)` rows in one transaction.
- [x] Use the bulk replacement helper from final artifact generation.
- [x] Run the focused finalization tests.

## Stage 4: Verification and Task Finalization
**Goal:** Verify the touched scope and update project tracking.
**Success Criteria:** Focused Meetings tests pass, Bandit completes for touched source, and Backlog task `TASK-9924` records results.
**Tests:** Focused pytest command plus Bandit on touched Meetings/DB source.
**Status:** Complete

- [x] Run focused Meetings pytest selections.
- [x] Run Bandit on touched source files.
- [x] Update `TASK-9924` with notes, checked acceptance criteria, final summary, and any known skips.
