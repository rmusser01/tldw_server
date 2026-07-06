# Watchlists Core Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix validated Watchlists core review findings from TASK-12790 without broad module refactors.

**Architecture:** Keep the changes at existing boundaries: filter safety stays in `Watchlists/filters.py`, tenant/status/cancellation behavior stays in pipeline and scheduler handler code, ownership checks stay in `Watchlists_DB.py`, and output enrichment scheduling reuses the existing `output_enrichment_handler.enrich_output` worker. Regression coverage is focused on the specific broken contracts.

**Tech Stack:** FastAPI, pytest, SQLite-backed WatchlistsDatabase tests, existing `regex` dependency for bounded regular-expression execution.

---

## Stage 1: Regex Safety And Cancellation Contracts

**Goal:** Prevent unsafe regex filters from blocking evaluation and stop treating task cancellation as recoverable noise.

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/filters.py`
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Test: `tldw_Server_API/tests/Watchlists/test_filters_matching.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py`

**Success Criteria:** Unsafe or oversized regex filters do not match, normal regex filters still work, and `asyncio.CancelledError` is no longer in Watchlists recoverable exception tuples.

**Tests:** `python -m pytest tldw_Server_API/tests/Watchlists/test_filters_matching.py tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py -q`

**Status:** Complete

## Stage 2: Tenant-Aware Pipeline And Scheduler Status

**Goal:** Preserve tenant context from Scheduler payloads into URL egress policy checks and return the real pipeline status from scheduler handlers.

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Modify: `tldw_Server_API/app/core/Scheduler/handlers/watchlists.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_scheduler_handler.py`

**Success Criteria:** RSS and scrape-rule fetchers receive the explicit tenant ID, scheduler passes `tenant_id`, and cancelled pipeline results are reported as cancelled instead of succeeded.

**Tests:** `python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py tldw_Server_API/tests/Watchlists/test_watchlists_scheduler_handler.py -q`

**Status:** Complete

## Stage 3: Source Association Ownership And Scope Failure Handling

**Goal:** Enforce source ownership for tags, groups, seen-item helpers, and source deletion, and ensure malformed scopes cannot leave runs stuck.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_db_user_scope.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py`

**Success Criteria:** Cross-user source association helpers raise or no-op without mutating owner rows, group assignment rejects foreign groups, malformed scopes are skipped safely, and unexpected pipeline failures update the run as failed.

**Tests:** `python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_db_user_scope.py tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py -q`

**Status:** Complete

## Stage 4: Output Enrichment Scheduling And Verification

**Goal:** Schedule the existing enrichment worker when output creation stores pending enrichment metadata.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
- Update: `backlog/tasks/task-12790 - Harden-Watchlists-core-review-findings.md`

**Success Criteria:** Creating an output with briefing summary or topic grouping schedules `enrich_output` after artifact creation, existing output behavior remains unchanged, focused Watchlists tests pass, and Bandit reports no new findings on touched Python files.

**Tests:** `python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q`; `python -m bandit -r tldw_Server_API/app/core/Watchlists tldw_Server_API/app/core/Scheduler/handlers/watchlists.py tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/app/api/v1/endpoints/watchlists.py -f json -o /tmp/bandit_watchlists_core_review_fixes.json`

**Status:** Complete
