# Resource Governance Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify TASK-12007 Resource Governance review findings against current code and fix every validated issue with focused regression tests.

**Architecture:** Keep the existing Resource_Governance public API intact while tightening backend correctness. Prefer small helpers inside the existing module over a large redesign; only split behavior where it reduces risk for Redis accounting, daily caps, route audit, or request identity.

**Tech Stack:** FastAPI/Starlette middleware, async Python, pytest, SQLite/AuthNZ test helpers, RedisResourceGovernor fake/stub-backed tests.

---

## Stage 1: Validate Findings and Add Regression Tests
**Goal**: Convert each confirmed review finding into a focused failing test before production edits.
**Success Criteria**: At least one targeted test fails for each validated P1/P2 behavior that changes code.
**Tests**: New or updated tests under `tldw_Server_API/tests/Resource_Governance/`.
**Status**: Complete

## Stage 2: Quota and Accounting Correctness
**Goal**: Fix token, concurrency, daily-cap, and idempotency behavior without changing the ResourceGovernor interface.
**Success Criteria**: Redis and memory governors consistently deny over-limit token reservations, atomic concurrency reservations cannot overbook, daily caps are consumed idempotently, and reserve/commit op records no longer collide.
**Tests**: Redis governor tests, memory governor tests, daily-cap tests, and targeted call-site tests where needed.
**Status**: Complete

## Stage 3: Enforcement and Policy Administration
**Goal**: Make middleware fail-mode behavior and policy admin version writes match the intended safety contract.
**Success Criteria**: Governed middleware requests fail closed when configured to do so, and concurrent expected-version policy updates cannot silently overwrite each other.
**Tests**: Middleware tests and AuthNZ policy admin backend tests.
**Status**: Complete

## Stage 4: Coverage and Tenant Scope Accuracy
**Goal**: Replace blanket route audit assumptions and wire tenant scope derivation into request identity.
**Success Criteria**: Coverage audit distinguishes mapped/unmapped routes, and tenant-scoped policies can be enforced from trusted request/auth context.
**Tests**: Coverage audit tests, dependency/entity derivation tests, and middleware tenant-scope tests.
**Status**: Complete

## Stage 5: Cleanup, Verification, and Backlog Finalization
**Goal**: Remove validated dead/test-only artifacts where safe, run targeted verification, run Bandit on touched backend code, and record results in TASK-12007.
**Success Criteria**: Targeted tests and security scan pass or any environmental blockers are documented.
**Tests**: Targeted pytest selection plus `python -m bandit -r <touched_paths> -f json -o /tmp/bandit_resource_governance_12007.json`.
**Status**: Complete

## Stage 6: PR Review Comment Follow-up
**Goal**: Rebase PR #2497 on latest `dev` and address all actionable Qodo review threads.
**Success Criteria**: Policy admin SQL lives behind `DB_Management`, coverage helpers have docstrings, tenant/entity derivation is consistent for endpoint callers, daily caps fail open on DB outages and consume atomically across workers, and Redis concurrency retry-after reflects multi-unit deficits.
**Tests**: Focused Resource Governance regressions for atomic daily-cap consumption, fail-open DB errors, tenant snapshot derivation, Redis concurrency retry-after, and existing policy admin backend selection.
**Status**: Complete
