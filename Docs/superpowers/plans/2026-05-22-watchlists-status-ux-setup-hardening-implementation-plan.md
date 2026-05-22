# Watchlists Status UX And Setup Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each behavior change and superpowers:verification-before-completion before claiming completion.

**Goal:** Make `/watchlists` more reliable for first-time setup and status observability by adding variable Quick Setup cadence, normalizing successful run statuses, surfacing actionable audio health, and rendering no-task audio diagnostics.

**Backlog:** TASK-486

## Stage 1: Quick Setup Cadence

**Goal:** Quick Setup can create manual, interval, daily, weekdays, weekly, and advanced cron jobs without leaving `/watchlists`.

**Success Criteria:**
- `QuickSetupValues` accepts a cadence draft in addition to legacy presets.
- `toQuickSetupJobPayload()` generates expected cron payloads for every N minutes/hours, weekly, and advanced cron.
- The Quick Setup UI exposes interval/weekly/advanced controls using existing schedule helpers and keeps legacy defaults working.

**Tests:**
- Add failing `quick-setup.test.ts` cases for interval, weekly, and advanced cadence payloads.
- Add or extend `OverviewTab.quick-setup.test.tsx` to verify the beginner UI exposes variable cadence controls.

**Status:** Complete

## Stage 2: Run Status Normalization

**Goal:** Frontend treats backend `succeeded` as a successful terminal status anywhere `/watchlists` displays or reacts to run status.

**Success Criteria:**
- A shared helper normalizes run statuses and identifies success/terminal states.
- Status tags render `succeeded` as a successful state.
- Overview preview lookup, Run Detail polling, Runs polling utilities, and notifications use the same semantics.

**Tests:**
- Add helper unit tests for `succeeded`, `completed`, failure, cancellation, and running states.
- Update focused existing tests for status tags, polling utilities, and notifications.

**Status:** Complete

## Stage 3: Overview Audio Health

**Goal:** Overview health flags actionable audio setup/queue issues without marking intentional disabled audio as broken.

**Success Criteria:**
- `queue_unavailable` and `configuration_required` increment output/audio issue counts.
- `disabled` does not increment audio issues.
- `skipped_no_items` remains informational unless an existing stronger failure signal is present.

**Tests:**
- Add `watchlists-overview.test.ts` cases for the new audio statuses.

**Status:** Complete

## Stage 4: No-Task Audio Diagnostics

**Goal:** Requested audio without an enqueued task is visible in API and Run Detail.

**Success Criteria:**
- `GET /api/v1/watchlists/runs/{run_id}/audio` returns `200` with status/reason for requested no-task audio states.
- The same endpoint preserves `404 no_audio_briefing_for_run` for runs with no audio request/status evidence.
- Run Detail renders an audio briefing panel from run stats when no task ID exists.
- Live polling and artifact rendering continue to work when a task ID exists.

**Tests:**
- Add backend tests in `test_audio_output_delivery.py`.
- Add `RunDetailDrawer.stream-lifecycle.test.tsx` coverage for metadata-only audio status.

**Status:** Complete

## Stage 5: Verification And Handoff

**Goal:** Verify the narrow change set and leave a clear PR handoff.

**Success Criteria:**
- Focused frontend and backend tests pass.
- Bandit is run against touched Python files.
- Backlog task records tests, changed files, and final summary.
- Plan statuses are updated.

**Tests:**
- Focused Vitest runs for touched frontend suites.
- Focused Pytest runs for touched backend suites.
- Bandit on touched backend files.

**Status:** Complete
