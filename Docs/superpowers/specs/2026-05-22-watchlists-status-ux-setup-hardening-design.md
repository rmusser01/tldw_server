# Watchlists Status UX And Setup Hardening Design

Status: Approved for implementation
Date: 2026-05-22
Backlog: TASK-486
Parent PRD: `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
Follows: `Docs/superpowers/specs/2026-05-22-watchlists-durable-audio-artifact-projection-design.md`

## Summary

This follow-up keeps the core digest-to-audio briefing workflow inside `/watchlists` and fixes the remaining status/setup gaps that make the feature brittle for demos and daily use.

The scope is intentionally narrow:

- Add variable cadence controls to the first-time Quick Setup path.
- Normalize Watchlists run statuses so backend `succeeded` is treated like frontend `completed`.
- Flag actionable audio setup/queue issues in Overview health.
- Show requested audio states even when no audio task was enqueued.

## Current Evidence

- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts` already models `WatchlistCadenceDraft`, but `QuickSetupValues` only accepts a small `schedulePreset` set.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx` exposes only Manual, Hourly, Daily, and Weekdays in the beginner Quick Setup schedule selector.
- `tldw_Server_API/app/core/Watchlists/pipeline.py` records completed runs as `succeeded`, while some frontend paths still look only for `completed`.
- `apps/packages/ui/src/components/Option/Watchlists/shared/StatusTag.tsx`, `RunsTab/polling-utils.ts`, `RunsTab/run-notifications.ts`, and `RunDetailDrawer.tsx` need one shared interpretation of terminal/success run statuses.
- `apps/packages/ui/src/services/watchlists-overview.ts` detects older audio issue strings, but does not treat `queue_unavailable` or `configuration_required` as attention states.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py` returns `404 no_audio_briefing_for_run` from `GET /runs/{run_id}/audio` when no task ID exists, even if run stats say audio was requested and skipped because configuration or queueing was unavailable.
- `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx` hides the audio panel when no task ID exists, so users cannot see why a requested briefing did not start.

## Goals

- Make Quick Setup support the same beginner-relevant timing model as the rest of Watchlists: manual, every N minutes/hours, daily, weekdays, weekly, and advanced cron.
- Keep variable timing inside `/watchlists`; advanced editing may hand off to Jobs, but the MVP path must not require leaving `/watchlists`.
- Use one frontend run-status normalization contract for successful and terminal states.
- Make actionable audio issues visible in Overview health without treating intentionally disabled audio as broken.
- Return and render meaningful audio status for requested audio with no task ID.
- Preserve existing power-user OSINT/news-junkie workflows and advanced controls.

## Non-Goals

- No broad `/watchlists` IA redesign.
- No new audio artifact storage model.
- No removal of existing raw template, source, job, filter, or advanced schedule controls.
- No new scheduler backend for cadence; this should reuse the existing schedule payload helpers.
- No podcast/script editor expansion in this slice.

## Decisions

### 1. Quick Setup Uses The Existing Cadence Draft

Quick Setup should use `WatchlistCadenceDraft` as the source of truth for schedule creation, not invent another timing model. Existing presets can remain as UX shortcuts, but payload generation must accept a cadence draft so "every 5 hours" and weekly schedules are first-class.

### 2. Backend `succeeded` Is A Successful Terminal Run

The frontend should normalize `succeeded` to the same semantic bucket as `completed`. Display copy may remain "Completed" or "Succeeded" where context demands it, but previews, polling termination, status tags, and notifications must not treat `succeeded` as unknown.

### 3. Audio Health Separates Actionable Issues From Intentional States

Overview health should flag:

- `queue_unavailable`
- `configuration_required`
- `enqueue_failed`
- `failed`
- `error`

It should not flag `disabled` as an issue. `skipped_no_items` is informational when there were no digest items, not a setup failure.

### 4. No-Task Audio Is Still User-Visible When Requested

When run stats or output metadata say audio was requested but no task exists, `/runs/{run_id}/audio` should return a compact status response for known no-task states such as:

- `configuration_required`
- `queue_unavailable`
- `skipped_no_items`
- `disabled`

The endpoint should keep returning `404 no_audio_briefing_for_run` when no audio request/configuration/status evidence exists.

### 5. Run Detail Should Render Metadata-Only Audio Status

Run Detail should show the audio briefing panel when stats indicate an audio request/status, even without a task ID. It should explain the status and reason, avoid misleading retry affordances when retry cannot help, and continue to poll live task/audio status when a task ID exists.

## UX Impact

First-time users can create a realistic news digest cadence from Quick Setup without needing to know cron or leave the page. They can also see why audio did not start instead of seeing silence.

Power users get less status ambiguity: successful runs sort/render consistently, Overview calls out actual audio setup/queue failures, and Run Detail preserves no-task diagnostics for triage.

## Test Strategy

- Unit-test Quick Setup payload creation for interval, weekly, and advanced cadence values.
- Unit-test run status normalization and wire it into existing badge, polling, and notification tests.
- Unit-test Overview health issue detection for actionable audio statuses and disabled audio.
- Backend-test `/runs/{run_id}/audio` no-task responses and preserved 404 for true no-audio runs.
- Component-test Run Detail rendering of metadata-only audio status.
- Run focused Vitest/Pytest suites plus Bandit on touched Python scope.
