# Task 2D — Watchlist Seed Report

**Status:** PASS — the owned, isolated Watchlist browser journey is green with real persisted briefing state and repeat controls.

## Real lifecycle and isolation

The journey uses only supported Watchlist interfaces:

- `POST /api/v1/watchlists`
- `POST /api/v1/watchlists/sources`
- `POST /api/v1/watchlists/jobs`
- `POST /api/v1/watchlists/jobs/{job_id}/run`
- `PATCH /api/v1/watchlists/sources/{source_id}`
- `GET /api/v1/watchlists/runs/{run_id}/briefing`
- `GET /api/v1/watchlists/briefings/latest?watchlist_id={watchlist_id}`
- `GET /api/v1/watchlists/items` with exact run, Watchlist, and ingested-status filters
- `DELETE /api/v1/watchlists/sources/{source_id}/seen`
- supported `DELETE` job, source, and Watchlist cleanup

It verifies matching Watchlist/job/run/occurrence/output identities before browser navigation. Each run creates a tokenized Watchlist, RSS source, and job. The seed and UI-run feeds have distinct GUIDs and use the supported full-RSS-content path, so they are portable across the configured WebUI origin and do not depend on one development port for article extraction. The local mock provider was launched with this worktree's `mock_openai_server` on `PYTHONPATH`. The briefing and item endpoints were never intercepted or stubbed, and no database rows or in-memory briefing projections were seeded.

The final isolated graph used Redis `62460`, mock provider `18092`, API `62461`, WebUI/RSS `62462`, and one retained temporary per-user database directory. Two consecutive passes used that same graph and database. State is recorded immediately after each create response; `finally` attempts every seen-state/job/source/Watchlist cleanup even if an earlier cleanup fails, then reports all failures together. The seen-state reset also runs immediately after source creation so a prior interrupted process cannot make a reused source ID classify the owned GUID as a duplicate. A process-level hard kill can still bypass JavaScript `finally`, but the next run repairs its own dedup state before ingestion.

## RED/GREEN and browser evidence

Initial RED established that the browser route alone has no persisted briefing/repeat controls. Public lifecycle setup then produced real persisted briefing state: exact run-briefing and latest-Watchlist-briefing calls returned `200` with matching ownership. Independent review then exposed a false-positive article assertion: the initial run's item could satisfy the UI-run check. The repair gives the UI run a distinct feed identity and polls the public items endpoint for the exact completed run, job, source, Watchlist, `ingested` status, title, and content.

The first live repair attempt produced an exact `304` trace for the newly selected UI-run feed. The source URL update had retained the old URL's ETag/Last-Modified validators and backoff state. A focused database regression failed on the retained ETag, then passed after `WatchlistsDatabase.update_source` was changed to clear URL-bound conditional state only when the URL actually changes.

Focused RED proved a direct child/store `setActiveTab("runs")` left progressive layout on the primary surface without expanding Activity. The minimal central `WatchlistsPlaygroundPage` repair normalizes direct progressive secondary transitions: `jobs` → `sources` + Monitors, `runs` → `items` + Activity, and `templates` → `outputs` + Templates. `LatestBriefing` now exposes stable real-action identifiers `watchlists-repeat-actions` and `watchlists-repeat-open-runs`.

The corrected isolated journey passed twice consecutively against the same retained graph and database: `12.3s`, then `10.8s`, with Playwright tracing enabled. Each pass proved:

- the selected shell contains the exact tokenized Watchlist;
- Help → command palette → Monitors reaches the owned job;
- the real Run Now response supplies the exact new run ID, and its persisted briefing is read from supported endpoints;
- the public item projection ties the new title/content to that exact Run Now run, job, source, Watchlist, and `ingested` status;
- Overview's accessible `Inspect run …` control opens `Run Details`;
- closing that real dialog reveals the centrally normalized, selected `Updates` surface;
- Activity has exactly one row containing `watchlists-run-open-outputs-${completedRunId}`, the owned job text, and its exact Open Reports control;
- the exact deterministic title/content appears in the owned item reader; and
- the notification inbox shows `Unread: 1`, `Run completed`, and the exact owned run message.

## Notification trace finding (reviewer concern)

Trace diagnostics found a broad existing page-object side effect: `NotificationsPage.goto()` calls `waitForConnection()`, whose `dismissConnectionModals()` uses generic `getByRole('button', { name: /dismiss/i })`. On the real inbox, that matches the notification item's own **Dismiss** control, sends `POST /api/v1/notifications/8/dismiss`, and removes the primary list item, leaving `Show snoozed (1)` with `Unread: 0`.

Task 2D uses a deliberately scoped workaround only: it navigates to the real `/notifications` route with `page.goto(..., { waitUntil: 'domcontentloaded' })`, then retains the existing page-object loaded/list/unread/owned-message assertions. The generic helper was not refactored because that is outside Task 2D scope; reviewers should consider its broad Dismiss selector separately.

## Gates

- Focused Watchlists component suite: **PASS** — checked-in UI runner: 2 files, 35/35 tests (`orientation-guidance` 8; `LatestBriefing` 27).
- Watchlist source-contract guard: **PASS** — its case is green; the full guard file remains 7/8 because of the documented unrelated Audiobook wrapper baseline.
- Source URL conditional-state regression: **PASS** — focused pytest 1/1 after a confirmed RED on the retained ETag.
- Isolated real Watchlist journey: **PASS** — two consecutive 1/1 runs in `12.3s` and `10.8s` against the same retained graph/database, with trace enabled and public-API cleanup above.
- Focused Prettier and ESLint: **PASS**.
- Frontend typecheck: **known unrelated baseline failures only** — `DocumentationPage.tsx` `DocImportMap` typing and skills-certification test typings; no touched Watchlists error.
- `git diff --check`: **PASS**.
- Bandit: **PASS** — no findings in the touched Watchlists database module.
- Transient traces/videos/test-results: **not staged**.
