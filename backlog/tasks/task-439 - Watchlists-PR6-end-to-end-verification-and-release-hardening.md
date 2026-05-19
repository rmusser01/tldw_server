---
id: TASK-439
title: Watchlists PR6 end-to-end verification and release hardening
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 11 from the Watchlists digest/audio implementation plan: run focused frontend/backend/security verification, perform browser-observed /watchlists QA where feasible, and patch only verified release-hardening issues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused Watchlists frontend gates pass or failures are classified with fixes/skips.
- [x] #2 Focused Watchlists backend verification passes or failures are classified with fixes/skips.
- [x] #3 Bandit runs on touched Watchlists backend scope or an explicit non-code skip is documented.
- [x] #4 Browser-observed `/watchlists` QA is performed where local services can be started, or the blocker is documented.
- [x] #5 Any release-hardening code changes are verified with targeted regression tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Branch started from `origin/dev` after PR #1864 merge commit `0a892b044`. Scope follows Task 11 in `Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md`.

Verification notes:
- Frontend gates passed after `bun install` in `apps/`: `bun run test:watchlists:typecheck` (1 file, 3 tests), `bun run test:watchlists:scale` (7 files, 53 tests), and `bun run test:watchlists:a11y` (12 files, 85 tests). The a11y suite emitted expected negative-test stderr but passed.
- Backend focused verification passed: `python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_full_pipeline_integration.py tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q` -> 66 passed, 1 xpassed, 5 warnings.
- Bandit initially flagged B405 stdlib XML imports in Watchlists RSS/OPML/WebSub parser files. Those imports were replaced with `defusedxml.ElementTree`, then Bandit passed with 0 findings on `watchlists.py`, `watchlists_schemas.py`, and `app/core/Watchlists`.
- Parser-adjacent regressions passed after the XML parser patch: OPML API/edge/export tests (6 passed) and RSS/fetcher/WebSub tests (42 passed).
- Browser QA used an isolated backend on `127.0.0.1:18002` and WebUI on `127.0.0.1:8082`. `/watchlists` rendered the Watchlists page with `Imported Watchlist`; observed Watchlists API calls returned 200; no console errors, request failures, or page errors in the final page-load pass. Create Watchlist opened the guided setup modal without errors. Overview shortcuts opened Feeds and Monitors content; telemetry/notification stream requests aborted only during scripted navigation/teardown.
- `git diff --check` passed before browser QA and should be rerun before commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed PR6 release-hardening verification for Watchlists. The only code change required was removing remaining stdlib XML imports from Watchlists RSS, OPML, and WebSub parsing paths so Bandit no longer reports B405 while existing defusedxml parsing remains intact. Focused frontend, backend, parser-adjacent, Bandit, and browser-observed `/watchlists` checks passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
