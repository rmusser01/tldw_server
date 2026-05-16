---
id: TASK-408
title: Fix broad Watchlists backend sweep failures
status: Done
labels:
- watchlists
- tests
- qa
priority: high
modified_files:
- tldw_Server_API/tests/Watchlists/test_admin_runs_ui_smoke.py
- tldw_Server_API/tests/Watchlists/test_runs_csv_export.py
- tldw_Server_API/tests/Watchlists/test_tts_brief_optional.py
- backlog/tasks/task-408 - Fix-broad-Watchlists-backend-sweep-failures.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and address the broad Watchlists backend sweep failures from the first-class Watchlists PR: stale admin runs UI smoke expectation, CSV export SQLite disk I/O failures, and optional TTS brief test reaching real providers in restricted-network runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Broad Watchlists backend sweep failures are root-caused before changes are made.
- [x] #2 Focused failing tests pass after fixes.
- [x] #3 Broad Watchlists backend sweep passes or any remaining failures are documented with exact cause and command output.
- [x] #4 Bandit or relevant security check is run if production Python code changes.
- [x] #5 PR #1775 is updated with the fixes and verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the broad Watchlists backend sweep failures after root-cause investigation. Admin smoke now asserts the actual RoutePlaceholder contract for `/admin/watchlists-runs` with a Watchlists CTA instead of the stale `/admin/server` RouteRedirect expectation. CSV export tests now use a per-test `tmp_path` USER_DB_BASE_DIR and close the managed SQLite backend for that test path, preventing stale open connections to unlinked SQLite files. Optional TTS brief tests now patch `tldw_Server_API.app.services.outputs_service.get_tts_service_v2`, which is the symbol used by `_write_tts_audio_file`, so tests no longer initialize real KittenTTS/HuggingFace/OpenAI providers. Verification: focused repro set passed 10 tests with 5 warnings; broad sweep `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists tldw_Server_API/tests/test_watchlist_alert_rules.py -q` passed 473 tests, 9 skipped, 1 xpassed, 159 warnings in 737.15s. `git diff --check` passed. No production Python changed; test-only Bandit with B101 skipped had zero high/medium/low findings in `/tmp/bandit_watchlists_broad_failures_tests_no_b101.json`, while unfiltered test-only Bandit reports expected pytest assert B101 findings.
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
