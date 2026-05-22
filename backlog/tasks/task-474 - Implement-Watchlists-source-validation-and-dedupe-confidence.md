---
id: TASK-474
title: Implement Watchlists source validation and dedupe confidence
status: Done
labels:
- watchlists
- webui
- backend
- ux
- pr-c
priority: high
references:
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/1925
modified_files:
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx
- apps/packages/ui/src/types/watchlists.ts
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/app/core/Watchlists/fetchers.py
- tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py
- tldw_Server_API/tests/Watchlists/test_preview_endpoint.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 8 from the Watchlists demo remediation implementation plan: preserve source settings while editing, pass draft scrape/extraction/dedupe settings into source tests, and surface fetch/selector/sample/dedupe diagnostics in /watchlists source validation flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source settings helper tests prove unknown existing settings survive typed scrape/extraction/dedupe patches.
- [x] #2 Source test UI shows fetch/status diagnostics, selector diagnostics when available, sample item count, and dedupe identity preview.
- [x] #3 Source test calls include normalized draft settings so diagnostics reflect what the user is about to save.
- [x] #4 Backend source-test/preview diagnostics expose selector-rule diagnostics where missing without changing persisted source contracts unnecessarily.
- [x] #5 Focused backend and frontend Watchlists source validation tests pass, with verification recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Task 8 in Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md: inspect current source form/test/preview behavior, add failing helper and UI assertions first, implement source settings helpers and draft-settings source test payloads, extend backend diagnostics only if missing, run focused frontend/backend verification, update plan/task, commit, push, and prepare PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Existing dev already had source settings preservation, draft source-test settings, selector diagnostics, sample-count summary, and dedupe preview coverage.
- This slice filled the remaining fetch observability gap by adding fetch status/error fields to source preview diagnostics and wiring scrape-rule HTTP observations through a non-persisted fetcher callback.
- PR review follow-up aligned scrape-rule HTTP failures with RSS diagnostics: non-2xx status events now carry an error string, the reducer keeps the first failure status/error pair instead of drifting to the last failure, and 304 remains non-error.
- Verification:
  - `./node_modules/.bin/vitest run src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx --maxWorkers=1 --no-file-parallelism` (12 passed)
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py tldw_Server_API/tests/Watchlists/test_preview_endpoint.py -q` (15 passed, 5 warnings)
  - `git diff --check` (passed)
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/core/Watchlists/fetchers.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_watchlists_source_validation.json` (0 findings)
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the remaining Task 8 source validation gap by surfacing fetch status/error diagnostics through source preview responses and rendering them in the /watchlists source test modal alongside existing selector, sample-count, and dedupe diagnostics. Added regression coverage for UI rendering, fetcher HTTP-status observation, endpoint fetch-error propagation, endpoint fetch-status propagation, first-failure diagnostic selection, and 304 non-error handling. Verification: focused Watchlists source Vitest suites passed (12 tests), focused backend source preview pytest suites passed (15 tests), git diff --check passed, and Bandit reported zero findings for touched backend files.
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
