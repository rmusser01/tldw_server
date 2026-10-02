---
id: TASK-13417
title: Qualify PR3071 on cookie-model latest dev and current CI
status: In Progress
created_date: 2026-10-02 02:39
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/3071
documentation:
- IMPLEMENTATION_PLAN_pr3071_latest_dev_ci_2026_10_01.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue PR3071 against dev dcae0cbd while preserving cookie-auth model discovery, owner-scoped recovery, fresh capability checks and handled-error logging. Resolve the TASK-13408 history collision without losing either record. Verify affected checks, a source-bound production build and real Chrome CDP acceptance using live services without mocks; publish to the existing draft PR without merging. Current CI and unavailable PostgreSQL remain explicit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preserve both task histories without an active ID collision and integrate frozen dev dcae0cbd.
- [ ] #2 Affected regression checks, TypeScript, production gates and real no-mock Chrome acceptance are freshly qualified.
- [ ] #3 Publish the existing draft PR with unchanged human Change summary and truthful current CI and PostgreSQL status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve tracker history and merge the frozen dev revision. 2. Run affected checks and source-bound production gates. 3. Run real native Chrome acceptance, publish evidence and inspect current CI without merging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
