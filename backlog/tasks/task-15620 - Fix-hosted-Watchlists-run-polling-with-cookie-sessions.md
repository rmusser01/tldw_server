---
id: TASK-15620
title: Fix hosted Watchlists run polling with cookie sessions
status: Done
assignee:
  - '@codex'
created_date: '2026-11-23T20:57:00Z'
updated_date: '2026-11-23T21:14:00Z'
labels:
  - frontend
  - watchlists
dependencies: []
references:
  - https://github.com/rmusser01/tldw_server/pull/3580
modified_files:
  - >-
    apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
  - >-
    apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
type: bug
ordinal: 14969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hosted WebUI sessions authenticate through cookies and keep the browser access token empty. The Watchlists run drawer expects a token for its WebSocket and stops updating after a successful initial read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active hosted runs refresh through the existing cookie-authenticated run-details request and display their terminal status.
- [x] #2 Polling stops on completion, close, and Live off. Late responses keep the selected run and its actions consistent.
- [x] #3 The regression covers a cookie session with no browser token and the existing self-hosted WebSocket tests pass.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Port the verified Watchlists fix and tests onto the current public dev branch.
2. Run the focused Watchlists and hosted-auth checks against the public base.
3. Open a draft pull request against dev and record the validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the previously reviewed drawer and regression changes onto public dev 2aa1f81c37. Both public main and dev still required a browser JWT for the run stream in hosted cookie mode before this change.

Local verification passed 79 Watchlists run and hosted-auth tests, 106 Watchlists accessibility tests, 54 Watchlists scale tests, and git diff --check. Bandit has no Python scope in this TypeScript change. The maintainer's human-written Change summary is required before merge.

Opened public draft PR #3580 against dev. Hosted browser acceptance belongs to the private deployment and is pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The public Watchlists drawer polls cookie-authenticated run details while the hosted run is active. Regression coverage confirms the no-token cookie path, completion, cleanup, and stale response handling. Focused Watchlists, accessibility, and scale suites passed.
<!-- SECTION:FINAL_SUMMARY:END -->
