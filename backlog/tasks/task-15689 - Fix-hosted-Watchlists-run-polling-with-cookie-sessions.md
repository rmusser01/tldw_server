---
id: TASK-15689
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
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
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
- [x] #4 Global and job-scoped run lists revalidate their status filters and totals when a hosted run changes status.
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ported the previously reviewed drawer and regression changes onto public dev 2aa1f81c37. Both public main and dev still required a browser JWT for the run stream in hosted cookie mode before this change.

Local verification passed 79 Watchlists run and hosted-auth tests, 106 Watchlists accessibility tests, 54 Watchlists scale tests, and git diff --check. Bandit has no Python scope in this TypeScript change. The maintainer's human-written Change summary is required before merge.

Opened public draft PR #3580 against dev. Hosted browser acceptance belongs to the private deployment and is pending.
Rebased #3580 onto dev while retaining its Watchlists React Query migration. Hosted sessions reuse the scoped run-details query and refresh the exact items page when the observed run reaches a terminal state. ADR required: no. This uses the existing account-scoped query keys, hosted transport, and persisted run state.
Renumbered for PR #3580 after backend-required found that TASK-15620 was already assigned to the quick-launch production WebUI build on dev. This is the same original Watchlists work record under a free ID.
Independent review found that patching a completed row in a Running list could leave the row and total stale and stop that list polling. Both filter shapes failed the new regression on the original branch. Updated the account-scoped patch to refetch filtered membership after status transitions while keeping compatible row patches for other pages. Watchlists verification on the prepared dev replay: 112 accessibility tests, 56 scale tests, and 3 static checks passed. No Python files changed. The previously recorded Change summary waiver applies to PR #3580. Hosted deployment acceptance remains pending.
A second independent review reproduced three timing failures. Delayed active details could replace a terminal Activity row, reopening the same run during a cancel could strand its synthetic snapshot, and changing items page mid-cancel refreshed the old page. Reproduced all three failures before fixing them. Terminal list rows now reject active regressions. Successful hosted cancellation reconciles the exact run and its item pages in the same account, with view-specific UI guarded independently. Exact prepared replay passed 23 drawer tests, 115 accessibility tests including the drawer suite, 56 scale tests, 3 static checks, and WebUI TypeScript.
Renumbered during CI preflight because open PR #3655 had already claimed the interim TASK-15672. This remains the same Watchlists record from #3580.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The public Watchlists drawer polls cookie-authenticated run details while the hosted run is active. Regression coverage confirms the no-token cookie path, completion, cleanup, and stale response handling. Focused Watchlists, accessibility, and scale suites passed.
<!-- SECTION:FINAL_SUMMARY:END -->
