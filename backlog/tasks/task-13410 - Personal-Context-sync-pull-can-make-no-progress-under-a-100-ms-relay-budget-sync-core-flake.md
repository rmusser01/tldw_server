---
id: TASK-13410
title: >-
  Personal Context sync pull can make no progress under a 100 ms relay budget
  (sync-core flake)
status: To Do
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-01 17:55'
labels:
  - bug
  - sync
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Sync/test_sync_v2_endpoints.py::test_personal_context_endpoints_use_real_factory_bootstrap_and_complete_flow fails intermittently: after a successful push it polls /api/v1/sync/pull up to 10 times back-to-back and every response is envelopes=[], next_cursor='0', has_more=true. Seen in CI (sync-core, #3063 run 36693239728, 2026-09-30) and locally 3/4 and 2/4 failures, including on a frontend-only tree. PersonalContextRelay.relay_profile bounds each pull to row_budget=100 and wall_time_ms=100. If the fixed per-pull work exceeds 100 ms on a slow host, the cursor never advances, which is a possible liveness bug and not only a test issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: whether a pull can exhaust its budget without advancing the relay cursor
- [ ] #2 Either the relay guarantees forward progress per pull, or the test waits on relay state with a deadline instead of 10 tight polls; the test passes reliably
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
