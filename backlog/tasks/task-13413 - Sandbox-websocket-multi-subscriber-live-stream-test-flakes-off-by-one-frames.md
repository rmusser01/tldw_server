---
id: TASK-13413
title: Sandbox websocket multi-subscriber live stream test flakes (off-by-one frames)
status: To Do
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-01 17:55'
labels:
  - bug
  - sandbox
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_Server_API/tests/sandbox/test_ws_multi_subscribers.py::test_ws_multi_subs_live_stream failed in CI (platform-sandbox-ws-streams, #3058 run 36693231386, 2026-09-30): assert [1, 2, 3, 4] == [2, 3, 4, 5]. A subscriber saw a frame published before it attached, or missed the last one. Unrelated to #3058's audio route change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Subscription and replay ordering race identified and the test passes reliably
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
