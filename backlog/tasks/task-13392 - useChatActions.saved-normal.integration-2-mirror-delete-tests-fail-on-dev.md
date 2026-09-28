---
id: TASK-13392
title: 'useChatActions.saved-normal.integration: 2 mirror-delete tests fail on dev'
status: To Do
assignee: []
created_date: '2026-09-28 00:20'
labels:
  - bug
  - chat
  - webui
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
apps/packages/ui useChatActions.saved-normal.integration has 2 failing cases on origin/dev (reproduced 2026-09-27 without any branch change): 'deletes a qualified mirror row and clears its local/server reply target using the canonical request ID'. Found while verifying TASK-13389. Not caught by the frontend ratchet because it only fails on head-vs-base regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified (stale test vs product regression) and the 2 cases pass
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
