---
id: TASK-13392
title: 'useChatActions.saved-normal.integration: 2 mirror-delete tests fail on dev'
status: In Progress
assignee: []
created_date: '2026-09-28 00:20'
updated_date: '2026-09-30 01:20'
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
- [x] #1 Root cause identified (stale test vs product regression) and the 2 cases pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already fixed on dev; no code change needed. Root cause: stale test, not a product regression. 977e118e57 (fix(chat): preserve ordinary actions and owner fences in H1 review, 2026-09-27) deliberately routed server-mirrored row deletes through removeAcknowledgedServerMirrorMessage (apps/packages/ui/src/db/dexie/server-chat-mirror.ts), which fails closed unless the local mirror history is bound to the server chat (server_chat_id + server_scope_key) and the row uniquely maps to the server message ID. The test only set UI state, so the guard refused and removeMessageById was never called (Number of calls: 0). Fixed by 9ac6709cd3 (test(chat): repair three chat tests that fail on main and dev), merged via PR #3046 on 2026-09-28: seeds the bound mirror history and its unique row, adds delete to the Dexie mock, and asserts the row is actually gone (stronger behavioral check than the old spy assertion). Verification: at ca3b7f834a (parent of fix) 'bunx vitest run src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx' -> 2 failed | 101 passed (both reply kinds of 'deletes a qualified mirror row ...'); at origin/dev 607431154c -> 103 passed; all sibling useChatActions tests 'bunx vitest run src/hooks/chat/__tests__/useChatActions' -> 10 files, 262 passed. Bandit: n/a (TypeScript test-only, no Python touched).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
