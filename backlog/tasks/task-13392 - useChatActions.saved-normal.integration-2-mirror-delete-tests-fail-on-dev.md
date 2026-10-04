---
id: TASK-13392
title: 'useChatActions.saved-normal.integration: 2 mirror-delete tests fail on dev'
status: Done
assignee: []
created_date: '2026-09-28 00:20'
updated_date: '2026-09-30 01:21'
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Already fixed on dev; no code change needed. Root cause: stale test, not a product regression. 977e118e57 (fix(chat): preserve ordinary actions and owner fences in H1 review, 2026-09-27) deliberately routed server-mirrored row deletes through removeAcknowledgedServerMirrorMessage (apps/packages/ui/src/db/dexie/server-chat-mirror.ts), which fails closed unless the local mirror history is bound to the server chat (server_chat_id + server_scope_key) and the row uniquely maps to the server message ID. The test only set UI state, so the guard refused and removeMessageById was never called (Number of calls: 0). Fixed by 9ac6709cd3 (test(chat): repair three chat tests that fail on main and dev), merged via PR #3046 on 2026-09-28: seeds the bound mirror history and its unique row, adds delete to the Dexie mock, and asserts the row is actually gone (stronger behavioral check than the old spy assertion). Verification: at ca3b7f834a (parent of fix) 'bunx vitest run src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx' -> 2 failed | 101 passed (both reply kinds of 'deletes a qualified mirror row ...'); at origin/dev 607431154c -> 103 passed; all sibling useChatActions tests 'bunx vitest run src/hooks/chat/__tests__/useChatActions' -> 10 files, 262 passed. Bandit: n/a (TypeScript test-only, no Python touched).

PR: https://github.com/rmusser01/tldw_server/pull/3062 (backlog-only record; the fix itself landed in #3046).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already fixed on dev by 9ac6709cd3 (#3046): the test predated 977e118e57, which routes mirror deletes through removeAcknowledgedServerMirrorMessage and requires a linked mirror; the test now seeds one and asserts the row is gone. Verified: saved-normal.integration 103 passed and all 10 useChatActions files 262 passed on 607431154c. No docs or Bandit (test-only, no code change). No known skips.
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
