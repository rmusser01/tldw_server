---
id: TASK-15626
title: Fix local chat import failing with searchTokens not included in parent transaction
status: In Progress
labels:
- webui
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings > Data > Import Data fails for every chat backup with 'Import error: Table searchTokens not included in parent transaction.' and imports nothing (observed live on dev 2aa1f81c37 with a 120-message chat). importPageAssistData (apps/packages/ui/src/libs/export-import.ts) wraps the restore in one rw Dexie transaction; importChatHistoryV2 opens a nested transaction that, since 5108fd4293 (TASK-13523, schema v17), also lists db.searchTokens, which the enclosing list lacks, so Dexie rejects the nested scope before any write. Add searchTokens to the enclosing transaction and pin the real import path with a scope-enforcing unit test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A chat backup with one message and one with 120 messages both import through importPageAssistData without a transaction-scope error
- [x] #2 Imported messages are findable by content search afterwards (search tokens written)
- [x] #3 A failing row rolls the whole import back (no histories, messages or tokens left behind)
- [x] #4 Every Dexie transaction in apps/packages/ui/src that can reach search-token maintenance is audited and any with the same gap fixed
- [x] #5 The PR states whether e2e/local-chat-import.spec.ts runs in CI
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. A one-table scope fix to an existing Dexie transaction; no durable architecture rule changes (Docs/ADR/README.md has no Dexie transaction-scope ADR; ADR-008 covers workspace persistence, unrelated).
Root cause: 5108fd4293 (TASK-13523, 2026-10-23) added db.searchTokens to importChatHistoryV2's own transaction (db/dexie/chat.ts:957) but not to the enclosing import transaction in libs/export-import.ts:57 (added by eb0c8c9e52, 2026-10-19, which fixed the same failure for compareStates). Dexie rejects a nested transaction whose tables are not a subset of the parent's, independent of message count, so every chat import fails.
Transaction audit (every db.transaction in apps/packages/ui/src that writes histories/messages or reaches search-token code): libs/export-import.ts:57 MISSING searchTokens (fixed). Already include it: chat.ts:453/471/500/957, helpers.ts:486 restoreChat, history-selection.ts:237 withLocalMessageMutation and :262 tables() (fork, admit, cursor paths), chat-persistence-transaction.ts:61 runChatPersistenceTransaction (chat saves, mirror reconcile), migration.ts:104 (legacy import). Never reach token code: chat-persistence-transaction.ts:12, chat.ts:309/324/402/556/943/1057, history-selection.ts:158/164/192 and the historySelections-only ones, server-chat-mirror.ts:33/111, migration.ts:72/153/168/185/227, services/chat-settings.ts:822/836, store/folder.tsx, audiobook/drafts/stt/tts/fork-operations.
Why nothing caught it: the memory-dexie harness joins any nested transaction without Dexie's parent-scope check, so 5108fd4293's unit tests passed; e2e/local-chat-import.spec.ts (eb0c8c9e52) is in the chromium Playwright project but no workflow or package script runs it (CI runs named specs only), and wiring it needs a workflow edit, so it is left as is. Real Dexie 4.2.1 in headless Chromium fails the one-message case the same way, so the spec would fail on dev if run.
Verification: new src/libs/__tests__/export-import.chat-restore.test.ts red (3/3 SubTransactionError: Table searchTokens not included in parent transaction.) then green (3/3). Neighbours green: export-import.prompt-account, chat-search-index, search-tokens.schema, chat-persistence-transaction, migration.retry (60 tests); DataManagementSettings, SystemSettings.i18n (7). eslint (lint:ui config) clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chat import from Settings > Data failed for every backup because the enclosing import transaction (libs/export-import.ts) lacked db.searchTokens, which importChatHistoryV2's nested transaction has listed since schema v17 (5108fd4293). Added the table to the enclosing list and a unit test that runs the real import, token index and content search under Dexie's parent-scope rule (one-message, 120-message, search-after-import, rollback on a failing row).
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
