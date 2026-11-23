---
id: TASK-15626
title: Fix Dexie nested-transaction scope gaps in chat import and continue saves
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
- [x] #6 The shared memory-dexie harness enforces Dexie 4's parent-scope rule (missing table, rw child of r parent) with a self-test, and every harness-consuming suite still passes
- [x] #7 Every operation run inside runChatPersistenceTransaction is traced; each reachable nested-transaction gap is fixed failing-test first and confirmed against real Dexie
- [x] #8 The import rollback test fails if chats and tokens were not covered by the enclosing transaction
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. A one-table scope fix to an existing Dexie transaction; no durable architecture rule changes (Docs/ADR/README.md has no Dexie transaction-scope ADR; ADR-008 covers workspace persistence, unrelated).
Root cause: 5108fd4293 (TASK-13523, 2026-10-23) added db.searchTokens to importChatHistoryV2's own transaction (db/dexie/chat.ts:957) but not to the enclosing import transaction in libs/export-import.ts:57 (added by eb0c8c9e52, 2026-10-19, which fixed the same failure for compareStates). Dexie rejects a nested transaction whose tables are not a subset of the parent's, independent of message count, so every chat import fails.
Transaction audit (every db.transaction in apps/packages/ui/src that writes histories/messages or reaches search-token code): libs/export-import.ts:57 MISSING searchTokens (fixed). Already include it: chat.ts:453/471/500/957, helpers.ts:486 restoreChat, history-selection.ts:237 withLocalMessageMutation and :262 tables() (fork, admit, cursor paths), chat-persistence-transaction.ts:61 runChatPersistenceTransaction (chat saves, mirror reconcile), migration.ts:104 (legacy import). Never reach token code: chat-persistence-transaction.ts:12, chat.ts:309/324/402/556/943/1057, history-selection.ts:158/164/192 and the historySelections-only ones, server-chat-mirror.ts:33/111, migration.ts:72/153/168/185/227, services/chat-settings.ts:822/836, store/folder.tsx, audiobook/drafts/stt/tts/fork-operations.
Why nothing caught it: the memory-dexie harness joins any nested transaction without Dexie's parent-scope check, so 5108fd4293's unit tests passed; e2e/local-chat-import.spec.ts (eb0c8c9e52) is in the chromium Playwright project but no workflow or package script runs it (CI runs named specs only), and wiring it needs a workflow edit, so it is left as is. Real Dexie 4.2.1 in headless Chromium fails the one-message case the same way, so the spec would fail on dev if run.
Verification: new src/libs/__tests__/export-import.chat-restore.test.ts red (3/3 SubTransactionError: Table searchTokens not included in parent transaction.) then green (3/3). Neighbours green: export-import.prompt-account, chat-search-index, search-tokens.schema, chat-persistence-transaction, migration.retry (60 tests); DataManagementSettings, SystemSettings.i18n (7). eslint (lint:ui config) clean.
PR: https://github.com/rmusser01/tldw_server/pull/3577 (base dev).
Review round 1 (PR #3577, verdict 'with fixes'): scope widened to the bug class.
Harness: createMemoryDexie now rejects a nested transaction naming a table its immediate parent lacks, or a rw child of an r parent, with SubTransactionError and Dexie 4.2.1's messages. ! and ? mode flags are not modelled: no caller uses them (grep). Self-test memory-dexie.scope.test.ts: 3 of 4 red on the old harness, 4 of 4 green. All 35 harness-consuming suites pass before and after the rule (356 tests); the import test's private scope wrapper is gone.
Second gap, same class: runChatPersistenceTransaction (chat-persistence-transaction.ts:61) lacked userSettings. Continuing a saved local chat calls updateMessage inside it (chat-helper saveMessageOnSuccess :702); updateMessage reaches withLocalMessageMutation (history-selection.ts:237), whose nested transaction names userSettings, so the continued reply never persisted. Present since da0f1cd3a3 (2026-09-23) routed updateMessage through withLocalMessageMutation; the save had run in this transaction since 9d2de72b1a. Real Dexie in Chromium: dev list throws SubTransactionError and leaves the row unchanged; with userSettings it commits. Fixed by adding userSettings to the list; continue-persistence.scope.test.ts red (2/2 SubTransactionError userSettings) then green.
runChatPersistenceTransaction callers traced: chat-helper saveMessageOnSuccess (continue: reachable, threw, fixed); chat-helper saveMessageOnError scoped branch (stop-then-continue: would throw the same way and is fixed by the same change, but unreachable today because no continue caller passes servicePromptSnapshot, so the non-transactional branch runs); server-chat-mirror link/ack/reconcile/remove/cursor (chatHistories, messages, searchTokens only: fine); useChatActions compare saves :4784 (saveHistory + saveMessage) and :4964 (last-used model/prompt, createdAt): fine; useDocumentChat :80 (getHistoryInfo, getHistoryByDocId, saveHistory, saveMessage): fine; useActiveChatTitle :156 (updateHistory): fine.
Import rollback test now fails in importPromptsV2 (a prompt without an id) after chats are written and indexed; a non-atomic mutation (chats imported before the enclosing transaction) leaves histories 1, messages 120, tokens 82, so the token assertion now matters. sessionFiles stays in the import list with a comment naming what the list must cover.
e2e/CI: local-chat-import.spec.ts runs nowhere. It is matched only by the chromium project; CI uses that project only with explicit file + --grep filters (login.spec.ts, chat-cockpit.real-server.spec.ts), and all other jobs select tier-1..5/journeys/ux-regression projects or named files. Owner options (not done): append it to the frontend-required.yml step that runs e2e/manual-api-key-persistence.spec.ts (one workflow line), or move it under e2e/workflows/tier-1-critical/ (no workflow edit; runs in frontend-e2e-tiers critical).
Verification round 2: 99 suites, 3447 tests green (harness consumers, every persistence-module test, all db/dexie tests, new tests); no scope error in any log. eslint lint:ui clean.
DoD: Bandit not applicable (TypeScript-only change, no Python touched). No documentation describes these transaction lists, so no docs change. Known skip: e2e spec not wired into CI (needs a workflow edit; owner decision, options above).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Two Dexie nested-transaction scope gaps fixed and the class now fails in unit tests. Import Data failed for every backup because the enclosing import transaction (libs/export-import.ts) lacked searchTokens (since 5108fd4293). Continuing a saved local chat never persisted because runChatPersistenceTransaction lacked userSettings, which updateMessage's withLocalMessageMutation names (since da0f1cd3a3). The shared memory-dexie harness now enforces Dexie 4's parent-scope rule, so 37 harness suites and the new real-path tests catch this class.
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
