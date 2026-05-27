---
id: TASK-516
title: Prevent Research Workspace migration from re-persisting deleted local content
status: Done
labels:
- research-workspace
- migration
- webui
- bugfix
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live CDP validation after TASK-515 showed the backend can authorize deletion and the WebUI can send client-delete-ack, but valid hydrated Research Workspace state can re-persist tldw-workspace and split snapshot/chat localStorage keys after the migration driver deletes them. Fix the WebUI persistence/migration interaction so successful true-move deletion is durable and does not silently restore covered legacy content payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A live/current WebUI migration with valid workspace snapshot and chat content declares all covered content chunks, finalizes with client_delete_eligible=true, sends client-delete-ack, writes a tombstone, and leaves no tldw-workspace or tldw-workspace:workspace:* content keys in localStorage after subsequent page activity.
- [x] #2 A focused failing test reproduces re-persistence of deleted local content after successful migration before production code changes.
- [x] #3 The fix preserves recovery behavior: if local inventory is blocked or server client_delete_eligible=false, content keys are retained and no client-delete-ack is sent.
- [x] #4 Existing focused Research Workspace migration backend/frontend tests still pass, plus any new regression coverage.
- [x] #5 Live CDP/Playwright validation records eligible delete and blocked/retained behavior against current backend/WebUI.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a regression in `apps/packages/ui/src/store/__tests__/workspace.test.ts` that seeds a verified Research Workspace migration tombstone and asserts that a later workspace persistence write does not recreate `tldw-workspace`, the split snapshot key, or the split chat key.
- Updated `apps/packages/ui/src/store/workspace.ts` so split workspace persistence checks verified migration tombstones before writing. Tombstoned workspace IDs are removed from active workspace metadata, saved/archive lists, split snapshots, split chat sessions, and existing local/IndexedDB records.
- Reconstructing split workspace storage now ignores tombstoned workspace IDs, so stale split envelopes cannot rehydrate content after a successful true-move deletion.
- Blocked/recovery paths are preserved because suppression only activates when a valid tombstone with `contentRetained:false`, matching `legacyWorkspaceId`, and non-empty `migrationId` exists.
- Live Playwright/CDP validation against current backend/WebUI confirmed eligible delete leaves no covered content keys after page activity and blocked inventory retains content without sending `client-delete-ack`. The live app normalized the seeded split chat key before migration, so live receipts covered the main index plus snapshot; the focused regression test covers the chat-key deletion/re-persist path directly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented durable true-move deletion for Research Workspace migration. After a verified migration tombstone exists, workspace persistence no longer rewrites covered local content for that workspace and also ignores tombstoned IDs during split-storage rehydrate. Focused Vitest coverage and live Playwright validation now show eligible migrations ack/delete durably while blocked migrations retain local content and skip ack.

Verification:
- `bunx vitest run src/store/__tests__/workspace.test.ts --maxWorkers=1 --no-file-parallelism --testNamePattern "does not re-persist covered workspace content after a migration tombstone exists"`: passed.
- `bunx vitest run src/store/__tests__/workspace.test.ts src/store/__tests__/workspace-migration.test.ts --maxWorkers=1 --no-file-parallelism`: 56 tests passed.
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx --maxWorkers=1 --no-file-parallelism`: 30 tests passed.
- Live Playwright/CDP eligible run: POST create, PUT chunks, POST finalize with `client_delete_eligible=true`, GET finalized, POST `client-delete-ack`, tombstone written, no covered content keys after idle wait.
- Live Playwright/CDP blocked run: POST create, PUT chunks, POST finalize/GET finalized, no `client-delete-ack`, no tombstone, content keys and unknown workspace-prefixed key retained.
- `git diff --check`: passed.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`: failed on unrelated baseline errors in `CharacterListContent.design-system.test.tsx` and `sidepanel-flashcards.test.tsx`.
- Bandit: skipped because this task touched TypeScript/WebUI, plan, and Backlog only; no Python runtime code changed.
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
