---
id: TASK-12163
title: Add explicit streaming emote directives for character chat portraits
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-06 20:04'
labels:
  - frontend
  - character-chat
  - emotes
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-06-character-chat-streaming-emote-directives-design.md
  - >-
    Docs/superpowers/plans/2026-07-06-character-chat-streaming-emote-directives-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement v1 character chat emote control: parse standalone Emote: <state> directives from assistant responses, strip them from visible/stored text, update character portraits live during streaming, persist final mood_label plus emote_events metadata, and demote heuristic mood detection to fallback when explicit directives exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Streaming character chat can change the character portrait multiple times within one assistant response when valid Emote directives arrive.
- [x] #2 Raw Emote directive lines never appear in rendered chat or persisted assistant content, including partial/chunked streaming cases.
- [x] #3 Explicit emote directives override heuristic mood detection; detectCharacterMood only runs when no valid directive is present.
- [x] #4 Non-streaming character responses are also parsed and stripped before display/persist.
- [x] #5 Invalid, unsafe, duplicate consecutive, or over-cap directives are stripped but do not fire/store emote events.
- [x] #6 Missing emote image assets do not break rendering; the UI keeps the current/base portrait.
- [x] #7 Final emote, defined as the last accepted event, persists as mood_label and optional emote_events are stored in metadata_extra.
- [x] #8 History reload restores the final emote and does not replay beat events in v1.
- [x] #9 Parser, streaming-buffer, integration, and minimal UI behavior tests cover the directive flow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-06-character-chat-streaming-emote-directives-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 2 complete in commit 48df2747c7: custom safe emote image slugs resolve via character mood image maps while classifier labels remain unchanged. Red: bunx vitest run src/utils/__tests__/character-mood.test.ts --maxWorkers=1 failed on smug returning empty string. Green: bunx vitest run src/utils/__tests__/character-mood.test.ts src/utils/__tests__/character-emotes.test.ts --maxWorkers=1 passed 23 tests. Bandit not run: touched files are frontend TypeScript only.

Task 6 final verification complete. Targeted frontend suite: bunx vitest run src/utils/__tests__/character-emotes.test.ts src/utils/__tests__/character-mood.test.ts src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts src/hooks/__tests__/useServerChatLoader.test.ts src/db/dexie/__tests__/helpers.character-emotes.test.ts src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx --maxWorkers=1 passed 7 files / 82 tests. Targeted backend suite: python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py -q passed 39 tests with 4 warnings. Bandit touched backend scope: no errors and no findings in /tmp/bandit_character_emotes.json. App-wide frontend typecheck was attempted with bun run typecheck and failed on existing baseline errors outside the character-emote touched files: TimelineEditor referrerPolicy, ScheduledTasks definitions, Skills Manager checkbox aria-label, scheduled task service param types, MCP hub path type, voice cloning ArrayBuffer, and e2e fixture/spec typing.

Code review follow-up reopened: fixing explicit-mood classifier short-circuit and stream persist assistant_content sanitization before merge.

Code review follow-up fixed in this branch: Message now skips detectCharacterMood when an explicit moodLabel is present, and the stream persist endpoint sanitizes raw Emote directives before fingerprint/idempotency/storage while deriving emote_events when the client did not send them. Red checks: Message routing test failed on detectCharacterMood being called once; backend persist sanitization test failed before endpoint sanitization. Green checks: Message routing suite passed 12 tests; targeted frontend emote suite passed 7 files / 82 tests; targeted backend emote suite passed 40 tests with 3 warnings; Bandit review-fix report had no errors and no findings. bun run typecheck was rerun and still fails on the known unrelated baseline files outside this feature.

PR review follow-up after rebase on latest dev: addressing Qodo comments for docstrings/types, moving emote prompt business logic to core, capping prompt state lists, pytest markers, and validating persisted emote event offsets.

PR review follow-up complete after focused rebase onto origin/dev. Addressed Qodo comments by adding emote core docstrings/type annotations, moving prompt instruction/state extraction into core Character_Chat logic, capping prompt state lists at 25 with (+N more), adding module-level pytest markers, validating persisted client emote offsets against sanitized UTF-16 content length with non-decreasing offsets, and removing a rebase duplicate mock declaration in the frontend integration test. Verification: backend targeted suite passed 44 tests with 3 warnings; frontend targeted emote suite passed 7 files / 85 tests; apps/tldw-frontend bun run typecheck passed; Bandit touched backend report /tmp/bandit_character_emotes_pr_review.json had no errors and no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented explicit character chat Emote directives across backend and WebUI. The parser strips standalone directives from visible and persisted assistant text, streams accepted events live to portrait mood state, stores strict emote_events metadata plus final mood_label, resolves safe custom emote image slugs, keeps heuristic mood detection as fallback only when no explicit directive exists, sanitizes non-streaming responses, and restores final explicit emotes from history without replaying beat events in v1. Follow-up B remains tracked separately as TASK-12164.
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
