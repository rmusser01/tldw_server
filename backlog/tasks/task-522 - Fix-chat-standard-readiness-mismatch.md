---
id: TASK-522
title: Fix chat standard readiness mismatch
status: Done
labels:
- chat
- ux
- bug
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire standard /chat model/provider readiness into cockpit runtime/status/composition surfaces so setup blockers do not conflict with selected model route state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When no usable chat models are available, /chat cockpit surfaces do not report runtime Ready for a selected stored/default model.
- [x] #2 Standard chat uses the existing chat model usability contract outside character mode.
- [x] #3 No-provider/no-model setup notices remain aligned with cockpit runtime/status/composition state.
- [x] #4 Focused Playground cockpit and chat-model availability tests pass.
- [x] #5 Verification and known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: `PlaygroundChat` rendered setup notices from `providersStatus` and `chatModels`, but the cockpit runtime/status/composition surfaces only received `modelUsability` in character workflow. Standard chat therefore fell through to runtime `Ready` for a stored/default selected model such as `tldw:gpt-4o` even after the chat-model catalog resolved empty.

TDD: Added a failing `Playground.cockpit-shell.test.tsx` regression that sets standard chat selected model to `tldw:gpt-4o`, makes `fetchChatModels` resolve `[]`, and asserts the runtime rail/status strip show blocked model state and `No chat models configured` rather than `Ready`.

Implementation: Reused the existing `buildChatModelUsability` result as standard chat readiness in `Playground.tsx`, then passed the active model usability state/message into the composition preview, runtime inspector, status strip, and composer model feedback props. Character workflow still layers character readiness copy on top when active.

Browser sanity check: Started the local WebUI on `127.0.0.1:18015`, mocked `/api/v1/llm/models/metadata` as empty plus provider/status endpoints, skipped the first-run overlay, and confirmed `/chat` text included runtime `Error` and `No chat models configured` for `tldw:gpt-4o` instead of runtime `Ready`. The temporary dev server was stopped afterward; `lsof -nP -iTCP:18015 -sTCP:LISTEN` returned no listener.

Verification:
- RED: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --testNamePattern "does not report standard chat runtime ready"` failed because the runtime rail still showed `Runtime Ready`.
- GREEN focused regression: same command passed after the implementation.
- Focused suite passed: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/playground-composition-preview.test.ts src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx src/utils/__tests__/chat-model-availability.test.ts` (6 files, 139 tests).
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false` still fails only on the known unrelated `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)` `GalleryCardDensity` baseline.
- Bandit skipped because this slice touched frontend TypeScript, frontend tests, and Backlog metadata only; no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the standard `/chat` cockpit readiness mismatch. The cockpit runtime rail, status strip, composition preview, and composer model feedback now use the existing chat-model usability result in standard chat, so an empty/unusable chat model catalog blocks the selected route instead of showing runtime `Ready`. Added regression coverage for the `tldw:gpt-4o` selected-model/no-chat-models case.
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
