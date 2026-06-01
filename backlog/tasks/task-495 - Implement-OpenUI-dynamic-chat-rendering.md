---
id: TASK-495
title: Implement OpenUI dynamic chat rendering
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-01 15:57
labels: []
dependencies: []
references:
- TASK-491
- TASK-493
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
- Docs/superpowers/plans/2026-06-01-openui-dynamic-chat-rendering-implementation-plan.md
- https://github.com/pewdiepie-archdaemon/odysseus/pull/151
documentation:
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
- Docs/superpowers/plans/2026-06-01-openui-dynamic-chat-rendering-implementation-plan.md
modified_files:
- Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md
- Docs/superpowers/plans/2026-06-01-openui-dynamic-chat-rendering-implementation-plan.md
- apps/tldw-frontend/package.json
- apps/packages/ui/package.json
- apps/bun.lock
- apps/extension/package.json
- apps/packages/ui/src/types/dynamic-ui.ts
- apps/packages/ui/src/utils/dynamic-ui.ts
- apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts
- apps/packages/ui/src/utils/message-variants.ts
- apps/packages/ui/src/utils/__tests__/message-variants.test.ts
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts
- apps/packages/ui/src/routes/sidepanel-chat.tsx
- apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx
- apps/packages/ui/src/components/Common/DynamicUI/__tests__/OpenUIRenderer.test.tsx
- apps/packages/ui/src/store/option.tsx
- apps/packages/ui/src/utils/dynamic-ui-openui-prompt.ts
- apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts
- apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts
- apps/packages/ui/src/hooks/chat/useChatActions.ts
- apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts
- apps/packages/ui/src/hooks/chat/useDynamicUIActionBridge.ts
- apps/packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.dynamic-ui-action.integration.test.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChat.dynamic-ui-action.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx
- apps/packages/ui/src/public/_locales/en/playground.json
- apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.dynamic-ui-fallback.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/body.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/body.dynamic-ui-fallback.test.tsx
- apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts
- backlog/tasks/task-495 - Implement-OpenUI-dynamic-chat-rendering.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan for OpenUI dynamic chat rendering. Scope includes runtime feasibility, shared Dynamic UI metadata/utilities, persistence, renderer registry, OpenUI adapter, /chat request mode, action bridge, shared-surface fallbacks, verification, and Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 0 runtime feasibility completed and recorded before OpenUI dependency use
- [ ] #2 Dynamic UI envelope/types/validation/persistence implemented with test-first coverage
- [ ] #3 /chat supports temporary OpenUI request mode and active rendering only on opted-in web chat surface
- [ ] #4 OpenUI actions round-trip as visible user messages with host-attached provenance metadata
- [ ] #5 Extension sidepanel and workspace render source fallback unless explicitly enabled later
- [ ] #6 Focused unit/build/browser/security verification completed or documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
2026-06-01 Task 7: added the one-send /chat OpenUI request-mode control, passed Dynamic UI request overrides through Playground submit, and reset the mode after resolved dispatch. Red command: `bunx vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx` failed for the expected missing accessible OpenUI button. Green focused command passed 3/3 tests: `bunx vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts`. Package `bunx tsc --noEmit --pretty false` still fails on existing baseline errors; no Task 7 implementation files or the new OpenUI-mode test appear in the output. `git diff --cached --check` passed. Bandit not applicable because Task 7 touched TypeScript/TSX, locale JSON, and markdown only.

2026-06-01 Task 8: forwarded Dynamic UI metadata to workspace and extension sidepanel message surfaces, setting explicit fallback-only surface IDs `workspace` and `extension-sidepanel`. Red command: `bunx vitest run src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.dynamic-ui-fallback.test.tsx src/components/Sidepanel/Chat/__tests__/body.dynamic-ui-fallback.test.tsx` failed because the fallback mock never received metadata/surface props. Green focused command passed 2/2 after forwarding those props. Broader focused unit command passed 22/22: `bunx vitest run src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.dynamic-ui-fallback.test.tsx src/components/Sidepanel/Chat/__tests__/body.dynamic-ui-fallback.test.tsx`. Browser smoke passed 1/1 outside sandbox after the sandbox denied local dev-server binding: `bun run e2e:pw e2e/smoke/chat-openui-dynamic-ui.spec.ts --reporter=line --workers=1`. Frontend `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build:dev` passed outside sandbox after Turbopack hit an internal sandbox port-binding failure; warnings were the known documentation glob/traced-file warnings and token-sync reported OK. Extension `bun run compile` passed. Package `bunx tsc --noEmit --pretty false` still fails on existing baseline errors; the new Task 8 fallback tests no longer appear after fixing a test-only mock type, while existing `WorkspaceChatPanel.test.tsx` baseline errors remain. Bandit not applicable because Task 8 touched TypeScript/TSX, Playwright test, and Backlog markdown only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-01 Task 3: added shared Dynamic UI registry, source fallback, error boundary, placeholder OpenUI renderer, `/chat` surface opt-in, and source-first guards. Red command: `bunx vitest run src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts src/components/Common/Playground/__tests__/Message.error-recovery.guard.test.ts` failed for the expected missing renderer/default-surface integration. Green rerun passed 8/8 tests. Existing metadata suites passed 22/22 tests. Package-wide `bunx tsc --noEmit --pretty false` still fails on existing baseline issues outside Task 3; a filtered rerun showed no errors in Task 3 touched TypeScript files. Bandit not applicable because Task 3 touched TS/TSX/Backlog markdown only.
- 2026-06-01 Task 2 quality fix: added focused failing coverage for `metadataExtra` in message variant helpers and shared message metadata normalization. Red command: `bunx vitest run src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts src/utils/__tests__/dynamic-ui.test.ts src/utils/__tests__/message-variants.test.ts` failed with 5 expected failures for missing `normalizeMessageMetadataExtra` and dropped/stale variant `metadataExtra`. Green rerun passed 44/44 tests after copying/clearing variant metadata and using shared normalization in playground and sidepanel hydration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-06-01 Task 4: replaced the placeholder OpenUI renderer with the real @openuidev/react-lang Renderer using the @openuidev/react-ui/genui-lib chat library plus package CSS imports. Added focused red/green coverage for source/state/library/action forwarding in OpenUIRenderer.test.tsx; red failed against the placeholder fallback, green passed after adapter implementation. Also re-exported MessageMetadataExtra from the store barrel after package TypeScript exposed earlier Dynamic UI imports depending on that public type boundary. Focused Dynamic UI tests pass 28/28. Frontend build:dev passed with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 when rerun outside the sandbox; warnings were pre-existing documentation glob and trace-copy warnings. Extension build:chrome:dev reached WXT but stuck at zero CPU for several minutes and was terminated; extension compile passed and a resolver check confirmed the OpenUI runtime specifiers resolve from the shared UI adapter path.

2026-06-01 Task 4 review fixes: rebuilt the OpenUI chat library through createLibrary with chart/style-injection components filtered out, added a bounded themed OpenUI shell, normalized raw OpenUI ActionEvent payloads into the app dynamic action shape, and added raw-event regression coverage. Added OpenUI peer runtime dependencies to apps/extension and refreshed apps/bun.lock so the extension package can resolve the lazy adapter imports while active rendering remains disabled/fallback there. Verification: focused Dynamic UI suite passed 10/10, extension compile passed, Bun resolver from apps/extension passed for react-lang/genui-lib/CSS imports, frontend build:dev passed outside sandbox with known docs/tracing warnings, touched-file diff check passed. apps/packages/ui tsc still fails on unrelated baseline errors with no touched dynamic UI files in output. extension build:chrome:dev still hangs in WXT/Vite after startup and was terminated; tracked as a non-OpenUI-specific build caveat for later surface enablement.

2026-06-01 Task 4 Bandit: not applicable for this slice because touched files are TypeScript/TSX, package manifests/lockfile, and Backlog markdown only; no Python code paths were modified.

2026-06-01 Task 5: added OpenUI request-mode prompt injection and metadata tagging in the chat pipeline. Red command: `bunx vitest run src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts` failed for the expected missing OpenUI prompt injection. Green focused command passed 22/22 tests: `bunx vitest run src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts src/utils/__tests__/dynamic-ui.test.ts`. Package `bunx tsc --noEmit --pretty false` still fails on existing baseline errors; after touched-file fixes there are no new errors for Task 5 files, while an existing error remains in `chatModePipeline.conversation-id.test.ts`. Bandit not applicable because Task 5 touched TypeScript/TSX and markdown only.

2026-06-01 Task 6: added the OpenUI action bridge, conservative sensitive-value blocking, `/chat` bridge wiring, and submit-path provenance coverage. Red commands: `bunx vitest run src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx` failed on missing hook, and `bunx vitest run src/components/Option/Playground/__tests__/PlaygroundChat.dynamic-ui-action.guard.test.ts` failed on missing `/chat` bridge wiring. Green focused command passed 9/9 tests: `bunx vitest run src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx src/hooks/chat/__tests__/useChatActions.dynamic-ui-action.integration.test.tsx src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.dynamic-ui-action.guard.test.ts`. Package `bunx tsc --noEmit --pretty false` still fails on existing baseline errors; after fixing the Task 6 fixture type, no Task 6 touched files remain in the typecheck output. Bandit not applicable because Task 6 touched TypeScript/TSX and markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tasks 0-1 complete. Task 0 committed OpenUI runtime feasibility review and dependency gate in ec3e71c6b7, with follow-up Backlog status commit ef2035486a. Task 1 added Dynamic UI shared types, validation helpers, source preflight, action normalization, sensitive-value blocking, and focused tests across commits 2e08f1e38c, 8ec5a81a2b, and 69d1d36890. Focused utility suite passes with 17 tests. Task 1 passed spec-compliance and code-quality review.
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
