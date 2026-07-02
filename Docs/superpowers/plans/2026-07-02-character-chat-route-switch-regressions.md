## Stage 1: Reproduce Character Route State Split
**Goal**: Capture the direct character route regression where a tracked character assistant is active but legacy character state is empty.
**Success Criteria**: A focused chat shell test fails before the fix and verifies the active character label, readiness, and recent-session state use the tracked assistant.
**Tests**: `bun run test src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --reporter=verbose`
**Status**: Complete

## Stage 2: Fix Tracked Character Switching
**Goal**: Ensure selecting a different tracked character from an active server chat clears the stale chat before applying the new character.
**Success Criteria**: Picker behavior clears history, messages, server chat id, and server chat participant metadata before the new tracked assistant selection is persisted.
**Tests**: `bun run test src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --reporter=verbose`
**Status**: Complete

## Stage 3: Resolve Effective Assistant Precedence
**Goal**: Prevent stale server-tracked metadata from overriding a newly selected tracked character/persona draft.
**Success Criteria**: Effective assistant resolution returns the new tracked draft when it differs from the active tracked chat metadata.
**Tests**: `bun run test src/hooks/chat/__tests__/effective-assistant-state.test.ts --reporter=verbose`
**Status**: Complete

## Stage 4: Verification And UAT
**Goal**: Verify the targeted tests, compile/type safety, and browser UAT for direct Miku route plus two separate character chat sessions.
**Success Criteria**: Focused tests pass, TypeScript check has no new relevant errors, UAT does not reproduce the stale character/chat bug, and backlog is updated with verification notes.
**Tests**: Focused Vitest suite, UI package TypeScript check, browser walkthrough against local WebUI and llama.cpp.
**Status**: Complete

## Stage 5: Clear Remaining Console Warnings
**Goal**: Remove the remaining `/openapi.json` 404, optional per-chat settings 404, and AntD deprecation warnings observed during chat UAT.
**Success Criteria**: Quickstart WebUI does not probe same-origin `/openapi.json`, missing optional chat settings records are treated as empty without request warning noise, and affected AntD components use current popup APIs.
**Tests**: Focused Vitest guards for OpenAPI discovery, chat settings sync, and AntD picker/popover props.
**Status**: Complete

## Verification Notes
- `git diff --check` passed.
- `bun run test src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/__tests__/useSelectedAssistant.test.tsx src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx --reporter=verbose` passed 10 files / 140 tests.
- `bun run test src/services/__tests__/tldw-api-client.connection-sync.test.ts src/entries/shared/__tests__/background-init.test.ts src/services/__tests__/chat-settings.sync.test.ts src/services/__tests__/background-proxy.test.ts src/components/Common/__tests__/character-picker-surface.guard.test.ts --reporter=verbose` passed 5 files / 68 tests.
- `bun run test src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/__tests__/useSelectedAssistant.test.tsx src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/services/__tests__/tldw-api-client.connection-sync.test.ts src/entries/shared/__tests__/background-init.test.ts src/services/__tests__/chat-settings.sync.test.ts src/services/__tests__/background-proxy.test.ts src/components/Common/__tests__/character-picker-surface.guard.test.ts --reporter=verbose` passed 15 files / 208 tests.
- `rg -n "overlayClassName|dropdownRender" apps/packages/ui/src --glob '!**/__tests__/**'` found no production usages.
- `NODE_OPTIONS=--max-old-space-size=8192 bun run tsc --noEmit -p tsconfig.json` still exits 2 on existing UI-package baseline errors outside this task's touched files. The local `antd` symlink had to be pointed at the installed Bun store hash for this check; with that fixed, the remaining diagnostics were Notes, AudioStudio, ResearchWorkspace, ScheduledTasks, Setup, Skills, Dexie, background, scheduled task services, MCP Hub, and voice cloning baseline errors.
- Browser UAT used Playwright outside the sandbox because Chromium launch is blocked by macOS Mach-port sandbox permissions. Fresh Miku-to-Ashley switch and direct Ashley sessions both showed Ashley active, sent the prompt, and did not show the `speaker_character_name must reference` error.
- Backend verification for the two fresh UAT chat ids returned `character_id: 4`, `assistant_kind: "character"`, `assistant_id: "4"`, and `source: "webui-character-chat"`.
- Browser-console recheck for the remaining warning items was attempted with a local quickstart WebUI dev server, but Next/Turbopack in this worktree failed before rendering `/chat` with `Cannot resolve 'antd'` for shared UI imports. Node and the UI package TypeScript resolver can resolve the repaired `antd` symlink; this appears to be a local dev-server dependency resolution issue rather than a regression in the touched code.
- Bandit was not run because this slice changes TypeScript/React UI code and Backlog/plan markdown only.
