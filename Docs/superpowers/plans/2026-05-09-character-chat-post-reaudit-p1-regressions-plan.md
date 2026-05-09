# Character Chat Post-Re-Audit P1 Regression Fix Plan

Backlog task: TASK-173

## Stage 1: Reproduce Re-Audit Regressions
**Goal**: Capture the two remaining P1 failures in focused tests.
**Success Criteria**: Tests fail because the WebUI first-run gate preempts character-chat intent and row-level chat can navigate away when model availability is unresolved.
**Tests**: `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`; `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx`
**Status**: Complete

## Stage 2: Preserve Character Intent Through WebUI First-Run Gate
**Goal**: Let character-chat routes reach the package-level onboarding lane instead of showing the generic assistant splash.
**Success Criteria**: `/characters` and explicit `intent=character-chat` routes bypass the generic gate; non-character routes retain existing setup behavior.
**Tests**: Focused app layout tests.
**Status**: Complete

## Stage 3: Keep Row Chat Local When Model Readiness Is Unknown
**Goal**: Prevent stale selected-model state from making the row action navigate to generic home before the model catalog confirms availability.
**Success Criteria**: Row `Chat as` preserves the selected character and shows the character-chat setup blocker when no model catalog is available.
**Tests**: Focused Characters manager tests.
**Status**: Complete

## Stage 4: Verification And Backlog Closeout
**Goal**: Verify the focused frontend surface and record outcomes.
**Success Criteria**: Focused tests, UI typecheck, diff hygiene, and Backlog TASK-173 are updated.
**Tests**: Focused Vitest runs plus `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false`.
**Status**: Complete

## Verification

- RED: `bunx vitest run __tests__/app/app-layout.test.tsx --testTimeout=30000` failed the two new character-chat app-gate tests because character routes did not bypass the generic first-run gate.
- RED: `bunx vitest run src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --testTimeout=30000` failed the new `FirstRunGate` bypass test and stale-selected-model row-chat test.
- GREEN: `bunx vitest run src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx -t "first-run setup is bypassed" --testTimeout=30000` passed.
- GREEN: `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "stale selected model" --testTimeout=30000` passed.
- GREEN: `bunx vitest run __tests__/app/app-layout.test.tsx -t "character-chat" --testTimeout=30000` passed.
- GREEN: `bunx vitest run src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx --testTimeout=30000` passed.
- GREEN: `bunx vitest run __tests__/app/app-layout.test.tsx --testTimeout=30000` passed.
- GREEN: `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "row chat intent|stale selected model|first-run template" --testTimeout=30000` passed.
- GREEN: `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false` passed from `apps/packages/ui`.
- GREEN: `node /private/tmp/character-p1-smoke.mjs` passed with Puppeteer/Chrome and saved evidence under `Docs/Reviews/assets/2026-05-09-character-chat-p1-smoke`.
- GREEN: `git diff --check` passed.
- Baseline note: direct `apps/tldw-frontend` TypeScript check still fails on pre-existing broad app errors outside this patch.
- Bandit: skipped because the touched runtime code is TypeScript/React and no Python files changed.
