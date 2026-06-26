# WebUI Stage 11 Speech No-Provider Capability Recovery Plan

## Stage 1: Lock Current Gap
**Goal**: Add a focused Speech playground regression test for the server TTS no-provider state.
**Success Criteria**: Test expects a shared setup-required state instead of the local alert-only banner.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
**Status**: Complete

## Stage 2: Adopt Shared State
**Goal**: Replace the Speech route-local no-provider alert with the shared `StatePanel` setup-required state.
**Success Criteria**: The no-provider state uses user-facing setup language, preserves the settings link, and keeps the provider strip and disabled action behavior intact.
**Tests**: Focused Speech component test.
**Status**: Complete

## Stage 3: Verification And Task Closure
**Goal**: Verify the focused slice and record completion on `TASK-12040`.
**Success Criteria**: Focused test, touched-file lint, and whitespace checks pass; Bandit is documented as not applicable for TS/TSX/docs-only changes.
**Tests**: Focused Speech test, ESLint touched files, `git diff --check`.
**Status**: Complete
