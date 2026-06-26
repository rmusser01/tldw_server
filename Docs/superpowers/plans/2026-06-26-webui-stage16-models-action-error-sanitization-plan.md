## Stage 1: Lock Models Action Error Sanitization
**Goal**: Capture the Models settings notification leak with a focused refresh failure regression.
**Success Criteria**: A failed refresh with raw endpoint, filesystem path, and secret-like text renders a sanitized notification description.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx -t "sanitizes refresh failure notifications"`
**Status**: Complete

## Stage 2: Reuse Sanitized Action Error Copy
**Goal**: Add a small local formatter and use it for Models refresh and OpenAI OAuth action notifications.
**Success Criteria**: User-facing notification descriptions redact API paths, filesystem paths, and secret-like tokens while preserving concise diagnostic context.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx`
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run scoped verification and record the result in Backlog.
**Success Criteria**: Focused Models tests pass, touched files lint clean, whitespace checks pass, and `TASK-12045` is finalized.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
