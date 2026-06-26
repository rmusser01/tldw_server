## Stage 1: Lock STT Action Error Sanitization
**Goal**: Capture STT save-to-notes and history-load action notification leaks with focused regressions.
**Success Criteria**: Failed actions with raw endpoint, filesystem path, and secret-like text render sanitized notification descriptions.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx -t "sanitizes"`
**Status**: Complete

## Stage 2: Reuse Sanitized STT Action Error Copy
**Goal**: Reuse the shared server error sanitizer for STT action error notifications and extend it for secret-like token redaction.
**Success Criteria**: User-facing notification descriptions redact API paths, filesystem paths, and secret-like tokens while preserving concise diagnostic context.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run scoped verification and record the result in Backlog.
**Success Criteria**: Focused STT tests pass, touched files lint clean, whitespace checks pass, and `TASK-12046` is finalized.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
