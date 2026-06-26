## Stage 1: Lock TTS Voice-Cloning Action Error Sanitization
**Goal**: Capture TTS voice-cloning action notification leaks with focused regressions.
**Success Criteria**: Failed upload and voice actions with raw endpoint, filesystem path, and secret-like text render sanitized notification descriptions.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx`
**Status**: Complete

## Stage 2: Reuse Sanitized Voice-Cloning Error Copy
**Goal**: Reuse the shared server error sanitizer for TTS voice-cloning upload, encode, delete, and preview notifications.
**Success Criteria**: User-facing notification descriptions redact API paths, filesystem paths, URLs, and secret-like tokens while preserving concise diagnostic context.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx`
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run scoped verification and record the result in Backlog.
**Success Criteria**: Focused TTS tests pass, touched files lint clean, whitespace checks pass, and `TASK-12047` is finalized.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
