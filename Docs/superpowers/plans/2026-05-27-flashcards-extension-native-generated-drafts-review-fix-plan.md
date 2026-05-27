## Stage 1: Review Comment Verification
**Goal**: Verify PR #2079 review comments against the current sidepanel Flashcards branch after rebasing on `origin/dev`.
**Success Criteria**: Unresolved GitHub review threads and PR checks are inspected; still-valid comments are listed in TASK-520.
**Tests**: `gh pr view 2079`, GraphQL review-thread query, `gh pr checks 2079`.
**Status**: Complete

## Stage 2: Regression Tests
**Goal**: Add coverage for duplicate generation activation, unexpected generation responses, and localized generated-card count copy.
**Success Criteria**: New tests fail against the pre-fix implementation for the reviewed behavior.
**Tests**: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`.
**Status**: Complete

## Stage 3: Sidepanel Review Fixes
**Goal**: Make sidepanel generation re-entrant-safe, tolerate malformed generation responses, and avoid hardcoded English plural labels in translation interpolation.
**Success Criteria**: Both generation entry points share a synchronous in-flight guard; native generation normalizes optional flashcard results; generated/save/queue status copy uses localized whole-message branches.
**Tests**: Focused sidepanel Vitest suite.
**Status**: Complete

## Stage 4: Metadata, Verification, and PR Update
**Goal**: Restore Backlog metadata and record verification before pushing review fixes.
**Success Criteria**: TASK-519 has a creation timestamp; TASK-520 records the review fix summary and verification; review-fix changes are ready to push to PR #2079.
**Tests**: `git diff --check`, focused Vitest, TypeScript check where practical, Bandit applicability review.
**Status**: Complete
