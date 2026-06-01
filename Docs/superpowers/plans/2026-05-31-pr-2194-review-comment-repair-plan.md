## Stage 1: Review Comment Inventory
**Goal**: Confirm every unresolved PR #2194 review thread and map each one to a backend, frontend, docs, or schema fix.
**Success Criteria**: Every actionable comment has an implementation target or a documented no-op rationale.
**Tests**: PR review thread inspection via GitHub CLI.
**Status**: Complete

## Stage 2: Backend Contract Tests
**Goal**: Add focused regression coverage for setup sanitizer safety, setup write access after terminal first-run states, local provider hostnames, provider response redaction, and completion ordering.
**Success Criteria**: New tests fail before the runtime changes and pass after implementation.
**Tests**: Targeted pytest runs for setup API, provider validation, first-run state, and audio health touched paths.
**Status**: Complete

## Stage 3: Backend Review Fixes
**Goal**: Repair setup state transitions, first-run store async usage, sanitizer false positives, audio pack import scope, local endpoint validation, exception placement, and documentation gaps.
**Success Criteria**: Backend behavior matches PR review expectations without weakening secret/path validation.
**Tests**: Targeted pytest plus Bandit on touched Python files.
**Status**: Complete

## Stage 4: Frontend Review Fixes
**Goal**: Preserve provider setup state across wizard resume, save all configured selected providers, improve error handling, and redact provider secrets from service responses.
**Success Criteria**: Frontend tests cover the changed wizard/provider/chat/service behavior and no API keys survive in returned provider payloads.
**Tests**: Targeted Bun/Vitest tests for onboarding components and setup services.
**Status**: Complete

## Stage 5: Verification And PR Thread Closeout
**Goal**: Run targeted verification, update Backlog task state, commit/push changes, and resolve or reply to all addressed PR threads.
**Success Criteria**: Relevant tests pass, task records are current, PR review threads no longer contain unaddressed actionable comments.
**Tests**: GitHub PR thread re-query and status check snapshot.
**Status**: In Progress
