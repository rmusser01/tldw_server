# VN Command Recovery Implementation Plan

Task: TASK-13385. Design: Docs/Design/VN_PENDING_COMMAND_RECOVERY.md.
Base: f2830058d5f2ce697c9128551623e12547aaf13f. Prior completion commit
709b351e25a1a7071f1f716cb314a417568124cc preserved on its original branch and
cherry-picked as 6bbe71ab9e on this branch.

## Stage 1: Recovery Contract
**Goal**: Closed, credential-free session journal and verified authority lifecycle.
**Success Criteria**: Invalid records cannot be replayed; storage failures fail closed.
**Tests**: Roundtrip, malformed/version/IDs/payload, authority isolation, storage errors.
**Status**: Complete

## Stage 2: Workbench Recovery
**Goal**: Explicit original-request recovery after reload without automatic POST.
**Success Criteria**: Same key/body survives changed status; acknowledgement clears;
boundaries fence late responses and preserve other-pack concurrency.
**Tests**: Remount Start/Retry, changed source, logout/account/server, lost responses.
**Status**: Complete

## Stage 3: Qualification
**Goal**: Verify approved frontend-only scope and document limitations.
**Success Criteria**: Full VN tests, typecheck, scoped lint and browser checks pass;
task records evidence; no backend or shared dependency changes.
**Tests**: Focused and full VN Vitest, TypeScript, ESLint, desktop/mobile smoke.
**Status**: In Progress
