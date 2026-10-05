# Character greeting creation transport — TASK-13454.3

Baseline8abb33ed03; fresh origin/dev025627214c unchanged. The approved explicit character-opening design requires the existing create endpoint's supported query options. Reuse createChat, its query builder and captured scope transport; keep unrelated callers unchanged. No new API, provider sender, store or owner resolver.

## Stage 1: Reproduce the missing query contract
**Goal**: failing direct client and real transport checks.
**Success Criteria**: default/selected alternate and explicit false/zero retain owner, signal and scope while remaining outside JSON.
**Tests**: createChat mutation/request-scope and real storage/fetch boundary.
**Status**: Complete

## Stage 2: Extend the existing client
**Goal**: typed optional seed query options and validation.
**Success Criteria**: exact backend query names/values, invalid alternate index prevented before dispatch, unchanged absent-options path.
**Tests**: red cases pass plus existing create/update/request-scope regressions.
**Status**: Complete

## Stage 3: Verify, review and locally commit
**Goal**: reviewable bounded delivery with honest limits.
**Success Criteria**: relevant tests/types/extension compile/lint/diff and both independent reviewers; associated task normalized and locally committed.
**Tests**: Bandit N/A to TypeScript/docs-only production scope. Actual greeting UI, preview separation, stored rows, zero-provider startup, repeat/unknown recovery and native/packaged checks remain open. Completed .7 plan archived8abb33ed03 can be removed here.
**Status**: Complete

Final:3suites/65frontend tests;4existing SQLite/API tests; WebUI typecheck/extension entrypoint compile; no added lint5TS; both independent reviews; clean diff. Scope/cancellation IO doubles are stated; no frontend greeting/storage/provider/Sync success claim. Bandit N/A. Prior own .7 plan archived8abb33ed03 and removed.
