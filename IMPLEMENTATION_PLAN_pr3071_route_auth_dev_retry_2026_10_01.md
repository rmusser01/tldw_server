# PR3071 Route-Auth Dev Retry

Tracking: TASK-13418. Existing draft PR3071; no merge authorization.
Frozen dev: 3caebcfc1e16596f4a352744a9f017017d766165.
Preserve live services, Chrome profiles/tabs/data, stashes and unrelated files.

## Stage 1: Preserve And Integrate
**Goal**: Preserve the prior qualification TASK-13417 byte-identically before
integrating the incoming unrelated route-map TASK-13417 and route-auth fixes.
**Success Criteria**: Both histories retained without active ID collision;
clean latest-dev merge and unchanged Chat Workspace frontend/backend scope.
**Tests**: Merge preview, archive SHA256, incoming record comparison and source
comparison against the qualified prior integration.
**Status**: Complete

## Stage 2: Qualify The Delta
**Goal**: Verify incoming auth/ratchet behavior and the real workspace without
rerunning or relabeling source-identical historical results as fresh tests.
**Success Criteria**: Owning/adjacent gate tests pass; scoped Bandit has no new
changed-code findings; live native Chrome loaded-state/draft/reload checks pass
without mocks, state injection or completion resends.
**Tests**: Route-auth lint, benchmark auth/error mapping, changed source lint,
Bandit comparison and actual Chrome/CDP verification on preserved services.
Add failing subprocess inventory tests for JSON enable lists and dotenv-only
config selection; fix the shared inspection loader using existing config APIs.
**Status**: Complete

## Stage 3: Retry And Publish
**Goal**: Retry PostgreSQL/current-head CI and normally publish verified evidence
to the existing draft PR while preserving the requester Change summary.
**Success Criteria**: Exact-head checks reported honestly; official PG fixture
unavailability remains a skip, not a pass; normal hooks/push and PR readback pass.
**Tests**: Official PG fixtures, exact-head CI logs, original data/tab/stash
preservation, unchanged-source binding and final PR head/body/draft readback.
**Status**: In Progress

Hosted backend-required on head7274 failed only the blocking OpenAPI drift
check: 2105 paths/3259 schemas versus the checked-in3245 schemas. Its mypy
step is explicitly non-blocking. Regenerate the fingerprint and frontend types
with the existing exporter, review the contract delta, then rerun the drift gate.
The completed auth-postgres shard ran56 real tests, but not the pending durable
turn/image recovery PostgreSQL cases; it does not close that qualification gap.

Initial retry: seven selected PostgreSQL cases skip through official fixtures
because the backend is unavailable; Docker's read-only info probe times out and
127.0.0.1:5432 refuses connections. No Docker restart, shared-container removal,
replacement DB or fixture bypass. Prior head7274b91cec has103 successful checks,
16 running and169 queued with no failures at the first retry snapshot; these
are head-specific observations, not certification of the next integration.

Final delta:87 passed/1 inherited skip; owning ratchet17 passed after four
expected red regressions. Independent follow-up review has no actionable
findings; changed Ruff is clean and four-module Bandit has zero findings/errors.
Isolated CI-version OpenAPI export reproduces the exact failed hosted hash;
frozen-dev control reproduces the old fingerprint. Reviewed14 added schemas,
three changed models and nine intended path contracts with no removals.
Fingerprint/types regenerated; fresh drift gate passes. New native Chrome
reload/mobile passes with preserved draft and zero completion dispatches.
Hosted head7274 auth-integration-b-z JUnit reports149 passed/0 skipped,
including all six actual durable-turn PostgreSQL cases. The separate image
recovery PG case and next-head hosted checks remain unqualified. Local official
fixture skips are retained and not relabeled as hosted successes.
