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
**Status**: In Progress

## Stage 2: Qualify The Delta
**Goal**: Verify incoming auth/ratchet behavior and the real workspace without
rerunning or relabeling source-identical historical results as fresh tests.
**Success Criteria**: Owning/adjacent gate tests pass; scoped Bandit has no new
changed-code findings; live native Chrome loaded-state/draft/reload checks pass
without mocks, state injection or completion resends.
**Tests**: Route-auth lint, benchmark auth/error mapping, changed source lint,
Bandit comparison and actual Chrome/CDP verification on preserved services.
**Status**: Not Started

## Stage 3: Retry And Publish
**Goal**: Retry PostgreSQL/current-head CI and normally publish verified evidence
to the existing draft PR while preserving the requester Change summary.
**Success Criteria**: Exact-head checks reported honestly; official PG fixture
unavailability remains a skip, not a pass; normal hooks/push and PR readback pass.
**Tests**: Official PG fixtures, exact-head CI logs, original data/tab/stash
preservation, unchanged-source binding and final PR head/body/draft readback.
**Status**: Not Started

Initial retry: seven selected PostgreSQL cases skip through official fixtures
because the backend is unavailable; Docker's read-only info probe times out and
127.0.0.1:5432 refuses connections. No Docker restart, shared-container removal,
replacement DB or fixture bypass. Prior head7274b91cec has103 successful checks,
16 running and169 queued with no failures at the first retry snapshot; these
are head-specific observations, not certification of the next integration.
