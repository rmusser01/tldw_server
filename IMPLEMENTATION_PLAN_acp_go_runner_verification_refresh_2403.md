# ACP Go Runner Verification Refresh Plan

## Stage 1: Inspect Runner Verification Surface
**Goal**: Confirm the expected `tools/tldw-agent` verification commands, host prerequisites, and current dev commit.
**Success Criteria**: Verification commands are identified before execution and toolchain/OS context is captured.
**Tests**: Read `tools/tldw-agent/scripts/verify-local-build.sh`, `go version`, and current commit metadata.
**Status**: Complete

## Stage 2: Run Go Runner Build And Test Gates
**Goal**: Execute the runner build/test verification on current dev with local cache paths that are safe in this environment.
**Success Criteria**: Host and ACP binaries build, and Go tests either pass or produce classified failures.
**Tests**: `tools/tldw-agent/scripts/verify-local-build.sh` and focused/full `go test` as needed.
**Status**: Complete

## Stage 3: Classify Results And Update Trackers
**Goal**: Record command evidence, failure classification if any, and final status on `TASK-2389`, #2403, and parent #2398.
**Success Criteria**: The GitHub child issue and parent tracker contain reproducible evidence and final status.
**Tests**: Backlog/issue evidence review and final git status.
**Status**: Complete
