## Stage 1: Host Runtime Evidence
**Goal**: Record the release host, available sandbox runtime tools, and why a runtime can or cannot be selected for ACP sandbox support.
**Success Criteria**: Docker, Lima, and macOS virtualization availability are checked with exact commands and results.
**Tests**: Host/runtime probe commands and targeted read review of existing sandbox runtime docs/tests.
**Status**: Complete

## Stage 2: Release Posture
**Goal**: Decide whether this release has a passing sandbox-backed ACP runtime claim or explicitly declines that claim.
**Success Criteria**: Documentation states the selected posture and keeps untested runtimes caveated.
**Tests**: Read review of ACP readiness, compatibility, certification, and user setup docs.
**Status**: Complete

## Stage 3: Documentation Updates
**Goal**: Add a durable #2400 evidence artifact and connect it from the ACP readiness and setup surfaces.
**Success Criteria**: Evidence includes host OS, runtime versions or absence, command, commit, result, and remaining next steps.
**Tests**: `git diff --check` and targeted doc grep/read review.
**Status**: Complete

## Stage 4: Verification and Handoff
**Goal**: Verify touched docs and relevant fail-closed tests, then open the PR against `dev`.
**Success Criteria**: Verification output is recorded in Backlog and PR notes; #2400 and parent #2398 are linked.
**Tests**: Targeted pytest for ACP runtime policy/sandbox runner behavior when feasible, plus docs checks.
**Status**: Complete
