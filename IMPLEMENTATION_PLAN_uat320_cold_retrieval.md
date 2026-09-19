# UAT320 cold Knowledge QA transport repair

Backlog: TASK13260.258. Authorized repair before full matrix rerun.

## Stage 1: Establish the timing failure
**Goal**: Explain why first model loading leaves the native QA page searching.
**Success Criteria**: Preserve original cold failure and reproduce with controlled latency.
**Tests**: Frozen63d native cold request; warm restart; diagnostic190-second first-load delay, thread/async stacks and installed proxy configuration.
**Status**: Complete

The model download/load exceeds quickstart's180-second proxy idle limit. No retrieval body event is emitted before completion. The direct browser stream starts its timeout only after fetch resolves. The warm control completes in4seconds; the190-second controlled load reproduces the indefinite page even after owned backend retrieval drains.

## Stage 2: Preserve retrieval transport and enforce response acquisition deadlines
**Goal**: Keep a slow active retrieval connected and bound an unresponsive initial fetch.
**Success Criteria**: Retrieval heartbeat uses the existing owned task queue; cancellation still drains work; client timeout uses its configured budget and cleans timers/listeners on every exit. No automatic generation replay or generation-stage heartbeat.
**Tests**: Causal slow-prefetch and pre-response timeout controls; progress enabled/disabled, cancellation, immediate result/error, caller abort, response/body failure and successful streams.
**Status**: Complete

Causal four missing-heartbeat controls and one unbounded initial-response control reproduced. Final98backend unit/endpoint and126frontend transport tests pass;86backend unit controls rechecked after lint-compatible iterator spelling. Production Bandit0; existing synthetic test-password finding retained. Ruff/ESLint pass; TypeScript93identical baseline errors. Bounded independent review reports no actionable findings. First green's two test fixture errors were corrected against the actual streaming endpoint contract; no sources-only product change needed.

## Stage 3: Verify native timing and clean up
**Goal**: Verify the repaired timing path and preserve evidence without claiming a full matrix pass.
**Success Criteria**: Controlled native slow retrieval completes or reports a truthful bounded error, exact source revision and owned runtime cleanup recorded.
**Tests**: Focused backend/frontend suites, Bandit, lint baseline comparison, independent review, fresh immutable and labeled controlled native acceptance.
**Status**: In Progress
