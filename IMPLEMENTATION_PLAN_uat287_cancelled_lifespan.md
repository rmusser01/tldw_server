# UAT287: cleanup after cancelled application lifespan

Backlog: TASK-13260.224. Native931e55 reproduction in `.tmp/uat286-native` shows Python finalization waiting for worker threads after the second SIGINT; the lifespan's bare yield bypasses shutdown on cancellation.

## Stage 1: causal regression
**Goal**: Exercise the real main lifespan with a non-daemon worker owned by mocked startup/shutdown boundaries.
**Success Criteria**: Cancellation propagates but must release the worker; normal exit and body exception follow the same cleanup path.
**Tests**: Lifecycle unit regression red before the change.
**Status**: Complete

## Stage 2: lifecycle correction
**Goal**: Run the existing shutdown sequence from finally without changing shutdown order or resource policy.
**Success Criteria**: Causal regression and affected lifecycle suites pass; no new lint/Bandit findings; independent review complete.
**Tests**: Lifespan startup/shutdown, resource cleanup, main readiness and owned-worker tests as appropriate.
**Status**: Complete

## Stage 3: native PostgreSQL shutdown
**Goal**: Verify the original open-browser/two-interrupt sequence and ordinary shutdown on committed source.
**Success Criteria**: Owned API process and resources exit without additional termination; source integrity and official fixture cleanup recorded.
**Tests**: Native browser connections, SIGINT/SIGINT and ordinary closed-browser shutdown; process/listener receipts.
**Status**: In Progress
