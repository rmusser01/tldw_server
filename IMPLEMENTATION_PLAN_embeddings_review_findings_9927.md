## Stage 1: Verify and Capture Regressions
**Goal**: Confirm the current Embeddings findings and add focused failing tests for the validated hot-path defects.
**Success Criteria**: Tests demonstrate unsafe artifact paths, unsafe Chroma fallback/dimension behavior, orphaned Redis enqueue failures, DLQ encryption downgrade, and sensitive logging.
**Tests**: Focused pytest cases under `tldw_Server_API/tests/Embeddings_isolated/` and `tldw_Server_API/tests/ChromaDB/unit/`.
**Status**: Complete

## Stage 2: Harden Artifact and Queue Boundaries
**Goal**: Ensure job artifacts stay under the per-user artifact directory and Redis enqueue failures are surfaced to the root job.
**Success Criteria**: Payload-provided artifact paths are rejected unless safely confined, generated artifact names are sanitized, and enqueue infrastructure failures do not leave root jobs queued without work.
**Tests**: New/updated Embeddings job worker and jobs adapter tests.
**Status**: Complete

## Stage 3: Harden Chroma Storage Behavior
**Goal**: Prevent silent in-memory fallback and implicit collection deletion on dimension mismatch.
**Success Criteria**: Persistent Chroma initialization fails closed unless explicitly allowed, and dimension mismatches raise without deleting collections.
**Tests**: New/updated ChromaDB unit tests.
**Status**: Complete

## Stage 4: Harden DLQ and Logging Privacy
**Goal**: Fail closed for configured DLQ encryption when crypto is unavailable and remove sensitive payload text from logs.
**Success Criteria**: DLQ encryption no longer returns `alg=none` for configured encryption, provider error logs omit response bodies, and vector search logs omit query text.
**Tests**: New/updated Embeddings DLQ, connection pool, and ChromaDB logging tests.
**Status**: Complete

## Stage 5: Dead-Code Cleanup Notes and Verification
**Goal**: Quarantine or document inactive sharding/request-signing helpers and run focused verification.
**Success Criteria**: Runtime-facing inactive helpers are clearly marked or removed from active paths, focused tests pass, Bandit is run on touched Embeddings files, and TASK-9927 is updated with results.
**Tests**: Focused pytest commands plus Bandit touched-scope scan.
**Status**: Complete
