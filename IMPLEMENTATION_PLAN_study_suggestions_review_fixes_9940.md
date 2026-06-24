## Stage 1: Collision-Safe Action Fingerprints
**Goal**: Replace delimiter-joined selection fingerprints with bounded, structured fingerprints that cannot collide on topic delimiters.
**Success Criteria**: Distinct delimiter-containing topic selections produce distinct fingerprints; pending target IDs remain usable.
**Tests**: Add focused action fingerprint tests in `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py`.
**Status**: Complete

## Stage 2: Recency-Aware Anchor Status
**Goal**: Make anchor status prefer the newest relevant state instead of letting stale failed jobs mask newer ready snapshots.
**Success Criteria**: A newer active snapshot reports `ready` even when an older failed refresh job exists; newer pending/failed jobs still surface.
**Tests**: Add status regression coverage in `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py`.
**Status**: Complete

## Stage 3: Real Live Evidence Availability
**Goal**: Hydrate snapshot evidence against backing note DB targets instead of treating frozen source IDs as available by definition.
**Success Criteria**: Deleted/missing flashcard deck and quiz targets report unavailable while existing targets remain available.
**Tests**: Add endpoint/core evidence coverage in `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`.
**Status**: Complete

## Stage 4: Finalization Failure Cleanup
**Goal**: Prevent generated follow-up artifacts from being silently orphaned when generation-link finalization fails.
**Success Criteria**: Failed finalization releases the pending reservation and cleans up generated flashcard decks.
**Tests**: Add action endpoint regression coverage in `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`.
**Status**: Complete

## Stage 5: Verification And Closeout
**Goal**: Run focused tests, touched-scope Bandit, and update task tracking.
**Success Criteria**: Relevant tests pass; Bandit reports no new findings for touched backend files; Backlog task records verification.
**Tests**: `python -m pytest` focused StudySuggestions tests, `python -m bandit` on touched backend scope.
**Status**: Complete

## Verification
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py -q --tb=short`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -q --tb=short`

**Security**:
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/StudySuggestions tldw_Server_API/app/api/v1/endpoints/study_suggestions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -f json -o /tmp/bandit_study_suggestions_9940.json`

**Result**: StudySuggestions worker/API suites passed; Bandit reported zero findings.
