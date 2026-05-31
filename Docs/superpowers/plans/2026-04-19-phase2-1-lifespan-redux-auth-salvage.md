## Stage 1: Lock Contract
**Goal**: Preserve the extracted AuthNZ startup behavior with explicit unit coverage.
**Success Criteria**: `test_startup_auth.py` exists and fails before implementation.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_startup_auth.py -v`
**Status**: Complete

## Stage 2: Extract Auth Startup Helper
**Goal**: Move the auth database pool, schema ensure, PG extras, RBAC seed, and provider override loading into a dedicated service helper.
**Success Criteria**: `main.py` delegates to `init_auth_services()` and the helper matches the tested call order.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_startup_auth.py -v`
**Status**: Complete

## Stage 3: Guard Existing Lifespan Behavior
**Goal**: Verify the extraction does not regress existing startup lifecycle expectations.
**Success Criteria**: Relevant existing service lifecycle tests remain green.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`
**Status**: Complete

## Stage 4: Security Verification And Commit
**Goal**: Validate touched paths with Bandit and commit only after fresh evidence.
**Success Criteria**: Bandit reports no findings on touched files and the branch is ready for the next 2.1 salvage slice.
**Tests**: `python -m bandit -r tldw_Server_API/app/services/startup_auth.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_phase2_1_auth_redux.json`
**Status**: Complete
