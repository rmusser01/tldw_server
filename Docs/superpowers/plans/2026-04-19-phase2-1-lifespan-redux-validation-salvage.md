## Stage 1: Lock Contract
**Goal**: Define explicit unit coverage for the setup warning and AuthNZ integrity preflight seam.
**Success Criteria**: `test_startup_validation.py` exists and fails before implementation.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_startup_validation.py -v`
**Status**: Complete

## Stage 2: Extract Startup Validation Helper
**Goal**: Move the first-time setup warning and AuthNZ integrity preflight into `app/services/startup_validation.py`.
**Success Criteria**: `main.py` delegates to `run_startup_validations()` and the helper preserves current warning and exception behavior.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_startup_validation.py -v`
**Status**: Complete

## Stage 3: Guard Lifespan Behavior
**Goal**: Verify the additional extraction does not change broader lifecycle expectations.
**Success Criteria**: Existing lifecycle tests remain green after the refactor.
**Tests**: `python -m pytest tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`
**Status**: Complete

## Stage 4: Security Verification And Commit
**Goal**: Re-run Bandit on the newly touched startup files and commit only after fresh evidence.
**Success Criteria**: Bandit reports no findings on the touched scope and the branch is ready for the next 2.1 decision.
**Tests**: `python -m bandit -r tldw_Server_API/app/services/startup_validation.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_phase2_1_validation_redux.json`
**Status**: Complete
