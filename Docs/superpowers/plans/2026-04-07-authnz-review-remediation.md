## Stage 1: Regression Tests
**Goal**: Add focused failing tests for the three reviewed findings before touching production code.
**Success Criteria**: Symlink persistence test fails for overwrite/fail-open behavior; refresh rotation failure-path test fails by returning tokens instead of error; missing-scope API key test fails by authorizing `read`.
**Tests**: `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py -k symlink_persistence`; `python -m pytest -q tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`; `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py -k missing_scope`
**Status**: Complete

## Stage 2: Minimal Fixes
**Goal**: Make the session-key, refresh-rotation, and API-key-scope paths fail closed.
**Success Criteria**: Session key persistence rejects symlinks, refresh helpers raise instead of returning rotated tokens when revocation persistence fails, and missing scope no longer grants `read`.
**Tests**: Stage 1 focused tests
**Status**: Complete

## Stage 3: Verification
**Goal**: Verify the touched AuthNZ slice with targeted tests and a security scan.
**Success Criteria**: Focused pytest selection passes; Bandit runs on touched files or the environment limitation is explicitly captured with evidence.
**Tests**: `python -m pytest -q ...`; `python -m bandit -r tldw_Server_API/app/core/AuthNZ/session_manager.py tldw_Server_API/app/core/AuthNZ/jwt_service.py tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
**Status**: Complete

## Stage 4: Blind Spots Review
**Goal**: Revisit the earlier blind spots after the fixes to see whether any additional issues remain in migrations/database/repo-sensitive AuthNZ paths.
**Success Criteria**: Additional review completed with either new findings or an explicit residual-risk statement.
**Tests**: Read-only review plus any targeted verification needed
**Status**: Complete
