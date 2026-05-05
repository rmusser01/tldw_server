## Stage 1: Verify and Reproduce
**Goal**: Confirm which PR #854 review findings are still valid on the current branch and capture them with focused failing tests.
**Success Criteria**: Stale findings are identified; new or existing tests fail for the still-valid redaction, probe, SSH, and unsupported-mode issues.
**Tests**: `pytest` on targeted `tldw_Server_API/tests/VLLM_Management/*` and `tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py` slices.
**Status**: Complete

## Stage 2: Security and API Hardening
**Goal**: Fix the verified security and API-surface issues with minimal, local changes.
**Success Criteria**: Sensitive values are redacted, unsafe probe/SSH inputs are rejected, and unsupported execution modes are blocked.
**Tests**: Focused `pytest` targets for API, jobs service, SSH runner, and subprocess logging behavior.
**Status**: Complete

## Stage 3: Follow-up Triage
**Goal**: Reassess the remaining open review findings after Stage 2 and decide which are already stale versus which need a second implementation slice.
**Success Criteria**: Remaining findings are documented as fixed, stale, or deferred to a clearly bounded next slice with evidence from current code/tests.
**Tests**: Focused verification only for any still-valid follow-up changes.
**Status**: Complete

## Stage 4: Managed Route Authorization
**Goal**: Close the explicit `provider_instance_id` selection gap for chat and embeddings without broadening this PR into per-instance ACL design.
**Success Criteria**: Non-admin/non-single-user callers cannot select arbitrary managed instances; existing default managed routing still works.
**Tests**: `test_vllm_instance_routing.py` coverage for chat and embeddings explicit-selection authorization paths.
**Status**: Complete

## Stage 5: DB Ownership Boundary
**Goal**: Address the remaining PR #854 review thread that flagged SQLite driver ownership under `core/VLLM_Management`.
**Success Criteria**: Raw `sqlite3` usage for managed vLLM persistence lives under `core/DB_Management`; `core/VLLM_Management/sqlite_repo.py` is only a compatibility import; a regression test guards against reintroducing SQLite driver imports under `core/VLLM_Management`.
**Tests**: Red/green static boundary test in `test_repository.py`; focused vLLM backend suite covering repository, resolver, API, jobs, worker, provider listing, and routing.
**Status**: Complete

Verification notes:
- Red check: `test_vllm_management_core_does_not_own_sqlite_driver_imports` failed with `sqlite_repo.py` as the offender before the move.
- Green checks: `test_repository.py` passed 5 tests; focused vLLM backend slice passed 62 tests after the move.
