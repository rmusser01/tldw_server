## Stage 1: Red Tests
**Goal**: Add focused failing coverage for evaluation dataset list and synthetic queue canonical pagination.
**Success Criteria**: New/updated tests fail because canonical `pagination` and true totals are not yet present.
**Tests**: `python -m pytest tldw_Server_API/tests/Evaluations/test_evaluations_unified.py -k "list_datasets" -q`, `python -m pytest tldw_Server_API/tests/Evaluations/integration/test_synthetic_eval_api.py -k "queue" -q`, `python -m pytest tldw_Server_API/tests/Evaluations/test_synthetic_eval_service.py -k "true_queue_total" -q`
**Status**: Complete

## Stage 2: Narrow Count Seams And Response Wiring
**Goal**: Add explicit count seams for dataset listing and synthetic draft queue, then wire canonical nested pagination into the two list responses.
**Success Criteria**: Route responses preserve legacy fields and add correct `pagination`.
**Tests**: Stage 1 tests plus touched evaluation route/service files.
**Status**: Complete

## Stage 3: Verification And Commit
**Goal**: Run focused/full touched tests, Bandit, and local hygiene, then commit the tranche.
**Success Criteria**: All touched tests green, Bandit clean, worktree clean except intended changes, local commit created.
**Tests**: targeted pytest selections, full touched eval files, `git diff --check`, Bandit on touched scope.
**Status**: Complete
