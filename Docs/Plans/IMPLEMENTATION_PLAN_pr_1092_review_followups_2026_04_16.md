# PR 1092 Review Follow-Ups Implementation Plan

## Stage 1: Review Triage
**Goal**: Verify every still-open PR #1092 review item against the current branch head and isolate the changes that are still technically required.
**Success Criteria**: Remaining work is reduced to a concrete, code-backed list; already-fixed comments are explicitly ruled out.
**Tests**: `gh pr view 1092 --repo rmusser01/tldw_server --json headRefOid,latestReviews,comments,reviews`
**Status**: Complete

## Stage 2: Shared Content Backend Retirement Safety
**Goal**: Replace immediate close-on-rotation with retire-then-drain behavior so in-flight borrowers can finish on the superseded backend.
**Success Criteria**: Rotating or clearing the shared content backend no longer closes an old pool while live references still exist, and pools still close once the retired backend is no longer referenced.
**Tests**: `python -m pytest tldw_Server_API/tests/DB_Management/test_content_backend_cache.py -k "content_backend or reset_media_runtime_defaults" -v`
**Status**: Complete

## Stage 3: Watchlists Operation-Scoped Backend Pinning
**Goal**: Ensure public Watchlists operations use one resolved backend for the full method call, including multi-query read and write paths.
**Success Criteria**: Mid-operation backend refreshes do not split a single Watchlists API call across old and new backends.
**Tests**: `python -m pytest tldw_Server_API/tests/DB_Management/test_content_backend_cache.py -k "watchlists" -v`
**Status**: Complete

## Stage 4: Verification and Security Gate
**Goal**: Re-run the touched DB-management tests and Bandit on the modified scope before closing out the PR work.
**Success Criteria**: Targeted pytest commands pass; Bandit reports no new findings in touched files.
**Tests**: `python -m pytest tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py -v`
**Tests**: `python -m bandit -r tldw_Server_API/app/core/DB_Management/content_backend.py tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py -f json -o /tmp/bandit_pr1092_review_followups.json`
**Status**: Complete
