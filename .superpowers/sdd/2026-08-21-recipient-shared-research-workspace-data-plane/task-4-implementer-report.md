# Task 4 Implementer Report

## Status

DONE. Task 4 adds authoritative shared-workspace authorization and reusable bounded Jobs and source-preview read helpers. The parent recipient data-plane task remains in progress for Tasks 5-7.

Starting commit: `ad1fbe49e79424fc2bbcd2353b6c201d7ae37c54`

Task commit: `feat(sharing): authorize canonical shared workspace reads` (this report is included in that commit)

## Implementation

- Added `SharedWorkspaceRepo.get_active_share_for_user` and `list_active_shares_for_user` as the sole authoritative share/membership reads for this path.
- Used parameterized backend-compatible SQL for SQLite and PostgreSQL.
- Required an unrevoked share plus either ownership or active membership in the exact active scope.
- Required team shares to have an active team and an active parent organization; organization shares require an active organization.
- Excluded shares owned by the current user from Shared-with-me listings while allowing owners to open their own active share URL.
- Added `SharedWorkspaceAccessService` with authorization-before-owner-lookup ordering, neutral not-found errors, operational unavailable errors, bounded owner display names, active-workspace validation, and the explicit recipient deny-by-default action projection.
- Extracted bounded workspace source Jobs enrichment without changing the two query families, 500-row limits, deduplication identity, ordering, or optional fail-open behavior.
- Extracted local source-preview projection without changing the local response contract. Added non-negative centered focus support while retaining the 12,000-character and 10-chunk bounds and active-only chunk reads.
- Rewired the local workspace endpoint to the public Jobs and preview helpers.

## TDD RED

Access service RED:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py -q
```

Result: expected collection failure because `shared_workspace_access_service` did not exist.

Repository RED:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py -q --tb=short
```

Result: `31 passed, 3 failed`; failures were the absent authoritative `get_active_share_for_user` and `list_active_shares_for_user` methods.

Helper extraction RED:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py -q --tb=short
```

Result: two expected collection errors because `job_status` and `source_preview` did not exist.

## GREEN Verification

Focused deterministic authorization/repository/helper target:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py -q --tb=short
```

Result: `59 passed, 2 warnings`.

Exact Task 4 matrix:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py -q
```

Result: `59 passed, 6 skipped, 12 warnings in 189.55s`. The six PostgreSQL tests collected and skipped through the standard repository fixture because local PostgreSQL was unavailable.

Focused existing local endpoint regression target:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py -q
```

Result: `22 passed, 4 warnings`.

Final post-format combined verification:

```text
source .venv/bin/activate && TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py -q
```

Result: `81 passed, 6 skipped, 16 warnings in 16.85s`. Only the PostgreSQL fixture cases skipped.

Ruff:

```text
source .venv/bin/activate && python -m ruff check <all changed Python files except workspaces.py>
source .venv/bin/activate && python -m ruff check --ignore BLE001 tldw_Server_API/app/api/v1/endpoints/workspaces.py
```

Result: both commands passed. The endpoint retains four pre-existing whole-file `BLE001` findings unrelated to Task 4; all Task 4-owned code passes Ruff.

Bandit:

```text
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py tldw_Server_API/app/core/Workspaces/job_status.py tldw_Server_API/app/core/Workspaces/source_preview.py -f json
```

Result: zero findings, zero errors, and zero skipped tests.

`git diff --check`: passed.

## PostgreSQL State

PostgreSQL integration coverage was collected but not executed because the standard AuthNZ fixture reported local PostgreSQL unavailable. No alternate database setup was introduced. Deterministic SQLite/repository/service coverage passed, and the PostgreSQL tests exercise the same parameterized repository queries when the fixture is available.

## Files

Production:

- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- `tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py`
- `tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py`
- `tldw_Server_API/app/core/Workspaces/job_status.py`
- `tldw_Server_API/app/core/Workspaces/source_preview.py`

Tests:

- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_job_status.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`

Tracking/reporting:

- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-4-implementer-report.md`

## Self-Review

- Authorization is database-derived and does not read `User.team_ids`, `User.org_ids`, JWT claims, or request membership state.
- Denied, revoked, missing, out-of-scope, inactive-scope, removed-member, suspended-member, deleted-workspace, and archived-workspace paths do not disclose owner data.
- A denied share performs neither owner-user lookup nor owner ChaCha loading.
- Repository/backend failures become `SharedWorkspaceUnavailable` before any owner database is opened; neutral target failures become `SharedWorkspaceNotFound`.
- Owner identity is loaded and sanitized only after authorization; non-printable/blank names fall back to `Workspace owner`, and display names are bounded to 128 characters.
- Owners receive the same recipient action projection when opening their own active share URL.
- Scope fields remain internal to the access context and are not serialized by Task 4.
- Shared-with-me preserves owner omission and deterministic ordering.
- Jobs extraction preserves both query families, 500-row bounds, dedupe identity, and fail-open enrichment.
- Preview extraction preserves local source/status/output behavior, rejects negative focus, and respects unchanged character/chunk bounds.
- Recipient-facing exception messages contain no raw exceptions or database/share/user/workspace identifiers.
- The two unrelated untracked watchlist templates were not touched or staged.

## Concerns

No implementation blocker or unresolved Task 4 concern. Live PostgreSQL execution remains an environment skip and should run in CI or on a host with the standard fixture available.
