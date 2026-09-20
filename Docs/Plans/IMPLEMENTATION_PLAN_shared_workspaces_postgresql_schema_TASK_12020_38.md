# Shared Workspaces PostgreSQL Schema Parity Implementation Plan

**Task:** TASK-12020.38 / GitHub #2736

**Goal:** Make a clean PostgreSQL AuthNZ deployment initialize and use the canonical sharing repository with the same active workspace, token, audit, and configuration contract as SQLite.

**Architecture:** Add one idempotent PostgreSQL DDL list and one fail-closed ensure helper in `pg_migrations_extra.py`. Invoke it from the existing PostgreSQL bootstrap after core user/org/team tables exist. Reuse one catalog-contract query from startup and `SharedWorkspaceRepo` so stale schemas fail with actionable issues. Reuse the existing repository and services; add no new migration framework or database abstraction.

## Stage 1: Specify PostgreSQL DDL

**Goal:** Capture the current sharing schema contract as focused failing unit tests.

**Success Criteria:** Tests require all four tables, active indexes and uniqueness/check constraints, native PostgreSQL booleans/timestamps, and the current `prototype_workspace` token resource type. A backend execution error returns `False`; a non-PostgreSQL pool is skipped.

**Tests:**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_sharing.py -q
```

**Status:** Complete

## Stage 2: Implement And Wire Bootstrap

**Goal:** Add the smallest idempotent PostgreSQL ensure path and make bootstrap failure explicit.

**Success Criteria:** `ensure_sharing_tables_pg()` executes the DDL on PostgreSQL, is safe to call repeatedly, and returns `False` on backend errors. `setup_database()` calls it after core tables and fails bootstrap when it cannot ensure the sharing schema.

**Tests:** Re-run Stage 1 plus the focused initializer integration assertion.

**Status:** Complete

## Stage 3: Verify The Real Repository

**Goal:** Prove sharing behavior against a clean real PostgreSQL database rather than SQL-string inspection alone.

**Success Criteria:** Backend-aware `ensure_tables()` succeeds; share create/list/update/revoke, token create/read/claim/release/revoke, audit/config operations, and a second ensure preserve existing rows. Missing tables still fail closed with an actionable error.

**Tests:**

```bash
source .venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py \
  --strict-markers -q
```

**Status:** Complete

## Stage 4: Quality And Closeout

**Goal:** Complete scoped static, security, documentation, and task verification.

**Success Criteria:** Focused unit/integration tests pass with no skip, Ruff and Bandit pass on touched Python, `git diff --check` passes, and TASK-12020.38 records exact evidence and residual risks.

**Tests:**

```bash
source .venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/api/v1/endpoints/sharing.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_sharing.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py \
  tldw_Server_API/tests/Sharing/test_sharing_endpoints.py
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/sharing.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py \
  -f json -o /tmp/bandit_task_12020_38.json
git diff --check
```

**Status:** Complete

## Review Remediation

The first real-PostgreSQL pass exposed and fixed audit-filter typing and global-config null uniqueness. Review then identified and resolved the following blockers before closeout:

- Normalize native PostgreSQL timestamps to the existing string API contract.
- Upgrade and validate legacy PostgreSQL sharing schemas without losing sentinel rows.
- Convert raw asyncpg DDL failures into the ensure helper's fail-closed `False` result.
- Map PostgreSQL duplicate-share violations to the existing conflict response.
- Prevent nested sharing endpoint test patches from leaking across randomized test order.
- Run the required sharing ensure from normal FastAPI PostgreSQL startup and abort startup when the canonical contract cannot be established.
- Limit automatic token-check repair to the exact known legacy expression; preserve and report compound or weakened drift.
- Match only the canonical SQLite scope uniqueness failure when mapping duplicate shares to `409`.

Final evidence:

- Focused startup and sharing unit tests: `19 passed`.
- Real PostgreSQL sharing integration tests: `5 passed` with the required fixture, including the normal FastAPI startup ensure path from an empty sharing schema.
- Sharing plus startup-auth regression: `169 passed` across four randomized workers.
- Ruff: all touched Python files passed.
- Bandit: `0` findings and `0` scan errors across touched production files.
- Independent remediation re-review: no remaining actionable issue or blocker.
- `git diff --check`: passed before closeout and rerun after documentation updates.

The live server reached healthy startup on PostgreSQL 18 in three isolated attempts. The authenticated probe did not complete because the validation harness successively hit CSRF protection, the admin seed helper's 422 response, and reserved-domain email validation. Per the three-attempt rule, the corrected complete live persona run is deferred to TASK-12020.37 after these review remediations and TASK-12020.39-.41 land.
