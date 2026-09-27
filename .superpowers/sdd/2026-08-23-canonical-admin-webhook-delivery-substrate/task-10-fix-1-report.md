# Task 10 Fix Round 1 Implementer Report

## Coordinates

- Branch: `codex/admin-webhooks-delivery-substrate`
- Fix base: `d083aaca14dfb4c3c876070f9130b0df23a33d09`
- Commit message: `fix(admin-webhooks): harden delivery API contracts`

## RED Evidence

Tests for all seven accepted defects were added before production edits. The
authoritative RED command used the brief's exact PostgreSQL environment:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 TEST_DB_HOST=127.0.0.1 \
TEST_DB_PORT=5432 TEST_DB_NAME=tldw_test TEST_DB_USER=tldw_user \
TEST_DB_PASSWORD='TestPassword123!' RUN_JOBS=1 PYTHONPATH=. \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short -p no:cacheprovider --show-capture=no \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  -k 'history_loads_only_public_columns or history_uses_one_consistent_snapshot or redelivery_key_family_conflict_contract or redelivery_exact_replay_row_contract or history_service_localizes_repository_not_found_mapping or delivery_openapi_declares_exact_mutation_and_success_headers or unexpected_route_errors_reach_the_global_handler or missing_delivery_history_is_404 or mutation_if_match_omission'
```

Result: `11 failed, 2 passed, 97 deselected, 0 skipped, 23 warnings`. The
expected failures covered cross-source key reuse on both backends, malformed
redelivery replay rows on both backends, forbidden history loads on both
backends, inconsistent history snapshots on both backends, shared rather than
local repository NOT_FOUND mapping, inaccurate OpenAPI headers, and route-local
capture of unexpected exceptions. The two passing assertions proved the
existing route-level 404/audit envelope and runtime omitted-`If-Match` 428 path;
their paired service/OpenAPI assertions failed as expected.

A test-fixture-only correction then assigned the second cross-source operation
a distinct deterministic UUID so the intended idempotency conflict, rather than
the unrelated delivery primary-key constraint, was the observed RED. No
production file had been edited.

## Implementation

- Redelivery lookup digests now use an actor/operation/webhook key family while
  persisted scope and fingerprint retain the full source delivery and route.
- Redelivery replay decoding accepts only the exact in-progress or completed
  202 row shape and rejects malformed coordinates before mutable registration
  or key reads.
- History SQL and mappers use explicit public-only delivery/attempt allowlists,
  and all count/page/attempt reads share a repeatable PostgreSQL or deferred
  SQLite snapshot.
- OpenAPI now declares exact mutation request constraints and success headers
  while preserving the service-owned runtime 428 response.
- Unexpected exceptions again reach the global sanitized 500 handler.
  Repository NOT_FOUND is mapped only by delivery history, preserving its 404
  and denied best-effort read audit without changing unrelated operations.

## GREEN Evidence

The same focused selector passed: `13 passed, 97 deselected, 0 skipped, 23
warnings in 10.94s`. Additional focused contracts passed as follows:

- Redelivery dual-backend contracts: `4 passed, 0 skipped, 6 warnings`.
- Public history/snapshot dual-backend contracts: `6 passed, 0 skipped, 8 warnings`.
- HTTP/global-handler/OpenAPI boundary contracts: `5 passed, 0 skipped, 6 warnings`.

The complete required command used the same exact PostgreSQL environment and
ran both repository wrappers plus audit, synchronous test, Task 10 service,
route, and OpenAPI suites. Result: `138 passed, 0 skipped, 222 warnings in
46.63s`. A restricted-sandbox attempt was unable to open `127.0.0.1:5432` and
was interrupted after the fixture reported PostgreSQL unavailable; it is not
product evidence. The superseding host-loopback run above passed with every
PostgreSQL test collected and zero skips.

- Task 9 dual-backend regressions: `18 passed, 44 deselected, 0 skipped, 21
  warnings in 17.90s`.
- Event expansion regressions: `24 passed, 0 skipped, 4 warnings in 7.28s`.
- Warning-enabled route/OpenAPI review: `46 passed, 0 skipped, 6 warnings in
  8.07s`.
- OpenAPI fingerprint refresh and drift check passed. Paths remain `2,043`,
  schemas remain `3,050`, and SHA-256 is
  `f8e39a35a9837a8fcad6cd638483317e361a3085a0b74b8551a2f8b0d2e3214e`.
- Ruff passed the six Task 10 production modules and all five changed Python
  test files. Python 3.10 compilation passed all six production modules.
- Raw Bandit reviewed 9,989 lines and reported `43` findings, `0 High`: three
  B105 Low/Medium constant-name false positives and 40 B608 Medium/Low fixed-SQL
  fragment reports. The new explicit history query contains only fixed
  allowlisted columns and bound values. The follow-up excluding the two reviewed
  categories passed with zero findings.
- Direct Jobs/runtime, legacy-service import, migration/schema, direct
  route/service SQL, sensitive history-load, public repr, and changed-file
  scope scans passed. History query/mapping tests additionally prove hidden
  Jobs, lease, claim, disposition, test-token, idempotency, and protected fields
  are neither selected nor read and cannot corrupt public history.
- `git diff --check` passed.

Warnings are inherited startup/test-framework provenance: SWIG deprecation,
legacy test API-key and isolated database fallback notices, and existing
Pydantic/FastAPI warnings during full application/OpenAPI import. No warning
originates from a Task 10 changed module or causes a skip.

## Changed Files

- `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`
- `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py`
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`
- `Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md`
- `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md`
- This report.

No load-bearing file outside the fix brief's expected production/test surface
was required. The generated fingerprint, controller-owned plan/Backlog records,
and required report are evidence artifacts, not expanded product scope.

## Residual Risk

The repository's existing low-confidence B608 baseline remains reviewed rather
than mass-suppressed. Snapshot consistency is proven against deterministic
concurrent commits on both supported backends, but backend lock/snapshot
behavior still depends on the deployed SQLite and PostgreSQL transaction
implementations. Runtime delivery activation and actual Jobs admission remain
intentionally deferred to Task 11.
