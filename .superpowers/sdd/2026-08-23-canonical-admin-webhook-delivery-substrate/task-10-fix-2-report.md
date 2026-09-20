# Task 10 Fix Round 2 Implementer Report

## Coordinates

- Branch: `codex/admin-webhooks-delivery-substrate`
- Fix base: `081470b7e295ec2d9bd474a24f77bb87db0ba877`
- Scope: remaining single-item sanitized-history leak and unused public attempt
  projection only

## Finding Verification

The re-review finding was valid. `AdminWebhookRepository.get_delivery_history_item()`
opened a plain read connection, and its UoW method selected `delivery.*`, mapped
through `_stored_delivery_from_row()`, then delegated attempts to
`list_delivery_attempts()`. Exact redelivery replay calls this path for both the
created and source deliveries. Hidden enqueue/test coordinates could therefore
invalidate otherwise valid public history and exact replay. The public attempt
column list also selected unused `attempt.created_at`.

## RED Evidence

Before production edits, three shared deterministic contracts were added to
both the SQLite and required-PostgreSQL wrappers:

- single-item SQL contains only public columns, never invokes internal delivery
  or attempt mappers, remains two bounded queries, and omits attempt `created_at`;
- exact replay remains successful after independently corrupting hidden attempt
  and delivery execution coordinates;
- a concurrent attempt commit between the delivery and attempt statements does
  not enter the returned old snapshot.

The RED command was:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 TEST_DB_HOST=127.0.0.1 TEST_DB_PORT=5432 \
TEST_DB_NAME=tldw_test TEST_DB_USER=tldw_user \
TEST_DB_PASSWORD='TestPassword123!' RUN_JOBS=1 PYTHONPATH=. \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short -p no:cacheprovider --show-capture=no \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  -k 'single_history_item or redelivery_replay_ignores_hidden_history_coordinates'
```

Result: `6 failed, 60 deselected, 0 skipped, 8 warnings in 12.12s`.
Both backends failed all three intended contracts: internal mapper calls were
observed, malformed hidden attempt coordinates produced
`admin_webhook_delivery_unavailable`, and the concurrent attempt appeared in
the single-item result.

## Implementation

- `get_delivery_history_item()` now uses `_read_snapshot()`, giving SQLite a
  deferred read transaction and PostgreSQL a read-only repeatable-read
  transaction across both statements.
- The delivery query selects `_HISTORY_DELIVERY_COLUMNS` plus the bounded event
  type and maps only through `_history_delivery_from_row()`.
- The attempt query selects `_HISTORY_ATTEMPT_COLUMNS`, preserves ownership and
  ascending attempt ordering, and maps only through
  `_history_attempt_from_row()`.
- `attempt.created_at` was removed from `_HISTORY_ATTEMPT_COLUMNS`.

The path remains two bounded queries. Missing/foreign ownership still returns
`None`; exact redelivery replay semantics and audit cardinality are unchanged.

## GREEN Evidence

The same focused selector passed after the production fix and again after final
test import cleanup: `6 passed, 60 deselected, 0 skipped, 8 warnings in 10.60s`.

The exact complete brief gate was rerun with the required PostgreSQL environment:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 TEST_DB_HOST=127.0.0.1 TEST_DB_PORT=5432 \
TEST_DB_NAME=tldw_test TEST_DB_USER=tldw_user \
TEST_DB_PASSWORD='TestPassword123!' RUN_JOBS=1 PYTHONPATH=. \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py
```

Final result: `144 passed, 0 skipped, 228 warnings in 51.32s`. Every required
PostgreSQL test was collected and executed.

- Ruff `--no-cache` passed the changed production module and three changed test
  files.
- Python 3.10 compilation passed the changed production module.
- Raw Bandit over the repository module reported 42 Medium/Low B608 findings,
  zero High and no other category. The two changed query sites interpolate only
  fixed module-owned public column allowlists and bind all caller values. The
  follow-up excluding reviewed B608 exited zero with no findings.
- Query-shape/no-leak inspection confirmed the single-item method contains no
  wildcard, internal mapper, internal attempt-list delegation, hidden field, or
  attempt `created_at` projection. Base-to-head sensitive-addition scan found
  zero hidden-coordinate matches.
- Base-to-head self-review found no ownership/not-found, ordering, replay,
  audit, backend-parity, or scope regression.
- `git diff --check` passed.

The 228 warnings have inherited startup/test-framework provenance: SWIG
deprecation, legacy test API-key and isolated database fallback notices, and
existing Pydantic/FastAPI import warnings. No warning caused a skip or points to
the changed repository path.

## Changed Files

- `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- `Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md`
- `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md`
- This report.

The Backlog file included the controller's pre-existing re-review ruling and was
updated through the Backlog CLI after verification. No load-bearing file
outside the binding fix surface was required.

## Residual Risk

No Task 10-specific concern remains from this fix round. The repository's
existing low-confidence B608 baseline remains reviewed rather than suppressed.
Task 11 runtime activation and Jobs admission remain intentionally untouched.
