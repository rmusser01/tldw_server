# Admin Webhooks PR 2 Verification

## Verification Identity

- Branch: `codex/admin-webhooks-delivery-substrate`
- Literal scope base: `52774a0453b24123cd4cfb3b2a1a38ebc2496f3e`
- Tested source head: `a5ec2cfb9d7553cf81f848982078ca7f54588b22`
- Definitive Step 1 source commit:
  `a5ec2cfb9d7553cf81f848982078ca7f54588b22`
- Observed `origin/dev` metadata, without fetch:
  `54448ef08970e4a348478bdf47be5715c875241c`
- Merge base of the literal scope base and tested head:
  `52774a0453b24123cd4cfb3b2a1a38ebc2496f3e`
- Verification date: `2026-08-30`
- Host: macOS 26.5.2, arm64
- Project Python: 3.11.13
- Compatibility Python: 3.10.20
- Pytest: 8.4.1
- PostgreSQL server: 18.6 (`Debian 18.6-1.pgdg13+2`)
- Pytest random seed: `20260829`

The PostgreSQL version was read through the project's `asyncpg` driver with a
read-only `SHOW server_version` query. Connection credentials, DSNs, private
receiver addresses, tokens, and private payloads are intentionally omitted.

The branch was rebased with approval onto the immutable scope base above before
the definitive gates. The first post-rebase Step 1 run collected 1,523 tests and
reported 1,521 passing with two test-contract failures: the exact startup-worker
set omitted two upstream Notes workers, and four delivery-mode guard tests lacked
direct unit markers. Commit `a5ec2cfb9d` made only those test corrections after
focused RED/root-cause confirmation and independent specification and quality
reviews. All definitive gates below ran at that commit.

The observed `origin/dev` metadata later advanced by six commits to `54448ef089`
while this verification remained pinned to `52774a0453`. Those commits are not
part of this evidence range. No fetch, rebase, push, or merge occurred after the
base was pinned; integration with the newer upstream state requires a separate
review decision.

## Result Summary

| Gate | Result |
| --- | --- |
| Complete SQLite/API/security matrix | PASS: 1,523 passed, 0 skipped, 2,722 warnings |
| Required PostgreSQL/four-backend matrix | PASS: 327 passed, 0 skipped, 656 warnings |
| Deterministic protocol/security matrix | PASS: 424 passed, 0 skipped, 613 warnings |
| Ruff | PASS: `All checks passed!` |
| Python 3.10 compatibility compile | PASS: all 77 changed Python files |
| Committed and worktree diff checks | PASS |
| Sensitive logger/metric scan | PASS: no matches |
| Bandit | REVIEWED PASS: raw exit 1, 61 accounted findings, 0 High |
| PR 3 exclusion and legacy isolation scans | PASS: no matches |
| OpenAPI fingerprint and drift | PASS |
| Default-off/no-release gate | PASS: delivery remains disabled by default |

The three pytest gates executed 2,274 test instances, emitted 3,991 warning
instances, and took 2,238.36 seconds (`37:18.36`) in total. Steps 2 and 3
deliberately repeat high-risk tests from the complete Step 1 union, so 2,274 is
an execution count, not a unique-test count. No counted test was skipped.

## Exact Pytest Gates

Every counted command forced required PostgreSQL, enabled Jobs, used the project
interpreter, retained short tracebacks and skip reporting, set a 90-second
per-test timeout, and used seed `20260829`. Localhost PostgreSQL and loopback
transport tests ran with host network permission.

### Step 1: Complete SQLite/API/security matrix

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short -ra --show-capture=no --timeout=90 --randomly-seed=20260829 --cache-clear \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Security/test_egress.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py
```

Result: exit 0, `1,523 passed, 2,722 warnings in 983.54s (0:16:23)`,
zero skips.

### Step 2: Required PostgreSQL and four-backend crash matrix

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short -ra --show-capture=no --timeout=90 --randomly-seed=20260829 \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py
```

Result: exit 0, `327 passed, 656 warnings in 781.82s (0:13:01)`, zero
skips.

### Step 3: Deterministic protocol and security matrix

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short -ra --show-capture=no --timeout=90 --randomly-seed=20260829 \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py
```

Result: exit 0, `424 passed, 613 warnings in 473.00s (0:07:52)`, zero
skips.

## Backend And Crash-Convergence Proof

`test_recovery_backend_matrix.py` parametrizes every integration contract over
all four independent persistence combinations:

| AuthNZ backend | Jobs backend | Result |
| --- | --- | --- |
| SQLite | SQLite | PASS |
| SQLite | PostgreSQL | PASS |
| PostgreSQL | SQLite | PASS |
| PostgreSQL | PostgreSQL | PASS |

The matrix proves the following boundaries on every pair:

| Boundary | Primary proof |
| --- | --- |
| Six enqueue crash points from pre-claim through queued commit | `test_enqueue_six_crash_boundaries_converge_across_backend_matrix` |
| Created-but-unattached Jobs work expires into exact cancellation recovery | `test_before_attach_crash_then_expiry_preserves_exact_cancel_recovery` |
| Terminal work is revalidated before Jobs admission | `test_enqueue_revalidates_terminal_work_before_admission_across_backend_matrix` |
| Orphan prepare/cancel crashes retain exact claim and disposition identity | `test_terminal_orphan_crashes_recover_with_exact_claim_and_disposition` |
| Foreign claim, missing/queued/processing cancel, and pre/post-create expiry | `test_enqueue_foreign_claim_cancellation_and_expiry_matrix` |
| Lost AuthNZ acknowledgement for complete/retry/fail/cancel/defer | `test_authnz_disposition_lost_ack_reconciles_across_backend_matrix` |
| Infrastructure/recovery defers leave historical markers and reacquire correctly | `test_no_ack_defer_marker_is_historical_across_backend_matrix` |
| Exact queued cancel supersedes only the matching historical marker | `test_queued_cancel_replaces_only_an_exact_historical_marker` |
| Six worker crash points across committed receiver outcomes | `test_worker_authnz_outcome_crash_cross_product_across_backend_matrix` |
| Four receiver calls are the hard lifetime cap | `test_worker_hard_cap_is_four_receiver_calls_across_backend_matrix` |
| A late exact-token writer cannot replace committed stale recovery | `test_exact_late_writer_cannot_replace_stale_recovery_across_backend_matrix` |

The worker-specific suite additionally proves reservation before I/O, no
overlapping attempts, no duplicate I/O after crash, lease-horizon enforcement,
lost-lease deferral, stale-attempt `outcome_unknown`, idempotent callback replay,
configuration/disable/rotation races, terminal monotonicity, and no fifth
network attempt.

## Protocol And Security Mapping

| Requirement | Proof |
| --- | --- |
| DNS answers are complete, canonical, public-only, and pinned | `test_http_hop_contract.py` DNS-set tests and `test_http_hop_transport.py` peer tests |
| Private, reserved, mixed, or changed DNS results are rejected | `test_egress.py`, DNS-set rejection, selected-peer equality, and post-TLS peer verification |
| Redirects are never followed | `test_returns_redirect_without_another_connect_or_request` |
| Ambient proxies and client state are ignored | `test_status_only_pins_dns_preserves_host_ignores_proxies_and_does_not_redirect`, `test_ambient_http_client_state_is_ignored` |
| TLS uses the original hostname and verified peer | HTTPS hostname/context, TLS-evidence, and post-TLS peer tests |
| Connect, TLS, read, and whole-hop timeouts fail closed | HTTP-hop transport/streaming timeout tests and executor timeout-bound tests |
| Receiver body is not buffered in status-only mode | `test_status_only_closes_without_reading_any_response_body`, coalesced-body discard, and projection guards |
| Response framing and decompression stay bounded | raw-wire, content-length, chunked, EOF, parser, decoder, and gzip-bomb tests |
| URL, exception detail, receiver content, and wire secrets are redacted | contract error-text, executor exception, repr, and HTTP-core log tests |
| Retry authority is closed and `Retry-After` is bounded | executor status/transport classification and strict status-only `Retry-After` tests |
| Signature and headers use exact raw bytes | `test_published_signature_vector_and_exact_request_headers` |
| Synchronous test is one attempt, replayable, and never enters Jobs | `test_test_delivery.py`, including stale recovery and no-Jobs assertions |

### Published synthetic signature vector

This is the public synthetic vector from the delivery runbook, not private
receiver data:

```text
secret: whsec_1111111111111111111111111111111111111111111111111111111111111111
timestamp: 1787443200
body: {"api_version":"2026-07-01","created_at":"2026-08-23T00:00:00Z","data":{"synthetic":true},"id":"00000000-0000-4000-8000-000000000001","type":"user.created"}
signature: v1=294bc280642cfd89fd011f606fbbe39633a77372db8ae9efd4281b2a3e509811
```

## Static, Sensitive-Data, And Scope Gates

Ruff scanned the canonical package plus every touched shared production
boundary. It returned `All checks passed!` from this exact command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

Every Python file changed from the immutable base also compiled with Python
3.10.20 while bytecode was redirected outside the worktree:

```bash
git diff --name-only -z 52774a0453b24123cd4cfb3b2a1a38ebc2496f3e HEAD -- '*.py' | \
  env PYTHONPYCACHEPREFIX=/tmp/admin-webhooks-py310-cache xargs -0 \
  /Users/macbook-dev/.local/bin/python3.10 -m py_compile
```

Both of these passed:

```bash
git diff --check 52774a0453b24123cd4cfb3b2a1a38ebc2496f3e HEAD
git diff --check HEAD
```

The fail-closed sensitive scan returned the expected clean no-match status:

```bash
rg -n "logger\..*(url|secret|signature|payload|response|ciphertext)|labels=.*(id|host|url|email|secret|payload)" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

The exact raw Bandit command was:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/Jobs/migrations.py \
  tldw_Server_API/app/core/Jobs/pg_migrations.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

The path and import scans found no committed, tracked-worktree, or untracked
admin UI, user/incident producer, legacy admin service, generic Jobs-webhook
service, or legacy import. The exact forbidden path pattern was:

```text
(^|/)(admin-ui|users|incidents|admin_system_ops_service|admin_webhooks_service|jobs_webhooks_service)
```

The legacy import pattern was:

```text
services\.(admin_webhooks_service|jobs_webhooks_service)|from .*admin_webhooks_service|from .*jobs_webhooks_service
```

The exact path/import invocations, run through status-aware helpers that make a
match fail and propagate Git/`rg` errors, were:

```bash
forbidden_path_pattern='(^|/)(admin-ui|users|incidents|admin_system_ops_service|admin_webhooks_service|jobs_webhooks_service)'
require_no_path_matches "$forbidden_path_pattern" git diff --name-only 52774a0453b24123cd4cfb3b2a1a38ebc2496f3e HEAD
require_no_path_matches "$forbidden_path_pattern" git diff --name-only HEAD
require_no_path_matches "$forbidden_path_pattern" git ls-files --others --exclude-standard
require_no_rg_matches -n "services\.(admin_webhooks_service|jobs_webhooks_service)|from .*admin_webhooks_service|from .*jobs_webhooks_service" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

### Bandit classification

The exact planned recursive Bandit path set returned raw status 1. Its summary
was 17 Low, 44 Medium, 0 High. A JSON-format repeat over the identical path set
contained exactly 61 results. Every result is accounted for below; no result
was suppressed or omitted by Task 12.

| Rule | Count | Exact locations | Classification |
| --- | ---: | --- | --- |
| B608, Medium/Low-confidence | 44 | `admin_webhooks_repository.py`: 2789, 2806, 2820, 3003, 3052, 3071, 3109, 3125, 3138, 3164, 3171, 3193, 3214, 3239, 3246, 3283, 3332, 3364, 3435, 3456, 3486, 3521, 3572, 3591, 3729, 3801, 3881, 3937, 3956, 4120, 4141, 4261, 4437, 4538, 4564, 4750, 4797, 4811, 4868, 4965, 5000, 5070, 5255; `Jobs/manager.py`: 9656 | Reviewed fixed SQL. Repository interpolation is limited to module column constants, backend-selected lock/null-safe/due literals, two closed attempt predicates, a fixed terminal SET clause, generated `?` placeholders, and an explicit five-table allowlist. The Jobs query accepts only two private, internally constructed clause shapes: `WHERE id = ANY(%s)` or the prune method's locally assembled placeholder clause. Caller values remain bound parameters. |
| B110/B112, Low/High-confidence | 14 | `control_plane.py`: 623, 633, 650, 1543, 1562; `delivery.py`: 1098, 1800; `observability.py`: 140, 626; `reconciler.py`: 585, 917, 956; `worker.py`: 216, 240 | Intentional fail-open metric registration/emission and status-probe observers. Each is downstream of durable truth or preserves a fail-closed unavailable probe; metrics cannot change commits, recovery, delivery, or API outcomes. |
| B105, Low/Medium-confidence | 3 | `schemas/admin_webhooks.py`: 143, 144; `domain.py`: 161 | False positives on numeric/boolean schema example fields named `secret_version`/`secret_rotation_required` and the closed enum value `canceled_secret_rotation`; none is a password or signing secret. |

The inventory contains the established Task 11 baseline of 43 repository
fixed-query reports, 14 intentional fail-open observer reports, and 3
enum/schema false positives, plus the one reviewed canonical Jobs prune query
added by the final implementation-review correction. Task 12 added no
suppression. Bandit also emitted diagnostics for
rule-specific suppressions already present in the broader previously reviewed
Jobs/SQL baseline; no such suppression changed after the definitive Step 1
source commit.

## OpenAPI Review

Authoritative commands:

```bash
CI_LOCAL_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make openapi-fingerprint
CI_LOCAL_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make openapi-drift-check
git diff 52774a0453b24123cd4cfb3b2a1a38ebc2496f3e HEAD -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
git diff HEAD -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Result: exit 0. The current-worktree diff is empty and drift check reports that
the checked-in snapshot matches. The reviewed committed delta is limited to the
approved test, manual-redelivery, delivery-history, and status contracts:

```text
path_count:   2048 -> 2051
schema_count: 3052 -> 3067
sha256:       7175f7d0e4413843b6f720586e6a6d7526604ad085c3fd6c972b2ef3fd2c0df9 -> dca38a546b4ce102ba701aee2c9fea396a11b2790ef74e52b04758087e4da567
```

The plain host command uses Python 3.9.6 and is non-authoritative. It exits 2
while importing existing `@dataclass(..., slots=True)` code because Python 3.9
does not support that argument. The project-Python commands above are the
authoritative passing result and leave the fingerprint unchanged.

## Warning And Invalid-Run Accounting

Warnings remained visible and unsuppressed. Their established sources are the
repository/dependency baseline: the unknown pytest `plugins` configuration,
Starlette/httpx test-client deprecation, passlib's Python `crypt` deprecation,
shared AuthNZ PostgreSQL fixture shutdown deprecations, optional OpenTelemetry
components, FastAPI's deprecated `example` parameter, and existing Pydantic
field-shadow warnings. Shared PostgreSQL setup/teardown causes the warning count
to scale with parametrized backend tests. No warning was converted into a skip.

The following outputs are not counted:

- restricted-network PostgreSQL probes and any sandbox-denied loopback run;
- interrupted, overlapping, partial, or stale-cache aggregate attempts recorded
  during the direct-marker correction;
- the first post-rebase aggregate run (`1,521 passed`, two failed) that exposed
  the stale startup-worker expectation and missing direct markers;
- the expected pre-fix marker and stale route-selection failures;
- a verifier process terminated by its agent usage limit before it could retain
  terminal output.

Every counted result above is a complete, serial, controller-owned run with a
terminal exit status.

## Scope And Release State

- `TLDW_ADMIN_WEBHOOKS_MODE` still defaults to `off`.
- The canonical runtime starts only when mode is `on` and canonical routing is
  selected; the lifecycle task remains `admin_webhook_delivery_runtime_task`.
- No user/incident producer, admin UI, generic outgoing-webhook UI/service, or
  final release activation is included. Those remain PR 3 scope.
- No production edit was made during final verification. The test-only
  `a5ec2cfb9d` correction was completed and independently reviewed before all
  definitive gates were rerun.
- A user-approved fetch/rebase occurred before the immutable base was pinned.
  No later fetch, rebase, merge, push, force-push, or history rewrite occurred.

## Records

- Backlog: [`TASK-13111`](../../backlog/tasks/task-13111%20-%20Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md)
- Design: [Canonical admin outgoing webhooks](../Design/2026-07-12-canonical-admin-outgoing-webhooks.md)
- Implementation plan: [Canonical delivery substrate plan](../superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md)
- Operations: [Admin Webhooks delivery runbook](../Admin_Webhooks_Delivery_Runbook.md)
- PR 1 dependency: [#2828](https://github.com/rmusser01/tldw_server/pull/2828)
- PR 2: created only after the independent Task 13 review; append its URL to
  `TASK-13111` when opened.

## Conclusion

The canonical admin-webhook delivery substrate is ready for independent review
against the pinned base. Because `origin/dev` subsequently advanced and touches
three PostgreSQL webhook test modules, reconcile and re-verify that newer
upstream state before PR creation. This evidence does not authorize merge,
production activation, durable event producers, or PR 3 UI work.
