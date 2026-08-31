# Admin Webhooks PR 2 Verification

## Verification Identity

- Branch: `codex/admin-webhooks-delivery-substrate`
- Literal scope base: `52774a0453b24123cd4cfb3b2a1a38ebc2496f3e`
- Original definitive source head:
  `a5ec2cfb9d7553cf81f848982078ca7f54588b22`
- Original definitive Step 1 source commit:
  `a5ec2cfb9d7553cf81f848982078ca7f54588b22`
- Post-review correction base head:
  `aa2ede91d83aa7309e6a65f9821d9c68f93f5529`
- Post-review correction state: reviewed, uncommitted working tree based on the
  correction base head above
- Observed `origin/dev` metadata, without fetch:
  `54448ef08970e4a348478bdf47be5715c875241c`
- Merge base of the literal scope base and tested head:
  `52774a0453b24123cd4cfb3b2a1a38ebc2496f3e`
- Original verification date: `2026-08-30`
- Post-review correction verification date: `2026-08-31`
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
reviews. The original definitive gates ran at that commit.

Subsequent review found delivery-integrity and target-parsing gaps. The
post-review correction tree now rejects insufficient public lease-horizon
guarantees, validates persisted event bodies before automatic and manual
delivery, preserves each stored event's API version during validation and
replay, and shares one strict target parser between registration and execution.
The parser rejects explicit empty ports, all fragment syntax, malformed or
ambiguous percent escapes, encoded controls/backslashes, and leading `//`
paths. Focused RED/GREEN tests cover every correction.

The first complete post-review Step 1 run was intentionally not accepted as
evidence: it reported `1,526 passed, 16 failed, 2,760 warnings in 991.38s`.
All failures came from the four-backend recovery matrix seeding a persistence-
only placeholder body that the new delivery-boundary validator correctly
rejected. The fixture was changed to seed the canonical event body; the 20
affected cross-backend variants then passed (`56 deselected`, `42 warnings in
227.83s`). The three complete gates below were rerun only after that correction.
Two independent focused re-reviews then reported zero Critical, Important, or
Minor findings.

A later whole-correction-tree review found three Important defects and one
Minor evidence gap. Historical source-command replay still compared a stored
event to the current global API version; registration admitted invalid IDNA
A-labels and legacy numeric host spellings rejected by the HTTP-hop contract;
raw and percent-encoded UTF-8 C1 controls remained deliverable; and the Python
3.10 command did not enumerate the uncommitted correction tree. Strict RED
reproduced one SQLite historical replay failure, six registration failures, and
three executor successes that should have failed. The minimal corrections
removed the command-unowned version comparison, added canonical IDNA round-trip
and legacy-numeric host checks, and validated decoded UTF-8 controls. Focused
GREEN passed 36 parser/executor cases and both SQLite/PostgreSQL historical
source-replay nodes. The complete impacted union then passed 488 tests with zero
skips and 978 warnings in 764.11s (`0:12:44`).
Final independent closure review confirmed all three behavioral findings closed
with zero Critical and zero Important findings. Its one Minor finding was the
missing Round 3 completion entry in `TASK-13111`; the append-only Backlog note
now records the final results and resolves that chronology gap.
After the aggregate gates, Ruff required only an explicit `from None` exception
chain on the already covered invalid-IDNA mismatch branch. The 25-case invalid-
target regression and full static gates passed after that non-control-flow edit.

The observed `origin/dev` metadata later advanced by six commits to `54448ef089`
while this verification remained pinned to `52774a0453`. Those commits are not
part of this evidence range. No fetch, rebase, push, or merge occurred after the
base was pinned; integration with the newer upstream state requires a separate
review decision.

## Result Summary

| Gate | Result |
| --- | --- |
| Complete SQLite/API/security matrix | PASS: 1,553 passed, 0 skipped, 2,782 warnings |
| Required PostgreSQL/four-backend matrix | PASS: 329 passed, 0 skipped, 660 warnings |
| Deterministic protocol/security matrix | PASS: 433 passed, 0 skipped, 613 warnings |
| Ruff | PASS: `All checks passed!` |
| Python 3.10 compatibility compile | PASS: original branch set and all 15 post-review changed/new Python files |
| Committed and worktree diff checks | PASS |
| Sensitive logger/metric scan | PASS: no matches |
| Bandit | REVIEWED PASS: raw exit 1, 61 accounted findings, 0 High |
| PR 3 exclusion and legacy isolation scans | PASS: no matches |
| OpenAPI fingerprint and drift | PASS |
| Default-off/no-release gate | PASS: delivery remains disabled by default |

The three pytest gates executed 2,315 test instances, emitted 4,055 warning
instances, and took 2,311.64 seconds (`38:31.64`) in total. Steps 2 and 3
deliberately repeat high-risk tests from the complete Step 1 union, so 2,315 is
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

Result: exit 0, `1,553 passed, 2,782 warnings in 1007.41s (0:16:47)`,
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

Result: exit 0, `329 passed, 660 warnings in 812.25s (0:13:32)`, zero
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

Result: exit 0, `433 passed, 613 warnings in 491.98s (0:08:11)`, zero
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

The original committed range compiled with Python 3.10.20 while bytecode was
redirected outside the worktree:

```bash
git diff --name-only -z 52774a0453b24123cd4cfb3b2a1a38ebc2496f3e HEAD -- '*.py' | \
  env PYTHONPYCACHEPREFIX=/tmp/admin-webhooks-py310-cache xargs -0 \
  /Users/macbook-dev/.local/bin/python3.10 -m py_compile
```

The post-review correction was still uncommitted, so the final-tree compile
used an explicit list that includes every changed and new Python file:

```bash
env PYTHONPYCACHEPREFIX=/tmp/admin-webhooks-review-py310-cache \
  /Users/macbook-dev/.local/bin/python3.10 -m py_compile \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/Admin_Webhooks/executor.py \
  tldw_Server_API/app/core/Admin_Webhooks/target.py \
  tldw_Server_API/app/core/Admin_Webhooks/worker.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py
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
| B110/B112, Low/High-confidence | 14 | `control_plane.py`: 623, 633, 650, 1543, 1562; `delivery.py`: 1129, 1825; `observability.py`: 140, 626; `reconciler.py`: 585, 917, 956; `worker.py`: 216, 240 | Intentional fail-open metric registration/emission and status-probe observers. Each is downstream of durable truth or preserves a fail-closed unavailable probe; metrics cannot change commits, recovery, delivery, or API outcomes. |
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
- the first complete post-review aggregate run (`1,526 passed`, 16 failed)
  that exposed the recovery matrix's noncanonical persistence-only fixture;
- the first PostgreSQL source-replay probe whose fixture setup could not access
  local PostgreSQL inside the restricted sandbox; the authorized host-loopback
  rerun passed and is the counted result;
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
- The original test-only `a5ec2cfb9d` correction remains historical. The later
  post-review production corrections are limited to lease guarantees, stored-
  body validation/version preservation, and shared target parsing; all were
  independently re-reviewed before this record was finalized.
- The statements above preserve the original pinned-base verification record.
  The later user-approved integration rebase is recorded separately below; no
  merge or production activation occurred.

## Post-Rebase Integration Verification

This addendum records the approved integration pass performed on `2026-08-31`.
It supplements rather than rewrites the original pinned-base evidence above.

### Integration identity and overlap review

- Fetched integration base: `3eb568b478a637adc2482e101cd1379b4a19f48a`
- Pre-rebase verified branch head: `374c77537fd28bb0ba0b9779a13750daeb2f0c1c`
- Tested rebased source head: `2b728686ff49cae76445d3a5bad97caf87f880b9`
- Replayed commits: 52
- Merge base after rebase: exact integration base above

The rebase stopped only for three successive revisions of the generated
`apps/tldw-frontend/lib/api/openapi.fingerprint.json`. Each conflict represented
concurrent OpenAPI contract growth and was resolved by running the authoritative
`make openapi-fingerprint` exporter against the combined tree. No application,
migration, repository, worker, security, or test implementation was resolved by
choosing one side wholesale.

The old merge-base audit found only two paths changed by both upstream and the
branch: the generated fingerprint and
`tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py`. The latter
merged without conflict and retains both the upstream
`tldw_Server_API.tests._plugins.authnz_full_fixtures` fixture path and all PR 2
migration coverage. The complete PostgreSQL migration module passed in both
aggregate gates that include it.

### Post-rebase results

| Gate | Result |
| --- | --- |
| Complete SQLite/API/security matrix | PASS: 1,553 passed, 0 skipped, 2,782 warnings in 1,001.83s |
| Required PostgreSQL/four-backend matrix | PASS: 329 passed, 0 skipped, 660 warnings in 791.21s |
| Deterministic protocol/security matrix | PASS: 433 passed, 0 skipped, 613 warnings in 484.50s |
| Direct-marker audits | PASS: 2 passed |
| Ruff | PASS: `All checks passed!` |
| Python 3.10 compatibility compile | PASS: every Python path changed from the integration base |
| Committed and worktree diff checks | PASS |
| Sensitive logger/metric scan | PASS: no matches |
| Bandit | REVIEWED PASS: raw exit 1, 61 accounted findings, 0 High |
| PR 3 exclusion and legacy isolation scans | PASS: no matches |
| OpenAPI fingerprint generation and drift | PASS |
| Default-off/no-release gate | PASS: delivery remains disabled by default |

The three aggregate commands executed 2,315 test instances with zero skips,
emitted 4,055 warning instances, and took 2,277.54 seconds (`37:57.54`) in
total. They used the same required PostgreSQL, Jobs, seed, timeout, and visible
skip-reporting controls documented in the exact command blocks above. No failed,
partial, interrupted, or sandbox-restricted run is included in these totals.

The scoped Bandit command again returned 17 Low, 44 Medium, and 0 High findings:
3 B105 enum/schema false positives, 12 B110 and 2 B112 intentional fail-open
observers, and 44 B608 fixed-query findings. The complete Bandit-scanned
production path set is byte-identical between the pre-rebase verified head and
the tested rebased source head, so the existing per-finding classification
remains applicable without a new suppression.

The combined OpenAPI fingerprint is:

```text
path_count:   2048 -> 2051
schema_count: 3052 -> 3067
sha256:       b2bb5273d1eda95a44866f58bc19c309aa2f163c0dfaa30b7e6c92a0bcbb1029 -> 0f00a5210305d35df7b5638f4b15cd6ad5e67b0a9175daf3ac6f30e1585f15fa
```

The authoritative fingerprint regeneration left the worktree copy unchanged,
and the drift check passed. `TLDW_ADMIN_WEBHOOKS_MODE` still defaults to `off`.
No durable user/incident producer, admin UI, final route activation, push,
force-push, PR merge, or production activation occurred during this integration
verification.

### Post-rebase independent review

A fresh read-only review inspected the complete integration-base-to-tested-head
diff, the design, implementation plan, Backlog task, runbook, and this
uncommitted integration addendum. It reported `0 Critical`, `0 Important`, and
`0 Minor` findings and returned `Ready for push/PR preparation: Yes`.

The reviewer retained the intended residual-risk record: delivery is
at-least-once; cross-database convergence uses durable fencing and reconciliation
rather than a distributed transaction; crash tests use deterministic fault
injection rather than arbitrary process suspension; and controlled resolver and
loopback tests cannot reproduce every production DNS, NAT, proxy, or certificate
edge. The reviewer did not rerun tests; the controller-owned post-rebase commands
and terminal results above remain the authoritative verification evidence.

## Records

- Backlog: [`TASK-13111`](../../backlog/tasks/task-13111%20-%20Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md)
- Design: [Canonical admin outgoing webhooks](../Design/2026-07-12-canonical-admin-outgoing-webhooks.md)
- Implementation plan: [Canonical delivery substrate plan](../superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md)
- Operations: [Admin Webhooks delivery runbook](../Admin_Webhooks_Delivery_Runbook.md)
- PR 1 dependency: [#2828](https://github.com/rmusser01/tldw_server/pull/2828)
- PR 2: created only after the independent Task 13 review; append its URL to
  `TASK-13111` when opened.

## Conclusion

The corrected canonical admin-webhook delivery substrate is rebased onto the
fetched integration base and has completed the full post-rebase verification
matrix with zero skips plus a clean independent diff review. It is ready for
user-approved push/PR preparation. This evidence does not authorize merge,
production activation, durable event producers, or PR 3 UI work.
