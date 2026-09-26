# Scoped PostgreSQL Email Search Benchmark

Tracking: TASK-13369 (backend support); TASK-13376.4 (full-scale closeout).
Updated: 2026-09-26. Owner: project owner / maintainer (single owner).

The existing `Helper_Scripts/benchmarks/email_search_bench.py` writes a
synthetic Media/email graph and times normalized operator searches. Its
factory already supports PostgreSQL through content-backend configuration,
but the CLI assumed SQLite and had no request scope. PostgreSQL needs an
explicit user scope because Media joins are protected by forced RLS.

The benchmark adds `--backend sqlite|postgresql` (default `sqlite`) as an
expected-backend check, not a connection-string argument. PostgreSQL uses
the existing `TLDW_CONTENT_DB_BACKEND` and `TLDW_CONTENT_PG_*` settings. It
requires a positive `--scope-user-id`, derives the default client and tenant
as that user ID and `user:<id>`, and keeps the standard `scoped_context`
active through fixture creation and both query passes. A conflicting client
ID or actual backend fails before fixture writes. Both fixture and query-only
modes use ordinary Media handle construction, including its schema checks and
migrations; the benchmark does not call schema initialization a second time.

Reports identify the actual backend and user scope. PostgreSQL reports omit
the SQLite path and never serialize a DSN or password. PostgreSQL date
bounds are converted to ISO strings so the dataset profile remains JSON
compatible.

The NFR gate follows `Docs/Product/Email_Search_Benchmark_Protocol.md`: at least
1,000,000 stored messages, all ten populated query classes (`from`, `to`,
`subject`, `label`, `has:attachment`, `before`, `after`, relative age, free-text
with meaningful unary negation and explicit `OR`), and aggregate warm p50 <=
250 ms / p95 <= 900 ms. Per-case latency is diagnostic. The report retains the
stricter `operator_latency_met` diagnostic, but it is not a separate release SLO.
A bounded or incomplete fixture can satisfy latency thresholds while
`nfr_performance_gate_met` remains false.

The default fixture calls Media and email graph APIs directly. The explicit
`--fixture-loader bulk` alternative uses the bounded DB-management fixture
abstraction and requires an empty disposable synthetic target. PostgreSQL bulk
targets must match `email_content_<10hex>`, have an unused Media sequence and
use `--tenant-id email-benchmark:<scope-user-id>` with a matching non-admin
`--scope-user-id`. The operations probe provisions a non-superuser, non-BYPASSRLS
role and verifies forced-RLS isolation. Use the existing content-backend settings;
do not embed credentials in the command or report. The fixture setup rate does not
measure ingestion throughput.

Neither fixture mode measures EML/ZIP/MBOX parsing, HTTP authentication, archive
throughput, multi-worker behavior or production parity. The cold pass creates
fresh MediaDatabase handles; backend pools may reuse physical connections.
PostgreSQL server and host filesystem caches are unchanged, so handle-reopen
diagnostics do not establish cache-cold performance. The operations report must
state these limits and the exact fixture/query protocol.

The 2026-09-25 10,000-message PostgreSQL report is historical bounded evidence.
TASK-13376.4 records actual million-message results after SQLite, without
inferring a production rollout or the owner's release approval.
Final fixtures must demonstrate ordinary legacy title/body FTS as well as native
query correctness. The earlier SQLite million-message run is superseded after
TASK-13376.4 corrected canonical Media FTS maintenance in the bulk loader; fresh
final SQLite and PostgreSQL evidence belongs in
`Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`.

The complete SQLite certificate now passes at aggregate warm p50 202.23 ms /
p95 725.74 ms after canonical FTS and planner-statistics maintenance. PostgreSQL's
120-message fixture smoke separately verifies non-superuser/non-BYPASSRLS scope,
forced RLS, full legacy/native/indexed-body parity, cross-user exclusion and owner
scope restoration. Normal append after explicit fixture IDs uses Media/native
IDs 121/121 and preserves a 16,384-byte native subject. These PostgreSQL checks
establish fixture/sequence/subject safety; they do not substitute for the complete
certificate below or sustained HTTP validation.

The first actual PostgreSQL 1,000,000-message diagnostic failed warm aggregate
p50/p95 at 510.89/1,841.72 ms while all fixture parity/index/identity checks and
forced-RLS isolation passed. It used PostgreSQL 18.6 Debian in the local Docker
VM (18 CPUs, 16,746,053,632 bytes RAM) on the Apple M5 Max host (18 logical CPUs,
128 GiB RAM). Guarded synthetic reuse and planner tuning were subsequently validated;
the core closeout report records this failed initial measurement separately from
the complete PostgreSQL certificate.

The `aac02e68e8` implementation preserves the original substring predicates,
binds matching label IDs, uses statement InitPlans for RLS scope expressions,
and adds expression statistics plus a native multicolumn `pg_trgm` GIN index.
These acceleration objects are optional when extension/DDL privileges are absent;
search remains functional, but a performance claim must identify the actual
objects and settings. Forced RLS and non-admin isolation remain required.

The selected probe connection profile is `PGOPTIONS='-c work_mem=64 MB'`.
Record observed `shared_buffers=128 MB`, `hash_mem_multiplier=2` and
`max_parallel_workers_per_gather=2` alongside it; no global service configuration
was changed. A short 30-sample comparison measured p50/p95 191.71/516.82 ms at
64 MB versus 248.01/962.52 ms at 4 MB on the same code. It is diagnostic, not the
final 200-sample certificate. The next full attempt aborted during fixture parity
before timing with SQLSTATE `53100`, as parallel wide body/version joins exceeded
the Docker container's 64 MiB POSIX shared-memory budget. Commit `1fee95629a`
sets `max_parallel_workers_per_gather=0` with `SET LOCAL` only in the verification
transaction, retaining all parity/security checks. Timed searches use the restored
two-worker / 64 MB profile. The second full attempt at this revision passed parity
but failed `label:Inbox` COUNT on iteration 11 with SQLSTATE `53100`. On the same
pooled connection, transaction-local `plan_cache_mode='force_custom_plan'`
completed 23 label searches. A generic-plan transition is inferred from the
iteration boundary; counters showed five custom / zero generic plans at iteration
10, and no post-failure generic counter survived psycopg's rollback cleanup.
Commit `455fad64fb` makes native PostgreSQL search select custom planning
inside its transaction, retaining bound parameters, substring predicates and RLS
and restoring the session mode on success/error. The 22-case regression passes;
no SLO pass or global configuration change is inferred from the bounded diagnostic.

The earlier full certificate at frozen `455fad64fb` passed aggregate warm p50/p95 at
**248.63 / 659.39 ms** over 200 samples (maximum 725.26 ms), with all ten cases
populated and meaningful negation 166,667 -> 165,000. It verifies 1,000,000
native/Media/version/identity/indexed-body rows, 200,000 attachments, 23 labels,
200/500 sender/recipient pools and a 365-day span. Its role remains
non-superuser/non-BYPASSRLS with forced RLS and zero other-user rows; guards are
zero and 21 measured source hashes unchanged. Ten cold-handle diagnostics are
272.64/514.79 ms; pooled connections and caches may remain warm. The stricter
per-operator diagnostic is false and the documented aggregate gate true.
Historical evidence is `Docs/Operations/evidence/email_core_closeout_13376/million_postgres_455fad64fb_historical.json`
and `Docs/Operations/evidence/email_core_closeout_13376/million_postgres_parity_455fad64fb_historical.json`.

This is historical search evidence for the recorded source/environment and is
superseded for current-source acceptance. The earlier HTTP attempt uploaded
1,700 messages over 60.9880 request seconds at 27.87 messages/sec and failed
legacy/native parity (0 versus 1,700); its retained report is explicitly diagnostic.
Legacy parameter/phrase/rank corrections restored exact 1,700/1/1 parity.

The combined legacy, bounded SQL rewrite/placeholder cache and identity changes
are committed as `ae46f643ab`. Its full HTTP certificate passes at 92.06
messages/sec over 60.8291 request seconds, 5,600 messages / 56 batches, complete
parity/retry/RLS/metrics/rollback and zero guards. Evidence is
`Docs/Operations/evidence/email_core_closeout_13376/sustained_postgres_ae46f643ab_historical.json`.
The fresh shared-service search load later aborted with POSIX shared-memory
SQLSTATE 53100 at 64 MiB; statistics/index/visibility checks were present, and
64/32/16/8 MB work_mem all reproduced the same subject COUNT failure.

The final reference service is disposable `tldw_email_closeout_13376_pg`,
loopback 5435 with 256 MiB shared memory, the same PostgreSQL 18.6 image digest and
64 MB connection budget. Shared5434 remains intact. Probe-only configuration at
`880d690a93` validates the explicit port/manifest and includes the provisioner in
23 measured source hashes; 91 helper regressions pass. Production queries/RLS are
unchanged. Fresh search passes warm **220.92/495.49 ms** over 200 samples and cold
handle diagnostics 281.54/502.76 ms, complete 1M parity/shape/isolation and meaningful
negation. Setup275.5685 seconds includes 33.0244 seconds ANALYZE, no reuse/extra
maintenance. Final HTTP independently passes61.71 messages/sec over 61.5734 request
seconds; 3/38 batches below 50 (minimum 38.96) remain diagnostics under the aggregate
gate. Canonical reports are `million_postgres.json`, `million_postgres_parity.json`
and `sustained_postgres.json` in the evidence directory. Both probes have23 stable
hashes/zero guards; final cleanup removes only owned resources and preserves the
shared service. The single owner's release decision remains unrecorded.

Original PostgreSQL loading took 207.49777 seconds including 5.62657 seconds of
`ANALYZE`. Later maintenance took 19.002133 seconds for schema/index/RLS changes
and 34.987543 seconds for `ANALYZE`; the retained
`Docs/Operations/evidence/email_core_closeout_13376/postgres_tuning_maintenance.json`
keeps those actions separate. A reused fixture must retain its original guarded
report, matching private resource manifest, validated parity and maintenance
receipt. New fixtures use the loader's automatic post-load `ANALYZE`.

TASK-13370 also makes PostgreSQL FTS backfills conditional on the stored vector
being distinct from the computed vector. Repeated Media handle bootstrap still
checks schemas and scans vectors, but it does not rewrite already current rows.
