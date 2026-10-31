# Email Search Benchmark Protocol (M1-009)

Last Updated: 2026-09-26
Owner: Project owner / maintainer (single owner)
Related PRD: `Docs/Product/Email_Ingestion_Search_PRD.md`
Current core closeout: TASK-13376.4 (SQLite first, then PostgreSQL).

## Purpose

Define a reproducible protocol for Stage-1 email operator search performance claims, including:

1. Hardware/software profile capture.
2. Dataset shape requirements.
3. Fixed query-mix methodology.
4. Warm and cold run measurement method.
5. Report artifact location and required fields.

## Harness and Artifacts

1. Benchmark harness script:
   - `Helper_Scripts/benchmarks/email_search_bench.py`
2. Sample query mix fixture:
   - `Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc`
3. Optional workload trace:
   - User-supplied synthetic JSON array or `{"queries": [...]}` with `query`, `count` and optional `name` entries. No sample workload trace is committed.
4. Recommended report output path:
   - `.benchmarks/email_search_report.json`

## Environment Profile Requirements

Every report must include:

1. OS/platform string.
2. CPU model and logical core count.
3. Python version.
4. Database backend and DB file path.
5. Timestamp (UTC ISO-8601).

The harness writes these fields automatically in `environment` and `benchmark`.

Record available RAM and the code revision alongside the report. PostgreSQL
evidence must identify the scoped disposable database and verify that the role is
neither superuser nor `BYPASSRLS`; do not publish credentials or a DSN.
Capture connection `work_mem`, server `shared_buffers`, `hash_mem_multiplier`
and `max_parallel_workers_per_gather`. TASK-13376's selected PostgreSQL profile
uses `PGOPTIONS='-c work_mem=64 MB'` per connection with observed values 128 MB,
2 and 2 respectively; no global service configuration was changed.

## Dataset Shape Requirements

Target benchmark profile for performance claims in PRD NFR:

1. Message count: 1,000,000 (single tenant) for final NFR signoff.
2. Attachment ratio: record actual ratio in report (recommended representative range 15% to 35%).
3. Label cardinality: record distinct label count.
4. Participant cardinality: use realistic sender/recipient pools (not a single sender).
5. Time span: include min/max `internal_date` in report.

Record sender and recipient pool sizes, actual native/legacy row counts and
attachment count. A matching message target argument is not proof that the
dataset contains that many messages.

For developer iteration, smaller fixtures are acceptable (for example 10k to 100k), but must not be used as final NFR evidence.

## Query Mix Requirements

Use at least one query for each operator class:

1. `from:`
2. `to:`
3. `subject:`
4. `label:`
5. `has:attachment`
6. `before:`
7. `after:`
8. `newer_than:` or `older_than:`
9. free-text with unary negation (`-label:...` or similar)
10. explicit `OR`

Every required case must return at least one matching message. The negation case
must remove matching records and retain some matches; a predicate that excludes
no relevant records or returns only an empty result is not representative evidence.
Retain the exact queries and result counts in the report.

After the timed cold and warm passes, the harness compares the uniquely named
`mixed_text_and_negation` case with the same query after removing its negated
terms. It preserves quoted terms and OR branches; a branch containing only
negated terms makes the positive baseline the unrestricted mailbox. The case
must contain positive free text and retain fewer matches than its baseline.
These extra reads do not enter latency samples or warm up the measured passes.
The `negation_validation` report block retains the original query, positive
baseline and both counts. Missing, ambiguous, empty or inert custom cases have
`meaningful_negation_met=false` and an explanatory `reason`, preventing final
NFR certification while retaining the measured results.

The harness auto-builds a mix from observed dataset values, or accepts a fixed file via `--query-mix-file`.
For M3 index/planner tuning, prefer a trace-derived mix via `--workload-trace-file` and tune with top-N production-like queries.

## Warm and Cold Methodology

Run both paths in a single benchmark execution:

1. Cold pass:
   - Open a fresh MediaDatabase handle for each query.
   - Execute each query once.
   - Report aggregate p50/p95 over cold samples.
   - This opens a fresh MediaDatabase handle per query; backend pools can reuse physical connections. Host filesystem and PostgreSQL server caches are not flushed, so this diagnostic does not establish cache-cold performance.
2. Warm pass:
   - Keep one MediaDatabase handle for the full pass.
   - Per query: run `warmup_runs` unmeasured calls.
   - Per query: run `runs` measured calls.
   - Report per-query and aggregate p50/p95.

Default harness values:

1. `warmup_runs=3`
2. `runs=20`
3. `limit=50`

## Command Recipes

Activate the project virtual environment and use `PYTHONPATH=$PWD` before running
the harness. Use synthetic disposable targets; fixture writes must not touch an
existing mailbox. Network/model guards and resource cleanup belong to the guarded
operations probes, not the search timing harness alone.

1. Build a small synthetic fixture for developer iteration (not final NFR evidence):

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path .benchmarks/email_search_bench.sqlite \
  --ensure-fixture \
  --fixture-messages 20000 \
  --runs 30 \
  --warmup-runs 5 \
  --out .benchmarks/email_search_report.json
```

2. Build an actual million-message SQLite fixture with the bounded bulk loader:

```bash
source .venv/bin/activate
PYTHONPATH=$PWD python Helper_Scripts/benchmarks/email_search_bench.py \
  --backend sqlite \
  --db-path /path/to/disposable/email-million.sqlite \
  --tenant-id email-benchmark:1 \
  --ensure-fixture \
  --fixture-loader bulk \
  --fixture-messages 1000000 \
  --attachment-ratio 0.2 \
  --label-cardinality 20 \
  --sender-pool 200 \
  --recipient-pool 500 \
  --runs 20 \
  --warmup-runs 3 \
  --out /path/to/evidence/email-million-sqlite.json
```

The bulk loader requires an empty synthetic target and creates legacy/native rows
through the DB-management fixture abstraction. It preserves ordinary FTS hooks
but deliberately bypasses the HTTP ingestion pipeline. Its setup duration is not
evidence for archive parsing or sustained ingestion throughput. PostgreSQL bulk
setup additionally requires a generated disposable `email_content_<10hex>`
database, an unused Media sequence and the matching non-admin user scope; see
`Docs/Design/email-search-postgresql-benchmark.md`.

After its bounded load batches complete, the bulk loader automatically runs
`ANALYZE` for SQLite and PostgreSQL so the measured planner has statistics for the
generated rows. The fresh-fixture command above includes this maintenance through
the loader; record its duration with fixture setup, separately from search samples
and HTTP ingestion throughput. Query-only reuse evidence must identify its
validated fixture and any post-load statistics maintenance and updated source
hashes. TASK-13376.4's final SQLite certificate reuses the original complete
million-message fixture after this maintenance; its complete 200-sample warm pass
meets the aggregate gate at p50 202.23 ms / p95 725.74 ms. The retained certificate
is `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite.json`.
Targeted diagnostics alone do not establish the final aggregate gate.

Validate actual legacy title/body FTS searches as well as native operator results
before accepting the fixture. TASK-13376.4 corrected missing canonical Media FTS
maintenance in the SQLite bulk path; earlier million-message measurements made
with that incomplete fixture are historical and are superseded by fresh final
runs. Native row count and latency alone cannot certify fixture parity.

3. Build a guarded million-message PostgreSQL fixture using the private manifest
from `Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py` on a
dedicated disposable reference service: loopback port 5435, PostgreSQL 18.6 and
256 MiB shared memory. Provision with `EMAIL_PROBE_PG_PORT=5435` and the matching
`EMAIL_PROBE_PG_CONTAINER`; preserve unrelated shared services. The manifest must identify only the provisioner's
disposable databases and role; it must not be published with evidence:

```bash
source .venv/bin/activate
EMAIL_PROBE_PG_PORT=5435 \
EMAIL_PROBE_PG_MANIFEST=/private/path/to/generated-manifest.json \
PGOPTIONS='-c work_mem=64 MB' PYTHONPATH=$PWD \
python Docs/Operations/probes/email_million_search_13376.py \
  --backend postgresql \
  --messages 1000000 \
  --runs 20 \
  --warmup-runs 3 \
  --out /path/to/evidence/email-million-postgresql.json
```

The fresh loader includes post-load `ANALYZE`. To reuse the same guarded synthetic
fixture, retain its original report and matching private manifest and add
`--postgres-existing-report /path/to/original-report.json`. If later statistics,
schema or index maintenance was performed, also add
`--postgres-maintenance-report /path/to/maintenance-receipt.json`; this argument
records completed work and does not execute maintenance. Reuse validates actual
fixture parity and role/RLS scope before timing and records source identity.
The `aac02e68e8` profile retains substring predicates with bound label IDs, RLS
statement InitPlans, expression statistics and a native multicolumn `pg_trgm`
GIN index. Record whether the optional acceleration objects are present.
The short 30-sample 64 MB/4 MB comparison is diagnostic and cannot replace the
required final 200-sample certificate.

Commit `1fee95629a` confines full body/version fixture verification to a serial
transaction with `SET LOCAL max_parallel_workers_per_gather=0`, avoiding the
local Docker container's 64 MiB POSIX shared-memory limit. It preserves the
complete parity/security checks and restores the two-worker / 64 MB settings
before timed search. Record verification settings separately from timed settings;
this does not change global service configuration or the acceptance protocol.

The subsequent repeated `label:Inbox` COUNT failed on iteration 11 despite valid
fixture parity. PostgreSQL native search in `455fad64fb` uses
`SET LOCAL plan_cache_mode='force_custom_plan'` inside its transaction to retain
parameter-specific plans, with the prior session mode restored on success/error.
Predicates, parameter binding, RLS, two-worker budget and 64 MB `work_mem` are
unchanged. A bounded 23-call pooled comparison and 22 passing regressions are
diagnostics; final acceptance still requires the complete guarded protocol.
The generic-plan transition is inferred from the prepared-execution boundary;
rollback cleared counters before a post-failure generic count could be observed.

The earlier complete PostgreSQL certificate at frozen `455fad64fb` passed the same
200-sample protocol at warm p50 **248.63 ms** / p95 **659.39 ms**. It validates
the actual million-message fixture, all ten populated cases, meaningful negation,
forced-RLS isolation and unchanged 21 measured source hashes. Cold-handle
diagnostics are **272.64 / 514.79 ms**, with pools and caches potentially reused.
Per-operator latency remains diagnostic and is false; aggregate acceptance is
true. Retained evidence is
`Docs/Operations/evidence/email_core_closeout_13376/million_postgres_455fad64fb_historical.json` and
`Docs/Operations/evidence/email_core_closeout_13376/million_postgres_parity_455fad64fb_historical.json`.
This search certificate is historical/superseded for current-source acceptance.
Final legacy/SQL-cache/identity corrections at `ae46f643ab` have a passing full
sustained HTTP/local release certificate (92.06 messages/sec over 60.8291 request
seconds), separate from the earlier failed 27.87 messages/sec HTTP/parity run.
The fresh shared-service `ae46f643ab` search run aborted with POSIX shared-memory
SQLSTATE 53100 at 64 MiB and did not establish acceptance. The final dedicated 256 MiB
profile at `880d690a93` preserves production queries/RLS and 64 MB work_mem. Its
fresh 500 batches of 2,000 took 275.5685 seconds including 33.0244 seconds ANALYZE, with
no reuse or extra maintenance. Final warm 200 p50/p95 **220.92/495.49 ms** passes;
ten cold-handle diagnostics are 281.54/502.76 ms. Full shape/parity/isolation,
all ten populated cases, meaningful negation and 23 stable hashes pass.
Canonical evidence is `Docs/Operations/evidence/email_core_closeout_13376/million_postgres.json`
and `Docs/Operations/evidence/email_core_closeout_13376/million_postgres_parity.json`.
Final HTTP independently passes its minute-wide aggregate gate at 61.71 messages/sec
over 61.5734 request seconds. Three of 38 batches are below 50 (minimum 38.96), a
batch diagnostic, not the aggregate criterion. All owned cleanup is verified in
the core closeout record; single-owner approval remains unrecorded.

4. Benchmark an existing populated synthetic tenant (no fixture writes):

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path /path/to/media.db \
  --tenant-id user:1 \
  --runs 20 \
  --warmup-runs 3 \
  --out .benchmarks/email_search_report.json
```

5. Use fixed query mix:

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path .benchmarks/email_search_bench.sqlite \
  --query-mix-file Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc \
  --out .benchmarks/email_search_report.json
```

6. Use workload trace and include SQLite query-plan capture:

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path .benchmarks/email_search_bench.sqlite \
  --workload-trace-file /path/to/synthetic-workload-trace.json \
  --workload-top-n 15 \
  --capture-query-plans \
  --out .benchmarks/email_search_report.json
```

## Pass/Fail Criteria for Stage-1 Target

Warm-pass aggregate targets (from PRD NFR):

1. p50 <= 250 ms
2. p95 <= 900 ms

The final gate requires at least 1,000,000 stored messages, all ten populated
operator classes, meaningful negation and both aggregate warm targets.
Per-operator p50/p95 values remain diagnostics; `operator_latency_met` is a stricter diagnostic field, while
`nfr_performance_gate_met` applies the documented aggregate protocol. Smaller or
incomplete runs cannot certify the final gate even when their latency is low.

Cold pass is recorded for observability/regression tracking and is not the primary SLO gate.

## Reporting Requirements

Any published benchmark claim must include:

1. Full JSON report artifact.
2. Command used.
3. Dataset profile summary.
4. Whether p50 and p95 warm-pass targets were met.
5. Date and commit SHA used for run.
6. Complete ten-class result counts, aggregate gate outcome and per-query diagnostics.
7. Guard observations, scoped isolation checks and cleanup confirmation when using an operations probe.

Record modified-source identity when a measurement uses uncommitted changes.
Historical 10,000-message or five/six-case reports remain bounded measurements;
do not relabel them as final million-message evidence. One owner records the
technical results and separate release sign-off in
`Docs/Operations/Email_Release_Checklist_and_Rollback.md`.

When `--capture-query-plans` is enabled for M3 planner/index work, include the `warm_pass.query_plan_summary` block (captured query count and index-hit coverage) in the published evidence.
