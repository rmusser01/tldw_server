# Scoped PostgreSQL Email Search Performance — 2026-09-25

Tracking: TASK-13369; FTS bootstrap follow-up: TASK-13370.

The repository benchmark ran against a disposable PostgreSQL content database
using synthetic mail only. The role owned that database but was neither a
superuser nor a `BYPASSRLS` role. The benchmark kept user scope 42 active through
fixture creation and queries. A direct check confirmed `media` RLS enabled and
forced: user 42 saw 10,000 rows and search results; user 43 saw zero of either,
including a search explicitly naming user 42's tenant.

## Dataset and protocol

The fixture was built in three steps: 100, 1,000, then 10,000 messages. It had
2,487 attachment rows and 23 distinct labels. Seed 42 resets at each expansion;
the date anchor uses the current clock, so later reproduction is not
byte-identical. The committed JSON preserves exact date bounds, query text,
match counts and measured results.

The client ran on macOS 26.5.2 / arm64 / Python 3.11.13 with 18 logical CPUs.
PostgreSQL used the local `postgres:18` Docker image. Docker reported 18 CPUs
and 16,746,053,632 bytes of memory; the container had no separate CPU or memory
limit configured. This is a local test profile, not a selected production
deployment.

Each query used limit 50 and offset 0, five warmups, then 15 measured runs.
The default ten-query mix includes two zero-result cases. A separate run used
the six required operators, all with positive matches. The cold pass opens a
fresh Media handle per query; it does not flush PostgreSQL or filesystem caches.
Handle construction/schema work is outside the timed search call.

| Workload | Warm samples | Warm p50 / p95 | Cold p50 / p95 |
| --- | ---: | ---: | ---: |
| Default ten-query mix | 150 | 33.39 / 106.86 ms | 38.32 / 114.42 ms |
| Six required operators | 90 | 29.90 / 96.85 ms | 35.41 / 97.85 ms |

| Required operator | Matches | Warm p50 / p95 |
| --- | ---: | ---: |
| `from` | 50 | 13.08 / 16.57 ms |
| `subject` | 1,667 | 29.05 / 37.54 ms |
| `label` | 10,000 | 95.15 / 98.26 ms |
| `has:attachment` | 2,487 | 44.17 / 48.95 ms |
| `after` | 5,744 | 26.72 / 43.69 ms |
| `before` | 4,256 | 24.01 / 40.72 ms |

Artifacts:

- `Docs/Operations/Email_Search_Benchmark_PostgreSQL_10k_2026-09-25.json`
- `Docs/Operations/Email_Search_Benchmark_PostgreSQL_10k_NFR_2026-09-25.json`
- Published fixture/query runner: `Helper_Scripts/benchmarks/email_search_bench.py`
- Backend/scope contract: `Docs/Design/email-search-postgresql-benchmark.md`

## Reproduction

Activate the project virtual environment and configure an isolated PostgreSQL
target through `TLDW_CONTENT_PG_*` settings before running. Keep the role
non-superuser and without `BYPASSRLS`. Connection credentials are not CLI
arguments or report fields.

```bash
export TLDW_CONTENT_DB_BACKEND=postgresql
python Helper_Scripts/benchmarks/email_search_bench.py \
  --backend postgresql --scope-user-id 42 --ensure-fixture \
  --fixture-messages 100 --runs 2 --warmup-runs 1 --out /tmp/email_pg_100.json
python Helper_Scripts/benchmarks/email_search_bench.py \
  --backend postgresql --scope-user-id 42 --ensure-fixture \
  --fixture-messages 1000 --runs 15 --warmup-runs 5 --out /tmp/email_pg_1000.json
python Helper_Scripts/benchmarks/email_search_bench.py \
  --backend postgresql --scope-user-id 42 --ensure-fixture \
  --fixture-messages 10000 --runs 15 --warmup-runs 5 --out /tmp/email_pg_10000.json
jq '[.query_mix[] | select(.name == "from_filter" or .name == "subject_filter" or .name == "label_filter" or .name == "has_attachment" or .name == "after_date" or .name == "before_date")]' \
  /tmp/email_pg_10000.json > /tmp/email_pg_nfr_mix.json
python Helper_Scripts/benchmarks/email_search_bench.py \
  --backend postgresql --scope-user-id 42 --query-mix-file /tmp/email_pg_nfr_mix.json \
  --runs 15 --warmup-runs 5 --out /tmp/email_pg_nfr.json
```

The recorded successful 10k passes were query-only reruns on the preserved
fixture after the storage recovery described below. Both non-loopback socket
guards recorded zero attempts. Gmail and connector workers were disabled. The
fixture calls Media/email graph APIs directly and does not invoke model processing.

## Failures, fixes and remaining limits

The first 10k build completed, adding 9,000 records in 542.25 seconds, but its
query pass failed during FTS bootstrap with a PostgreSQL WAL `fsync` I/O error.
The host had only 119 MiB free, and Docker file reads also failed. That failed
pass contributes no latency measurements to the committed reports.

Code inspection found unconditional FTS vector refreshes on each Media handle
bootstrap. TASK-13370 adds a null-safe difference predicate: missing/stale
vectors are repaired, while current vectors do not generate row writes.
Twenty-four related non-integration tests and two real PostgreSQL tests passed;
the latter verified unchanged row versions on repeated setup and existing FTS
search behavior. This change does not prove that FTS caused the host's disk
exhaustion, and bootstrap still performs schema checks and scans vectors.

After freeing disposable email artifacts and restarting the local Docker engine,
PostgreSQL recovered and both 10k passes and the direct RLS check succeeded.
Host storage became unstable again while saving evidence. Final database/role
cleanup is **not confirmed** because Docker inspection failed. The private
manifest remains at `/tmp/email_pg_manifest_13364.json`; the disposable cleanup
script is `/tmp/email_pg_databases_13364.py`. Restore stable host/Docker storage,
then clean these resources before starting another large fixture.

The benchmark CLI, PostgreSQL date serialization and report-gate regressions
passed eight focused tests including existing helper import checks. Ruff
check/format passed for the benchmark files; Bandit found zero issues in the
benchmark and FTS implementation scopes. Test `assert` findings were treated
as test-only B101 warnings, not production security defects.

Ten thousand messages are only 1% of the specified mailbox size. The new report's
`nfr_performance_gate_met` remains false despite passing these latency thresholds.
This evidence does not measure archive parsing/ingestion throughput, HTTP latency,
multi-worker behavior, production parity, or the 1M-message target. Those release
gates remain open.

## Cleanup follow-up

Later on 2026-09-25, the host reported 237 GiB free. Docker Desktop started
successfully on the same `desktop-linux` context, but the previous test store
was gone: both container and volume inventories were empty, the old
`tldw_postgres_test` container was absent, and port 5434 refused connections.
The disposable databases and role no longer exist in that removed store.
This confirms resource absence, not successful execution of the earlier SQL
cleanup script. The obsolete private credential manifest was removed.
This follow-up did not delete unrelated Docker resources. TASK-13369 is closed;
the benchmark's scale and deployment limitations above remain unchanged.
