# Email Search M3-003 Index and Planner Tuning

Last Updated: 2026-09-26
Owner: Backend and Search Team
Related PRD: `Docs/Product/Email_Ingestion_Search_PRD.md`

## Scope

This note captures the M3-003 search planner/index tuning changes for normalized email search and the benchmark protocol updates used for trace-driven validation.

## Index Changes

Applied in `tldw_Server_API/app/core/DB_Management/Media_DB_v2.py` (`_EMAIL_INDICES_SQL`):

1. `idx_email_messages_tenant_date_id` on `(tenant_id, internal_date DESC, id DESC)`
   - Aligns with default sort path and tenant-scoped paging.
2. `idx_email_messages_tenant_has_attachments_date` on `(tenant_id, has_attachments, internal_date DESC, id DESC)`
   - Improves `has:attachment` + recent-date query paths.
3. `idx_email_message_participants_message_role` on `(email_message_id, role, participant_id)`
   - Reduces participant-role EXISTS lookup cost for `from:/to:/cc:/bcc:` operators.

Validation test coverage:

- `tldw_Server_API/tests/DB_Management/test_email_native_stage1.py::test_email_search_m3_indexes_exist_on_sqlite`

## Workload Trace and Planner Capture

`Helper_Scripts/benchmarks/email_search_bench.py` now supports:

1. `--workload-trace-file` (JSON trace query inputs with counts)
2. `--workload-top-n` / `--workload-min-count` filtering
3. `--capture-query-plans` (SQLite `EXPLAIN QUERY PLAN` capture for each benchmark query)

Provide a synthetic workload trace JSON matching the benchmark protocol:

- `/tmp/synthetic_email_workload_trace.json` (user-supplied; no trace sample is committed)

## Benchmark Command (Trace-Driven)

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path .benchmarks/email_search_bench.sqlite \
  --ensure-fixture \
  --fixture-messages 20000 \
  --workload-trace-file /tmp/synthetic_email_workload_trace.json \
  --workload-top-n 15 \
  --warmup-runs 5 \
  --runs 30 \
  --capture-query-plans \
  --out .benchmarks/email_search_report_m3_003.json
```

## Benchmark Evidence (2026-02-10)

Executed against a 3,000-message fixture tenant with trace-derived top-5 queries and SQLite query-plan capture:

The following command and results are historical. Its referenced trace sample is
not present in the current checkout, so this run is not reproducible from current
files and does not certify the actual million-message, ten-class NFR gate. Current
validation is tracked in TASK-13376.4 and the benchmark protocol.

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path /tmp/email_m3_003_bench.sqlite \
  --tenant-id bench-tenant \
  --workload-trace-file Helper_Scripts/benchmarks/email_search_workload_trace.sample.json \
  --workload-top-n 5 \
  --warmup-runs 1 \
  --runs 5 \
  --capture-query-plans \
  --query-plan-statements-max 4 \
  --out /tmp/email_m3_003_report_v2.json
```

Observed report highlights:

1. Historical bounded warm summary: `p50_ms=9.51`, `p95_ms=10.89`; the 3,000-message/top-five workload does not meet the million-message coverage gate.
2. Planner capture summary:
   - `captured_queries=5`
   - `queries_with_index_hits=5`
   - `total_explained_statements=11`
   - `total_index_hit_rows=22`
3. Captured plans showed new/updated indexes in use, including:
   - `idx_email_messages_tenant_date_id`
   - `idx_email_messages_tenant_has_attachments_date`
   - `idx_email_message_participants_message_role`

## Acceptance Mapping

M3-003 deliverables and acceptance mapping:

1. Index tuning and planner optimization:
   - Implemented via new composite indexes and query-plan capture in the benchmark harness.
2. Documented and benchmarked:
   - This document + updated benchmark protocol/README.
3. NFR target validation:
   - Use report targets from `email_search_bench.py` (`p50 <= 250ms`, `p95 <= 900ms`) on representative corpus before cutover.

## Actual Million-Message Validation (TASK-13376.4)

SQLite was validated first, then PostgreSQL. Both synthetic fixtures contain one million native/Media/version/indexed-body identities, 200,000 attachments, 23 labels, 200 senders and 500 recipients over 365 days. Each certificate measures all ten populated operator classes with 20 calls and three warmups per case at limit 50. Negation reduces 166,667 matches to 165,000. The aggregate warm targets pass:

| Backend | Warm p50 | Warm p95 | Retained certificate |
| --- | --- | --- | --- |
| SQLite | 202.23 ms | 725.74 ms | `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite.json` |
| PostgreSQL | 220.92 ms | 495.49 ms | `Docs/Operations/evidence/email_core_closeout_13376/million_postgres.json` |

The current native indexes live in `media_db/schema/email_schema_structures.py`. SQLite uses covering reverse participant/label links and explicit post-load statistics. PostgreSQL retains exact independent substring predicates, optional expression statistics and multicolumn trigram indexing, statement-scoped RLS setting reads, bound label IDs and transaction-local custom planning. The fresh PostgreSQL certificate at `880d690a93` uses a dedicated disposable reference container with 256 MiB shared memory and a 64 MB connection work-memory budget; full fixture parity inspection alone runs serially, with two parallel workers restored before timing. The earlier 64 MiB shared-service container could not reliably hold the parallel hash/bitmap allocation and remains preserved. Forced RLS and pooled scope checks pass. No global planner settings or search predicates are weakened.

Fresh-handle diagnostics retain existing caches: SQLite p50/p95 197.33/694.77 ms and PostgreSQL 281.54/502.76 ms. They are not cache-cold SLOs. Per-operator latencies remain diagnostics; the documented acceptance gate is the 200-sample aggregate. The final PostgreSQL run contains a fresh 275.568-second load, including 33.024 seconds of statistics maintenance, with 23 unchanged measured file hashes. The earlier `455fad64fb` result is retained separately as historical evidence. Measured revisions, immutable file hashes, exact fixture maintenance, guarded network/model counts and cleanup receipts are retained with the certificates. See `Docs/Design/Email_Search_Scale_Tuning_13376_4_2026-09-26.md` and `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md` for the verified implementation and deployment boundary.
