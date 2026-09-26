# Scoped PostgreSQL Email Search Benchmark

Tracking: TASK-13369.

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

The existing aggregate latency-threshold fields remain available. A new
explicit NFR gate also requires at least 1,000,000 messages, non-empty
`from`, `subject`, `label`, attachment, `after` and `before` query cases,
and each required case meeting p50 <= 250 ms and p95 <= 900 ms. A bounded
fixture can satisfy latency thresholds while this gate remains false.

The fixture calls Media and email graph APIs directly. It does not measure
EML/ZIP/MBOX parsing, HTTP authentication, archive throughput, multi-worker
behavior or production parity. A fresh Media handle in the cold pass does
not flush the PostgreSQL server or host filesystem cache. The operations
report must state these limits and the exact fixture/query protocol.

TASK-13370 also makes PostgreSQL FTS backfills conditional on the stored vector
being distinct from the computed vector. Repeated Media handle bootstrap still
checks schemas and scans vectors, but it does not rewrite already current rows.
