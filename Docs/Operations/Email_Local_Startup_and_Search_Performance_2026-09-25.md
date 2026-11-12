# Local Email Startup and Search Performance — 2026-09-25

Tracking: TASK-13361. Base commit: `8c6489007804894b18fc06c9a4da698aa340c3b3`.

The main FastAPI app completed its actual lifespan startup and shutdown in a
focused pytest integration case. During that lifespan, a real AuthNZ API key
read synthetic email search and detail, and the media search route delegated an
email operator query. The test used separate temporary SQLite Media databases
for two users, required the second user's results to stay scoped, and rejected
even caught socket, DNS, shared HTTP-client, model and background-job attempts.
It did not bind a network socket or exercise a deployed process. The Gmail
connector and connectors worker were disabled; no personal mail was used.

Run from the repository root after activating `.venv`:

```bash
python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_authenticated_access.py::test_main_app_lifespan_serves_scoped_email_and_media_routes \
  -q
```

The isolated run passed **1 test** in 10.52 seconds. The fixture's teardown
asserted zero outbound/model attempts. The previous authenticated upload and
search cases remain separate evidence; this new test closes only their
application-lifespan gap.

## Bounded synthetic benchmark

`Helper_Scripts/benchmarks/email_search_bench.py` built a 10,000-message,
single-tenant SQLite fixture on macOS 26.5.2 / arm64 / Python 3.11.13 (18 logical
CPUs). The fixture contained 2,476 attachment rows and 23 distinct labels. Its
default mix covers sender, recipient, subject, label, attachment, date, free
text with negation, OR, and a relative window. Across 15 measured runs per query
after five warmups, 150 warm queries measured **p50 15.39 ms, p95 30.85 ms**;
the ten-query cold pass measured p50 16.22 ms, p95 33.89 ms. The default mix
includes two zero-result queries, so its aggregate is not a production workload
sample. The full report, including the exact query mix, per-query match counts,
latencies and hardware profile, is
`Docs/Operations/Email_Search_Benchmark_10k_2026-09-25.json`.

A second run on the same fixture selected six non-empty cases corresponding to
the NFR-PERF-001 operators: `from`, `subject`, `label`, `has:attachment`, `after`
and `before`. Across 15 measured runs per query after five warmups, 90 warm
queries measured **p50 9.91 ms, p95 21.19 ms**; the six-query cold pass measured
p50 11.50 ms, p95 22.27 ms. Each case had at least 50 matches, and the label
case matched all 10,000 messages. The full report is
`Docs/Operations/Email_Search_Benchmark_10k_NFR_2026-09-25.json`.

The recorded fixture began with a 1,000-message smoke run and was expanded to
10,000 messages in the same database. Reproduction of that two-step shape and
query mix, using an isolated temporary database and no provider access:

```bash
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path /tmp/email_search_bench_13257.sqlite \
  --ensure-fixture --fixture-messages 1000 \
  --runs 5 --warmup-runs 2 \
  --out /tmp/email_search_bench_13361_1000.json
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path /tmp/email_search_bench_13257.sqlite \
  --ensure-fixture --fixture-messages 10000 \
  --runs 15 --warmup-runs 5 \
  --out /tmp/email_search_bench_13361_10000.json
jq '[.query_mix[] | select(.name == "from_filter" or .name == "subject_filter" or .name == "label_filter" or .name == "has_attachment" or .name == "after_date" or .name == "before_date")]' \
  /tmp/email_search_bench_13361_10000.json > /tmp/email_search_nfr_mix_13361.json
python Helper_Scripts/benchmarks/email_search_bench.py \
  --db-path /tmp/email_search_bench_13257.sqlite \
  --query-mix-file /tmp/email_search_nfr_mix_13361.json \
  --runs 15 --warmup-runs 5 \
  --out /tmp/email_search_bench_13361_nfr.json
```

The fixture date anchor uses the current clock, and the recorded two-step build
resets its seeded attachment sequence at expansion. A later run can therefore
match the protocol and workload shape without producing byte-identical rows or
latencies. The committed JSON files preserve the exact results of this run.

The harness directly writes Media and normalized email graph records for its
fixture. It does not measure `.eml`/ZIP/MBOX parsing, API ingestion throughput,
or the NFR-PERF-002 archive ingestion target. Its 1,000-message smoke run was
p50 5.17 ms, p95 6.77 ms; the 10,000-message run is still only 1% of the
specified 1M-message mailbox. Neither run certifies the NFR-PERF-001 target at
that scale. The intended deployment, backend, workload trace and hardware are
not yet selected, so no production-performance or parity claim follows.

The Docker daemon is available on this host, but this task did not start a
PostgreSQL deployment or validate its email tenant isolation. PostgreSQL/RLS,
real server socket readiness, rollout flags in the chosen environment, 1M-message
performance, archive ingestion throughput and owner sign-off remain open core
release gates. Live Gmail/OAuth is optional and remains deferred.
