# Synthetic Email Archive Ingestion Throughput — 2026-09-25

Tracking: TASK-13371. PostgreSQL correctness fix: TASK-13372, commit `41684adbe3`.

The full main app ingested three synthetic 100-message MBOX archives over
authenticated loopback HTTP, first on SQLite and then PostgreSQL. Each archive
was 36,280 bytes, with unique Message-IDs, subjects and plain-text bodies, zero
attachments and one archive-group keyword per batch. Archives stayed within the
existing 100-member guardrail. No Gmail account or personal mail was accessed.

Both backends passed all 300 child-persistence and native-search ID assertions,
first/last detail subject checks, a 100-message rerun preserving IDs, and
cross-user search/detail isolation. Model/background hooks and non-loopback
socket/DNS guards recorded zero attempts. PostgreSQL also confirmed a
non-superuser/non-bypass role, Media RLS enabled and forced, and direct owner
rows 300 versus other-user rows 0.

## Timing protocol and results

Timing starts before each `POST /api/v1/media/add` and ends after its response.
It includes HTTP/auth/quota checks, parsing, legacy Media persistence, native
email persistence, and response serialization. Fixture generation, startup,
search/detail assertions and the retry are excluded. Analysis, claims,
chunking, auto-chunk LLM, embeddings, original-file retention and attachment
ingestion were disabled. This is an end-to-end metadata-only upload reference,
not a persistence-only microbenchmark or heavy-attachment test.

Hardware: macOS 26.5.2 arm64, Python 3.11.13, 18 logical host CPUs. PostgreSQL
used the `postgres:18` image in the local repository-fixture service on port 5434
in `tldw_workspace_12020_50_pg`, with no per-container CPU or memory limit. Docker
reported 18 CPUs and 16,746,053,632 bytes of memory. The cluster can serve other local tests; no exclusive
hardware or production topology is claimed. Each app used one loopback Uvicorn
worker, multi-user AuthNZ, isolated storage, Gmail/connectors disabled, worker
sidecar mode and deferred heavy startup.

| Recorded run | Per-batch messages/sec | Aggregate messages/sec |
| --- | --- | ---: |
| SQLite initial baseline | 31.84, 28.18, 28.06 | 29.26 |
| SQLite factory diagnostic | 43.82, 47.93, 45.39 | 45.65 |
| SQLite published probe | 49.59, 54.68, 56.74 | 53.49 |
| PostgreSQL first fixed run | 3.30, 6.21, 7.54 | 5.03 |
| PostgreSQL published probe | 6.32, 7.23, 7.30 | 6.92 |

A main-thread cProfile run on SQLite also passed correctness at 32.57 msg/sec;
it is not a reference timing because profiling adds overhead and does not
capture executor-thread work. The factory diagnostic measured 100 worker
handles per batch but only 0.054–0.059 seconds total construction time per batch,
roughly 2–3% of upload time. Repeated SQLite handle construction is not established
as the main bottleneck. Host/cache variability is visible across runs.

The 50 msg/sec target is **not certified**: PostgreSQL remains far below it, and
SQLite has one aggregate pass but earlier misses and a below-target first batch.
Three small archives do not establish sustained large-archive throughput.
The probes' `all_batches_meet_target` is false on both published runs.

## Failure and correction

Before TASK-13372, PostgreSQL returned a nominal success response but persisted
only one of the first archive's 100 children. The probe rejected it. Server
errors confirmed 99 duplicate `media_pkey` failures. Routine Media bootstrap
reset sequences from row maxima before worker scope was installed; forced RLS
hid existing rows and reset the next ID to 1. Removing routine sequence repair
fixed the repeated-handle/tenant case while preserving explicit v18 migration
maintenance. The failing pass provides no valid throughput measurement.

Regression evidence: 75 schema unit tests passed, including the retained v18
migration check; three sequence/FTS cases passed, two using live PostgreSQL. The sequence
unit boundary and real PostgreSQL repeated-handle regression failed before the
fix. The subsequent authenticated archive probes passed every persistence,
retry and RLS assertion.

## Artifacts and reproduction

Recorded probes are operations validation artifacts, not production ingestion
tools. They create synthetic users and temporary local data and expect a fresh
target. Activate the project virtual environment and set `PYTHONPATH` to the
checkout containing the probes. The database helper reads local Docker test
credentials internally, stores its generated probe manifest with mode 0600,
prints no password and validates generated resource names before cleanup.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
EMAIL_PROBE_OUT=/tmp/email_archive_sqlite.json \
  python Docs/Operations/probes/email_archive_throughput_sqlite_2026_09_25.py

# Use the existing repository-fixture PostgreSQL service, not a production cluster.
export EMAIL_PROBE_PG_CONTAINER=tldw_workspace_12020_50_pg
export EMAIL_PROBE_PG_MANIFEST=/tmp/email_archive_private_manifest.json
python Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py setup
EMAIL_PROBE_OUT=/tmp/email_archive_postgres.json \
  python Docs/Operations/probes/email_archive_throughput_postgres_2026_09_25.py
python Docs/Operations/probes/email_archive_probe_databases_2026_09_25.py cleanup
```

The PostgreSQL probe assigns synthetic user roles after app startup because
startup seeds the default PostgreSQL RBAC roles. It requires the manifest's
isolated AuthNZ/content databases and generated non-superuser role. The helper
does not start or delete the shared fixture container. The local reference uses
127.0.0.1:5434; other ports require deliberate adaptation of this frozen probe.
The JSON reports print temporary root paths; remove only those probe roots
after shutdown and evidence capture. Failed probes also require cleanup.

JSON evidence:

- `Email_Archive_Throughput_SQLite_Baseline_2026-09-25.json`
- `Email_Archive_Throughput_SQLite_Instrumented_2026-09-25.json`
- `Email_Archive_Throughput_SQLite_2026-09-25.json`
- `Email_Archive_Throughput_PostgreSQL_FirstFixed_2026-09-25.json`
- `Email_Archive_Throughput_PostgreSQL_2026-09-25.json`

The diagnostic added a timed wrapper around `persistence.create_media_database`
and summed construction timings for each upload; it changed no database behavior.
Published probe Ruff check/format and Bandit passed. Bandit B101 is excluded only
for assertions in these validation artifacts/tests. Six cleanup target tests
passed, rejecting unrelated roles, databases, hosts and ports. Production
sequence-fix Bandit had zero findings/errors without that exclusion.

Cleanup was confirmed by the published helper's catalog checks: both disposable
databases and the generated role were absent before manifest removal. All nine
private roots created by this validation round, including failed probes, were
removed after shutdown. Older validation roots and the shared fixture container
were preserved.

Remaining gates: sustained archive throughput, 1M-message search scale, intended
deployment topology and production parity. Optional live Gmail stays deferred.
