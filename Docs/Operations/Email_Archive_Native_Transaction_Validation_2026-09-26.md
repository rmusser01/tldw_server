# Email archive native transaction validation — 2026-09-26

TASK-13375 follows the archive-local worker reuse in TASK-13373 (`27969ae 722`).
The remaining archive path read saved Media metadata/body through separate
connections before starting normalized graph persistence. It now performs
those reads and the graph write inside one native-only transaction per message.
The legacy Media write has already committed. Native errors are caught outside
the new transaction, so a failed graph rolls back while accepted Media rows
remain available for retrieval and retry. The primary-message and attachment
paths retain their existing behavior.

## Evidence and timing limits

The full main app ingested three synthetic 100-message MBOX archives through
authenticated loopback HTTP in multi-user mode, SQLite first then PostgreSQL.
Each archive was 36,280 bytes; all messages had unique synthetic IDs/subjects,
plain-text bodies and zero attachments. Analysis, claims, chunking, embeddings,
auto-chunk LLM, attachment ingestion and original retention were disabled.
The probes assert that model/background hooks and non-loopback sockets/DNS
receive zero attempts. Personal Gmail was not accessed.

| Unprofiled run | Per-batch messages/sec | Aggregate messages/sec |
| --- | --- | ---: |
| Fresh SQLite baseline |70.92,71.39,69.11|70.46|
| SQLite native transaction |63.02,54.92,61.01|59.44|
| Fresh PostgreSQL baseline |20.77,21.33,18.91|20.28|
| PostgreSQL native transaction |37.59,37.38,45.98|39.94|

Both before/after passes on both backends verified 300 persisted child IDs and
matching native-search IDs, first/last detail subjects, a 100-message retry
preserving IDs, other-user empty search and detail 404, and zero model/network
attempts. PostgreSQL directly verified Media RLS enabled/forced, a
non-superuser/non-bypass probe role, and owner rows 300 versus other-user rows 0.

The protocol includes HTTP/auth/quota checks, parsing, persistence, indexing and
response serialization. Startup, fixture generation, search/detail validation
and retry timing are excluded. Hardware is macOS 26.5.2 arm 64, Python 3.11.13,
18 logical CPUs, one Uvicorn worker and the shared repository-fixture PostgreSQL
service on 127.0.0.1:5434 (`tldw_workspace_12020_50_pg`, postgres 18), using fresh
isolated generated AuthNZ/content databases for each pass.

PostgreSQL's observed aggregate is 1.97 times its fresh baseline. SQLite's
unprofiled after run is slower than its fresh baseline, despite fewer measured
setup calls. Host/cache variability is substantial: the preceding day's
unprofiled references were 56.46/33.67 and today's profiled before/after HTTP
runs were 54.08/84.16 for SQLite and 15.32/25.54 for PostgreSQL. Profiling adds
overhead; those are diagnostic runs, not reference timings. These small
sequential passes cannot attribute the full wall-clock difference to this code
change or certify sustained throughput. PostgreSQL remains below 50 msg/sec,
and the 1M-message search, attachment-heavy and deployment-parity gates remain
unverified. Optional live Gmail stays deferred.

## Worker profiles and regression checks

Temporary wrappers called `cProfile.Profile.runcall(operation, db)` inside the
archive worker and combined the first 300 initial-message profiles, excluding
retry profiles and database factory construction. Raw diagnostic HTTP reports
and selected worker counters are retained in the profile-summary JSON.

| Counter across 300 messages | Before | After |
| --- | ---: | ---: |
| SQLite connection configuration calls |1,814|1,214|
| PostgreSQL connection borrows/scope setup calls |1,200|600|
| SQLite saved-payload read cumulative seconds |1.485|0.121|
| PostgreSQL saved-payload read cumulative seconds |3.408|0.548|

Call-count reductions establish removed setup work; time differences are
subject to profile overhead and host variability. Total profiled worker time
was 4.755 → 2.901 s on SQLite and 17.652 → 10.238 s on PostgreSQL.

The shared-connection and late native-error SQLite regressions failed before
the production change. Both PostgreSQL late Python/SQL failure cases also
failed before the change because a graph row remained committed. After the
change,25 focused SQLite/archive/chunk/native-graph tests and five sequence/FTS/
native-rollback cases passed, four against live PostgreSQL. They cover stable
retry IDs, declined overwrite retaining saved body/metadata, graph rollback
preserving legacy rows, legacy metadata fallback, concurrent request scope,
thread ownership and repeated-cancellation cleanup. Independent code review
reported no important findings.

Bandit reported zero findings/errors on the touched production module without
exclusions, and on touched tests with B 101 excluded only for test assertions.
Ruff passed all touched tests and reported no new production findings;13
unrelated inherited module findings remain, confirmed against the starting
revision. The two new/extended test files were Ruff-formatted. Whitespace
validation passed. See the design at
`Docs/Design/Email_Archive_Native_Transaction_2026-09-26.md`.

## Reproduction and cleanup

Use the frozen guarded SQLite/PostgreSQL probes and generated-database helper
under `Docs/Operations/probes/`, with the project virtual environment activated
and `PYTHONPATH` pointing at this worktree. The complete protocol and commands
are in `Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md`.

Four generated PostgreSQL database/role rounds were cleaned with helper catalog
checks before private manifest removal. Eight synthetic data roots from the
baseline, after and diagnostic passes were removed after app shutdown. The
shared fixture service and unrelated data were preserved. The completed
TASK-13375 implementation plan was removed as required.

JSON evidence:
- `Email_Archive_Native_Transaction_SQLite_Baseline_2026-09-26.json`
- `Email_Archive_Native_Transaction_SQLite_After_2026-09-26.json`
- `Email_Archive_Native_Transaction_PostgreSQL_Baseline_2026-09-26.json`
- `Email_Archive_Native_Transaction_PostgreSQL_After_2026-09-26.json`
- `Email_Archive_Native_Transaction_Profiles_2026-09-26.json`
