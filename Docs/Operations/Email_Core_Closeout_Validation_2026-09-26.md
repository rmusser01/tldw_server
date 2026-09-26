# Core Email Closeout Validation — 2026-09-26

Tracking: TASK-13376. Owner: project owner / maintainer (single owner).
Status: All core technical gates verified for the local reference scope, including
fresh dedicated PostgreSQL search/HTTP/release evidence and complete cleanup.
Earlier shared-service HTTP/search snapshots are retained separately. This is an
evidence record for owner review, not recorded release approval or production rollout.

## Scope

The selected scope is a local reference deployment with one loopback Uvicorn
worker, synthetic users and synthetic mail. SQLite is measured first, followed by
PostgreSQL on a dedicated disposable reference service at loopback port 5435 with
256 MiB shared memory, scoped databases and a non-superuser/non-BYPASSRLS role.
The existing shared service at port 5434 is preserved; its earlier measurements
are historical diagnostics. Model, background-work and non-loopback
socket/DNS guards apply to the probes. Personal Gmail and personal mail are
excluded; optional live Gmail remains deferred and does not block core release.

Core flags are `EMAIL_NATIVE_PERSIST_ENABLED=true`,
`EMAIL_OPERATOR_SEARCH_ENABLED=true`, `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
and `EMAIL_GMAIL_CONNECTOR_ENABLED=false`. The local rehearsal separately checks
`auto_email`, disables native/search flags, confirms retained legacy data and
restores the baseline. It reloads settings in a disposable app; it does not
certify process restarts, another host or a multi-worker deployment.

## Verified Implementation

Stage 1 is committed as `b880be1b53`: attachment policy, bounded ingestion/native
metrics and sensitive logging. Explicit metadata-only mode overrides the legacy
attachment switch; selective nested EML extraction has MIME allow/deny rules,
depth/count/size limits and persisted immediate-parent links. Unsupported binary
and PST/OST payloads remain metadata-only. Logging captures exclude synthetic
body/header/credential/metadata values, including echoed dependency errors and
traceback locals, while preserving exception propagation and transaction behavior.

The metrics implementation is covered by registry and operation tests. Both
backends' real authenticated-upload observations are recorded below.
These counters describe observed operations, not durable global commit counts.

Subsequent planner, fixture and PostgreSQL archive changes are committed in
`76f5b5caba` and `aac02e68e8`. The latter preserves the original substring
predicates while reducing scoped query and archive round trips; its diagnostic
evidence below does not close the final PostgreSQL gates.

The final PostgreSQL corrections are committed as `ae46f643ab`: legacy query
parameter order, quoted phrase/rank compatibility, bounded SQL rewrite/placeholder
caches and reduced identity round trips. Cache entries contain SQL text and flags,
with separate 512-entry bounds; request parameters and user content are not cached.
Independent identity/keyword review found no actionable issues. Current-source
shared-service HTTP/release evidence passes historically. The final probe-only
resource configuration is committed as `880d690a93`; it adds explicit scoped port
validation and the provisioner to 23 measured source files without changing
production queries. Dedicated current-source search and HTTP/release evidence pass below.

| Verification scope | Observed result | Evidence log |
| --- | --- | --- |
| Combined core parser, persistence, identity, attachment, metrics and query regressions | 328 passed; 2 optional real-PST fixture skips | `/tmp/email_core_regression_13376.log` |
| Focused logging, SQLite pool and access-log regressions | 83 passed; includes two real lower-backend schema failure captures | `/tmp/email_sqlite_schema_green_13376.log` |
| Additional legacy-search success/fallback/failure sentinel paths | Six sentinel paths in the 10-case focused suite | `/tmp/email_legacy_search_green_13376.log` |
| Selected legacy-search/read-contract regressions | 24 passed | `/tmp/email_legacy_search_regressions_13376.log` |
| PostgreSQL scope, sequence bootstrap, FTS and archive transaction regressions | 8 passed | `/tmp/email_postgres_core_regression_13376.log` |
| Mock connector, query metrics and scale regressions | 35 passed | `/tmp/email_final_mock_and_query_tests_13376.log` |
| Fixture correctness, canonical legacy title/body FTS and post-load statistics maintenance | 21 passed; earlier title/body captures had 2 expected pre-fix failures | `/tmp/email_fixture_analyze_green_13376.log`, `/tmp/email_bulk_legacy_fts_red_13376.log` |
| Probe guard/boundary regressions | 14 passed | `/tmp/email_probe_boundaries_verified_13376.log` |
| PostgreSQL substring/index compatibility | 51 passed, including 16 live PostgreSQL cases | `/tmp/email_pg_acceleration_compatibility_13376.log` |
| PostgreSQL RLS/scope and read contract | 28 passed; the combined run also had one read-contract fixture setup error. The read contract then passed separately with its proper fixture | `/tmp/email_pg_acceleration_rls_regressions_13376.log`, `/tmp/email_pg_read_contract_13376.log` |
| Guarded PostgreSQL fixture reuse | 55 passed | `/tmp/email_pg_reuse_verified_13376.log` |
| Archive graph round trips and identity/logging regressions | 18 graph cases and 96 identity/logging cases passed | `/tmp/email_graph_roundtrips_green_13376.log`, `/tmp/email_pg_graph_regressions_13376.log` |
| PostgreSQL settings/fixture observations | 10 passed | `/tmp/email_pg_settings_green_13376.log` |
| PostgreSQL verification memory budget and restored timed settings | 11 passed after the new live regression first failed; verification is serial within its transaction, timed settings restore | `/tmp/email_pg_parity_budget_green_13376.log` |
| PostgreSQL transaction-local custom plans and restored session settings | 22 passed; repeated pooled label searches and success/error restoration covered | `/tmp/email_pg_custom_plan_green_13376.log` |
| Final PostgreSQL combined regressions | 211 passed; 4 existing warnings | `/tmp/email_postgres_final_combined_regression_13376.log` |
| Identity and keyword compatibility | 51 passed | `/tmp/email_pg_identity_final_verified_13376.log` |
| Legacy parameter, phrase and rank compatibility | 28 passed | `/tmp/email_pg_legacy_final_regressions_13376.log` |
| Bounded SQL rewrite/placeholder caches | 65 passed | `/tmp/email_sql_rewrite_cache_green_13376.log` |
| Expanded 22-file measured source guard | 56 passed | `/tmp/email_source_sql_utils_green_13376.log` |
| Full helper/probe regressions including explicit port and 23-file source guards | 91 passed | `/tmp/email_pg_port_guard_full_green_13376.log` |
| Touched Python compilation | 62 files compiled successfully | `/tmp/email_final_quality_13376.json` |
| Touched Python Ruff | Clean | `/tmp/email_final_ruff_13376.log` |
| Touched production/probe Bandit | 38 files; zero findings and scanner errors | `/tmp/email_final_bandit_13376.json`, `/tmp/email_final_quality_13376.json` |

These suites overlap and their counts are not additive. Commands activate the
project virtual environment and run from the isolated worktree with
`PYTHONPATH=$PWD`. Final retained artifact paths, commands and measured-source
identity are recorded with the backend evidence below under TASK-13376.

Detailed contracts and audit scope:

- `Docs/API-related/Email_Attachment_Policy_13376.md`
- `Docs/Operations/Email_Sensitive_Logging_Audit_13376_2_2026-09-26.md`
- `Docs/Design/email-ingestion-metrics-13376-1.md`
- `Docs/Product/Email_Search_Benchmark_Protocol.md`

## Final Backend Measurements — TASK-13376

| Evidence | SQLite | PostgreSQL |
| --- | --- | --- |
| Retained report links, exact commands, UTC run date and measured revision/source hash | `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite.json`, `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite_parity.json` and `Docs/Operations/evidence/email_core_closeout_13376/sustained_sqlite.json`; search report generated 2026-09-26 18:06:28 UTC; base `b880be1b53` plus 19 unchanged measured file hashes and reuse provenance | Canonical `Docs/Operations/evidence/email_core_closeout_13376/million_postgres.json`, `million_postgres_parity.json` and `sustained_postgres.json`; search generated 2026-09-26 20:22:19.584478 UTC at `880d690a93`; 23 unchanged measured hashes in both final probes |
| Hardware/CPU, RAM, Python/backend versions and deployment topology | Apple M5 Max, 18 logical CPUs, 128 GiB RAM, macOS 26.5.2, Python 3.11.13; one loopback Uvicorn worker for HTTP | Same host/Python and Docker VM resources; dedicated PostgreSQL 18.6 at loopback port 5435 with **256 MiB POSIX shared memory**, same image digest as shared port 5434; timed 64 MB work_mem / 128 MB shared_buffers / hash multiplier 2 / parallel workers 2 / RLS on; shared service preserved |
| Actual Media/native messages, attachments and legacy title/body FTS validation | 1,000,000 each: native rows, owner Media rows, linked legacy rows, matching body versions, synthetic identities and legacy indexed bodies; 200,000 attachment rows | Same complete fresh 1,000,000-row body/index/identity/version parity checks; 200,000 attachments |
| Attachment ratio, labels, sender/recipient pools and internal-date span | 20%; 23 labels; 200 senders / 500 recipients; 2025-09-26 17:53:30 UTC through 2026-09-26 17:53:30 UTC (365 days) | 20%; 23 labels; 200 senders / 500 recipients; 2025-09-26 20:15:21 UTC through 2026-09-26 20:15:21 UTC (365 days) |
| Ten populated operator cases and meaningful negation baseline/result counts | All ten populated; negation reduces 166,667 matches to 165,000 | All ten populated; same meaningful negation 166,667 -> 165,000 |
| Warm aggregate p50/p95, per-case diagnostics and NFR gate outcome | **Pass:** 200 samples, **202.23 ms / 725.74 ms**; maximum 771.29 ms. Per-case diagnostics remain in the artifact; its stricter per-operator latency diagnostic is false and is not the aggregate gate | **Pass:** 200 samples, **220.92 ms / 495.49 ms**, maximum 519.40 ms; aggregate gate true, stricter per-operator diagnostic false |
| Cold-pass handle-reopen aggregate p50/p95 | 10 samples, **197.33 ms / 694.77 ms**; pooled connections and caches may remain warm | 10 samples, **281.54 ms / 502.76 ms**; same pool/cache limits |
| Bulk fixture setup duration/batches, separately from ingestion rate | 500 batches of 2,000; original complete load 270.7611 seconds plus 3.7664 seconds `ANALYZE` = **274.5275 seconds**; reused validated rows after maintenance | Fresh 500 batches of 2,000; **275.5685 seconds includes 33.0244 seconds ANALYZE**; no fixture reuse or extra maintenance |
| Sustained HTTP messages/batches, measured request seconds, total wall seconds and messages/sec | **Pass:** 7,900 messages / 79 batches; 60.5196 request seconds / 62.9324 wall seconds; **130.54 messages/sec** by request time / **125.53** by wall time; every batch >= 50 | **Aggregate pass:** 3,800 messages / 38 batches; **61.5734 request seconds / 62.4711 wall seconds**; **61.71 messages/sec** by request time / **60.83** by wall time. Batch diagnostic false: 3 batches < 50, minimum **38.96**; the gate is the minute-wide aggregate |
| Stable identities on retry, owner/other-user search/detail and forced-RLS checks | Exact first 100 IDs on retry; another user's search empty and detail 404 | Exact retry of first 100 IDs; other-user search 0/detail 404; role non-super/non-BYPASSRLS and forced RLS; owner scope restored |
| Baseline flags, email search/detail and media search | Guarded HTTP checks passed; restored flags equal baseline and detail 200 | Same baseline flags pass, restored flags equal baseline and detail 200 |
| Complete legacy/native result-ID parity, separate `auto_email` and restored `opt_in` | Equal 7,900-ID result sets; quoted title/body queries each matched the same single ID; `auto_email` matched all 7,900 explicit-operator IDs; `opt_in` restored | Exact 3,800/1/1 legacy/native IDs; `auto_email` same 3,800 explicit IDs; `opt_in` restored |
| Native/search flag rollback, all expected retained legacy IDs and restored detail | All 7,900 legacy IDs retained; email search/detail 404 and explicit operator bridge 422 while disabled; restored detail 200 | All 3,800 legacy IDs retained; disabled 404/404/operator 422; restored detail 200 and baseline flags |
| Actual parse/dedupe/persistence/search metric observations | 8,000 parse/Media/native successes including retry, 100 dedupe matches, and native search request/result/duration samples in the report | 3,900 parse/Media/native successes including retry; dedupe 100; 133 native searches (132 query-present/1 empty) with duration/result observations |
| Zero outbound/model attempts and blocked background work | Zero outbound/model attempts; guards active and source unchanged during the probe | Both final guards zero, background work blocked, 23 hashes unchanged at `880d690a93` |
| Probe-owned temporary roots, PostgreSQL databases/role and private manifest cleanup | Exact generated SQLite roots removed and confirmed absent in `Docs/Operations/evidence/email_core_closeout_13376/email_cleanup_roots_receipt_13376.json` | Three shared-service and two dedicated targets: remaining database/role counts 0, five manifests absent; owned container/anonymous volume removed, shared service preserved. All 55 exact roots absent; retained `postgres_cleanup.json`, `postgres_dedicated_cleanup.json` and root receipt in the evidence directory |

Search acceptance requires an actual 1,000,000-message single-tenant fixture, all
ten populated classes with meaningful negation, and aggregate warm p50 <= 250 ms /
p95 <= 900 ms using 20 measured calls and 3 unmeasured warmups per case at limit 50.
Per-case timings are diagnostics. The cold pass opens fresh MediaDatabase handles;
pools may reuse physical connections and OS/filesystem/PostgreSQL caches remain
unchanged. It is not proof of cache-cold performance and is not the primary SLO.

Sustained ingestion acceptance requires at least 50 messages/sec for metadata-only
persistence over at least 60 measured authenticated HTTP request seconds per
backend, with total wall time, identity/retry/isolation and native failure checks.
The sustained gate uses the complete measured request window; individual batch
rates are diagnostics. PostgreSQL's three below-target batches remain visible and
do not change the aggregate protocol.
Fixture setup rate does not measure HTTP ingestion throughput. Final artifacts
must identify any source changes made after Stage 1 rather than attributing the
whole measured run to `b880be1b53`.

The earlier SQLite million-message warm p50 191.52 ms / p95 714.60 ms result is
historical and superseded: its bulk fixture did not maintain canonical legacy
Media FTS. The fixture now has a retained title/body FTS regression and the final
complete SQLite certificate above supersedes that result. Historical 10,000-message search and three
100-message archive batches also do not satisfy the final scale/sustained gates.

The fresh complete SQLite fixture passed every parity, index and identity check,
but its initial full warm run failed the latency gate at p50 **698.82 ms** /
p95 **2,615.85 ms**. Exact query plans showed missing `ANALYZE` statistics. On the
identical rows, statistics maintenance took **3.76637 seconds** and changed
participant lookup to matching-address/reverse-link plans. Subsequent targeted
diagnostics measured sender **64.03 ms**, recipient **27.18 ms** and negation
**717.61 ms**, with unchanged result counts of **5,000**, **2,000** and **165,000**.
Those targeted diagnostics are not a final aggregate performance pass.

The bulk loader now runs `ANALYZE` automatically for SQLite and PostgreSQL after
loading. The final guarded SQLite certificate reuses the original fully validated
fixture after this maintenance and records the updated source hashes; its final
aggregate result passes at **202.23 ms / 725.74 ms**. New users follow the fresh-loader recipe in
the benchmark protocol, which now performs the same statistics step automatically.
Fixture loading/statistics time remains separate from measured HTTP throughput.

Three reviewed probe gaps—complete expected-ID rollback checks, explicit backend
selection and private-manifest resource cleanup—were corrected with 14 guard and
boundary regressions. The SQLite report verifies the complete rollback ID set.
The exact generated SQLite roots are confirmed absent in the cleanup receipt.
Final PostgreSQL cleanup is verified in the retained catalog, manifest,
container/volume and exact-root receipts.

## PostgreSQL Fixture Smoke Evidence

The scoped 120-message PostgreSQL smoke fixture passed native/Media/version/body
index/identity and shape checks with 24 attachments. Its role was neither
superuser nor BYPASSRLS; Media RLS was enabled and forced, another user's Media
count was zero, and owner scope was restored. A normal subsequent append used
Media/native IDs **121 / 121**, preserved a **16,384-byte** native subject and
retained zero other-user rows. Evidence: `/tmp/email_pg_small_fixture_13376.json`
and `/tmp/email_pg_append_validation_13376.json`. This establishes fixture,
sequence and long-subject safety on the chosen backend; its 120 rows cannot
certify million-message latency or sustained HTTP throughput. The complete final
dedicated certificates above supply those gates.

## PostgreSQL Initial Million-Message Diagnostic

The first actual PostgreSQL million-message run, generated 2026-09-26 18:17:54 UTC,
failed the aggregate warm latency gate at **p50 510.89 ms / p95 1,841.72 ms** over
200 samples. Its fixture passed all 1,000,000-row native/Media/version/synthetic
identity/indexed-body checks, with 200,000 attachments, 23 labels, 200 senders /
500 recipients, all ten populated cases and meaningful negation 166,667 ->
165,000. The role remained non-superuser/non-BYPASSRLS, Media RLS was enabled and
forced, another user's row count stayed zero, and owner scope was restored.
Guards recorded zero outbound/model attempts and unchanged measured source.

The host was Apple M5 Max with 18 logical CPUs / 128 GiB RAM; PostgreSQL 18.6
Debian ran in the Docker VM with 18 CPUs / 16,746,053,632 bytes RAM. Cold-pass
handle diagnostics were p50 505.44 ms / p95 2,253.67 ms with pooled physical
connections and caches potentially reused. Evidence is
`/tmp/email_million_postgres_final_13376.json`; despite that temporary filename,
this is a failed initial diagnostic, not a final passing certificate. Its original
load took **207.49777 seconds**, including **5.62657 seconds** of `ANALYZE`.

The `aac02e68e8` query changes retain the original substring predicates and bind
matching label IDs within the scoped transaction. PostgreSQL acceleration adds
expression statistics and a native multicolumn `pg_trgm` GIN index. RLS scope
expressions use statement InitPlans; forced RLS and non-admin isolation remain
required. The optional acceleration objects can be unavailable without changing
search semantics; their presence must be recorded for a performance claim.

The chosen connection profile is `PGOPTIONS='-c work_mem=64 MB'`, with observed
`shared_buffers=128 MB`, `hash_mem_multiplier=2` and
`max_parallel_workers_per_gather=2`. This changes memory for probe connections;
no global service configuration was changed. A short 30-sample comparison on
the same code measured **191.71 / 516.82 ms** p50/p95 at 64 MB versus
**248.01 / 962.52 ms** at 4 MB. These are diagnostic samples, not the required
200-sample final certificate.

Guarded reuse records later schema/index/RLS maintenance of **19.002133 seconds**
and another **34.987543 seconds** of `ANALYZE` separately from the original load
in `Docs/Operations/evidence/email_core_closeout_13376/postgres_tuning_maintenance.json`.
The retained bounded 100-message archive profiles are
`Docs/Operations/evidence/email_core_closeout_13376/postgres_archive_profile_before.json`
and `Docs/Operations/evidence/email_core_closeout_13376/postgres_archive_profile_after.json`:
SELECT statements fell from 1,931 to 1,531 and measured rates were 44.92 and
58.02 messages/sec respectively. This bounded profile does not certify sustained
authenticated HTTP throughput.

The subsequent guarded full-scale attempt aborted during fixture parity before
timing with SQLSTATE `53100`: the Docker container's 64 MiB POSIX shared-memory
budget could not hold two parallel wide body/version verification joins. It
produced no new SLO result. The verification fix is committed as `1fee95629a`:
only the fixture-inspection transaction sets
`SET LOCAL max_parallel_workers_per_gather=0`. All parity and security checks
remain intact; transaction completion restores the timed search profile of two
parallel workers and 64 MB `work_mem`. No global service setting changed.

The second full attempt at `1fee95629a` passed fixture parity but failed the
`label:Inbox` COUNT on iteration 11 with SQLSTATE `53100`. A bounded comparison
using the same pooled connection reproduced the failure, then completed 23
label searches with `SET LOCAL plan_cache_mode='force_custom_plan'`. This points
to a generic prepared plan losing parameter selectivity and requiring a large
parallel hash join, but that transition is inferred. Counters showed one custom
plan at iteration 6 and five custom / zero generic plans at iteration 10. No
post-failure generic count was observed because psycopg clears prepared counters
on rollback. The production correction is committed as `455fad64fb` and applies
custom planning only inside PostgreSQL native search's transaction, preserving bound parameters,
predicates and RLS and restoring the session mode on success or error. Evidence:
`Docs/Operations/evidence/email_core_closeout_13376/postgres_generic_plan_diagnostic.json`,
`/tmp/email_pg_prepared_search_diagnostic_13376.log` and the 22-case regression
above. These repeated-call checks establish the diagnosis and bounded fix;
they are not the final ten-class certificate. Independent source review found
no actionable issues. The earlier complete guarded certificate at frozen `455fad64fb`
subsequently passed at warm aggregate **248.63 / 659.39 ms**, with all 200 samples,
ten populated cases, meaningful negation, complete fixture parity and forced-RLS
checks. Its 21 measured source hashes remained unchanged and guards were zero.
The fixture-only transaction used zero parallel workers; timing retained two
workers / 64 MB. The million-message probe's generated databases, role, manifest
and exact roots were removed while preserving the local service. This certificate
is historical/superseded by the final dedicated `880d690a93` certificate. Its
retained reports are `million_postgres_455fad64fb_historical.json` and
`million_postgres_parity_455fad64fb_historical.json` in the evidence directory.

## PostgreSQL HTTP and Legacy Diagnostics — Historical

The earlier fresh HTTP attempt uploaded **1,700** messages over
**60.9880 measured request seconds**, aggregate **27.87 messages/sec**, below
the 50 messages/sec target. Its release check then failed
the `ArchiveThroughput` full-set parity assertion: **legacy 0 / native 1,700**.
The process exited 1 without a final JSON certificate. Evidence is
`Docs/Operations/evidence/email_core_closeout_13376/postgres_http_sustained_failed_diagnostic.json`;
it is explicitly not an acceptance certificate. The retained legacy diagnosis
and corrected checks are `postgres_legacy_parity_diagnosis.json` and
`postgres_legacy_parity_corrected.json` in the same evidence directory. After
parameter/phrase/rank corrections, exact parity was 1,700/1/1 with other-user 0.
The cache-only 300-message HTTP diagnostic reached 45.71 messages/sec; the later
identity correction reached 86.20 messages/sec over 300 messages. Both are bounded
diagnostics, not sustained acceptance.

The earlier full HTTP certificate at `ae46f643ab` passed **92.06 messages/sec**
over 60.8291 request seconds, with exact 5,600/1/1 parity, retry 100, `auto_email`,
rollback/restore, forced RLS, live metrics and 22 stable hashes. Its retained report is
`Docs/Operations/evidence/email_core_closeout_13376/sustained_postgres_ae46f643ab_historical.json`.
It is historical; canonical final HTTP uses the dedicated service/source below.

The fresh `ae46f643ab` run completed its million-message load but aborted with a
database error before producing a final JSON certificate. Diagnosis identified
SQLSTATE `53100` in `dsm_impl.c::dsm_impl_posix`: the shared container's 64 MiB
POSIX shared memory was insufficient. Statistics/index/visibility checks were
present; the same subject COUNT failed at 64/32/16/8 MB work_mem budgets. Evidence:
`Docs/Operations/evidence/email_core_closeout_13376/postgres_fresh_dsm_diagnostic.json`
and `postgres_fresh_shared_memory_diagnostic.json` in that directory.

## Dedicated PostgreSQL Reference Profile

The final profile uses disposable `tldw_email_closeout_13376_pg` at
`127.0.0.1:5435`, **256 MiB shared memory**, PostgreSQL 18.6 and the same
`postgres:18` image digest as the preserved shared service at port 5434.
`Docs/Operations/evidence/email_core_closeout_13376/postgres_dedicated_environment.json`
records the resource identity. This changes probe infrastructure capacity;
production queries, authorization and the 64 MB per-connection timing budget are
unchanged. Scoped port/manifest validation and provisioner source guards are
committed as `880d690a93`; 91 helper regressions pass.

The fresh final search certificate uses no fixture reuse or later maintenance:
500 batches of 2,000,275.5685 seconds including 33.0244 seconds of `ANALYZE`.
Complete body/index/version/identity/RLS checks pass, warm aggregate is
**220.92 / 495.49 ms**, cold-handle diagnostics **281.54 / 502.76 ms**, all ten
cases are populated and negation removes 1,667 relevant records. Its 23 source
hashes remain unchanged and guards zero. Final HTTP/release on the same dedicated
source/service passes the minute-wide aggregate gate at **61.71 messages/sec**,
with exact retry/parity/RLS/metrics/rollback evidence in the table. Its batch
diagnostic is false (3 below 50; minimum 38.96), separate from aggregate acceptance.
Cleanup confirms database/role counts 0 across five generated targets, five absent
private manifests, removed owned container/anonymous volume, preserved shared
service and 55 unique exact roots absent.

## Release Record

The single owner's technical review uses this record and
`Docs/Operations/Email_Release_Checklist_and_Rollback.md`. Logging and attachment
implementation checks are complete. SQLite million-message search, sustained
HTTP, live metrics, local rollout and temporary-root cleanup are recorded.
PostgreSQL fresh current-source million-message search now passes at
`880d690a93` on the dedicated profile. Final dedicated sustained HTTP, live metrics,
local release rehearsal and cleanup also pass. No core technical validation item
remains open for this selected local reference scope. Earlier shared-service HTTP
and `455fad64fb` search reports are historical evidence, retained as
`sustained_postgres_ae46f643ab_historical.json`,
`million_postgres_455fad64fb_historical.json` and
`million_postgres_parity_455fad64fb_historical.json` in the evidence directory.
Release approval is a separate human record in the checklist and has not been
inferred from implementation, test results or the instruction to finish the work.

Optional live Gmail OAuth/provider behavior and staging sync lag remain deferred.
Two real-PST fixture skips limit enabled-adapter evidence; synthetic adapter and
missing-parser checks remain separate. None of these optional live-mail checks
claims access to a personal account or authorizes downstream model processing.
