# Email ingestion metrics (TASK-13376.1)

This bounded change implements EMAIL-M0-003 under the owner-authorized remaining-email closeout. No provider, model, retention, transaction, or ingestion behavior changes.

## Contract

Register these persistent families in the existing MetricsRegistry and expose them through the existing JSON/Prometheus monitoring paths:

| Family | Type | Labels | Meaning |
| --- | --- | --- | --- |
| email_ingestion_parse_total | counter | format, outcome | parsed message or failed parse event |
| email_ingestion_parse_seconds | histogram, seconds | format, outcome | duration of the corresponding parse event |
| email_ingestion_persist_total | counter | backend, outcome | completed email Media repository operation, success or error |
| email_ingestion_persist_seconds | histogram, seconds | backend, outcome | full repository operation duration, including failed validation/transaction exit |
| email_ingestion_dedupe_total | counter | backend | actual existing identity match; emitted once on the selected lookup/recheck branch |
| email_native_persist_total | counter | path_kind, outcome | existing normalized graph attempt result |
| email_native_persist_seconds | histogram, seconds | path_kind, outcome | matching normalized graph attempt duration |

Formats: eml, zip, mbox, pst, ost, other. Parse outcomes: parsed, error. Backends: sqlite, postgresql, other. Persistence outcomes: success, error. Native paths: primary, attachment_child, archive_child, other. Native outcomes: success, noop, error, skipped_flag. Unknown caller values collapse to other/error; only enums enter labels. Durations are nonnegative finite seconds. No identities, filenames, content, headers, queries, exceptions, or tenant labels are accepted.

Message parse successes are counted once in process_email_task. ZIP/MBOX delegation must not add container success counts. PST synthesized messages also delegate to process_email_task and must not emit a second success event. Container guards/extraction failures count failed parsing events under their format (the failure count is events, not necessarily rejected-message count). The process_email_task boundary records each delegated message once; parser policy tests verify disabled extraction and parse failures without invoking models.

Media operation successes include duplicate returns and overwrite updates. Dedupe matches describe observed identity matches, including attempts that later fail; match counters are not committed-ingestion counters. The repository decorator observes the operation after its transaction context exits, including transaction failures. Caller-owned enclosing transactions may later roll back; metrics describe this operation, not global commit counts. Native graph errors remain separately identifiable from Media successes. Instrumentation errors use fixed debug logs and cannot break ingestion.

## Alternatives

Generic ingestion metrics lack email identity outcomes. Inferring outcomes from returned English messages is brittle. Instrumenting lookup branches and the repository operation boundary gives explicit outcomes with existing monitoring and no new dependency.

## Stage 1: Specify observable behavior
**Goal**: Real SQLite and registry tests reproduce missing metrics.
**Success Criteria**: New tests fail for absent registered/emitted families.
**Tests**: success, repeat import, distinct IDs/same body, failed validation, rollback, non-email exclusion.
**Status**: Complete

## Stage 2: Register and emit bounded metrics
**Goal**: Add helper and repository/persistence integrations.
**Success Criteria**: No outcome guessing or sensitive labels; monitoring exports counters and histograms.
**Tests**: registry exports/reset, bounded label normalization, native success/failure/skipped behavior, duration validation.
**Status**: Complete

## Stage 3: Verify and hand off parser hooks
**Goal**: Focused regressions, Ruff, Bandit, and parent integration notes.
**Success Criteria**: Record commands/results; review diff; preserve existing archive transaction behavior.
**Tests**: new metric tests plus identity/archive/native persistence regressions.
**Status**: Complete

## Verification and parent integration

TDD first reproduced missing families/counters/timers (10 failed, 1 passed), then the initial helper suite passed 11 cases. Expanded email checks passed 77 tests, including 18 metrics cases, real primary/archive/attachment persistence, failing native graph operations, rollback, concurrent lookup recheck, telemetry failures, identity regressions, and offline multipart upload/search. Central registry/export regressions passed 32 tests. Fresh additional timing/search-registration checks are recorded in Backlog notes after execution.

The integrated parser hooks, repository operation timers, explicit dedupe branches and native persistence timers are implemented and covered by the combined core regression suite (328 passed, two optional PST fixture skips). Touched persistence lint findings were fixed under TASK-13376.5; the full touched Python scope passed Ruff and production/probe scope passed Bandit with zero findings and scanner errors.

Parse metrics belong at process_email_task, rather than parse_eml_bytes: the message task boundary captures decoding and guard failures consistently while archive and synthesized PST messages delegate once. Container failures emit format-specific error events; successful containers do not add a second success count.

A real SQLite search failure exposed a mismatched extra error_type label that prevented cumulative request-error lookups from matching the registered schema. Removing the extra label preserves the fixed phase/query_present/include_deleted contract. Its regression deliberately removes the required index, verifies the raised database error and observes exactly one request-error event; the focused query/metrics suite passed 11 tests.

Live authenticated HTTP observations and flag-reload evidence are recorded in the final core validation report and backend throughput artifacts. Metrics represent observed operations, not durable global commits; caller-owned outer transactions can still roll back after an operation was measured.

## Native search registration audit

Parent requested a bounded audit of native search emissions. The query helper already emits requests (attempt/success/parse_error/error), parse failures, result-count histograms, and duration histograms using only fixed phase and true/false labels. Their generic logger bridge currently auto-registers them as transient metrics without deliberate units/buckets, so reset can remove the definitions. Register these existing names persistently in MetricsRegistry, with seconds latency buckets and message-count result buckets. Real SQLite search and invalid-query tests verify exported observations, privacy, and reset stability. The existing search behavior is unchanged; its error labels now match the persistent schema.

### Final owned-scope verification (2026-09-26)

Native search registration TDD: real search then reset produced 1 expected failure and 5 passed tests; after persistent definitions, the final lightweight suite passed 69 tests. That suite includes 20 ingestion-metrics cases, 6 real search/reset metrics cases, and existing email query/cursor and registry-bridge tests. Logs: /tmp/email_metrics_search_red_13376.log and /tmp/email_metrics_search_green_13376.log. Ruff helper, registry, both new test modules passed. Fresh Bandit helper/registry scan: 0 findings and 0 scanner errors (/tmp/email_metrics_search_bandit_13376.json). git diff --check passed. No commits; shared persistence/repository files are parent-owned. No further tests run after this handoff to keep benchmark timings free of concurrent test workload.

The four persistent existing search names are email_native_search_requests_total (counter; phase/query_present/include_deleted), email_native_search_parse_failures_total (counter; query_present/include_deleted), email_native_search_duration_seconds (histogram; seconds), and email_native_search_results_total (histogram; matching-message counts despite the historical suffix). Search labels are generated as fixed phase and true/false values by existing query code. No tenant, query, message, or exception data enter these labels.
