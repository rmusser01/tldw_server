# Email Ingestion and Optional Gmail Sync Operations Runbook

Audience: Owner / maintainer
Status (2026-09-26): All core technical gates and cleanup pass for the local reference scope: SQLite then dedicated PostgreSQL at `880d690a93`. Owner approval remains unrecorded; optional live Gmail validation remains deferred.

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Email_Release_Checklist_and_Rollback.md`
- `Docs/Operations/Email_Core_Validation_2026-09-13.md`
- `Docs/Product/Email_Search_Benchmark_Protocol.md`
- `Docs/API-related/Email_Attachment_Policy_13376.md`
- `Docs/Operations/Email_Sensitive_Logging_Audit_13376_2_2026-09-26.md`
- `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`
- `Docs/Operations/Env_Vars.md`

## Scope

One owner executes validation and approves each rollout scope. Core file ingestion
and search have their own gate. Gmail is an optional connector, and its deferred
live validation does not block that gate. Use synthetic email for development and
validation. Do not access the owner's personal Gmail or send personal mail to an LLM.

## Core Offline Validation

`POST /api/v1/media/process-emails` processes EML and enabled containers without
persisting them. Use `POST /api/v1/media/add` with `media_type=email` to save mail for
`GET /api/v1/email/search` and `GET /api/v1/email/messages/{id}`. The existing media
search compatibility surface is `POST /api/v1/media/search`.

Use these explicit request options for the tested path:

```text
perform_analysis=false
perform_claims_extraction=false
perform_chunking=false
auto_chunking_use_llm=false
generate_embeddings=false
keep_original_file=false
extract_attachments=false
```

Omit URLs and model providers. Set `accept_archives=true` for ZIP or
`accept_mbox=true` for MBOX. Attachment descriptors are parsed automatically.
Explicit `extract_attachments=false` retains metadata only and overrides
`ingest_attachments=true`; an omitted extraction field inherits the legacy switch.
Extraction supports nested EML only, with `message/rfc822` and legacy `.eml`
filename inference selected by default. Explicit MIME allowlists use declared
types; deny rules take precedence. Unsupported binaries and PST/OST payloads
stay metadata-only, and existing depth/count/size guards remain. Skipped child
mail does not enter the parent's body. Extraction status describes capture or a
skip reason, while a child result separately describes processing success.
See `Docs/API-related/Email_Attachment_Policy_13376.md` for form syntax and limits.
Extraction never enables models, claims or embeddings; retain all offline flags
above even when explicitly opting into nested EML processing.

Core settings: `EMAIL_NATIVE_PERSIST_ENABLED=true`,
`EMAIL_OPERATOR_SEARCH_ENABLED=true`, `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`,
`EMAIL_GMAIL_CONNECTOR_ENABLED=false`. No connectors worker is needed. Keep it off
in an isolated validation environment; do not stop other connectors in a shared
installation merely to test email uploads.

Run the committed synthetic integration harness:

```bash
source .venv/bin/activate
TEST_MODE=true AUTO_DOWNLOAD_MODELS=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py -q
```

The harness builds synthetic MIME files, uses temporary SQLite and production
processing/persistence/search functions, and intercepts model entry points,
background embedding dispatch, shared HTTP helpers and socket connections/DNS.
It asserts zero attempted calls even if application error handling catches an
interception exception. It excludes full application startup and auth/billing/quota
integration. These limits apply to the evidence; request flags alone are not proof.

Record both successful paths and defects. TASK-13251 now preserves separate
identities for identical-body messages; TASK-13253 adds optional cursor pagination.
These local regression results do not close the deployment and scale checks.

TASK-13255 adds real API-key authentication and per-user SQLite isolation coverage
without overriding auth or DB dependencies. See
`Docs/Operations/Email_Authenticated_Validation_2026-09-13.md` for the test command,
negative credential checks, main-app test-mode routing evidence and remaining
startup/upload-quota/PostgreSQL limits.

TASK-13256 extends the local check through the production `/media/add` router
with real API-key scope, RBAC and organization storage-quota dependencies. Use
`Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md` for the
synthetic fixture, results and remaining deployment limits. The quota guard
returns 503 on setup/check errors by default; `STORAGE_QUOTA_FAIL_OPEN=1` is an
explicit override. An allowed upload near its storage limit returns
`X-Storage-Warning`; billing limit headers also reach the final response.
For a user in multiple organizations, pass the intended organization's
`X-TLDW-Org-Id` on upload and email search/detail requests; membership is
validated, and the selected organization is used for quota, billing and content.
Without an explicit selector, a validated JWT active organization takes priority
over the first membership. An org-scoped API key cannot select outside its scope.

## Local Reference Release Validation

TASK-13376 uses one loopback Uvicorn worker with synthetic users and mail,
validating SQLite first and then scoped PostgreSQL on the dedicated reference
service at loopback 5435 with 256 MiB shared memory. Shared5434 is preserved; its
earlier measurements are historical. Existing one-message and three-batch HTTP reports are bounded historical
evidence; use the release checklist's final core evidence table for current gates.

1. Sustain authenticated metadata-only archive uploads for at least 60 measured
   request seconds, record total wall time and batch/message counts, and meet
   50 messages/sec on each backend. Verify distinct identities, retry stability
   and another user's empty search/detail rejection.
2. Build an actual 1,000,000-message synthetic fixture using the guarded disposable
   bulk-loader path. Its setup rate is not ingestion throughput. Record dataset
   shape and all ten populated operator cases; apply the warm aggregate p50 <=
   250 ms / p95 <= 900 ms gate from the benchmark protocol. Report per-case
   diagnostics and cold-pass handle-reopen values. Pools may reuse physical
   connections; OS/filesystem and PostgreSQL server caches remain unchanged, so
   these measurements do not establish cache-cold performance. Before timing,
   verify legacy title/body FTS and native fixture parity; the incomplete earlier
   SQLite bulk-fixture result is superseded by fresh final runs.
3. Exercise search/detail/media search with the baseline flags above, compare
   complete legacy/native result ID sets, separately enable `auto_email`, and
   restore `opt_in` after comparison with explicit operator results.
4. Rehearse disabling native persistence/operator search, verify the disabled
   surfaces and retained legacy data, then restore flags and verify detail.
   The local probe reloads settings; a real deployment follows its process-restart
   procedure. Neither rehearsal certifies another deployment topology.
5. Export actual registry observations for parse/dedupe/persistence and search,
   and run the sensitive-log sentinel regressions. INFO-and-higher diagnostics
   must not expose bodies, headers, filenames, credentials or metadata through
   echoed exceptions or traceback locals; retain bounded stage/error-type context.
6. Preserve reports, source identity, guard/isolation and cleanup evidence, then
   record the single owner's separate release decision. Remove only resources
   owned by the disposable probes.

Model, background-work and non-loopback socket/DNS guards are required during
these probes; flags alone are not proof of offline behavior. PostgreSQL evidence
includes non-superuser/non-BYPASSRLS role checks and forced-RLS isolation. No
personal mailbox or live Gmail source is part of this procedure.

The verified Stage 1 snapshot is `b880be1b53`: 328 passing core cases with two
optional real-PST skips, 83 focused logging/pool/access-log cases, and subsequent
legacy-search sentinel/regression coverage. See the audit and core closeout
record for scope, remaining measured gates and final artifacts. A test snapshot
does not stand in for the sustained or million-message measurements above.
TASK-13377 subsequently runs both real-PST cases without skips using a pinned
public synthetic fixture and isolated optional parser; see
[real PST validation and reproduction](Email_Real_PST_Validation_2026-09-26.md).
Actual OST containers and live provider behavior remain unverified.

PostgreSQL tuning in `aac02e68e8` preserves substring predicates and forced RLS,
using bound label IDs, statement InitPlans, expression statistics and a native
multicolumn trigram index. The selected probe connections use
`PGOPTIONS='-c work_mem=64 MB'`; record observed `shared_buffers=128 MB`,
`hash_mem_multiplier=2` and `max_parallel_workers_per_gather=2`. No global service
configuration changed. Follow the benchmark protocol's fresh/reuse recipes and
retain any maintenance receipts. Short diagnostic samples and bounded archive
profiles are separate from the final search and sustained HTTP gates.

The final SQLite report records 1,000,000 fully validated legacy/native/indexed
messages, warm aggregate p50 **202.23 ms** / p95 **725.74 ms**, and sustained HTTP
ingestion of **130.54 messages/sec** (125.53 by wall time). Complete parity,
`auto_email`, flag rollback/restore, retry and isolated-user checks passed with
zero outbound/model attempts; exact generated roots are confirmed absent. The
PostgreSQL 120-message role/RLS/index/body smoke and normal append checks pass.
Its next full-scale attempt aborted during fixture parity before timing and produced no
new SLO result. Commit `1fee95629a` serializes only the body/version verification
transaction to fit Docker's 64 MiB POSIX shared memory; all checks remain intact
and two parallel workers / 64 MB are restored for timed search. The next full run
passed parity but failed repeated label COUNT. Native PostgreSQL search in
`455fad64fb` uses transaction-local custom planning with session settings restored on success/error;
parameters, predicates and RLS remain intact. The bounded comparison and 22-case
regression do not replace a complete final certificate. A generic-plan transition
is inferred from the preparation boundary; no post-failure generic counter was
observed. The earlier complete certificate at frozen `455fad64fb` passed at warm
aggregate **248.63 / 659.39 ms** over 200 samples, with all ten populated classes,
meaningful negation, complete million-row parity, 21 unchanged source hashes,
forced RLS and zero guards. Cold-handle diagnostics are **272.64 / 514.79 ms**;
pools/caches may remain warm. Generated million-message resources were removed.
The earlier HTTP attempt uploaded 1,700 messages over 60.9880 request seconds
at 27.87 messages/sec and failed legacy/native parity (0 versus 1,700). The
retained diagnosis and corrected 1,700/1/1 exact-ID parity remain historical.
Final legacy/SQL-cache/identity corrections are committed as `ae46f643ab`.
Its earlier shared-service HTTP rerun passed at **92.06 messages/sec** (89.91 by wall time), with
5,600 messages / 56 batches over **60.8291 request seconds / 62.2837 wall seconds**,
every batch >= 50. Exact retry 100, other search 0/detail 404, non-super/non-BYPASSRLS
forced-RLS scope, 5,600/1/1 legacy/native parity, `auto_email`, complete rollback
and restored flags/detail pass. Live parse/Media/native successes are 5,700 with
dedupe 100 and 195 native searches; guards are zero and 22 hashes unchanged.
The earlier `455fad64fb` search certificate is superseded for current-source
acceptance. Fresh shared-service search then aborted with POSIX shared-memory
SQLSTATE 53100 at 64 MiB, without a certificate. The final dedicated 256 MiB service
at `880d690a93` retains64 MB connection settings, production queries and forced RLS;
fresh search now passes warm **220.92/495.49 ms**, cold handles281.54/502.76 ms and
complete 1M shape/body/version/index/identity/isolation. Its full HTTP passes
**61.71 messages/sec** (60.83 by wall time), 3,800/38 batches over 61.5734 request /
62.4711 wall seconds. Three batches below 50 (minimum 38.96) are diagnostics; the
minute-wide aggregate gate passes. Exact 3,800/1/1 parity/retry 100/auto_email/
rollback/restore and live 3,900-success/dedupe 100/search 133 observations pass,
with 23 stable hashes and zero guards. All 55 exact roots/five scoped targets and
manifests/owned container-volume are removed, shared service preserved. Use the
core closeout report for final per-backend state rather than bounded older runs.

## Optional Gmail Validation — Deferred

Nothing below is a prerequisite for core file-upload validation. Before future
live work, explicitly authorize a dedicated synthetic test mailbox, audit all
downstream model processing, and confirm the intended environment. Personal Gmail
is excluded. Label filtering is not an OAuth permission boundary, and read-only
provider access does not constrain downstream LLM/embedding use.

Optional flags:
- `EMAIL_NATIVE_PERSIST_ENABLED=true`
- `EMAIL_OPERATOR_SEARCH_ENABLED=true`
- `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
- `EMAIL_GMAIL_CONNECTOR_ENABLED=true`
- `CONNECTORS_WORKER_ENABLED=true`

OAuth uses `CONNECTOR_GMAIL_CLIENT_ID` and `CONNECTOR_GMAIL_CLIENT_SECRET`.
A scaffold authorization response does not prove a connected account.

Preflight for explicitly authorized future live work:
- Verify API and worker reachability, OAuth completion and dedicated source identity.
- Confirm access to `GET /api/v1/jobs/list` and source status.
- Confirm downstream processing options before triggering sync.

### Live-source checklist

- [ ] List the dedicated Gmail source with `GET /api/v1/email/sources`.
- [ ] Trigger initial sync with `POST /api/v1/email/sources/{source_id}/sync`.
- [ ] Find the job using `GET /api/v1/jobs/list?domain=connectors&limit=50`.
- [ ] Confirm complete backfill against the synthetic source inventory and a healthy final source state.
- [ ] Repeat sync and verify no duplicates; apply synthetic message/label changes and verify deltas and cursor advancement.
- [ ] Validate invalid-cursor recovery and explicit full-backfill-required state under controlled conditions.
- [ ] Validate provider throttling/failure recovery; no sustained retrying/failed state should remain.
- [ ] Verify email search, message detail and `POST /api/v1/media/search`.
- [ ] Measure lag over a recorded monitoring window with real controlled-source samples.
- [ ] Record source/job IDs, observations, evidence and owner approval in the optional Gmail checklist.

Mocked Gmail tests verify code against simulated provider responses. The offline
`email_m2_gate_validation.py` fixtures verify checker behavior only. Its output now
labels `evidence_source=offline_fixture` and leaves staging unverified. Live endpoint
sampling is labeled separately and still requires a representative monitoring window.

## Connector Incident Playbooks

### Retry/backoff saturation

If source state remains `retrying` or jobs report `backoff_active` or
`retry_budget_exhausted`, inspect source state and recent connectors jobs. Correct
the auth/provider problem before retrying. Review `EMAIL_SYNC_RETRY_MAX_ATTEMPTS`,
`EMAIL_SYNC_RETRY_BASE_SECONDS` and `EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS`.

### Invalid cursor / replay recovery

Inspect cursor errors and bounded-replay/full-backfill-required state. Review
`EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS` and `EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES`.
Re-trigger one controlled source and verify continuity and cursor progression;
a healthy status alone does not establish complete recovery.

### Provider quota / throttling

Reduce concurrency/polling pressure, allow backoff to drain, and validate one-source
recovery before broad retries. Review `CONNECTORS_POLL_INTERVAL_SECONDS`. Use the
connector-specific rollback steps in the release checklist if recovery fails.

## Completion Criteria

Core technical evidence is complete when the selected local reference scope has
recorded synthetic correctness, metrics/logging/attachment checks, million-message
search, sustained HTTP ingestion and parity/flag/rollback results for both
backends, with guard/isolation and cleanup observations. Historical smaller runs
do not substitute for those gates. Core rollout then requires its own single-owner
release record; technical verification does not silently grant human sign-off.

Optional Gmail validation completes only after live provider behavior, data
completeness, recovery and monitoring requirements are validated on an explicitly
authorized synthetic test source. This gate remains deferred.
