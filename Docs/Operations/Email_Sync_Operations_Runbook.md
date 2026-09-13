# Email Ingestion and Optional Gmail Sync Operations Runbook

Audience: Owner / maintainer
Status (2026-09-13): Synthetic core validation recorded; optional live Gmail validation deferred.

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Email_Release_Checklist_and_Rollback.md`
- `Docs/Operations/Email_Core_Validation_2026-09-13.md`
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
```

Omit URLs and model providers. Set `accept_archives=true` for ZIP or
`accept_mbox=true` for MBOX. Attachment metadata is parsed automatically;
`ingest_attachments=true` additionally enables nested EML ingestion within
`max_depth`. It does not imply binary attachment extraction or storage.

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

Core evidence is complete when the synthetic tests and audit are recorded, with
remaining defects and untested release criteria explicit. Core rollout requires
its own owner-approved release checklist.

Optional Gmail validation completes only after live provider behavior, data
completeness, recovery and monitoring requirements are validated on an explicitly
authorized synthetic test source. This gate remains deferred.
