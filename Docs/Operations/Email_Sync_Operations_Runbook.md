# Email Sync Operations Runbook

Audience: Maintainer / operator
Status: Ready for live-source validation

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Env_Vars.md`
- `tldw_Server_API/app/api/v1/endpoints/email.py`
- `tldw_Server_API/app/services/connectors_worker.py`

## Scope

This runbook covers the remaining live validation work for Gmail source sync in staging or a safe demo environment. It is written for a single-owner workflow: one maintainer can execute, verify, and sign off the entire checklist.

## Required Flags

- `EMAIL_NATIVE_PERSIST_ENABLED=true`
- `EMAIL_OPERATOR_SEARCH_ENABLED=true`
- `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
- `EMAIL_GMAIL_CONNECTOR_ENABLED=true`
- `CONNECTORS_WORKER_ENABLED=true`

Optional tuning knobs during incident triage:
- `EMAIL_SYNC_RETRY_MAX_ATTEMPTS`
- `EMAIL_SYNC_RETRY_BASE_SECONDS`
- `EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS`
- `EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS`
- `EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES`
- `CONNECTORS_POLL_INTERVAL_SECONDS`

## Preflight

- Confirm the API is running and reachable.
- Confirm the connectors worker is running.
- Confirm at least one Gmail account is connected and has a valid source.
- Confirm jobs admin access works for `GET /api/v1/jobs/list`.

## Live Validation Checklist

- [ ] Set the required email flags and restart the API / worker.
- [ ] List Gmail sources and confirm a live source exists:
  - `GET /api/v1/email/sources`
- [ ] Trigger a sync for one Gmail source:
  - `POST /api/v1/email/sources/{source_id}/sync`
- [ ] Verify a `connectors` job is created:
  - `GET /api/v1/jobs/list?domain=connectors&limit=50`
- [ ] Poll source status until it settles and confirm the source reaches `healthy`.
- [ ] Verify no sustained `retrying` or `failed` state remains for the validation source.
- [ ] Verify email search APIs remain healthy:
  - `GET /api/v1/email/search`
  - `GET /api/v1/email/messages/{id}`
- [ ] Verify existing `/api/v1/media/search` behavior remains healthy with delegation still at `opt_in`.
- [ ] Observe the validation source and job metrics for at least one monitoring window.

## Incident Classification

### Retry/backoff saturation

Indicators:
- source status remains `retrying`
- job results show `backoff_active` or `retry_budget_exhausted`

Actions:
- inspect the source state
- inspect recent `connectors` jobs
- correct the underlying auth/provider issue
- retry failed jobs only after the cause is understood

### Invalid cursor / replay recovery

Indicators:
- source error shows invalid or expired history cursor behavior
- sync falls back to bounded replay or explicit full-backfill-required state

Actions:
- verify cursor recovery settings
- narrow scope temporarily if needed
- re-trigger a single-source sync and confirm cursor progression

### Provider quota / throttling

Indicators:
- repeated failures under live provider load
- sync eventually succeeds only after backoff windows

Actions:
- reduce pressure
- allow backoff to drain
- validate a single-source recovery before broad retries

## Evidence To Record

Capture all of the following in the release checklist:
- validation date/time
- environment used
- Gmail source ID used
- job IDs created
- final source sync state
- any retries / incidents observed
- final outcome

## Exit Criteria

This runbook is complete when:
- one real Gmail source has synced successfully,
- source state is healthy,
- no blocking regressions are observed in search or media search behavior,
- evidence has been copied into the release checklist.
