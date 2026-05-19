# Email Release Checklist and Rollback

Audience: Owner / maintainer
Status: Pending live Gmail validation

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Email_Sync_Operations_Runbook.md`
- `Docs/Operations/Env_Vars.md`

## Single-Owner Workflow

This checklist assumes one maintainer owns implementation, operations, and release approval. References to backend, ops, and product are collapsed into a single owner sign-off.

## Release Checklist

- [ ] Offline evidence is up to date:
  - endpoint and worker regression tests
  - offline lag validation
  - search parity / benchmark artifacts if being used for release evidence
- [ ] Required flags are configured for validation:
  - `EMAIL_NATIVE_PERSIST_ENABLED=true`
  - `EMAIL_OPERATOR_SEARCH_ENABLED=true`
  - `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
  - `EMAIL_GMAIL_CONNECTOR_ENABLED=true`
  - `CONNECTORS_WORKER_ENABLED=true`
- [ ] A live Gmail account is connected and at least one Gmail source exists.
- [ ] The live-source validation checklist in `Docs/Operations/Email_Sync_Operations_Runbook.md` has been executed.
- [ ] `GET /api/v1/email/search` remains healthy.
- [ ] `GET /api/v1/email/messages/{id}` remains healthy.
- [ ] Existing `/api/v1/media/search` behavior remains healthy.
- [ ] If delegation promotion is in scope, `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=auto_email` has been validated separately.
- [ ] Release evidence has been written below.

## Release Evidence

- Validation date:
- Environment:
- Gmail source ID:
- Job ID(s):
- Final source state:
- Notes:

## Rollback Triggers

Rollback immediately if any of the following are observed during live validation or release:
- sustained source `failed` or `retrying` state without recovery
- repeated invalid cursor escalation without successful bounded recovery
- user-visible regression in email search or `/api/v1/media/search`
- provider quota / throttling severe enough to make the rollout non-viable

## Rollback Steps

1. Set `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`.
2. If needed, set `EMAIL_GMAIL_CONNECTOR_ENABLED=false`.
3. If needed, set `CONNECTORS_WORKER_ENABLED=false`.
4. Restart affected API / worker processes.
5. Verify legacy email search and media search behavior is stable.
6. Record rollback reason and outcome in this file.

## Owner Sign-off

- [ ] Owner release sign-off
- Name:
- Date:
- Notes:
