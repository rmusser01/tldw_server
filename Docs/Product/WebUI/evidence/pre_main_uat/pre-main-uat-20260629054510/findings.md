# Findings

- Run id: `pre-main-uat-20260629054510`
- Task id: `TASK-12064`
- Status: In Progress

## Findings

| ID | Severity | Status | Summary | Evidence | Fix / disposition |
| --- | --- | --- | --- | --- | --- |
| UAT-LB-001 | P1 | Fixed | Local WebUI runtime auth was unavailable behind loopback forwarded metadata, causing the first local quickstart pass to route into unauthenticated setup/settings behavior. | `local-single-user.md`; focused Vitest red/green output. | Fixed in commit `abe42060c1` by accepting loopback-only forwarded client metadata for runtime auth exposure while continuing to reject external forwarded values. |
| UAT-LB-002 | P1 | Fixed locally, pending commit | Quick ingest media jobs stayed queued because route-backed lifecycle workers required a truthy env flag instead of inheriting the route gate when unset. The user-facing docs-info capability also checked the heavy media-ingest worker instead of the normal media-ingest worker. | `media-ingest-job-2-after-worker-fix.json`; `docs-info-after-worker-fix.json`; focused pytest output in `local-single-user.md`. | Fixed in `tldw_Server_API/app/services/lifecycle_worker_specs.py` and `tldw_Server_API/app/api/v1/endpoints/config_info.py`; explicit false env values still disable the worker. Verification: 45 focused tests passed, Bandit reported zero findings, and `git diff --check` passed. |
| UAT-LB-003 | Test blocker | Open | Browser/mobile UI automation could not continue because the in-app Browser URL policy blocked local navigation for this workspace. | Browser tool rejection during local UAT continuation; API-level artifacts under `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/local/basic/`. | Not an app defect. Finish visual/mobile pass when Browser access to the local tab is available. |

## Verified Non-Issues / Skips

- The media search summary artifact does not contain the disposable tag because the list endpoint omits content from result items. The follow-up media detail artifact confirms the stored content contains `uat-basic-pre-main-uat-20260629054510`.
- Earlier observations about onboarding copy and a development-console `/openapi.json` 404 were not filed as findings in this update because current browser automation is blocked and they have not been reverified against the fixed runtime state.

## Passing Local API Checks

| Area | Result |
| --- | --- |
| Basic media ingest | Job `2` completed with `media_id=1`. |
| Media detail | Stored content contains `uat-basic-pre-main-uat-20260629054510`. |
| RAG search | HTTP 200 and response contains `uat-basic-pre-main-uat-20260629054510`. |
| OpenAI backend chat | HTTP 200 and response contains `uat-basic-pre-main-uat-20260629054510`. |
| llama.cpp backend chat | HTTP 200 and response contains `uat-basic-pre-main-uat-20260629054510`. |
| Character import | HTTP 201 for `UAT Character pre-main-uat-20260629054510`. |
| OpenAI roleplay chat | HTTP 200, saved response contains `pre-main-uat-20260629054510`. |
| llama.cpp roleplay chat | HTTP 200, saved response contains `pre-main-uat-20260629054510`. |
