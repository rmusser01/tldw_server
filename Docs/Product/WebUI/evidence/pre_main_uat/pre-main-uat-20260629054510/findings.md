# Findings

- Run id: `pre-main-uat-20260629054510`
- Task id: `TASK-12064`
- Status: Local single-user UAT passed after fixes

## Findings

| ID | Severity | Status | Summary | Evidence | Fix / disposition |
| --- | --- | --- | --- | --- | --- |
| UAT-LB-001 | P1 | Fixed | Local WebUI runtime auth was unavailable behind loopback forwarded metadata, causing the first local quickstart pass to route into unauthenticated setup/settings behavior. | `local-single-user.md`; focused Vitest red/green output. | Fixed by accepting loopback-only forwarded client metadata for runtime auth exposure while continuing to reject external forwarded values. |
| UAT-LB-002 | P1 | Fixed | Quick ingest media jobs stayed queued because route-backed lifecycle workers required a truthy env flag instead of inheriting the route gate when unset. The user-facing docs-info capability also checked the heavy media-ingest worker instead of the normal media-ingest worker. | `media-ingest-job-2-after-worker-fix.json`; `docs-info-after-worker-fix.json`; focused pytest output in `local-single-user.md`. | Fixed in `tldw_Server_API/app/services/lifecycle_worker_specs.py` and `tldw_Server_API/app/api/v1/endpoints/config_info.py`; explicit false env values still disable the worker. Verification: 45 focused tests passed, Bandit reported zero findings, and `git diff --check` passed. |
| UAT-LB-003 | Test blocker | Closed | In-app Browser automation could not continue because the local URL was blocked by Browser policy. | Browser tool rejection during local UAT continuation. | Not an app defect. Resolved by rerunning visual/mobile checks through a user-approved CDP-controlled Chromium session. |
| UAT-LB-004 | P3 | Fixed | Quickstart WebUI repeatedly requested same-origin `/openapi.json`, which returned HTTP 404 and generated console errors on home and chat routes. | `cdp-results.json` before fix showed repeated `http://127.0.0.1:8080/openapi.json` 404s. | Added `apps/tldw-frontend/pages/openapi.json.ts` to proxy the backend OpenAPI document and regression coverage in `__tests__/pages/openapi-json-proxy.test.ts`. Final CDP run reports `relevantConsoleEvents=[]`. |

## Verified Non-Issues / Skips

- The media search summary artifact does not contain the disposable tag because the list endpoint omits content from result items. The follow-up media detail artifact confirms the stored content contains `uat-basic-pre-main-uat-20260629054510`.
- The initial CDP build overlay for missing `antd` was a stale local Next dev process after dependency repair/rebase. Restarting WebUI from the rebased worktree cleared it; no repo code change was required.

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
| Desktop visual CDP | Home and chat rendered nonblank with no framework overlay. |
| Mobile visual CDP | Home and chat rendered at 390px with no framework overlay. |
| Browser console/network | Final CDP run reported no relevant console/network errors. |
