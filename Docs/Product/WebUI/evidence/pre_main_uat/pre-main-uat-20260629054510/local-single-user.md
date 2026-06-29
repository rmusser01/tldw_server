# Local Single User

- Run id: `pre-main-uat-20260629054510`
- Task id: `TASK-12064`
- Backlog task: `TASK-12066`
- Basic UAT task: `TASK-12068`
- Status: Basic local single-user API pass after fixes; browser/mobile UI pass blocked by Browser policy

## Runtime

- API: `http://127.0.0.1:8000`, detached process listening on loopback.
- WebUI: `http://127.0.0.1:8080`, detached Next dev process listening on loopback.
- Runtime config: `GET /api/_tldw-webui/runtime-config` returned HTTP 200 with `runtimeAuth.available=true`, `authMode=single-user`, and an API key present. The key was not logged.
- Backend proxy: `GET /api/v1/setup/readiness/status` through the WebUI origin returned HTTP 200.
- Browser: an earlier local browser check rendered the first-time setup screen with setup readiness lanes and setup path choices. Later Browser automation attempts were blocked by the in-app Browser URL policy for local navigation, so the remaining UI/mobile checks were not completed in this pass. This is recorded as an automation limitation, not an application failure.

## Runtime Auth Finding Fixed

- UAT exposed a quickstart bootstrap bug in `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`.
- The route rejected any forwarded metadata before exposing runtime auth. Next dev sends loopback forwarding metadata on local requests, so the WebUI stayed unauthenticated and routed to `/settings/tldw` with repeated 401s from readiness and notification requests.
- Fixed by allowing only loopback-only `Forwarded` and `x-forwarded-for` values, ignoring `x-forwarded-host` for the exposure decision, and continuing to reject external/empty forwarded client IP values and `x-real-ip`.
- Added regression coverage in `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`.

## Runtime Auth Verification

- Red test observed before implementation:
  - `bunx vitest run __tests__/pages/api/runtime-config.test.ts`
  - Failed on loopback forwarded metadata being rejected.
- Final focused tests:
  - `bunx vitest run __tests__/pages/api/runtime-config.test.ts __tests__/extension/runtime-bootstrap.test.ts`
  - Result: 2 files passed, 67 tests passed.
- Final local probes:
  - `GET http://127.0.0.1:8080/api/_tldw-webui/runtime-config` returned HTTP 200 with runtime auth available.
  - `GET http://127.0.0.1:8080/api/v1/setup/readiness/status` returned HTTP 200 through the WebUI proxy.
- Final browser check:
  - Page URL: `http://127.0.0.1:8080/`
  - Visible state: first-time setup, setup readiness, and setup path buttons.
  - Console: no auth/readiness errors after the fix.

## Basic User Journey

Raw artifacts are under `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/local/basic/`.

### Document Ingest

- Flow: WebUI quick ingest submitted a disposable Markdown source containing `uat-basic-pre-main-uat-20260629054510`.
- Initial result: the submitted media ingest job stayed `queued` because the normal media ingest worker was not running.
- Verified root cause in current code:
  - `route_enabled_predicate()` required `MEDIA_INGEST_JOBS_WORKER_ENABLED` to be truthy, so an unset env var disabled route-backed workers even when the route itself was enabled.
  - `/api/v1/config/docs-info` reported `hasMediaIngestWorker` from the heavy media-ingest worker path instead of the normal media-ingest worker path.
- Fix applied:
  - An unset worker env flag now inherits the route gate.
  - Explicit false values such as `0` still disable the worker.
  - `hasMediaIngestWorker` now reports the normal media ingest worker path.
- Post-fix API capability artifact: `docs-info-after-worker-fix.json` reports `hasMediaIngestJobs=true`, `hasMediaIngestJobEvents=true`, and `hasMediaIngestWorker=true`.
- Post-fix job artifact: `media-ingest-job-2-after-worker-fix.json` reports job `2` as `completed`, `progress_percent=100`, `media_id=1`, and `db_message="Media 'basic-user-source' added."`.

### Basic Search And Answer

| Check | Artifact | Result |
| --- | --- | --- |
| Media search | `media-search.json` | HTTP 200, one matching item. The list response does not include body content, so the tag is not present in this artifact. |
| Media detail | `media-detail.json` | HTTP 200, stored content contains `uat-basic-pre-main-uat-20260629054510`. |
| RAG search | `rag-search.json` | HTTP 200, response contains `uat-basic-pre-main-uat-20260629054510`. |
| Backend OpenAI chat | `chat-openai.json` | HTTP 200, response contains `uat-basic-pre-main-uat-20260629054510`. |
| Backend llama.cpp chat | `chat-llamacpp.json` | HTTP 200, response contains `uat-basic-pre-main-uat-20260629054510`. |

### Roleplay Character Chat

| Check | Artifact | Result |
| --- | --- | --- |
| Character import | `roleplay-character-import.json` | HTTP 201, imported `UAT Character pre-main-uat-20260629054510` as character id `3`. |
| Chat creation | `roleplay-chat-create.json` | HTTP 201, created character chat `00e95188-8a73-4a4a-984a-b10302377263`. |
| OpenAI character turn | `roleplay-chat-openai.json` | HTTP 200, provider `openai`, model `gpt-4o-mini`, saved response contains `pre-main-uat-20260629054510`. |
| llama.cpp character turn | `roleplay-chat-llamacpp.json` | HTTP 200, provider `llama.cpp`, model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`, saved response contains `pre-main-uat-20260629054510`. |

## Backend Fix Verification

- Red tests observed before implementation:
  - `tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py::test_route_enabled_predicate_inherits_route_gate_when_env_unset`
  - `tldw_Server_API/tests/Services/test_lifecycle_worker_specs.py::test_route_enabled_predicate_forwards_route_and_kwargs_when_env_unset`
  - `tldw_Server_API/tests/Config/test_docs_info_capabilities.py::test_docs_info_exposes_bulk_conference_ingest_capabilities`
  - `tldw_Server_API/tests/Config/test_docs_info_capabilities.py::test_docs_info_media_ingest_worker_capability_respects_env_disable`
- Initial green focused verification:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py::test_route_enabled_predicate_inherits_route_gate_when_env_unset tldw_Server_API/tests/Services/test_lifecycle_worker_specs.py::test_route_enabled_predicate_forwards_route_and_kwargs_when_env_unset tldw_Server_API/tests/Config/test_docs_info_capabilities.py::test_docs_info_exposes_bulk_conference_ingest_capabilities tldw_Server_API/tests/Config/test_docs_info_capabilities.py::test_docs_info_media_ingest_worker_capability_respects_env_disable -q`
  - Result: `4 passed, 4 warnings in 7.52s`.
- Broader focused verification:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py tldw_Server_API/tests/Services/test_lifecycle_worker_specs.py tldw_Server_API/tests/Config/test_docs_info_capabilities.py -q`
  - Result: `45 passed, 4 warnings in 8.09s`.
- Security and diff checks:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/services/lifecycle_worker_specs.py tldw_Server_API/app/api/v1/endpoints/config_info.py -f json -o /tmp/bandit_uat_worker.json`
  - Result: exit `0`; `/tmp/bandit_uat_worker.json` contains `results_count=0`.
  - `git diff --check`
  - Result: exit `0`.

## Notes

- The temporary UAT launcher must parse `export NAME=value` lines in the run-scoped `uat.env`; an earlier local restart missed the API key because the parser treated `export UAT_API_KEY` as the variable name.
- `bun install --frozen-lockfile` in `apps/` repaired a stale local `apps/packages/ui/node_modules/antd` symlink needed for the WebUI dev server to resolve shared UI imports. This dependency-state repair is left unstaged and is not part of the product fix.
- The OpenAI key was loaded from the local UAT env and was not printed or stored in artifacts.
- Browser/mobile visual checks remain pending until local Browser navigation is available for this workspace.
