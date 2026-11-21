# Test Coverage And Verification Gaps Specialist Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Test coverage and verification gaps
- In scope: missing tests for high-risk paths, weak assertions, unverified domain claims, feasible targeted verification, and coverage-relevant domain findings.
- Out of scope: remediation implementation and broad test-suite rewrites.
- Review mode: report-only specialist pass using static inspection, existing audit evidence, and one focused local pytest slice. No production code, tests, configs, Backlog task files, index files, command logs, or other reports were edited.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

No new `AUDIT-2026-06-27-TESTS-NNN` findings were added. The material test and verification gaps found in this pass are already represented by normalized findings in `findings-index.json` or by first-batch specialist escalations `AUDIT-2026-06-27-APIWEB-001` and `AUDIT-2026-06-27-REL-001`.

## Index Mapping

Use finding IDs like `AUDIT-2026-06-27-TESTS-001`. Set `evidence_tier` from the report section bucket (`confirmed_issue`, `likely_risk`, or `improvement_opportunity`) and `evidence_strength` from the schema allowed values. Set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/test-coverage-verification.md`, set `owner_domain` to this report owner, and include `affected_paths`, `recommendation`, `status`, and `validation_status` in each detailed finding.

No new index rows are requested from this specialist pass.

Existing normalized findings confirmed for coverage follow-up:

- `AUDIT-2026-06-27-AUTH-001`
- `AUDIT-2026-06-27-AUTH-002`
- `AUDIT-2026-06-27-AUTH-003`
- `AUDIT-2026-06-27-DB-001`
- `AUDIT-2026-06-27-DB-002`
- `AUDIT-2026-06-27-WEBUI-001`
- `AUDIT-2026-06-27-WEBUI-002`
- `AUDIT-2026-06-27-OPS-001`
- `AUDIT-2026-06-27-OPS-003`
- `AUDIT-2026-06-27-OPS-004`
- `AUDIT-2026-06-27-MEDIA-001`
- `AUDIT-2026-06-27-MEDIA-002`
- `AUDIT-2026-06-27-MEDIA-003`
- `AUDIT-2026-06-27-MEDIA-004`
- `AUDIT-2026-06-27-CHAT-001`
- `AUDIT-2026-06-27-CHAT-002`
- `AUDIT-2026-06-27-JOBS-001`
- `AUDIT-2026-06-27-JOBS-002`
- `AUDIT-2026-06-27-INTEGRATIONS-001`
- `AUDIT-2026-06-27-INTEGRATIONS-002`
- `AUDIT-2026-06-27-INTEGRATIONS-003`
- `AUDIT-2026-06-27-MCP-001`
- `AUDIT-2026-06-27-MCP-002`

First-batch specialist findings needing the same test-follow-up track if they are accepted into the normalized index:

- `AUDIT-2026-06-27-APIWEB-001`
- `AUDIT-2026-06-27-REL-001`

## Confirmed Issues

No new confirmed issues were added by this specialist pass.

Confirmed existing issues with clear test or verification follow-up:

- `AUDIT-2026-06-27-AUTH-001`: Confirmed coverage gap. `test_admin_impersonation.py` asserts the response default `expires_in_minutes == 15` and verifies only extra JWT claims on the mocked service. It does not decode a minted token or assert `exp - iat`, so the token lifetime contract still needs a regression test.
- `AUDIT-2026-06-27-AUTH-002`: Confirmed coverage gap on a high-severity audit boundary. Existing impersonation tests verify claim creation but do not exercise request auth resolution or durable audit attribution after the impersonation token is used. Follow-up should assert both actor and subject survive into downstream audit context.
- `AUDIT-2026-06-27-DB-001`: Already runtime reproduced. The missing regression coverage is a package-directory upgrade test for a representative legacy SQLite Media DB below v22, plus a documented unsupported-version test if the intended behavior is explicit rejection.
- `AUDIT-2026-06-27-DB-002`: Already runtime reproduced. Add a failing multi-statement migration regression asserting that the first DDL statement is rolled back and that ledger plus `schema_version` updates are atomic.
- `AUDIT-2026-06-27-WEBUI-001`: Confirmed contract-test gap. The OpenAPI verifier passes while warning about reviewed billing exceptions, and the backend removal test asserts no OSS billing routes. Add frontend behavior coverage that hides or disables Billing in normal OSS multi-user mode unless a hosted billing capability is present.
- `AUDIT-2026-06-27-WEBUI-002`: Confirmed. Backend helper coverage proves query-token auth is rejected by default for `audio.stream.tts`, but the browser path still needs client-side first-frame auth coverage. This is broadened by `AUDIT-2026-06-27-APIWEB-001`.
- `AUDIT-2026-06-27-OPS-001`: Confirmed CI test gap. Published worker and audio-worker images need PR build/smoke coverage or an explicit release-surface removal.
- `AUDIT-2026-06-27-OPS-003`: Confirmed CI test gap. `actionlint` should be run against all workflows and composite actions, or excluded workflows should be named with a reason.
- `AUDIT-2026-06-27-OPS-004`: Confirmed release verification gap. SBOM validation needs Bun lockfile coverage for the frontend/admin workspaces and should fail when an expected ecosystem is skipped.
- `AUDIT-2026-06-27-MEDIA-001`: Confirmed high-severity coverage gap. Existing permission-denial coverage includes media ingest jobs, but the processing-only endpoint set needs no-permission HTTP tests asserting 403 for each user-media or remote-input route.
- `AUDIT-2026-06-27-MEDIA-002`: Confirmed high-severity tenant-isolation test gap. Existing MediaWiki DB persistence tests assert the current shared `managed_media_database(client_id="mediawiki_import", kwargs={})` behavior; add multi-user tests proving MediaWiki DB writes and vector writes are scoped to the request user.
- `AUDIT-2026-06-27-MEDIA-004`: Confirmed no-op test. `test_download_audio_rejects_when_content_length_exceeds_limit` builds a fake oversized response but never calls `download_audio_file` and has no assertion. A focused run still passes, so this test does not currently protect the header-declared size boundary.
- `AUDIT-2026-06-27-CHAT-002`: Confirmed log-redaction coverage gap. Add `caplog` or logger-stub tests for main and advanced RAG search paths that assert raw query text is absent and hash/length metadata remains.
- `AUDIT-2026-06-27-MCP-001`: Confirmed WebSocket AuthNZ test gap. Existing ACP tests reject read-only API keys for stream and SSH sockets; they do not cover scoped AuthNZ JWT claim rejection for ACP stream, ACP SSH, or sandbox run stream.
- `AUDIT-2026-06-27-MCP-002`: Confirmed lifecycle test gap. Existing reconnect tests start a `WSBroadcaster` directly and stop it manually; add endpoint-level reconnect-disconnect coverage that asserts broadcaster tasks and event-bus subscribers are cleaned up.
- `AUDIT-2026-06-27-APIWEB-001`: Confirmed specialist escalation. Frontend tests currently assert query-token voice-chat URLs in places, which locks in the broken default contract. Add client tests for STT, voice chat, and TTS that assert the first sent frame is `auth` before config, prompt, or audio frames.

## Likely Risks

No new likely-risk finding was added by this specialist pass.

Existing likely risks that should remain in targeted reproduction status:

- `AUDIT-2026-06-27-AUTH-003`: Needs PostgreSQL-backed or backend-agnostic placeholder verification. A focused test should fail when raw `pool.acquire()` paths send `?` markers to asyncpg, then pass after the endpoint uses `DatabasePool.fetchone()` or repository methods.
- `AUDIT-2026-06-27-MEDIA-003`: Needs a compensating-cleanup unit test. Use a fake storage backend whose `store()` succeeds and a DB double whose `insert_media_file()` raises, then assert `storage.delete(storage_path)` is called and cleanup failures are logged.
- `AUDIT-2026-06-27-CHAT-001`: Needs virtual-key and scoped-token HTTP reproduction for alternate RAG, character completion, chat-document generation, and embeddings routes. Existing quota tests cover `/api/v1/chat/completions` and `/api/v1/rag/search`, not the adjacent routes named in the finding.
- `AUDIT-2026-06-27-JOBS-001`: Needs process-loss or durable-owner reproduction for workflow rows created before in-process daemon-thread execution is accepted. A cheap unit-level substitute can inject a scheduler/engine failure after row creation and assert a repairable durable state.
- `AUDIT-2026-06-27-JOBS-002`: Needs duplicate schedule-fire tests. Existing workflow scheduler tests validate single-process fire/history and owner resolution; add a test that submits the same logical workflow or ACP schedule fire twice and asserts one Scheduler task or one workflow run by deterministic idempotency key.
- `AUDIT-2026-06-27-INTEGRATIONS-001`: Needs egress/proxy tests for workflow research adapters, especially direct `pdf_url` download. Existing tests cover test-mode behavior and sanitized backend errors, not private/loopback URL denial or `trust_env=False`.
- `AUDIT-2026-06-27-INTEGRATIONS-002`: Needs tokenizer resolver egress/proxy tests. Existing unit tests monkeypatch `_http_post` for payload behavior and host guard cases, but do not prove central HTTP client use, private URL denial, or environment proxy avoidance.
- `AUDIT-2026-06-27-REL-001`: Needs accepted-then-lost continuation tests. Existing workflow tests cover successful checkpoint resume and pre-schedule `_schedule_resume` exceptions; they do not cover `asyncio.create_task(...)` acceptance followed by task failure, cancellation, or process shutdown before execution.

## Improvement Opportunities

No new improvement-opportunity finding was added by this specialist pass.

Coverage improvements that would reduce audit uncertainty:

- Treat WebSocket handshakes as first-class contract surfaces. OpenAPI path verification cannot represent first-frame auth, query-token policy, scoped JWT claims, or socket capability differences; add shared backend/client contract fixtures for audio, ACP, sandbox, watchlist, persona, and prompt streaming paths over time.
- Extend frontend contract verification beyond `ClientPath`. The API/WebUI specialist found newer code paths using `AllowedPath`, `PathOrUrl`, and `toAllowedPath` directly. Sampled setup paths were valid, but verifier coverage should make bypasses visible.
- Add one CI job that statically inventories published artifacts and required validation gates. `AUDIT-2026-06-27-OPS-001`, `AUDIT-2026-06-27-OPS-003`, and `AUDIT-2026-06-27-OPS-004` are all preventable by comparing release matrices, workflow files, and dependency lockfiles against gate coverage.
- Keep `AUDIT-2026-06-27-INTEGRATIONS-003` as a low-severity central-client consistency follow-up: fixed OpenWeather URLs make SSRF risk narrow, but an API-key-bearing request should still inherit central proxy and timeout defaults or explicitly test equivalent behavior.
- Reconcile `AUDIT-2026-06-27-REL-001` with `AUDIT-2026-06-27-JOBS-001` during index finalization. If it is not accepted as a standalone normalized finding, expand `AUDIT-2026-06-27-JOBS-001` to include fire-and-forget continuation paths and research wait resume marking.

## Coverage And Evidence

### Files Inspected

Audit artifacts:

- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- All domain reports under `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/`
- First-batch specialist reports: `security-boundaries.md`, `reliability-lifecycle.md`, and `api-webui-contracts.md`
- Current scaffold state of `dependency-static-analysis.md`
- Evidence inventories: `backend-test-inventory.txt`, `frontend-api-client-inventory.txt`, `endpoint-inventory.txt`, `db-migration-inventory.txt`, `dependency-manifest-inventory.txt`, `ci-deploy-ops-inventory.txt`, and `bandit-app-summary.txt`
- Domain evidence files: `db-migrations-data-durability-reproductions.txt`, `webui-extension-api-contracts-static-evidence.txt`, `integrations-providers-static-evidence.txt`, and `ci-deployment-operations-release-candidates.txt`

Source and test paths line-checked or searched for coverage relevance:

- `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py`
- `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`
- `tldw_Server_API/tests/Audio/ws_test_helpers.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py`
- `tldw_Server_API/tests/Workflows/test_workflows_api.py`
- `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
- `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`
- `tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py`
- `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py`
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py`
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py`
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py`
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`
- `apps/packages/ui/src/services/__tests__/voice-conversation.test.ts`
- `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx`
- `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`
- Targeted frontend inventory hits for `SpeechPlaygroundPage.tsx`, `useVoiceChatStream.tsx`, `voice-conversation.ts`, and extension STT background paths.

### Tests Or Scans Run

Focused local verification:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py -q
```

Result: `10 passed, 29 warnings in 8.74s`.

Interpretation:

- The existing audio download limit suite still passes even though the header-declared oversized download test never invokes the function under test.
- The shared audio WebSocket auth helper has default query-token rejection coverage for `audio.stream.tts`, but this does not cover the frontend STT and voice-chat first-frame ordering gaps from `AUDIT-2026-06-27-APIWEB-001`.

Static inspection commands used included:

- `git status --short`
- `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 3 -type f | sort`
- `jq` summaries over `findings-index.json` for severity, category, validation status, affected paths, and recommendations
- `sed -n ...` over all domain and first-batch specialist reports
- `wc -l` on the main evidence inventories and scoped domain evidence files
- Targeted `rg` over backend test inventory, frontend API-client inventory, endpoint inventory, and relevant source/test paths
- Targeted `nl -ba ... | sed -n ...` reads over the test and frontend files listed above

### Blocked Or Unverified Areas

- No production code, tests, configs, Backlog task files, `findings-index.json`, `command-log.md`, `inventory.md`, or other reports were edited.
- No Docker, network access, dependency installation, service startup, live browser automation, live provider/API calls, or Kubernetes manifest application was performed.
- No full backend, frontend, or repository-wide suite was run.
- PostgreSQL-specific behavior for `AUDIT-2026-06-27-AUTH-003` remains unverified at runtime.
- Multi-user MediaWiki cross-tenant visibility for `AUDIT-2026-06-27-MEDIA-002` remains statically confirmed but not dynamically reproduced.
- Workflow process-loss, multi-process APScheduler duplicate fire, and post-`create_task` continuation failure remain unverified at runtime.
- `actionlint`, container builds, image inspections, and SBOM generation were not run locally; the operations findings are based on static CI/release evidence.
- The dependency/static-analysis specialist report was still scaffold-only when inspected, so this pass did not rely on it for completed first-batch conclusions.

### Evidence Notes

- No existing normalized finding was refuted by this pass.
- No separate `test-coverage-verification-*` evidence file was created; the report relies on the existing evidence inventories, domain evidence files, line-checked source/test paths, and the focused pytest run recorded above.
- Existing domain reports already recorded useful focused test evidence: DB migrations were runtime reproduced, WebUI OpenAPI verification was run, integrations provider tests passed while missing egress assertions, and MCP/ACP tests passed while missing scoped-JWT coverage.
- Current worktree status before editing showed only the two known unrelated untracked watchlist templates. The focused pytest run wrote `Databases/system_logs.jsonl`, but it is not reported by `git status --short`.
