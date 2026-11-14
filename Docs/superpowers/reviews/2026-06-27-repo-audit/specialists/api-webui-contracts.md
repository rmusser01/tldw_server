# API And WebUI Contract Drift Specialist Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: API and WebUI contract drift
- In scope: endpoint/client mismatch, auth/setup/upload/streaming/job status flows, error recovery, schema drift, and contract-relevant domain findings.
- Out of scope: remediation implementation and visual redesign.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-APIWEB-001 | confirmed_issue | static_confirmed | medium | high | api_contract | Audio WebSocket query-token drift extends beyond Speech playground TTS to STT and voice chat | open | validated |

## Index Mapping

New specialist finding details for index ingestion:

- `id`: `AUDIT-2026-06-27-APIWEB-001`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/api-webui-contracts.md`
- `owner_domain`: `API and WebUI contract drift`
- `affected_paths`: `apps/packages/ui/src/entries/background.ts`, `apps/packages/ui/src/hooks/useTldwStt.ts`, `apps/packages/ui/src/services/tldw/voice-conversation.ts`, `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`, `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`, `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`, `tldw_Server_API/app/core/Audio/streaming_service.py`, `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`, `tldw_Server_API/tests/Audio/ws_test_helpers.py`
- `recommendation`: Replace browser audio WebSocket URL builders with a shared helper that opens the bare route and sends the backend-supported initial auth frame before config, prompt, or audio frames; add client and backend contract coverage for default query-token-disabled behavior on STT, voice chat, and TTS.
- `status`: `open`
- `validation_status`: `validated`

## Confirmed Issues

### AUDIT-2026-06-27-APIWEB-001: Audio WebSocket query-token drift extends beyond Speech playground TTS to STT and voice chat

- **Severity**: medium
- **Confidence**: high
- **Category**: api_contract
- **Evidence Tier**: confirmed_issue
- **Evidence Strength**: static_confirmed
- **Owner Domain**: API and WebUI contract drift
- **Source Report**: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/api-webui-contracts.md`
- **Status**: open
- **Validation Status**: validated
- **Escalates Existing Finding**: `AUDIT-2026-06-27-WEBUI-002`
- **Affected Paths**:
  - `apps/packages/ui/src/entries/background.ts`
  - `apps/packages/ui/src/hooks/useTldwStt.ts`
  - `apps/packages/ui/src/services/tldw/voice-conversation.ts`
  - `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`
  - `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
  - `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
  - `tldw_Server_API/app/core/Audio/streaming_service.py`
  - `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`
  - `tldw_Server_API/tests/Audio/ws_test_helpers.py`
- **Evidence**:
  - `AUDIT-2026-06-27-WEBUI-002` already proves the Speech playground opens `/api/v1/audio/stream/tts?token=...` and sends a `prompt` frame first, while the backend accepts query-token auth only when `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH` is enabled.
  - The extension STT bridge has the same default-contract mismatch: `useTldwStt` connects to the background `tldw:stt` port (`apps/packages/ui/src/hooks/useTldwStt.ts:18`, `:24`), and the background script opens `/api/v1/audio/stream/transcribe?token=${encodeURIComponent(token)}` without sending an auth frame before audio frames (`apps/packages/ui/src/entries/background.ts:2802`, `:2817`, `:2818`, `:2852`).
  - The voice-chat path also builds `/api/v1/audio/chat/stream?token=${encodeURIComponent(token)}` (`apps/packages/ui/src/services/tldw/voice-conversation.ts:360`) and opens it directly (`apps/packages/ui/src/hooks/useVoiceChatStream.tsx:476`). On open, the first structured message is `type: "config"`, not `type: "auth"` (`apps/packages/ui/src/hooks/useVoiceChatStream.tsx:512`, `:571`, `:573`).
  - The backend documents the same auth contract for all three affected audio WebSocket routes: `/stream/transcribe` says query-token auth is legacy and disabled by default (`tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py:864`, `:871`, `:874`, `:961`); `/chat/stream` says the same (`audio_streaming.py:1546`, `:1557`, `:1560`, `:1618`); `/stream/tts` says the same (`audio_streaming.py:2955`, `:2963`, `:2966`, `:3016`).
  - The shared auth helper gates query-string tokens behind `_allow_query_token_auth()`, whose default is false (`tldw_Server_API/app/core/Audio/streaming_service.py:92`, `:524`, `:526`, `:618`, `:697`). When that flag is absent, single-user mode falls through to an initial auth-message read (`streaming_service.py:719`), so query-token-only browser clients fail rather than authenticate.
  - Existing backend tests prove the default policy at the shared helper level for `audio.stream.tts` (`tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py:37`, `:56`, `:67`, `:87`). Several audio WebSocket tests that use `?token=` explicitly enable the legacy fallback through `ws_test_helpers.py:12-14`, which means those tests do not cover the browser clients' default production contract.
- **Impact**: Default deployments can break multiple browser audio experiences, not only the Speech playground TTS path already captured by `AUDIT-2026-06-27-WEBUI-002`: extension STT, live voice chat, and Speech playground streaming TTS all depend on query-token-only WebSocket authentication. Enabling the legacy backend flag restores those clients but reintroduces token-in-URL exposure that the backend intentionally disabled.
- **Recommendation**: Replace the browser audio WebSocket URL builders with a shared audio WebSocket auth helper that opens the bare route and sends the backend-supported initial `{"type":"auth","token":"..."}` frame before config, prompt, or audio frames. Add frontend/unit coverage for STT, voice chat, and TTS first-frame ordering. Add backend or contract tests that exercise the default `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH=0` policy for `/stream/transcribe`, `/chat/stream`, and `/stream/tts`.

### Existing Normalized Findings Confirmed Or Escalated

- `AUDIT-2026-06-27-WEBUI-001` confirmed. The local OpenAPI verifier still allows 10 reviewed exceptions, including all eight `/api/v1/billing/*` paths, while the settings UI renders and calls billing routes based only on multi-user login state. The backend billing removal test still asserts no public OSS billing routes.
- `AUDIT-2026-06-27-WEBUI-002` confirmed and escalated by `AUDIT-2026-06-27-APIWEB-001`. The same backend audio WebSocket auth contract mismatch affects STT and voice chat clients in addition to Speech playground TTS.
- `AUDIT-2026-06-27-MCP-001` confirmed as contract-relevant. ACP and sandbox WebSocket auth helpers accept JWTs outside the normal HTTP scoped-token enforcement path; sampled frontend ACP clients use the currently documented query-token contract, so the issue remains an AuthNZ WebSocket scope contract problem rather than a client path typo.
- `AUDIT-2026-06-27-MEDIA-001` confirmed as contract-relevant. The WebUI quick-ingest process-only paths call `/api/v1/media/process-web-scraping` and `/api/v1/media/process-documents`, which aligns with existing client expectations but reaches the processing-only endpoint set that the Media domain found lacks the `media.create` gate.
- `AUDIT-2026-06-27-CHAT-001` confirmed as contract-relevant. Adjacent RAG and chat generation routes expose similar provider-spending semantics with inconsistent token-scope/max-call enforcement, which is an API authorization contract drift across related client-callable surfaces.
- `AUDIT-2026-06-27-AUTH-001` confirmed as contract-relevant. The impersonation endpoint response advertises a 15 minute TTL while token creation uses the normal access-token lifetime, creating response schema/behavior drift.

## Likely Risks

No new specialist-specific likely-risk finding was added.

Existing likely-risk findings that deserve contract follow-up:

- `AUDIT-2026-06-27-AUTH-003`: PostgreSQL-backed impersonation can fail because raw connection calls use SQLite placeholders, which would violate the endpoint's advertised runtime contract in a supported DB mode.
- `AUDIT-2026-06-27-JOBS-002`: recurring workflow and ACP schedule submissions lack deterministic idempotency keys, so schedule API status can diverge from logical run semantics under duplicate fires.
- `AUDIT-2026-06-27-MEDIA-003`: original-file storage can orphan permanent files after a DB registration failure, which affects upload response/error recovery semantics.

## Improvement Opportunities

No standalone new improvement finding was added.

Recommended contract-maintenance improvements:

- Tighten the OpenAPI verifier path policy around `AUDIT-2026-06-27-WEBUI-001`. The verifier currently succeeds while warning about known missing billing paths. That is acceptable for documented hosted-only paths only if the shared OSS UI does not render or call those paths in normal self-hosted mode.
- Extend contract verification beyond `ClientPath` entries. Several frontend call sites use `AllowedPath`, `PathOrUrl`, or `toAllowedPath` directly, and audio WebSocket URLs are not represented in `ClientPath`. The setup admin audio installer paths were statically checked and do exist on the backend, so this is a guard-coverage gap rather than a runtime mismatch in that sampled area.
- Add a small WebSocket contract fixture that checks first-frame auth behavior for browser clients. The current OpenAPI verifier cannot represent WebSocket handshakes, and existing audio WebSocket backend tests commonly enable `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH=1` through test helpers.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/jobs-scheduler-workflows.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/frontend-api-client-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/webui-extension-api-contracts-static-evidence.txt`
- `apps/extension/scripts/verify-openapi-client-paths.mjs`
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
- `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- `apps/packages/ui/src/services/tldw/voice-conversation.ts`
- `apps/packages/ui/src/services/acp/client.ts`
- `apps/packages/ui/src/services/background-proxy.ts`
- `apps/packages/ui/src/entries/background.ts`
- `apps/packages/ui/src/hooks/useTldwStt.ts`
- `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`
- `apps/packages/ui/src/hooks/useACPSession.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Settings/tldw.tsx`
- `apps/packages/ui/src/components/Option/Settings/tldw-settings-tabs.tsx`
- `apps/packages/ui/src/components/Option/Setup/hooks/useAudioInstaller.ts`
- `apps/packages/ui/src/services/tldw/setup-readiness.ts`
- `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
- `apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx`
- `apps/packages/voice-assistant-sdk/src/client/VoiceAssistantClient.ts`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- `tldw_Server_API/app/core/Audio/streaming_service.py`
- `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`
- `tldw_Server_API/tests/Audio/ws_test_helpers.py`
- `tldw_Server_API/tests/Billing/test_billing_public_api_removed.py`
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- `tldw_Server_API/app/api/v1/endpoints/media/add.py`
- `tldw_Server_API/app/api/v1/endpoints/setup.py`
- `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/auth.py`
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- `tldw_Server_API/app/api/v1/endpoints/sandbox.py`

### Tests Or Scans Run

- `node ../../extension/scripts/verify-openapi-client-paths.mjs` from `apps/packages/ui`
  - Result: passed.
  - Verified 303 `ClientPath` entries and 49 `MEDIA_ADD_SCHEMA_FALLBACK` fields against generated OpenAPI from `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python`.
  - Warned that 10 reviewed exception paths are outside the current OSS OpenAPI contract, including all eight public billing paths.
- Static inspections with `rg`, `sed`, and `nl -ba` over the files listed above.

### Blocked Or Unverified Areas

- No production code, tests, configs, Backlog files, index files, command logs, or domain/specialist reports other than this report were edited.
- No network access, dependency installation, Docker, service startup, or browser automation was used.
- No live browser/server reproduction was performed for Billing, STT, voice chat, TTS, ACP, sandbox, setup, upload, or job-status flows.
- Full frontend and backend test suites were not run. This was a report-only static specialist pass plus the local OpenAPI verifier.
- The frontend surface is large and many newer modules use `AllowedPath`/`toAllowedPath`; the pass sampled high-risk contract areas instead of exhaustively proving every frontend path against OpenAPI.
- The two unrelated untracked watchlist template files present in the worktree were not touched.

### Evidence Notes

- `AUDIT-2026-06-27-APIWEB-001` is intentionally recorded as an escalation of `AUDIT-2026-06-27-WEBUI-002`, not a duplicate of the Speech playground TTS finding. It adds two additional default-broken client flows: extension STT and voice chat.
- The voice assistant SDK was checked as a comparator. It appends a query `token`, but also sends an initial `auth` message after connection (`apps/packages/voice-assistant-sdk/src/client/VoiceAssistantClient.ts:224`, `:274`), so it does not have the same default-auth failure as the STT, voice chat, and Speech playground TTS paths.
- Setup first-run and setup-readiness paths sampled in the UI are represented in `ClientPath` and exist on the backend. Admin audio installer paths exist on the backend but bypass `ClientPath` verification through `toAllowedPath`, making them useful evidence for verifier coverage limits rather than a runtime mismatch.
- Media ingest job submission/status paths sampled in the quick-ingest tests align with `/api/v1/media/ingest/jobs` and `/api/v1/media/ingest/jobs/{job_id}`. Job status endpoints enforce owner/admin checks in the backend sample. The contract-relevant media concern remains the existing processing-only permission gap in `AUDIT-2026-06-27-MEDIA-001`.
- No normalized finding was refuted in this specialist pass.
