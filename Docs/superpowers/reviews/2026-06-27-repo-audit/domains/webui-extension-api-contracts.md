# WebUI, Extension, And API Contracts Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: WebUI, Extension, and API Contracts
- In scope: frontend API usage, auth persistence, setup flows, uploads, streaming, background jobs, error recovery, WebUI/backend contract drift, extension handoff paths, and dependency risk.
- Out of scope: remediation implementation and visual redesign.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-WEBUI-001 | confirmed_issue | static_confirmed | medium | high | api_contract | Billing settings call public billing routes that the OSS API intentionally omits | open | validated |
| AUDIT-2026-06-27-WEBUI-002 | confirmed_issue | test_reproduced | medium | high | api_contract | Speech playground TTS streaming uses query-token WebSocket auth rejected by default | open | validated |

## Index Mapping

Use finding IDs like `AUDIT-2026-06-27-WEBUI-001`. Set `evidence_tier` from the report section bucket (`confirmed_issue`, `likely_risk`, or `improvement_opportunity`) and `evidence_strength` from the schema allowed values. Set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`, set `owner_domain` to this report owner, and include `affected_paths`, `recommendation`, `status`, and `validation_status` in each detailed finding.

## Confirmed Issues

### AUDIT-2026-06-27-WEBUI-001 / CANDIDATE-webui-extension-api-contracts-001: Billing settings call public billing routes that the OSS API intentionally omits

- **Severity**: medium
- **Confidence**: high
- **Category**: api_contract
- **Evidence Tier**: confirmed_issue
- **Evidence Strength**: static_confirmed
- **Owner Domain**: WebUI, Extension, and API Contracts
- **Source Report**: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`
- **Status**: open
- **Validation Status**: validated
- **Affected Paths**:
  - `apps/packages/ui/src/components/Option/Settings/tldw.tsx`
  - `apps/packages/ui/src/components/Option/Settings/tldw-settings-tabs.tsx`
  - `apps/extension/scripts/verify-openapi-client-paths.mjs`
  - `tldw_Server_API/tests/Billing/test_billing_public_api_removed.py`
- **Evidence**:
  - The settings page loads billing data whenever `authMode === 'multi-user' && isLoggedIn`, then calls `/api/v1/billing/plans`, `/subscription`, `/usage`, and `/invoices` without a hosted-deployment or capability guard (`apps/packages/ui/src/components/Option/Settings/tldw.tsx:214`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:223`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:227`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:231`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:302`).
  - Billing actions call `/api/v1/billing/checkout`, `/portal`, `/subscription/cancel`, and `/subscription/resume` from the same OSS-shared UI surface (`apps/packages/ui/src/components/Option/Settings/tldw.tsx:621`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:648`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:670`, `apps/packages/ui/src/components/Option/Settings/tldw.tsx:696`).
  - The billing panel and tab are displayed solely by multi-user login state (`apps/packages/ui/src/components/Option/Settings/tldw.tsx:885`, `apps/packages/ui/src/components/Option/Settings/tldw-settings-tabs.tsx:21`, `apps/packages/ui/src/components/Option/Settings/tldw-settings-tabs.tsx:102`).
  - The backend test asserts the public OSS app registers no `/api/v1/billing` routes (`tldw_Server_API/tests/Billing/test_billing_public_api_removed.py:14`), and the OpenAPI verifier lists every billing path as a reviewed exception outside the OSS contract (`apps/extension/scripts/verify-openapi-client-paths.mjs:41`).
  - Local verifier output: `node ../../extension/scripts/verify-openapi-client-paths.mjs` succeeded but warned that all eight billing paths are intentionally absent from the current OSS OpenAPI contract.
- **Impact**: A self-hosted or OSS multi-user extension/WebUI user can be shown a Billing tab and trigger requests that the paired backend intentionally does not serve. The result is a deterministic client/backend contract failure rather than a recoverable server outage.
- **Recommendation**: Hide or disable the billing tab and billing loaders/actions unless a hosted billing capability is advertised by the backend or `isHostedTldwDeployment()` is true. Keep hosted-only billing paths out of the OSS `ClientPath` contract, or isolate them behind a hosted-only client adapter so the OpenAPI verifier can fail unexpected public-route drift.

### AUDIT-2026-06-27-WEBUI-002 / CANDIDATE-webui-extension-api-contracts-002: Speech playground TTS streaming uses query-token WebSocket auth rejected by default

- **Severity**: medium
- **Confidence**: high
- **Category**: api_contract
- **Evidence Tier**: confirmed_issue
- **Evidence Strength**: test_reproduced
- **Owner Domain**: WebUI, Extension, and API Contracts
- **Source Report**: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`
- **Status**: open
- **Validation Status**: validated
- **Affected Paths**:
  - `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
  - `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
  - `tldw_Server_API/app/core/Audio/streaming_service.py`
  - `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`
- **Evidence**:
  - The Speech playground builds a WebSocket URL as `/api/v1/audio/stream/tts?token=${encodeURIComponent(token)}`, opens it directly, and sends the first frame as a `prompt` payload rather than an auth frame (`apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx:1475`, `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx:1487`, `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx:1488`, `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx:1526`, `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx:1538`).
  - The backend TTS WebSocket contract says query-string token auth is legacy and accepted only when `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH` is enabled; otherwise clients must use headers or an initial auth message (`tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py:2955`, `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py:2963`, `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py:3016`).
  - Core WebSocket auth disables query-token auth by default because URLs are captured in logs and browser history (`tldw_Server_API/app/core/Audio/streaming_service.py:524`) and only reads the query `token` when the explicit allow flag is enabled (`tldw_Server_API/app/core/Audio/streaming_service.py:526`, `tldw_Server_API/app/core/Audio/streaming_service.py:618`).
  - The backend unit test reproduces the default policy for this exact endpoint, asserting that `?token=single-user-secret` is rejected for `audio.stream.tts` unless `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH=1` is set (`tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py:37`, `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py:56`, `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py:67`).
- **Impact**: Streaming TTS from the Speech playground fails against the default backend auth policy in both single-user and multi-user modes unless the deployment enables the legacy query-token fallback. Enabling that fallback also reintroduces token exposure in browser history and logs.
- **Recommendation**: Change the browser client to use the backend-supported initial auth frame before sending the TTS request payload, or add a shared WebSocket auth helper matching the working voice-assistant protocol. Do not depend on query-string tokens for default WebSocket streaming flows.

## Likely Risks

No likely risks were promoted beyond the confirmed issues above.

## Improvement Opportunities

No standalone improvement opportunities recorded. The main improvement is covered by the recommendations for the two confirmed API contract findings.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/frontend-api-client-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
- `apps/packages/ui/src/services/tldw/deployment-mode.ts`
- `apps/packages/ui/src/components/Option/Settings/tldw.tsx`
- `apps/packages/ui/src/components/Option/Settings/tldw-settings-tabs.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/voice-assistant-sdk/src/client/VoiceAssistantClient.ts`
- `apps/packages/ui/src/entries/background.ts`
- `apps/packages/ui/src/services/background-proxy.ts`
- `apps/packages/ui/src/store/connection.tsx`
- `apps/extension/scripts/verify-openapi-client-paths.mjs`
- `apps/extension/tests/e2e/media-and-tts.spec.ts`
- `apps/extension/tests/e2e/tts-playground.spec.ts`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- `tldw_Server_API/app/core/Audio/streaming_service.py`
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- `tldw_Server_API/app/api/v1/endpoints/media/add.py`
- `tldw_Server_API/tests/Billing/test_billing_public_api_removed.py`
- `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`
- `tldw_Server_API/tests/frontend_e2e/server_e2e_tests`

### Tests Or Scans Run

- `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 3 -type f | sort`
- `wc -l Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/frontend-api-client-inventory.txt Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`
- `find apps -maxdepth 3 -type f | sort`
- `find tldw_Server_API/tests -path '*frontend_e2e*' -o -path '*server_e2e_tests*' -o -path '*contract*' | sort`
- `find apps/tldw-frontend apps/extension apps/packages -maxdepth 5 -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.mjs' -o -name '*.json' \) | wc -l`
- `rg -n "audio/stream/tts|billing|voice/assistant|setup|job|upload|apiSend" apps/packages/ui/src apps/extension -g '*.ts' -g '*.tsx'`
- `node ../../extension/scripts/verify-openapi-client-paths.mjs` from `apps/packages/ui`
- Focused `nl -ba`, `sed`, and `rg` inspections of the affected frontend, extension verifier, backend endpoint, auth service, and backend test files.

### Blocked Or Unverified Areas

- No network access, dependency installation, service startup, Docker, browser automation, or full frontend/backend test suite runs were performed per the domain-agent rules.
- The review used static inspection and the local OpenAPI verifier; it did not exercise the Billing tab or Speech playground against a live server.
- The primary app and extension surface is large. Broad scans covered the scoped frontend/API-client files, but not every TS/TSX file was line-reviewed.
- No source/production code was edited.

### Evidence Notes

- Additional scoped evidence: `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/webui-extension-api-contracts-static-evidence.txt`
- The OpenAPI verifier confirmed 303 `ClientPath` entries and 49 media-add fallback fields while warning about 10 reviewed exceptions. The eight billing exceptions are the only exceptions promoted to a finding because the current settings UI still exposes and calls them in the OSS multi-user flow.
- Voice assistant WebSocket auth was checked as a comparator. It sends an initial `auth` frame after connection, matching the backend protocol, so it was not promoted as a finding.
