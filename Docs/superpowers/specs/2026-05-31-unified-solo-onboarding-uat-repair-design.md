# Unified Solo Onboarding UAT Repair Design

Date: 2026-05-31
Status: Ready for user review
Owner: Codex brainstorming session
Backlog: TASK-576
Related PR: #2194

## Summary

This repair slice makes PR #2194 green by fixing the blockers found during the real first-time solo-user UAT walkthrough.

The release gate is no longer "the setup wizard can render" or "the setup APIs pass unit tests." The release gate is a complete first-time walkthrough: a fresh local user opens the WebUI, enters the unified setup flow, configures chat/provider defaults, selects `onnx-parakeet` for STT and `pocket-tts` for TTS, receives an actual successful chat response, adds a first source, and can both see and search that source in the WebUI without manually editing `.env`, `config.txt`, or visiting `/settings/tldw`.

The design keeps backend setup/readiness state authoritative, keeps `/setup` as the operator recovery surface, and treats the WebUI as the primary cohesive solo-user onboarding flow.

## UAT Findings To Repair

The fresh-install walkthrough exposed four blockers:

1. Root first-run entry is not unified. The first-run overlay's "Get Started" action sends a generic first-time user to `/persona`, which then causes login/settings friction instead of opening the unified setup wizard.
2. WebUI authenticated API handoff is incomplete. Even when the frontend knows the backend URL, normal post-setup APIs still require a manual `/settings/tldw` API-key save/test before first-source ingest can work.
3. First-source ingest is not reliable from the guided milestone. The milestone opens Quick Ingest, but ordinary web URLs and local markdown URLs can be routed through an unsupported document-download path. Arbitrary localhost ports are also correctly blocked by SSRF protection, so the first-source happy path cannot rely on a temporary localhost URL.
4. The real completion signal was only achieved with manual workarounds. Backend file ingest and media search worked after direct API calls, proving persistence/search are viable, but the cohesive WebUI journey remained broken.

## Goals

1. Make a clean first-time solo-user walkthrough possible through the WebUI without a settings detour.
2. Route generic first-run entry into the unified setup shell, while preserving explicit character-chat intent behavior.
3. Ensure the local quickstart path gives the browser enough authenticated API context to use normal APIs after setup.
4. Provide an inline WebUI fallback for manual/dev runs where authenticated API context is missing.
5. Make first-source ingest use supported, onboarding-safe source paths and defaults.
6. Fix Quick Ingest URL routing so ordinary web pages do not get treated as unsupported document downloads.
7. Add focused regression coverage for the broken contracts.
8. Require a fresh UAT pass before calling the PR green.

## Non-Goals

- Do not remove backend `/setup`; it remains the recovery and operator surface.
- Do not weaken SSRF protections to make arbitrary localhost URL ingestion work.
- Do not make RAG tuning, embeddings, advanced storage paths, or browser extension setup mandatory for first use.
- Do not install or manage Ollama, llama.cpp, or other local model runtimes.
- Do not redesign the full Quick Ingest UX beyond the routing/defaults needed for first-source reliability.
- Do not collapse multi-user setup into the solo-user wizard; multi-user remains an exit to the multi-user guide.

## Acceptance Journey

The repaired PR is acceptable only when this journey passes from a clean install root:

1. Start the backend and WebUI through the obvious local start path.
2. Open the WebUI root.
3. Click the first-run start action and land in the unified setup shell, not `/persona`.
4. Complete setup path, privacy/security, provider, ingest-default, and audio-default screens.
5. Configure OpenAI as the chat provider using an API key from the project `.env` for UAT.
6. Select `onnx-parakeet` as the STT model/provider default.
7. Select `pocket-tts` as the TTS provider default.
8. Send a first chat and require an actual successful model response before onboarding completion.
9. Immediately receive the "add your first source" milestone.
10. Add a supported first source through the WebUI.
11. See the source in the WebUI media surface and search for a unique phrase from it.
12. Stop services and confirm temporary UAT state, ports, and generated artifacts are cleaned up or intentionally retained as evidence.

## Design Decisions

### Entry Routing

Generic first-run entry must target the unified setup shell. The existing character/persona route may still be used for explicit character-chat entrypoints, but it cannot be the default route for a first-time solo user.

The route-selection contract should distinguish:

- generic first-run setup: route to the unified setup experience;
- explicit character-chat setup intent: preserve the existing character-chat handoff;
- completed or explicitly skipped setup: expose normal app navigation.

Regression tests should cover root, chat, character, and non-character routes so future setup changes do not drift back to `/persona` as the generic first-run target.

### Authenticated API Handoff

The WebUI setup endpoints can run without normal API auth because local setup access is backend-gated. That is not enough for the post-onboarding milestone because media and Quick Ingest APIs require normal authenticated API calls.

The repaired product contract is:

- the bundled quickstart/local start path should seed the WebUI with the single-user API auth context it needs for normal APIs;
- this seeding must be local-first and documented as a local single-user convenience, not a production secret-distribution pattern;
- automatic API-key handoff must be gated to single-user localhost/same-origin quickstart or equivalent explicitly local contexts, and must not be enabled for multi-user, advanced remote, LAN, reverse-proxy, or reusable public WebUI artifacts;
- prefer runtime browser-local seeding or same-origin local handoff over baking generated secrets into a long-lived public client bundle; if `NEXT_PUBLIC_X_API_KEY` remains part of the implementation, tests and docs must make its local/dev scope explicit;
- if a manual/dev launch lacks usable auth context, the setup flow should present inline recovery for the tldw server API key before the first-source milestone, instead of sending the user to `/settings/tldw`;
- inline recovery must update the normal WebUI API client configuration and verify an authenticated media-capable request before rendering the first-source milestone;
- stored browser config should remain masked in the UI and follow the existing `tldwConfig` storage conventions.

Implementation can satisfy this through quickstart environment wiring, runtime bootstrap changes, an inline setup recovery step, or a combination. The observable requirement is that the happy path does not require manual settings navigation.

### First-Source Milestone

The first-source milestone should be an onboarding-safe ingest profile, not a generic heavy Quick Ingest launch.

The first-source profile should:

- prefer upload, paste, or direct supported file source paths for UAT and first-use reliability;
- use a deterministic UAT fixture: a small Markdown/text source uploaded or pasted through the WebUI with a title such as `UAT onboarding source` and a unique phrase that can be searched after ingest;
- use lightweight defaults that avoid unnecessary analysis/provider dependencies;
- chunk and store enough content to prove value through search or media visibility;
- present SSRF-blocked local URLs as expected safety behavior with a clear alternative, such as upload the file directly;
- report completion only after the backend returns a durable media/job result that the WebUI can show.

This milestone may still open Quick Ingest, but it should pass enough context for Quick Ingest to use the first-source profile and completion callback.

### Quick Ingest URL Routing

Quick Ingest needs a clearer split between ordinary web pages and direct downloadable files.

The repaired routing contract is:

- direct files with supported extensions such as `.md`, `.markdown`, `.txt`, `.pdf`, `.epub`, audio, and video should follow the existing document/media ingest paths;
- ordinary web pages should follow the web scraping/process path or a durable web-ingest job path;
- ordinary web pages should not be normalized to `document` and sent to the document-download worker when that path rejects or misclassifies the input;
- when `storeRemote` or first-source mode is active, the chosen web path must persist a media record or job result that can be shown/searched; a process-only scrape response is not enough for first-source completion;
- backend errors from SSRF, unsupported content, auth, or provider failure should surface as user-actionable Quick Ingest results.

The implementation may either route web URLs directly through the existing web scraping endpoint or extend durable media ingest jobs to support `web`/`html` as first-class job types. The implementation plan should choose the smallest reliable option that preserves progress reporting for the first-source milestone.

### Backend Authority And Recovery

Setup state, readiness, skip state, and completion remain backend-authoritative. Frontend local state may only support temporary UI resume, dismissed tips, and local browser auth configuration.

Both `/setup` and the WebUI setup wizard must keep using shared backend setup/readiness APIs so the recovery surface and primary onboarding flow do not drift.

### Error Handling

The repaired flow should classify failures by user action:

- backend unreachable or wrong URL;
- setup access denied or remote setup not allowed;
- API auth missing or invalid for post-onboarding APIs;
- provider key invalid or selected model unavailable;
- local model endpoint unreachable or not OpenAI-compatible;
- first chat failed, which blocks completion;
- source rejected by SSRF or unsupported type;
- ingest job failed after acceptance.

Errors should explain the next action without exposing secrets or raw stack traces. Diagnostic detail can remain behind existing logs/details affordances.

## Components Likely To Change

The implementation plan should inspect and update these areas first:

- `apps/tldw-frontend/pages/_app.tsx` for first-run route selection and overlay handoff.
- `apps/tldw-frontend/__tests__/app/app-layout.test.tsx` for route regression coverage.
- `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts` and related auth storage/API client code for browser config seeding.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts` for config initialization and authenticated request readiness.
- `Makefile`, Docker compose WebUI wiring, or quickstart bootstrap scripts if the chosen auth handoff requires start-command changes.
- `apps/packages/ui/src/components/Common/QuickIngest*` for first-source context, source detection, defaults, and completion handling.
- `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts` and `media-routing.ts` for web/document URL routing.
- `tldw_Server_API/app/services/media_ingest_jobs_worker.py` only if durable web-ingest job support is selected.
- setup/readiness API tests only if backend readiness contracts need to expose additional auth or first-source readiness details.

## Testing And Verification

### Automated Regression Coverage

Add or update focused tests for:

- generic first-run start routes to the unified setup shell;
- explicit character-chat first-run intent still routes to character-chat setup;
- quickstart/runtime bootstrap can seed usable WebUI auth when provided;
- automatic WebUI auth seeding is disabled or rejected for non-local, multi-user, and advanced remote deployment contexts;
- missing WebUI auth produces inline setup recovery rather than a settings detour;
- inline setup recovery writes normal WebUI API client config and verifies authenticated media API readiness;
- first-source milestone opens Quick Ingest with onboarding-safe defaults;
- first-source UAT can upload or paste the deterministic Markdown/text fixture and later find its unique phrase;
- Quick Ingest distinguishes ordinary web URLs from direct supported file URLs;
- store-remote web ingest either persists visible/searchable media or fails with an actionable alternative;
- ingest failures surface actionable results instead of silent completion.

Backend tests are required if durable web-ingest job behavior changes.

### Real UAT

After implementation, rerun the full UAT walkthrough from a fresh isolated install/state using CDP browser control only:

- use the OpenAI key from the existing project `.env`;
- use `pocket-tts` for TTS;
- use `onnx-parakeet` for STT;
- require a successful chat response as the completion gate;
- add the deterministic Markdown/text first-source fixture through the WebUI without a temporary localhost file server;
- verify source visibility and searchability by title and unique phrase;
- capture evidence artifacts;
- stop all services and verify cleanup.

The PR should not be called green until this UAT passes without manual workarounds.

## Cleanup Requirements

The implementation and UAT process must clean up:

- temporary install roots;
- temporary test media/source files unless retained as named evidence artifacts;
- local ports used for backend, frontend, mock servers, or test file servers;
- generated browser profiles and local storage state;
- provider keys from logs, screenshots, task notes, and test artifacts.

Evidence artifacts may be retained under a temporary UAT artifact directory if they contain no secrets and are referenced in the final verification notes.

## Implementation Planning Notes

The implementation plan should sequence work so that each repair can be verified independently:

1. Route-selection tests and route repair.
2. Auth handoff tests and quickstart/runtime/inline fallback repair.
3. First-source profile tests and Quick Ingest milestone repair.
4. Web/document URL routing tests and ingest routing repair.
5. Fresh UAT and cleanup verification.

The implementation plan must name the exact local start path it will certify, the exact first-source fixture content, and whether ordinary web URLs will use direct web scraping or durable web-ingest jobs.

Each stage should include rollback-safe, focused changes and should not broaden into unrelated onboarding redesign.
