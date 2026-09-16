# Cycle4 read-only: Media model reversion and tab connection starvation

Status: diagnosis only. No product/test files, browsers, profiles, runtime, or Git state changed. Frozen source: `7c9409fad2`.

## Confirmed Media selection reversion

`AnalysisModal.tsx:87` reads/writes `useStorage('selectedModel')` directly; its Select invokes that raw setter. `ViewMediaPage.tsx:309` and WebLayout.tsx:194 concurrently mount `useMessageOption`, which calls `useSelectedModel`. In `hooks/chat/useSelectedModel.ts:59–70`, a nonempty Zustand model is authoritative: any differing storage value is rewritten to that old store model. Consequently the actual modal choice briefly enters storage and is then reverted. It is not a provider resolver changing the chosen request. `effectiveModelKey` has a first-catalog fallback, but that is not needed to reproduce this failure.

Private unchanged-source probe: `/private/tmp/uat-media-model-selection-readonly.config.ts` (log same stem). It transforms the existing AnalysisModal regression fixture in memory, substitutes the actual Next WebUI Storage/hook, mounts the real `useSelectedModel` and Zustand store, and retains existing AntD adapters and synthetic model/service transport mocks. Results: **1 PASS without the shared model owner; 1 FAIL with it**. Starting at `tldw:gemma3:1b`, choosing the exact advertised long Gemma GGUF option sends that selected model when isolated; with the actual owner, UI/storage/store revert to `gemma3:1b`, and the outgoing body sends that old model. No inference or server requests occur.

Native provenance: `uat-cycle4-single-analysis-retry-request.json` and `...request2.json` contain ollama/gemma3:1b streaming/nonstreaming bodies; `...retry-model-options.txt` confirms both choices were offered. That pre-click snapshot alone marks gemma3 selected and cannot prove actual option activation. The private interaction establishes the causal writer; root owns separated native click/snapshot confirmation.

Small repair after matrix: use the existing consolidated model-selection setter at the AnalysisModal explicit-selection boundary so store and storage agree. Preserve normalization of serialized persisted values, catalog loading/missing-model behavior, custom prompts, generation/error handling, and cancellation. Permanent actual modal + shared owner + real WebUI storage regression; normal explicit selection and unchanged-selection controls. Do not rewrite global model authority merely to fix this producer.

## Per-tab notification streams: confirmed lifecycle gap, socket exhaustion strongly supported

`WebLayout.tsx:866` wraps each full application tab with NotificationLifecycleProvider. Provider:247 opens one subscription after bootstrap list/unread reads; each instance has its own refs/controller. It has no visibility listener, cross-tab lease, or stream sharing. Polls continue every30s. `lib/api/notifications.ts:89` invokes `streamStructuredSSE`, which opens ordinary fetch and holds its reader until end/abort. `services/notifications.ts:210` owns a separate AbortController per subscriber and reconnects. Backend `endpoints/notifications.py:370` defaults max duration0→unlimited and sends heartbeat every10s. Unmount/auth/connection stop paths do release; ordinary hidden tabs do not.

Private provider probe: `/private/tmp/uat-notifications-hidden-stream-readonly.config.ts` and log. Actual lifecycle owner, existing auth/connection/API test doubles. **2 FAIL**: mounting while document.visibilityState=hidden still starts a stream; changing visible→hidden does not call unsubscribe. These test lifecycle behavior, not browser socket scheduling or a proposed sharing implementation.

Native evidence `uat-cycle4-single-prompt-network.txt` shows six open application tabs, notifications/stream200, then API aborts across health/notifications/Buddy/Prompt. `...prompt-after-sync-network.txt` includes subsequent healthy calls and prompt/project201 after root closed completed tabs. Root separately reports curl/docs remained200 during browser stalls. Thus starvation fits browser connection capacity rather than an unavailable API; exact all-six socket occupancy and negotiated protocol have not been inspected by this agent.

Primary corroboration: Chromium's current socket pool source uses6 normal connections per group (WebSockets have a different pool): https://chromium.googlesource.com/chromium/src/net/+/refs/heads/main/socket/client_socket_pool_manager.cc . This supports the inference for the recorded plain HTTP/uvicorn origin; it is not a claim that all HTTP versions have a six-request limit.

Small repair design after matrix: make notification transport visibility-aware and catch up on activation, preserving cursor/auth/account/permission boundaries and unread updates. Assess whether visible-window concurrency requires a shared owner; avoid introducing an account-unsafe global stream. Meaningful checks: hidden start, visible→hidden cancellation, foreground catch-up without duplicate events; existing no-overlap polling, StrictMode, credential rotation/invalidation and A→B→A; native six-tab ordinary API requests and mutation exactly once. Do not raise browser limits or replace the runtime as acceptance.

## Commands

From `apps/packages/ui`:
`./node_modules/.bin/vitest run --config /private/tmp/uat-media-model-selection-readonly.config.ts --maxWorkers=1 --no-file-parallelism`

From `apps/tldw-frontend`:
`./node_modules/.bin/vitest run --config /private/tmp/uat-notifications-hidden-stream-readonly.config.ts --maxWorkers=1 --no-file-parallelism`

Both exit1 intentionally: behavioral RED probes. Existing unrelated cases are filtered, not disabled in the repository. Vite transformed line numbers point into the original fixture, so use probe names and config bodies for provenance. No Bandit/lint needed for a read-only source investigation.
