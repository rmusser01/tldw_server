# Guided Onboarding Diagnostics And Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task-by-task. Use `superpowers:test-driven-development` before changing implementation code, and keep this checklist updated as work progresses.

**Backlog Tasks:** `TASK-513` (planning), `TASK-514` (implementation)

**Goal:** Make first-time solo onboarding recoverable when setup or first chat fails, using backend-authoritative setup/readiness signals and real UAT coverage through the existing isolated WebUI/backend/mock OpenAI harness.

**Scope:** PR2 of the solo onboarding V2 roadmap only. This PR adds diagnostics and recovery. It does not add starter questions, provider/model discovery UX, or the full local-model guided alternative beyond recovery copy/actions needed for the current setup shell. Provider/model selection actions should route to existing settings surfaces unless a narrow setup-shell affordance already exists.

**Current Architecture Notes:**
- The repeatable UAT harness exists under `apps/tldw-frontend/e2e/onboarding-uat/` and starts an isolated backend, WebUI, and repo `mock_openai_server`.
- `scenarios.ts` already declares `provider-retry-recovery`, but `recovery.spec.ts` is not implemented yet.
- Setup backend endpoints expose `/api/v1/setup/status`, `/config`, `/audio/readiness`, `/audio/recommendations`, and sanitized setup audio verification payloads.
- The WebUI setup shell already categorizes URL/auth/network errors in `OnboardingConnectForm.tsx` and `validation.ts`, but primary error UI still risks showing raw error detail and only offers a generic retry/docs path.
- First-chat errors already become structured encoded error payloads and render through `PlaygroundChatErrorBanner`, but the banner currently only offers Health & diagnostics plus dismiss.

---

## Stage 1: Harness-First Recovery Coverage
**Goal:** Add failing UAT scenarios that prove today’s blocked/recovery states before product code changes.

**Success Criteria:**
- `recovery.spec.ts` exists and runs through the real isolated WebUI/backend/mock OpenAI path.
- `provider-retry-recovery` uses `chat-fail-once.json`, captures the failed first-chat state, exercises an inline retry action, and captures successful recovery.
- A model-unavailable UAT scenario uses `model-unavailable.json` and proves the UI shows a visible recovery path without exposing raw stack traces, secrets, headers, or local filesystem paths.
- A setup connection failure scenario uses the real setup form with an unreachable local endpoint or invalid single-user key and captures the diagnostic panel state.
- The JSON summary and step artifacts identify scenario id, failure category, visible recovery actions, URL, and screenshot path.

**Tests:**
- `bun run e2e:onboarding:uat -- --scenario provider-retry-recovery --viewport desktop --mock-config chat-fail-once.json`
- Add runner/static fixture unit coverage for any new scenario ids or mock config files.

**Status:** Complete

---

## Stage 2: Diagnostic Mapping And Sanitization
**Goal:** Create a small pure diagnostic layer that maps existing setup, readiness, and chat failure categories to safe user-facing copy and action metadata.

**Success Criteria:**
- Add a pure helper near the onboarding setup code, for example `apps/packages/ui/src/components/Option/Onboarding/onboarding-diagnostics.ts`.
- Supported setup categories include at least: `auth_invalid`, `refused`, `dns_failed`, `timeout`, `cors_blocked`, `ssl_error`, `server_error`, `config_write_failed`, `restart_needed`, `network_unavailable`, `downloads_disabled`, and `package_installs_disabled`.
- Supported first-chat categories include at least: `provider_auth_failed`, `model_unavailable`, `provider_unavailable`, `rate_or_quota_limited`, `timeout`, and `unknown_chat_failure`.
- Each category returns status, plain-language cause, blocking severity, primary action, optional secondary actions, and sanitized detail policy.
- Raw exception details remain available only for logs/artifacts, not primary UI.

**Tests:**
- Unit tests for every category and fallback behavior.
- Tests assert secrets, request headers, stack traces, and local paths are not returned in primary diagnostic copy.

**Status:** Complete

---

## Stage 3: Setup Shell Diagnostic Panel
**Goal:** Replace the setup form’s current generic error block with a compact recovery panel that keeps the user inside the focused setup shell.

**Success Criteria:**
- Add an accessible diagnostic panel component with stable test ids:
  - `onboarding-diagnostic-panel`
  - `onboarding-diagnostic-title`
  - `onboarding-diagnostic-cause`
  - `onboarding-diagnostic-primary-action`
  - `onboarding-diagnostic-secondary-action-*`
- URL/auth/network errors show specific recovery actions: edit server URL, edit API key, retry connection, and open operator `/setup` recovery when relevant.
- The panel uses backend/store categories and does not invent readiness state on the client.
- Existing success path and first-run gate behavior are unchanged.

**Tests:**
- Component/unit tests for visible copy and actions.
- Regression test that raw `errorMessage` is not rendered in the primary setup error UI.
- Existing onboarding setup form tests still pass.

**Status:** Complete

---

## Stage 4: First-Chat Recovery Actions
**Goal:** Make first-chat failures recoverable because first chat is the onboarding completion gate.

**Success Criteria:**
- `PlaygroundChatErrorBanner` exposes inline actions for first-time onboarding contexts:
  - Retry first chat.
  - Edit provider/settings.
  - Switch provider/model.
- Skip stays hidden unless an existing explicit skip path is already available and backend completion rules allow it.
- Model-unavailable failures point to model/provider selection instead of only health diagnostics.
- Provider/auth failures point back to provider setup/editing.
- Retry reuses the existing composer/chat submit path and dismisses only the recovered error, not newer errors.

**Tests:**
- Update `PlaygroundChatErrorBanner` tests for the new action set.
- Add tests for retry/dismiss semantics so a successful retry clears the current error without hiding a later error.
- UAT `provider-retry-recovery` must pass after product changes.

**Status:** Complete

---

## Stage 5: Readiness Overlays And Optional Lanes
**Goal:** Surface backend readiness overlays without making optional RAG/storage/audio lanes block first chat unless the user opted into them.

**Success Criteria:**
- Setup/readiness status is normalized through the diagnostic helper.
- Restart-needed, network-unavailable, downloads-disabled, and package-installs-disabled states display as actionable warnings with operator-safe wording.
- RAG/storage path issues stay optional/non-blocking on first use unless selected by the user.
- Audio/STT/TTS issues use setup readiness fields and stay recoverable through existing audio setup actions.

**Tests:**
- Unit tests for blocking vs deferrable readiness states.
- Existing audio setup/readiness tests still pass if touched.

**Status:** Complete

**Implementation Note:** Completed as a reusable readiness diagnostic mapper with blocking/deferrable policy for setup, RAG/storage, and audio readiness categories. No broader readiness overlay was wired because `OnboardingConnectForm` does not currently consume a backend readiness issue payload beyond the connection/setup checks; that UI integration should remain a follow-up once the setup shell has the authoritative payload.

---

## Stage 6: Verification, Cleanup, And Commit Staging
**Goal:** Prove the recovery flow works and keep the PR staged as one cohesive diagnostics/recovery change.

**Success Criteria:**
- All changed frontend unit tests pass.
- Onboarding UAT happy path still passes on desktop and mobile.
- Recovery UAT scenarios pass on desktop and produce screenshots, JSON summary, and backend/frontend/mock logs.
- No secret leaks in UAT artifacts.
- Bandit is run for touched Python scope, or explicitly skipped because no Python code changed.
- Backlog task contains final summary, verification results, and known skips.

**Commands:**
- `bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__ apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx`
- `bunx vitest run apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`
- `bun run e2e:onboarding:uat -- --scenario hosted-openai-first-chat --viewport desktop --mock-config hosted-success.json`
- `bun run e2e:onboarding:uat -- --scenario hosted-openai-first-chat --viewport mobile --mock-config hosted-success.json`
- `bun run e2e:onboarding:uat -- --scenario provider-retry-recovery --viewport desktop --mock-config chat-fail-once.json`

**Status:** Complete

**Verification Notes:**
- `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/onboarding-diagnostics.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/validation.test.ts ../packages/ui/src/services/__tests__/tldw-auth.api-key-validation.test.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.provider-recovery.test.ts` - passed, 8 files / 56 tests.
- `bun run e2e:onboarding:uat -- --scenario setup-endpoint-recovery --viewport desktop --mock-config hosted-success.json` - passed; latest summary `apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T18-47-05-900Z-65vaxv/summary.json`.
- `bun run e2e:onboarding:uat -- --scenario provider-retry-recovery --viewport desktop --mock-config chat-fail-once.json` - passed; runner cleans prior artifacts on later runs.
- `bun run e2e:onboarding:uat -- --scenario model-unavailable-recovery --viewport desktop --mock-config model-unavailable.json` - passed; runner cleans prior artifacts on later runs.
- `bun run lint -- ...touched files...` from `apps/tldw-frontend` - exited 0; package UI files are outside the frontend lint base and were ignored, while frontend lint reported existing warnings only.
- `./apps/tldw-frontend/node_modules/.bin/tsc -p apps/packages/ui/tsconfig.json --noEmit` - failed on existing package-wide baseline type errors across unrelated tests/modules; one untouched onboarding design-system test also appears in the baseline.
- `git diff --check` - passed.
- Bandit not run because no Python files changed.

---

## Commit Plan
- Commit 1: Add recovery UAT scenarios and fixture/runner coverage.
- Commit 2: Add diagnostic mapping and setup shell recovery panel.
- Commit 3: Add first-chat recovery actions and tests.
- Commit 4: Add readiness overlay handling, cleanup, and final verification notes.

## Open Questions To Resolve During Implementation
- Whether `config_write_failed` can be exercised through the existing UAT runner by creating a read-only runtime profile without weakening cleanup.
- Whether provider/model switching should route to existing model settings or a narrower onboarding setup panel in PR2. Prefer existing settings route for this PR unless implementation shows a clean setup-shell action already exists.
- Whether skip should remain hidden in first-chat failures if backend setup completion strictly requires successful chat. Prefer hiding skip unless an existing explicit skip path is already present.
