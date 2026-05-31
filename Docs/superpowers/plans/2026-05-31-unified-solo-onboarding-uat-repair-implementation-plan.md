# Unified Solo Onboarding UAT Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the identified first-time solo onboarding blockers so a real new installation can move from obvious start command to WebUI setup wizard, successful first chat, first-source ingest, and visible/searchable value without manual config edits.

**Architecture:** Keep backend setup/readiness authoritative and the WebUI as the primary cohesive solo-user onboarding flow. Repair the seams around WebUI route entry, quickstart auth handoff, authenticated post-onboarding readiness, first-source Quick Ingest defaults, and web/document ingest routing. `/setup` remains an operator recovery surface and shares backend setup/readiness APIs.

**Tech Stack:** Next.js WebUI, React, Vitest, TypeScript packages under `apps/packages/ui`, existing `TldwApiClient`, Docker Compose quickstart, existing FastAPI media/web-scraping endpoints, Backlog.md task `TASK-576`.

---

## Stage 0: Preflight And Safety

**Goal:** Confirm the branch, task, and local disk state are safe before runtime code edits and before expensive UAT.

**Success Criteria:**
- Work happens on `codex/unified-solo-onboarding-pr` in the `unified-solo-onboarding-pr` worktree.
- Backlog task `TASK-576` is referenced in commits and kept current.
- There is enough free disk for focused frontend work and, separately, for a fresh quickstart UAT run.
- If disk is below threshold, work stops before implementation and asks the user to approve cleanup.

**Tests:** Read-only shell checks.

**Status:** Complete

### Tasks

- [ ] Run `git status --short --branch` from `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/unified-solo-onboarding-pr`.
- [ ] Run `df -h .` and `git count-objects -vH`.
- [ ] Require at least `2GiB` free before changing runtime code or running focused frontend tests.
- [ ] Require at least `10GiB` free before Docker image rebuilds, fresh install/UAT work, or any command expected to materialize large dependency/build artifacts.
- [ ] If free space is below the threshold for the next stage, stop and request approval for a targeted cleanup plan. Do not run `git prune`, remove worktrees, delete Docker data, or delete temp directories without explicit approval.
- [ ] Confirm the existing spec is present at `Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md`.
- [ ] Update `TASK-576` with the implementation-plan path and current status.

---

## Stage 1: First-Run WebUI Route Repair

**Goal:** Make a generic first-time solo-user WebUI entry land in the unified setup shell instead of the legacy Persona path, while preserving explicit character-chat onboarding.

**Success Criteria:**
- First-time generic entry from routes like `/chat`, `/media`, and `/notes` routes to `/`.
- `/` can render the unified setup experience directly.
- Explicit character-chat onboarding still routes to `/?intent=character_chat_onboarding` and then to the Persona/character flow as designed.
- Route bypass logic does not accidentally disable the first-run overlay on all non-Persona routes.

**Tests:** `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`

**Status:** Complete

### Files

- `apps/tldw-frontend/pages/_app.tsx`
- `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`

### Red Tests

- [ ] Add a test showing a first-time `/chat` start pushes `/`, not `/persona`.
- [ ] Add a test showing a first-time `/media` start pushes `/`.
- [ ] Add a test showing `/` can bypass the overlay because it is the unified setup host route.
- [ ] Add a test showing explicit `CHARACTER_CHAT_ONBOARDING_INTENT` still routes through the character onboarding route.
- [ ] Run the focused test file and confirm the new generic-route assertions fail before implementation.

### Implementation

- [ ] Change `buildFirstRunSetupRoute` so it only builds the character onboarding route for the explicit character intent:

```ts
function buildFirstRunSetupRoute(entryIntent: FirstRunEntryIntent | null): string {
  if (entryIntent === CHARACTER_CHAT_ONBOARDING_INTENT) {
    return buildFirstRunOnboardingRoute(entryIntent);
  }

  return "/";
}
```

- [ ] Replace the current Persona-based bypass condition with route-aware logic that only bypasses the first-run overlay for the actual setup host route, already-allowed bypass routes, or the explicit character onboarding route. The key invariant is that a non-setup route must still show the overlay until the user starts setup.
- [ ] Preserve existing auth-gate behavior for normal authenticated users and already-completed onboarding.
- [ ] Run the focused test file and confirm it passes.
- [ ] Commit with message `fix: route generic first run to unified setup`.

---

## Stage 2: Quickstart WebUI Auth Handoff

**Goal:** Make the Docker single-user quickstart produce a WebUI that can authenticate to the backend without requiring the user to edit `.env`, browser storage, or settings after startup.

**Success Criteria:**
- The product requirement remains one obvious command: `make quickstart`.
- Local verification can use `DOCKER_BUILD=true make quickstart` to rebuild current branch images.
- In single-user quickstart mode, the WebUI receives the generated `SINGLE_USER_API_KEY` as `NEXT_PUBLIC_X_API_KEY` only for the local quickstart deployment path.
- The API key is not printed in normal Makefile output beyond existing backend startup behavior.
- The frontend runtime bootstrap and `TldwApiClient` can seed a usable config on true first run when a quickstart URL and API key are available.
- Hosted or advanced remote deployments are not silently configured with a generated local key unless they explicitly provide that public key.

**Tests:**
- `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`
- `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`
- New or existing `apps/packages/ui/src/services/__tests__/tldw-api-client*.test.ts`
- Makefile dry-run or targeted command inspection with a fake `.env`

**Status:** Complete

### Files

- `Makefile`
- `Dockerfiles/docker-compose.webui.yml`
- `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`
- `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
- `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`

### Red Tests

- [x] Add a quickstart networking test showing the `start-docker-single` Makefile recipe supplies `NEXT_PUBLIC_X_API_KEY` to the WebUI compose command from the same `SINGLE_USER_API_KEY` source used by the backend.
- [x] Add a quickstart networking test or dry-run assertion showing the recipe does not print the resolved key value.
- [x] Preserve or tighten the existing runtime-bootstrap test coverage where `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`, `NEXT_PUBLIC_API_URL` is empty, and `NEXT_PUBLIC_X_API_KEY` is present. This may already pass; keep it as a regression check rather than forcing a redundant red test.
- [x] Preserve or tighten the existing runtime-bootstrap test showing no automatic generated-key behavior when deployment mode is not `quickstart` and no explicit public key is provided.
- [x] Add a `TldwApiClient` initialization test showing a first-run client with no stored config creates usable config from `NEXT_PUBLIC_X_API_KEY` and the quickstart API URL origin.
- [x] Add a `TldwApiClient` test showing missing key still returns the existing explicit missing-key error.
- [x] Run the focused tests and confirm the new Makefile handoff and `TldwApiClient` first-run seeding assertions fail before implementation.

### Implementation

- [x] Update `start-docker-single` so it reads `SINGLE_USER_API_KEY` from `$(TLDW_ENV_FILE)` and passes it to the WebUI compose as `NEXT_PUBLIC_X_API_KEY` only when the caller did not set `NEXT_PUBLIC_X_API_KEY`. Mirror the existing `Helper_Scripts/run-frontend-integration.sh` precedence: explicit `NEXT_PUBLIC_X_API_KEY` wins, otherwise use `SINGLE_USER_API_KEY`.
- [x] Keep the Makefile command structured so the key is not echoed. Use shell variable assignment inside the recipe rather than printing the resolved key.
- [x] If the env file has no usable `SINGLE_USER_API_KEY`, fail with a clear setup message instead of launching a WebUI that cannot authenticate.
- [ ] Preserve explicit caller overrides:

```make
API_KEY="$$(grep '^SINGLE_USER_API_KEY=' "$(TLDW_ENV_FILE)" | cut -d= -f2-)"; \
NEXT_PUBLIC_X_API_KEY="$${NEXT_PUBLIC_X_API_KEY:-$$API_KEY}" \
docker compose --env-file "$(TLDW_ENV_FILE)" ...
```

- [x] Keep `Dockerfiles/docker-compose.webui.yml` accepting `NEXT_PUBLIC_X_API_KEY` as a WebUI build arg and environment value; add a short comment if needed to document that Make local quickstart supplies it.
- [x] Do not expand this PR into a new Next.js server-side auth proxy. The current WebUI already documents `NEXT_PUBLIC_X_API_KEY` as browser-visible single-user auth; keep this repair scoped to making the existing local quickstart contract work, and record server-side proxy auth as a future hardening option if desired.
- [x] Extend `runtime-bootstrap.ts` to treat `NEXT_PUBLIC_X_API_KEY` as the local quickstart bootstrap key and persist it through the same existing config path.
- [x] Extend `TldwApiClient.getEnvApiKey()` so it also reads `process.env.NEXT_PUBLIC_X_API_KEY`.
- [x] Extend `TldwApiClient.initialize()` so a true first-run client creates a config when env URL/key material is present, instead of leaving `config=null`.
- [x] Ensure config seeding never logs secrets.
- [x] Run the focused tests and confirm they pass.
- [x] Commit with message `fix: seed quickstart webui auth config`.

---

## Stage 3: Authenticated Post-Onboarding Readiness Gate

**Goal:** After first chat completes, verify the normal authenticated media API path before offering first-source ingest, and provide inline recovery if the WebUI cannot authenticate.

**Success Criteria:**
- Backend setup completion remains gated by an actual successful chat response.
- The post-onboarding first-source milestone appears only after a normal authenticated media request succeeds.
- If the backend setup state is complete but the WebUI lacks usable API config, the user sees an inline recovery panel in the setup shell instead of being sent to Settings.
- Recovery accepts a single-user API key, validates it against the configured backend, persists it through `tldwClient.updateConfig`, and rechecks media readiness.
- Temporary UI state remains frontend-local only; readiness/completion truth remains backend-authoritative.

**Tests:**
- `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
- New component/hook tests for post-setup API recovery if split into separate files

**Status:** Not Started

### Files

- `apps/packages/ui/src/routes/option-index.tsx`
- `apps/packages/ui/src/components/Option/Onboarding/PostSetupApiRecovery.tsx`
- `apps/packages/ui/src/hooks/usePostOnboardingMediaReadiness.ts`
- `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`

### Red Tests

- [ ] Add a completed-setup test where authenticated media readiness succeeds and `FirstSourceMilestonePrompt` is shown.
- [ ] Add a completed-setup test where setup is complete but media readiness fails due to missing API key, and the inline recovery panel is shown.
- [ ] Add a recovery submission test where entering an API key calls `tldwClient.updateConfig`, then rechecks readiness, then shows `FirstSourceMilestonePrompt`.
- [ ] Add a test that first-source CTA is not rendered while readiness is unknown or failed.
- [ ] Run the focused route tests and confirm the new recovery/readiness assertions fail before implementation.

### Implementation

- [ ] Add `usePostOnboardingMediaReadiness` with these states: `checking`, `ready`, `needs_config`, `error`.
- [ ] Implement the readiness check by requiring both client config and a successful authenticated media request. Prefer a low-cost media endpoint already wrapped by `TldwApiClient`, such as `listMedia({ results_per_page: 1 })`, because the first-source flow depends on media API auth.
- [ ] Add `PostSetupApiRecovery` as a compact setup-shell panel with:
  - [ ] masked API key input,
  - [ ] submit button,
  - [ ] clear validation error state,
  - [ ] no detour to Settings as the primary path.
- [ ] On submit, call `tldwClient.updateConfig` with the existing or discovered backend URL, `authMode: "single-user"`, and the entered key.
- [ ] After successful update, immediately re-run the authenticated media readiness check.
- [ ] Modify `option-index.tsx` so `FirstSourceMilestonePrompt` renders only when setup state is complete and media readiness is `ready`.
- [ ] Keep `/setup` behavior unchanged except for sharing the same readiness/backend contract where it already does.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit with message `fix: gate first source on authenticated readiness`.

---

## Stage 4: First-Source Quick Ingest Defaults

**Goal:** Make the “add your first source” milestone open Quick Ingest in a friction-minimized, value-producing mode that works for an uploaded/pasted Markdown or text source.

**Success Criteria:**
- Clicking “add your first source” opens Quick Ingest without requiring navigation to another settings surface.
- The first-source flow defaults to a quick profile that stores content and creates searchable chunks, while skipping heavy analysis and OCR.
- The first-source context is explicit in the Quick Ingest open event so future analytics or UI copy can distinguish it from regular ingest without relying on route state.
- Ordinary Quick Ingest entry points keep their existing defaults.

**Tests:**
- `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
- Quick Ingest modal/session tests if existing coverage is present

**Status:** Not Started

### Files

- `apps/packages/ui/src/routes/option-index.tsx`
- `apps/packages/ui/src/utils/quick-ingest-open.ts`
- `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`
- `apps/packages/ui/src/components/Common/QuickIngest/presets.ts`
- Existing Quick Ingest tests under `apps/packages/ui/src/components/Common/QuickIngest*/` or route tests

### Red Tests

- [ ] Add a route test showing `FirstSourceMilestonePrompt` dispatches Quick Ingest with `source: "first_source_milestone"` and first-source preset metadata.
- [ ] Add a Quick Ingest session/modal test showing first-source metadata selects the quick first-source profile.
- [ ] Add a control test showing regular Quick Ingest opens with existing default preset behavior.
- [ ] Run the focused tests and confirm first-source preset assertions fail before implementation.

### Implementation

- [ ] Extend the Quick Ingest open detail type to allow:

```ts
type QuickIngestOpenDetail = {
  source?: "global" | "media" | "first_source_milestone";
  preferredPreset?: "quick" | "standard" | "deep";
  firstSource?: boolean;
};
```

- [ ] Dispatch first-source open details from `option-index.tsx`:

```ts
requestQuickIngestOpen(
  {
    source: "first_source_milestone",
    preferredPreset: "quick",
    firstSource: true,
  },
  { focusTrigger: true }
);
```

- [ ] Add a first-source quick ingest preset or preset override that keeps:
  - [ ] `storeRemote: true`,
  - [ ] chunking enabled,
  - [ ] analysis disabled,
  - [ ] OCR disabled,
  - [ ] upload/paste/manual content paths available.
- [ ] Apply the first-source preset only when the modal opens from `first_source_milestone`.
- [ ] Preserve existing default preset selection for every other Quick Ingest entry point.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit with message `fix: default first source ingest for fast value`.

---

## Stage 5: Web And Text Ingest Routing Repair

**Goal:** Fix Quick Ingest routing so first-source text/Markdown succeeds and ordinary web URLs use the web-scraping processor instead of the document job path.

**Success Criteria:**
- Uploaded or pasted `.md`, `.markdown`, and `.txt` content is treated as a document/text source and can be stored.
- Direct file URLs with `.md`, `.markdown`, `.txt`, `.pdf`, `.epub`, audio, or video extensions continue to use the media/document job path.
- Ordinary `http(s)` web pages use the existing `/api/v1/media/process-web-scraping` endpoint.
- Web-scrape persist responses are normalized so success surfaces a persisted media id when the backend returns `media_ids`.
- This repair does not introduce a new durable web-ingest jobs backend in this PR.

**Tests:** `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

**Status:** Not Started

### Files

- `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
- `apps/packages/ui/src/services/tldw/media-routing.ts`
- `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
- `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
- Quick Ingest UI tests for type detection if present

### Red Tests

- [ ] Add a test showing an ordinary URL like `https://example.com/article` calls `processWebScrape` and does not enqueue `/media/ingest/jobs`.
- [ ] Add a test showing a direct Markdown URL like `https://example.com/source.md` still follows the document/media route.
- [ ] Add a test showing a web-scrape persist response with `{ status: "persist-ok", media_ids: [123] }` is normalized as a successful result with media id `123`.
- [ ] Add a type-detection test for `.md`/`.markdown`/`.txt` if the UI detection logic has test coverage.
- [ ] Run the focused tests and confirm the new URL-routing assertions fail before implementation.

### Implementation

- [ ] Update `AddContentStep.tsx` URL detection so `.md`, `.markdown`, and `.txt` are detected as document/text sources instead of generic web sources.
- [ ] Keep ordinary URLs as web/html sources.
- [ ] In `quick-ingest-batch.ts`, route `resolvedType === "html"` or equivalent ordinary web source entries through `client.processWebScrape()` before the generic ingest job path, including when `storeRemote` is true.
- [ ] Use the backend’s existing persist mode for ordinary web URLs. Do not add a new durable web-job backend in this PR.
- [ ] Normalize the backend web-scrape persist response:

```ts
const mediaId = Array.isArray(response.media_ids) ? response.media_ids[0] : response.media_id;
```

- [ ] Ensure direct supported file URLs continue through the existing media/document job path.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit with message `fix: route quick ingest web sources correctly`.

---

## Stage 6: Focused Regression Verification

**Goal:** Prove the repaired contract with automated tests before manual UAT.

**Success Criteria:**
- All focused frontend tests pass.
- No relevant existing onboarding/Quick Ingest regression tests fail.
- Bandit requirement is addressed for touched scope. If no Python files are touched, explicitly record that Bandit is not applicable for this TS/Makefile-only implementation.
- Backlog task includes verification commands and results.

**Tests:** Focused command list below.

**Status:** Not Started

### Commands

- [ ] From `apps/tldw-frontend`, run:

```bash
bunx vitest run \
  __tests__/frontend-quickstart-networking.test.ts \
  __tests__/app/app-layout.test.tsx \
  __tests__/extension/runtime-bootstrap.test.ts \
  --reporter=default
```

- [ ] From `apps/packages/ui`, run:

```bash
bunx vitest run \
  src/routes/__tests__/option-index.unified-setup.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx \
  src/services/__tests__/quick-ingest-batch.test.ts \
  --reporter=default
```

- [ ] If any backend Python code is touched, activate `.venv` and run Bandit on the touched Python paths:

```bash
source .venv/bin/activate
python -m bandit -r <touched-python-paths> -f json -o /tmp/bandit_unified_onboarding_uat_repair.json
```

- [ ] If no backend Python code is touched, document “Bandit not applicable: touched scope is TS/React/Makefile only” in `TASK-576`.
- [ ] Run `git diff --check`.
- [ ] Commit any remaining test-only or documentation updates with a message tied to the changed behavior.

---

## Stage 7: Full Real UAT Walkthrough

**Goal:** Validate the exact end-to-end first-time solo-user path in a new installation using the existing OpenAI key, `pocket-tts`, and `onnx-parakeet`, with browser automation through CDP only.

**Success Criteria:**
- A fresh install starts through the product start path.
- WebUI first-run setup is reachable without manual settings or `.env` edits after startup.
- Wizard covers Chat/provider, ingest defaults, audio/STT/TTS, privacy/security, and optional advanced RAG/storage without forcing RAG/storage.
- OpenAI API key is entered through the wizard from the existing project `.env`; it is not logged or included in artifacts.
- TTS is configured as `pocket-tts`.
- STT is configured as `onnx-parakeet`.
- Completion requires and records a real successful chat response.
- Immediately after first chat, first-source milestone is offered.
- Uploading or pasting the deterministic Markdown fixture succeeds.
- The ingested source is visible in media/library UI and searchable by its unique phrase.
- Cleanup leaves no running quickstart containers/processes from the UAT install and removes temporary UAT artifacts that are safe to remove.

**Tests:** Manual UAT through CDP/in-app browser automation plus API checks where useful.

**Status:** Not Started

### UAT Fixture

- [ ] Use this deterministic first-source content:

```md
# UAT onboarding source

This source verifies first-time onboarding ingest.

Unique phrase: cobalt pine 731.
```

- [ ] Save it as `uat-onboarding-source.md` in a temporary UAT directory.

### Setup

- [ ] Confirm at least `10GiB` free disk remains with `df -h .`.
- [ ] Create a fresh temporary clone or worktree-based install isolated from the active development worktree. Prefer the smallest option that still exercises first-install semantics and does not reuse browser local storage.
- [ ] Copy only the needed OpenAI API key from the existing project `.env` into the wizard input or UAT environment. Do not print the key.
- [ ] Ensure ports for quickstart are available, or stop the previous UAT quickstart containers after approval if they belong to this task.
- [ ] Run the local product start path. For validating this branch’s WebUI image, use:

```bash
DOCKER_BUILD=true make quickstart
```

- [ ] Record the API and WebUI URLs actually used.

### Browser Walkthrough

- [ ] Use CDP/browser automation only. Do not use Computer Use.
- [ ] Open the WebUI URL in a clean browser context with cleared local/session storage for that origin.
- [ ] Verify the first visible product surface is the focused setup shell, not the legacy Persona path and not Settings.
- [ ] Select solo/local single-user path.
- [ ] Complete privacy/security selections.
- [ ] Select hosted OpenAI provider as the primary chat provider and enter the OpenAI key through the wizard.
- [ ] Configure ingest defaults using the wizard controls.
- [ ] Configure audio defaults with:
  - [ ] TTS provider/model: `pocket-tts`
  - [ ] STT model/provider: `onnx-parakeet`
- [ ] Leave optional RAG/storage advanced settings unset unless the wizard requires them; if required, record as a defect.
- [ ] Send the required first chat and verify an actual successful assistant response appears.
- [ ] Verify setup completion occurs only after the successful chat response.
- [ ] Verify the first-source milestone appears immediately after completion.
- [ ] Open Quick Ingest from the milestone.
- [ ] Upload or paste `uat-onboarding-source.md`.
- [ ] Submit ingest and wait for success.
- [ ] Navigate to the media/library surface and verify `UAT onboarding source` is visible.
- [ ] Search for `cobalt pine 731` and verify the ingested source is returned.

### Cleanup

- [ ] Stop quickstart containers/processes started for UAT.
- [ ] Remove only temporary UAT directories/files created for this run.
- [ ] Clear browser storage for UAT origins used in the test.
- [ ] Leave the existing project `.env` unchanged.
- [ ] Ensure no secrets are present in logs, screenshots, Backlog task notes, commits, or final response.
- [ ] Update `TASK-576` with UAT pass/fail evidence and any remaining defects.
- [ ] If UAT passes, commit final cleanup/doc updates if any.
- [ ] If UAT fails, add a new focused repair stage before declaring completion.

---

## Stage 8: Final Review And PR Readiness

**Goal:** Prepare the PR for review with traceable evidence and without claiming completion before verification.

**Success Criteria:**
- All planned stages are either complete or explicitly deferred with rationale.
- `TASK-576` has final status, touched files, test results, UAT evidence, and cleanup evidence.
- Git status is clean except for intentionally unstaged user-owned changes outside this worktree.
- PR description/comment includes a human-readable change summary, test evidence, UAT evidence, and residual risks.

**Tests:** Final status checks.

**Status:** Not Started

### Tasks

- [ ] Run `git status --short --branch`.
- [ ] Run `git log --oneline --decorate -5`.
- [ ] Review diff for secrets and unrelated changes.
- [ ] Update `TASK-576` with final summary and verification evidence.
- [ ] Push the branch if requested or already expected by the active PR workflow.
- [ ] Provide the user with a concise closeout including:
  - [ ] what changed,
  - [ ] what passed,
  - [ ] UAT result,
  - [ ] cleanup performed,
  - [ ] any residual risk.

---

## Risk Controls

- **Disk pressure:** Preflight requires at least `2GiB` free before focused implementation/tests and at least `10GiB` before Docker rebuild or fresh-install UAT. Cleanup requires explicit approval.
- **Secret exposure:** API keys are accepted through wizard/env only, never printed, screenshotted intentionally, or committed.
- **Route regressions:** Tests cover both generic first-run and character-chat-specific onboarding.
- **Hosted/local drift:** Quickstart auth seeding is scoped to explicit local quickstart/public key configuration.
- **Ingest scope creep:** Ordinary web URLs use the existing web-scraping persist endpoint; durable web ingest jobs remain outside this PR.
- **False completion:** Backend setup completion still requires an actual successful chat response; frontend readiness is an additional gate for first-source ingest only.
- **Manual browser control:** UAT uses CDP/browser automation only, per user instruction.
