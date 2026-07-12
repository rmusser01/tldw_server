# PR 2713 Authentication Persistence Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every production-readiness finding on PR 2713 and prove the corrected cookie, device, and session authentication flows through real WebUI and extension browser lifecycles.

**Architecture:** Add one storage-agnostic effective-auth resolver beside the existing credential policy. WebUI callers explicitly supply verified quickstart cookie eligibility; extension callers never do, but both hydrate origin-bound session keys through the same function. Keep persistence formats unchanged, make logout and clearing truthful/idempotent, and extend existing lifecycle suites to perform authenticated requests.

**Tech Stack:** TypeScript, Vitest, Plasmo storage, WXT, Next.js, Playwright, Python 3.11, FastAPI, pytest, GitHub Actions.

---

### Task 1: Shared effective-auth resolver

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/single-user-credential.ts`
- Modify: `apps/packages/ui/src/services/tldw/browser-networking.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts`

- [x] Add failing tests that require exact-origin session hydration, exact-origin cookie preference, explicit removal of API/Bearer fields from cookie configs, and fail-closed storage errors.
- [x] Run `bunx vitest run src/services/tldw/__tests__/single-user-credential.test.ts --maxWorkers=1` from `apps/packages/ui`; confirm the new cases fail because no shared resolver exists.
- [x] Export the existing cookie-config validator and implement the minimal storage-injected effective resolver without new persistence formats or dependencies.
- [x] Re-run the focused test and confirm all cases pass.
- [x] Run `git diff --check` and commit `fix(web): centralize effective auth configuration`.

### Task 2: Use the resolver in WebUI and extension transports

**Files:**
- Modify: `apps/packages/ui/src/services/background-proxy.ts`
- Modify: `apps/packages/ui/src/services/api-send.ts`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Modify: `apps/packages/ui/src/entries/shared/background-init.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Test: `apps/packages/ui/src/services/__tests__/background-proxy.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/background-proxy.web-refresh.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/api-send.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts`
- Test: `apps/packages/ui/src/entries/__tests__/background.stt-protocol.test.ts`
- Test: `apps/packages/ui/src/entries/shared/__tests__/background-init.test.ts`
- Create: `apps/packages/ui/src/entries/__tests__/background.effective-auth.test.ts`
- Test: `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts`
- Test: `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts`

- [x] Add failing direct-runtime tests proving a verified cookie marker overrides a preserved remote key and adds CSRF on POST/PATCH; add direct `bgUpload` and `bgStream` cases proving origin-bound session hydration without writing the key to persistent storage.
- [x] Add failing `apiSend` direct-fallback tests for the same cookie/session resolution contract.
- [x] Add failing worker-level tests proving session-only credentials authenticate ordinary requests, uploads, HTTP streams, STT WebSocket first-frame auth, and background-init OpenAPI requests without local persistence. Seed a cookie marker and prove extension contexts ignore it.
- [x] Add the authenticated WebUI and extension lifecycle assertions now, before implementation, and run them to confirm the production request paths fail even though form/storage hydration succeeds.
- [x] Run the focused Vitest and Playwright files and confirm failures occur at local-only config reads.
- [x] Replace credential-bearing local-only reads with the shared resolver; pass cookie eligibility only from an HTTP(S) same-origin quickstart WebUI.
- [x] Re-run the focused suites, then the changed shared-UI test matrix.
- [x] Run scoped ESLint on changed TypeScript files and commit `fix(web): hydrate effective auth across transports`.

### Task 3: Idempotent cookie logout and truthful secret clearing

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Modify: `apps/packages/ui/src/services/tldw/TldwAuth.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/single-user-credential.ts`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`
- Test: `apps/packages/ui/src/services/__tests__/tldw-auth.api-key-validation.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts`

- [x] Add failing backend tests for active, missing, stale, and infrastructure-failure DELETE responses, exact-session revocation, cookie clearing, and `Cache-Control: no-store` on POST/DELETE.
- [x] Add failing client tests proving cookie logout uses the effective WebUI runtime with CSRF, removes/invalidates the marker only after success, rehydrates the preserved manual connection even when marker removal fails, and preserves cookie state on network error.
- [x] Add failing clear-policy tests proving persistent read/write and session remove failures are surfaced while both stores are attempted.
- [x] Run the focused pytest and Vitest cases and confirm expected failures.
- [x] Remove the mandatory principal dependency from DELETE, strictly validate/revoke only an active canonical session, clear the cookie only after safe completion, and set no-store.
- [x] Wire client cookie logout and implement truthful two-store clearing with the smallest existing APIs.
- [x] Re-run focused tests and Bandit over the changed Python authentication scope.
- [x] Commit Task 3 and both review follow-ups (`05962a3802`, `8d8e960f47`, `3dfeb52393`).

### Task 4: Harden successful-bootstrap secret scrubbing

**Files:**
- Modify: `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
- Test: `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

- [x] Add failing tests for leftover session secrets beside device, cookie, ambiguous, different-origin, same-origin, and noncanonical connection metadata.
- [x] Run `bunx vitest run __tests__/extension/runtime-bootstrap.test.ts --maxWorkers=1` from `apps/tldw-frontend`; confirm the mismatched records survive before the fix.
- [x] Make the scrub predicate require complete active manual/session metadata, canonical matching origins, and a remote origin distinct from the quickstart WebUI.
- [x] Re-run the 24-case focused bootstrap suite and commit Task 4 plus its quality-review follow-up (`c7941f1dcb`, `e08dfb36fa`).

### Task 5: Authenticate through lifecycle E2E and required CI

**Files:**
- Modify: `apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts`
- Modify: `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts`
- Modify: `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts`
- Modify: `apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts`
- Modify: `.github/workflows/frontend-required.yml`

- [x] Extend the fixture with an authenticated protected endpoint and request counters that distinguish missing/wrong credentials.
- [x] Finalize the WebUI lifecycle assertions added in Task 2: call the protected endpoint after hard reload and profile reopen; session mode must authenticate before close and fail after reopen.
- [x] Finalize the extension lifecycle assertions added in Task 2: call the same endpoint through the production background request path after reload/reopen.
- [x] Extend cookie lifecycle coverage with an unsafe production-path mutation, preserved remote metadata, production client logout, status reset/manual rehydration, and stale-cookie idempotency.
- [x] Add the production extension build and three lifecycle suites to the frontend-changed branch of `frontend-required` because the existing security classifier omits frontend auth paths.
- [x] Re-run all three suites and commit Task 5 plus review follow-ups (`494e55268c`, `a112b604b0`, `b2154bcce1`).

### Task 6: Full verification and UAT

**Files:**
- Modify: `backlog/tasks/task-12948 - Address-PR-2713-authentication-persistence-review-findings-and-run-UAT.md`

- [ ] Run all changed backend tests and the full changed shared/frontend Vitest matrix.
- [ ] Run scoped ESLint, frontend typecheck, production Chromium extension build, Bandit on changed Python, `git diff --check`, and secret-marker artifact scans.
- [ ] Use the Playwright CLI prerequisite/wrapper for headed WebUI UAT where practical; use the repository Playwright extension harness for the loaded unpacked-extension profile because CLI does not expose extension launch flags.
- [ ] UAT WebUI cookie bootstrap, unsafe mutation, reload, logout, and stale-cookie recovery.
- [ ] UAT WebUI manual device/session save, reload, close/reopen, authenticated request, and session expiry.
- [ ] UAT the unpacked extension with the same device/session matrix and stable extension installation/profile.
- [ ] Record exact commands, versions, pass counts, artifacts, and unrelated baselines in TASK-12948; complete its acceptance criteria and Definition of Done.
- [ ] Run `git status --short`, commit the verification record, push the PR branch, and report CI status without claiming pending checks passed.
