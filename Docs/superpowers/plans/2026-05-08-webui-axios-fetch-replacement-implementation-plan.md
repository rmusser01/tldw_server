# WebUI Axios Fetch Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct WebUI/shared UI `axios` usage with fetch-backed helpers while preserving the current API-client behavior.

**Architecture:** Keep the first-party WebUI API surface centered on `apps/tldw-frontend/lib/api.ts`, but replace the Axios instance with a small local client object that exposes the same `apiClient` methods and `api.defaults.baseURL` mutation point used by `useConfig`. Keep the ElevenLabs external-origin helper separate in `apps/packages/ui/src/services/elevenlabs.ts` so it does not inherit first-party auth/CSRF behavior.

**Tech Stack:** TypeScript, React/Next.js, Bun workspaces, Vitest, Testing Library, platform `fetch`, `AbortController`, `Headers`, `FormData`.

---

## Files

- Modify: `apps/tldw-frontend/lib/api.ts`
- Modify: `apps/tldw-frontend/types/common.ts`
- Modify: `apps/tldw-frontend/hooks/useConfig.tsx`
- Create: `apps/tldw-frontend/lib/__tests__/api-client.fetch.test.ts`
- Modify or create: `apps/packages/ui/src/services/__tests__/elevenlabs.test.ts`
- Modify: `apps/packages/ui/src/services/elevenlabs.ts`
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/packages/ui/package.json`
- Modify: `apps/extension/package.json`
- Modify: `apps/bun.lock`
- Modify: `backlog/tasks/task-122 - Replace-WebUI-axios-with-fetch-helpers-for-issue-1346.md`

## Task 1: First-Party API Client Contract Tests

**Files:**
- Create: `apps/tldw-frontend/lib/__tests__/api-client.fetch.test.ts`
- Read: `apps/tldw-frontend/lib/api.ts`
- Read: `apps/tldw-frontend/lib/history.ts`
- Read: `apps/tldw-frontend/lib/authStorage.ts`
- Read: `apps/tldw-frontend/lib/session.ts`

- [x] **Step 1: Write failing tests for base URL, JSON, headers, auth, CSRF, credentials, and history**

  Cover:
  - `api.defaults.baseURL` mutation changes the request URL.
  - JSON requests send `Content-Type: application/json`.
  - `FormData` requests do not force `Content-Type`.
  - `Authorization`, `X-API-KEY`, session, and CSRF headers match existing rules.
  - `withCredentials: false` maps to `credentials: "omit"` and default browser credentials map to `credentials: "include"`.
  - successful responses call `addRequestHistory`.

- [x] **Step 2: Run RED**

  Run:
  `bunx vitest run lib/__tests__/api-client.fetch.test.ts`

  Expected: FAIL because the current Axios-backed client does not call the mocked global `fetch`.

- [x] **Step 3: Write failing tests for errors, redirects, retry-after, timeouts, abort signals, and response parsing**

  Cover:
  - 401 clears stored auth and redirects to `/login` only when no env/stored API auth is present and route is redirectable.
  - 403 CSRF detail rejects with the existing refresh-page message.
  - Non-2xx JSON bodies map to `ApiError.status`, `statusCode`, `detail`, and `retryAfter`.
  - Timeout uses `AbortController` and rejects with `ApiError`.
  - Caller-provided `signal` aborts the request.
  - `responseType: "arraybuffer"` returns an `ArrayBuffer`; empty `204` returns `undefined`.

- [x] **Step 4: Run RED**

  Run:
  `bunx vitest run lib/__tests__/api-client.fetch.test.ts`

  Expected: FAIL on fetch-specific expectations before implementation.

## Task 2: First-Party Fetch Client Implementation

**Files:**
- Modify: `apps/tldw-frontend/lib/api.ts`
- Modify: `apps/tldw-frontend/types/common.ts`
- Modify: `apps/tldw-frontend/hooks/useConfig.tsx`

- [x] **Step 1: Replace Axios types with local request/response types**

  Define local equivalents for the currently used config surface:
  - `headers?: HeadersInit`
  - `withCredentials?: boolean`
  - `signal?: AbortSignal`
  - `timeout?: number`
  - `responseType?: "json" | "text" | "arraybuffer" | "blob"`
  - `params?: Record<string, string | number | boolean | null | undefined>`

- [x] **Step 2: Implement request URL and request body handling**

  Preserve relative path behavior, query-string merging, JSON body serialization, and `FormData` content-type behavior.

- [x] **Step 3: Implement auth, CSRF, session, credentials, timeout, and abort handling**

  Reuse existing `getApiKey`, `getApiBearer`, `hasEnvApiAuth`, `getOrCreateSessionId`, and `captureSessionIdFromHeaders` helpers.

- [x] **Step 4: Implement response parsing, history logging, redirect, and error normalization**

  Keep the current `ApiError` class and reject with it for non-CSRF API failures.

- [x] **Step 5: Run GREEN**

  Run:
  `bunx vitest run lib/__tests__/api-client.fetch.test.ts hooks/__tests__/useConfig.networking.test.tsx`

  Expected: PASS.

## Task 3: ElevenLabs Fetch Helper

**Files:**
- Modify or create: `apps/packages/ui/src/services/__tests__/elevenlabs.test.ts`
- Modify: `apps/packages/ui/src/services/elevenlabs.ts`

- [x] **Step 1: Write failing tests for voices, models, speech, API-key headers, JSON payload, ArrayBuffer, and timeout**

  Mock `global.fetch` and assert requests go to `https://api.elevenlabs.io/v1`, include `xi-api-key`, use JSON content type for speech generation, parse voices/models, and return an `ArrayBuffer` for speech.

- [x] **Step 2: Run RED**

  Run:
  `bunx vitest run ../packages/ui/src/services/__tests__/elevenlabs.test.ts`

  Expected: FAIL because the current service uses Axios instead of `fetch`.

- [x] **Step 3: Replace Axios usage with a local external-origin fetch helper**

  Preserve `DEFAULT_ELEVENLABS_TIMEOUT_MS` for metadata calls and apply the same default timeout to speech unless tests show current behavior intentionally differs.

- [x] **Step 4: Run GREEN**

  Run:
  `bunx vitest run ../packages/ui/src/services/__tests__/elevenlabs.test.ts`

  Expected: PASS.

## Task 4: Manifest and Lockfile Cleanup

**Files:**
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/packages/ui/package.json`
- Modify: `apps/extension/package.json`
- Modify: `apps/bun.lock`

- [x] **Step 1: Confirm no direct Axios usage remains**

  Run:
  `rg -n "from ['\"]axios['\"]|import axios|AxiosRequestConfig|InternalAxiosRequestConfig" apps/tldw-frontend apps/packages/ui/src apps/extension`

  Expected: no direct import/type usage remains except documentation comments if any are intentionally kept.

- [x] **Step 2: Remove direct Axios declarations from audited manifests**

  Remove `axios` from `apps/tldw-frontend/package.json`, `apps/packages/ui/package.json`, and `apps/extension/package.json` only after the usage guard is clean.

- [x] **Step 3: Regenerate lockfile**

  Run:
  `bun install`

  From: `apps/`

  Expected: `apps/bun.lock` updates consistently.

- [x] **Step 4: Verify frozen install**

  Run:
  `bun install --frozen-lockfile`

  From: `apps/`

  Expected: PASS.

## Task 5: Final Verification and PR

**Files:**
- Modify: `backlog/tasks/task-122 - Replace-WebUI-axios-with-fetch-helpers-for-issue-1346.md`

- [x] **Step 1: Run focused tests**

  Run:
  `bunx vitest run lib/__tests__/api-client.fetch.test.ts hooks/__tests__/useConfig.networking.test.tsx ../packages/ui/src/services/__tests__/elevenlabs.test.ts`

- [x] **Step 2: Run changed-test sweep**

Run:
`NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx vitest run --changed=origin/dev`

Result: broad changed sweep currently fails on baseline UI tests unrelated to this axios slice. Representative failures reproduced with this patch stashed: `ReviewTab.queue-state.test.tsx` and `FlashcardsWorkspace.connection-state.test.tsx`.

- [x] **Step 3: Run lint, typecheck, and compile**

  Run:
  `bun run lint`

  Run:
  `bunx tsc --noEmit -p tsconfig.json --pretty false`

  Run:
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile`

  From: `apps/tldw-frontend`

- [x] **Step 4: Run extension/shared UI impact check**

  Run:
  `bun run compile`

  From: `apps/extension`

- [x] **Step 5: Run final guards**

  Run:
  `git diff --check`

  Bandit: skip with rationale if this slice touches only TypeScript/package metadata/Backlog documentation and no Python files.

- [x] **Step 6: Update Backlog task and open PR**

  Record verification and known skips/blockers in `TASK-122`, commit, push, and open a draft PR against `dev`.
