# WebUI Dependency Trimming Design

Date: 2026-05-07
Owner: Codex collaboration session
Status: Approved in chat, pending spec review and user review

## Summary

GitHub issue #1346 asks the project to reduce WebUI package usage where browser,
ECMAScript, Node, React, or Next.js built-ins can safely replace direct
dependencies.

The approved sequence is:

1. Audit first.
2. Remove unused or tiny low-risk packages.
3. Replace `axios` with fetch-backed helpers after the dependency map is
   documented.

This sequence keeps the work reviewable. The audit establishes ground truth,
the cleanup slice removes obvious dependency declarations, and the `axios`
replacement is treated as a separate behavioral change because the current
transport layer handles auth, CSRF, timeouts, request history, session headers,
redirects, and error normalization.

## References

- GitHub issue: `https://github.com/rmusser01/tldw_server/issues/1346`
- Backlog task: `TASK-100`
- Frontend app manifest: `apps/tldw-frontend/package.json`
- Shared UI manifest: `apps/packages/ui/package.json`
- WebUI lockfile: `apps/bun.lock`
- Extension manifest for shared-package impact checks:
  `apps/extension/package.json`

## Goals

1. Produce a dependency inventory for the WebUI package surfaces.
2. Classify each direct dependency as `keep`, `remove-now`, `replace-later`,
   `defer-design`, or `investigate-lockfile`.
3. Record whether each package is used by the web app, shared UI package,
   extension app, or only the lockfile.
4. Remove only low-risk packages whose usage is absent or trivially replaceable.
5. Replace `axios` only after preserving the current public WebUI client
   contract and behavioral edge cases.
6. Record measurable impact where practical: dependency count, lockfile delta,
   install size, or bundle/build output.
7. Split implementation into small PRs that are easy to review and verify.

## Non-Goals

1. No one-shot dependency purge.
2. No hand-rolled replacements for security-sensitive packages such as
   `dompurify`.
3. No hand-rolled replacements for complex parsers, PDF/ePub rendering, rich
   text editing, graph visualization, OCR, Monaco, Mermaid, KaTeX, tokenizers,
   schema validation, or archive/document tooling without a separate design.
4. No icon-stack consolidation in the platform-native replacement slice. Icon
   consolidation can be a later dependency reduction effort, but it is not a
   standard-library rewrite.
5. No behavior changes to WebUI API auth, CSRF, session propagation, or request
   logging as part of the audit or quick cleanup slices.

## Current Findings

Initial read-only scans found:

- `axios` has real usage in `apps/tldw-frontend/lib/api.ts`,
  `apps/tldw-frontend/types/common.ts`, and
  `apps/packages/ui/src/services/elevenlabs.ts`.
- `apps/tldw-frontend/lib/api.ts` is a behavioral transport wrapper, not a
  simple import. It owns base URL mutation, auth header construction, CSRF
  header injection, request timing/history, session header capture, 401 login
  redirects, 403 CSRF error normalization, and `retry-after` propagation.
- `apps/packages/ui/src/services/elevenlabs.ts` uses `axios` for external
  ElevenLabs metadata and audio requests. It should not be routed through the
  first-party WebUI API client.
- `pubsub-js`, `buffer`, and `stream-browserify` did not appear in direct source
  imports under `apps/tldw-frontend` or `apps/packages/ui` during the initial
  scan, excluding lockfiles and generated/vendor artifacts.
- `clsx` is used by `apps/tldw-frontend/lib/utils.ts`, while
  `apps/packages/ui/src/libs/utils.ts` already provides a local `cn` helper.
  Replacing `clsx` needs compatibility checks because `clsx` accepts a broader
  input shape than the current local helper.
- The WebUI currently uses multiple icon stacks: `lucide-react`,
  `@heroicons/react`, `@ant-design/icons`, and `react-icons`. This may be a
  valuable later cleanup, but it is not part of the platform-native replacement
  work.
- `@tldw/ui` is consumed by both `apps/tldw-frontend` and `apps/extension`.
  Removing a shared UI dependency or peer dependency needs an extension impact
  check even when the primary GitHub issue is scoped to WebUI dependency
  trimming.

These findings are not a complete audit. The first implementation slice should
make the inventory reproducible and reviewable.

## Work Units

### 1. Dependency Audit

The first PR should add a lightweight audit artifact at
`Docs/Design/WebUI_Dependency_Audit.md`.

The audit should cover:

- `apps/tldw-frontend/package.json`
- `apps/packages/ui/package.json`
- `apps/bun.lock`

Because `@tldw/ui` also feeds the browser extension, the audit should also check
`apps/extension/package.json` when a candidate package appears in the shared UI
package or lockfile. The extension manifest is an impact-check surface, not a
new primary target for issue #1346.

The audit table should include:

- package name
- declared package locations
- import count
- representative import sites
- consumer surface: web app, shared UI, extension, config/script, or lockfile
  only
- category
- decision
- risk
- expected impact
- follow-up slice

Use these decision values:

- `keep`: dependency is justified.
- `remove-now`: no source usage or only safely replaceable usage.
- `replace-later`: replacement looks useful but needs a dedicated PR.
- `defer-design`: package is complex or risky enough to require its own design.
- `investigate-lockfile`: source imports are absent, but lockfile/dependency
  graph details need confirmation.

The audit should explicitly retain or defer complex packages instead of leaving
them ambiguous. This prevents future cleanup work from drifting into unsafe
hand-rolled parsing, rendering, sanitization, editor, or document-processing
code.

### 2. Quick Cleanup

The second PR should remove only packages that meet at least one of these
criteria:

- No source, test, script, or config imports in `apps/tldw-frontend` or
  `apps/packages/ui`, excluding lockfiles and generated/vendor artifacts.
- The package is used only through a tiny local wrapper and the repo already has
  an equivalent helper.
- Removal is validated by install/build/test behavior, not just text search.

Likely candidates from the initial scan:

- `pubsub-js`
- `buffer`
- `stream-browserify`
- possibly `clsx`, if the local class-name helper is made compatibility-safe

`clsx` should not be removed by simply swapping to the existing shared `cn`
helper unless the accepted input shapes remain sufficient for WebUI call sites.
If compatibility needs to expand, add focused tests for class-name joining and
Tailwind conflict merging. If that compatibility work grows beyond a mechanical
helper replacement, split `clsx` into its own cleanup PR instead of bundling it
with unused declaration removals.

If a cleanup removes or changes a package declared by `apps/packages/ui`, also
check whether `apps/extension/package.json` still needs that package for direct
extension code or for shared UI entrypoints.

### 3. `axios` Replacement

The third PR should replace `axios` with platform-native `fetch` helpers.

The first-party client in `apps/tldw-frontend/lib/api.ts` should preserve the
current public surface:

- `apiClient.get`
- `apiClient.post`
- `apiClient.put`
- `apiClient.delete`
- `apiClient.patch`
- `getApiBaseUrl`
- `buildAuthHeaders`
- `shouldIncludeBrowserCredentials`
- `hasEnvAuthConfigured`
- `hasExplicitAuthHeaders`
- `ApiError`
- either the current default `api` export with `api.defaults.baseURL` semantics,
  or an explicit replacement setter/getter with all current callers migrated in
  the same PR
- mutable base URL behavior used by `apps/tldw-frontend/hooks/useConfig.tsx`

Do not keep `AxiosRequestConfig` or `InternalAxiosRequestConfig` as public types
after `axios` is removed. Define a small local request config type that supports
the options current call sites use, including per-request `headers`,
`withCredentials`, `signal`, and timeout behavior. If additional axios config
keys are discovered during implementation, either support them deliberately or
remove/migrate the call site in the same PR.

The replacement should preserve these behaviors:

- JSON request and response handling.
- `FormData` support without forcing `Content-Type`.
- timeouts through `AbortController`.
- browser credentials based on the current auth mode.
- per-request credential overrides such as notification calls that pass
  `withCredentials`.
- per-request header overrides from current callers.
- session header creation and response-header capture.
- bearer token, API bearer, and `X-API-KEY` header behavior.
- CSRF header injection for mutating requests when API-key auth is absent.
- request history logging for success and failure.
- 401 cleanup and login redirect rules.
- 403 CSRF message normalization.
- `retry-after` parsing on error responses.

The ElevenLabs service in `apps/packages/ui/src/services/elevenlabs.ts` should
use a small local fetch helper instead of the first-party WebUI client. It calls
an external origin and has its own API-key and timeout semantics. Preserve the
existing exported results:

- `getVoices(apiKey, options?) -> Promise<Voice[]>`
- `getModels(apiKey, options?) -> Promise<Model[]>`
- `generateSpeech(apiKey, text, voiceId, modelId, speed?) -> Promise<ArrayBuffer>`

## Data Flow

### Current first-party request flow

1. Caller uses `apiClient` or the default `api` instance.
2. `axios` applies base URL, JSON headers, credentials, and timeout defaults.
3. Request interceptor adds session, auth, API key, and CSRF headers.
4. Response interceptor captures session headers, records request history, and
   normalizes errors.

### Target first-party request flow

1. Caller uses the same `apiClient` methods.
2. The fetch-backed client resolves the current base URL and request URL.
3. The client builds headers with the existing auth and CSRF rules.
4. The client applies timeout cancellation through `AbortController`.
5. The client sends `fetch` with the correct `credentials` mode.
6. The client parses JSON, text, binary, or empty responses based on request
   options and response headers.
7. The client records request history and raises `ApiError` for non-2xx
   responses.

## Error Handling

The audit and quick cleanup slices should fail closed: if a package has
uncertain usage, mark it `investigate-lockfile` or `defer-design` instead of
removing it.

The fetch-backed client should map network failures, aborts, non-2xx responses,
invalid JSON, and empty responses deliberately. It should keep compatibility
with existing UI code that expects `ApiError.status`, `ApiError.statusCode`,
`ApiError.detail`, and `ApiError.retryAfter`.

Timeout errors should be distinguishable enough for callers and tests to avoid
misclassifying them as server responses.

## Testing And Verification

### Audit PR

- Run the import inventory script or documented `rg` commands used to populate
  the audit.
- Run `git diff --check`.
- Record that Bandit is skipped because this is docs-only.

### Quick Cleanup PR

- Run source import guards for each removed package.
- Run config/script/test import guards for each removed package.
- Confirm direct-vs-transitive ownership through the package manager or lockfile
  before removing direct declarations.
- Run `bun install` from `apps/` to update `apps/bun.lock`.
- Run focused tests for any changed helper behavior.
- Run a targeted frontend lint/build command if the current baseline allows it.
- If `apps/packages/ui` dependencies change, run at least one targeted
  extension compile or unit-test command that exercises shared UI imports, or
  record the current blocker.
- Run `git diff --check`.
- Record Bandit skip if no Python code is touched.

### `axios` Replacement PR

- Add or update tests for:
  - default `api` export compatibility or the replacement base-URL setter
  - auth header construction
  - CSRF header behavior
  - per-request headers and credential overrides
  - timeout and abort handling
  - request-history logging
  - base URL mutation through `useConfig`
  - 401 redirect behavior
  - 403 CSRF normalization
  - `retry-after` mapping
  - `FormData` requests
  - ElevenLabs JSON and array-buffer requests
- Run focused WebUI tests for `apps/tldw-frontend/lib/api.ts`,
  `apps/tldw-frontend/hooks/useConfig.tsx`, and ElevenLabs consumers.
- Run `bun install` from `apps/` after dependency removal.
- Run `git diff --check`.
- Record Bandit skip if no Python code is touched.

## Rollout Plan

1. Create and land the audit PR.
2. Create follow-up tasks from the audit table for each `remove-now` or
   `replace-later` package.
3. Land one quick cleanup PR for unused declarations and tiny local-helper
   replacements.
4. Land a separate `axios` replacement PR.
5. Leave icon-stack consolidation and complex renderer/parser packages as
   separate future design items.

## Acceptance Criteria

1. The design documents the approved A -> C -> B sequence.
2. The design identifies `apps/tldw-frontend/package.json`,
   `apps/packages/ui/package.json`, and `apps/bun.lock` as target surfaces.
3. The design includes guardrails for security-sensitive and complex-domain
   packages.
4. The design defines verification expectations for audit, quick cleanup, and
   `axios` replacement slices.
5. The design links to GitHub issue #1346 and `TASK-100`.
