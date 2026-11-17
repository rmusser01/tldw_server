# Implementation Plan: Web transport delegation — Stage A (TASK-13526)

Branch: `codex/web-transport-delegation-13526`
ADR check: **no new ADR required — governed by ADR-065** (its Follow-up section names this exact migration); recorded on TASK-13526.

## Deep-read conclusions (2026-10-14)
- `tldwRequest` returns a parsed envelope; web policy needs the raw `Response` (headers → session capture, `ApiResponse.headers`, sanitized error bodies). Full delegation therefore requires a raw-response mode + hooks in request-core — a contract change to the extension's load-bearing transport.
- The web auth pipeline (`applyBrowserHeaders`: localStorage/env bearer+apikey, web session header, method-gated CSRF) and the core's plan-auth (tldwConfig-driven, runtime override, cookie CSRF) draw from DIFFERENT credential domains. Convergence needs a credential-strategy interface, not a code move.
- Already shared: URL/base resolution (browser-networking), cancellation classification (`request-events`), and the error model (Stage 2).

## Stage A scope (this PR)
**Goal**: Extract the pure request-plan resolution in request-core and adopt the shared timeout policy web-side — the documented groundwork for full delegation — with zero extension behavior change.
**Changes**:
1. `lib/api.ts`: timeout via shared `deriveRequestTimeout({requestTimeoutMs: <web 30s default>}, url, config.timeout)` — the seeding keeps generic paths at today's 30s (the shared generic default is 10s; adopting it bare would regress the web), while per-path floors (slides 120s, model-metadata 60s) and future per-path config overrides now apply exactly as in the extension. ApiError.retryAfter populated via shared `parseRetryAfter` (adds HTTP-date support).
2. Tests: existing web api/auth/sse/notifications suites green untouched (behavior preservation); a source-contract test pins the shared-policy adoption.
3. The `resolveTldwRequestPlan` extraction is deliberately NOT in this PR: it refactors the extension's load-bearing transport with zero consumer-visible change — doubled review surface for groundwork alone. It is Stage B step 1.
**Success Criteria**: extension `tsc` compile clean; request-core + web api suites green; web `next` build green locally (lesson from the CSS incident).
**Status**: In Progress

## Stage B (follow-up PR, design recorded on TASK-13526)
Credential-strategy interface in `TldwRequestRuntime` → web adopts `resolveTldwRequestPlan` for headers; raw-response mode + `{onResponseHeaders, buildError}` hooks → web fetch loop replaced. Web policy (history, 401 logout, CSRF messaging, provider sanitization, VN mapping) stays in `lib/api.ts` permanently.
