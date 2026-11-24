---
id: TASK-15627
title: Stop the WebUI cookie sign-in retry storm when the browser drops the session
  cookie
status: Done
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live in Safari 26.5.2 (production WebUI, quickstart mode, API on plain-http loopback with SESSION_COOKIE_SECURE=True), Safari drops the Secure session cookie, so the runtime cookie bootstrap's profile probe returns 401 and runtime-bootstrap-recovery re-runs the whole bootstrap on every TldwApiClient.initialize(): 33 POST /api/v1/auth/single-user/session (all 200) and 32 401 profile probes in ~55 s, each POST minting a new 30-day AuthNZ session (~50,000 rows per day per idle tab). Bound the recovery: classify attempts as success, transient (network/timeout/408/429/5xx including 503: back off 5 s -> 60 s on the monotonic clock), cookie not kept or not accepted (session 200 + probe 401/403: stop after two attempts until reload or a saved connection), or refused (other 4xx: stop; a 401 names the WebUI/API SINGLE_USER_API_KEY mismatch in one console warning); after a session POST whose probe failed transiently, re-probe before minting another session; keep user clicks able to retry a transient failure; tell the user what to do on the connection settings page until the connection succeeds; report the backend session-count gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With the session POST 200 and the profile probe 401, repeated initialize() calls (5 s poll and request bursts) send at most two session POSTs per page load and then none
- [x] #2 Transient failures (network, timeout, 502) back off with a growing interval capped at 60 s and still recover without a reload once the API answers
- [x] #3 A user click re-arms a transient retry immediately; saving a connection still retires recovery; success path unchanged
- [x] #4 The connection settings page explains that the browser did not keep the sign-in cookie and how to fix it, and one console warning is logged
- [x] #5 Backend gap (no cap on live single-user cookie sessions; 30-day rows outlive hourly cleanup) is either fixed with a small tested cap or filed as a follow-up task
- [x] #6 A refused runtime sign-in (a 4xx other than 408/429) stops automatic attempts; a 503 is transient and backs off
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Bug fix that bounds the existing WebUI runtime cookie bootstrap recovery (runtime-bootstrap.ts / runtime-bootstrap-recovery.ts); no new or changed durable architecture rule. ADR-044 (cookie-session governance) is adjacent but not affected.
Root cause (verified): TldwApiClient.initialize() (TldwApiClient.ts:2244-2245) calls recoverRuntimeAuthBootstrap() first on every call, and initialize() runs from the 5 s disconnected health poll (useServerOnline -> connection.tsx checkOnce:700), every direct WebUI request (requestWithCurrentConfig:2127), getConfig() while config is null (:2652), getDomainCacheRevision (:1973), useChatDraftOwner focus/pageshow revalidation via resolveServicePromptScope, and settings/Test Connection. runtime-bootstrap-recovery.ts only shared an attempt that was still in flight, so every sequential initialize started a new one, and runtime-bootstrap.ts mapped a 200 session POST + 401 profile probe to retry. Red: with session 200 / probe 401 and a 5 s poll for 120 s, 25 session POSTs (one per poll). Fix: recovery backs off 5/10/20/40/60 s after each failed attempt (first retry after page load is immediate), a click re-arms it; cookie-not-kept stops after two attempts per page load and shows a settings notice plus one console warning; refused (4xx except 408/429, or 503) stops. Backend gap filed as TASK-15628.
PR: https://github.com/rmusser01/tldw_server/pull/3578. Bandit skipped: no Python changed. Not run: Playwright single-user-cookie-lifecycle.spec.ts (needs live backend + WebUI; Chromium keeps the cookie so its path is unchanged).
Review round 2 (2026-11-23, PR #3578 'with fixes'): (1) 503 from the session POST is now transient with the backoff. The Next route's own policy 503 has no body or Content-Type and the proxied backend 503 forwards only status and Content-Type, so CSRF-disabled cannot be told from a session-store, rate-limit-store or maintenance 503; none of them mints a session, so the cost is at most one request a minute. (2) Backoff uses performance.now(). (3) The click listener is added when recovery registers and removed in cancelRuntimeAuthBootstrapRecovery, which covers success, give-up, logout, saves and account changes. (4) The notice no longer clears when a save starts; TldwConnectionSettings hides it while the connection store reports connected (five settings test mocks of @/store/connection made callable). (5) Wording no longer blames the browser; SESSION_COOKIE_SECURE=0 advice is scoped to localhost/127.0.0.1-only use. (6) A 401 from the session POST logs one warning naming SINGLE_USER_API_KEY. (7) After a POST whose probe failed transiently, the next attempt re-probes first and only re-POSTs on 401/403. (8) TASK-15628 premise corrected. Red on the round-1 code: cookie-not-kept wording, 401 warning (0 calls), 502 and network-error probe re-POSTs ([0,0,5,15,...] vs [0]), wall clock stepped back (2 vs 3 POSTs), 503 backoff ([0] vs [0,0,5,15,35,75,135,195]), settings notice text and cancel behaviour.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bounded the WebUI runtime cookie sign-in recovery (PR #3578). Every TldwApiClient.initialize() used to re-run the bootstrap; now: a sign-in cookie that is not kept or not accepted (session 200, probe 401/403) stops after two attempts per page load, with a settings notice (shown until the connection succeeds) and one console warning; a refused sign-in (4xx other than 408/429) stops, and a 401 logs one warning naming SINGLE_USER_API_KEY; transient failures (network, timeout, 408, 429, 5xx including 503) back off 5, 10, 20, 40, then every 60 s on the monotonic clock, the first retry after page load running at once, and still recover without a reload; after a session POST whose probe failed transiently, retries re-probe and re-POST only on 401/403; a click re-arms a backed-off retry, and the click listener is removed whenever recovery retires. Backend session-count cap filed as TASK-15628 (premise corrected in review). Tests: 12 new and 1 updated in runtime-bootstrap.test.ts, 1 new in tldw-review-comments.test.tsx, five settings test store mocks made callable; 309 neighbouring tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
