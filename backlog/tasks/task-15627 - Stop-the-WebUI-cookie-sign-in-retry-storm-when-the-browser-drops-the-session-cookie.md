---
id: TASK-15627
title: Stop the WebUI cookie sign-in retry storm when the browser drops the session
  cookie
status: In Progress
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live in Safari 26.5.2 (production WebUI, quickstart mode, API on plain-http loopback with SESSION_COOKIE_SECURE=True), Safari drops the Secure session cookie, so the runtime cookie bootstrap's profile probe returns 401 and runtime-bootstrap-recovery re-runs the whole bootstrap on every TldwApiClient.initialize(): 33 POST /api/v1/auth/single-user/session (all 200) and 32 401 profile probes in ~55 s, each POST minting a new 30-day AuthNZ session (~50,000 rows per day per idle tab). Bound the recovery: classify attempts as success, transient (network/timeout/5xx: back off 5 s -> 60 s), cookie not kept (session 200 + probe 401/403: stop after two attempts until reload or a saved connection), or refused (4xx/503: stop); keep user clicks able to retry a transient failure; tell the user what to do on the connection settings page; report the backend session-count gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With the session POST 200 and the profile probe 401, repeated initialize() calls (5 s poll and request bursts) send at most two session POSTs per page load and then none
- [x] #2 Transient failures (network, timeout, 502) back off with a growing interval capped at 60 s and still recover without a reload once the API answers
- [x] #3 A refused runtime sign-in (4xx other than 408/429, or 503) stops automatic attempts
- [x] #4 A user click re-arms a transient retry immediately; saving a connection still retires recovery; success path unchanged
- [x] #5 The connection settings page explains that the browser did not keep the sign-in cookie and how to fix it, and one console warning is logged
- [x] #6 Backend gap (no cap on live single-user cookie sessions; 30-day rows outlive hourly cleanup) is either fixed with a small tested cap or filed as a follow-up task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Bug fix that bounds the existing WebUI runtime cookie bootstrap recovery (runtime-bootstrap.ts / runtime-bootstrap-recovery.ts); no new or changed durable architecture rule. ADR-044 (cookie-session governance) is adjacent but not affected.
Root cause (verified): TldwApiClient.initialize() (TldwApiClient.ts:2244-2245) calls recoverRuntimeAuthBootstrap() first on every call, and initialize() runs from the 5 s disconnected health poll (useServerOnline -> connection.tsx checkOnce:700), every direct WebUI request (requestWithCurrentConfig:2127), getConfig() while config is null (:2652), getDomainCacheRevision (:1973), useChatDraftOwner focus/pageshow revalidation via resolveServicePromptScope, and settings/Test Connection. runtime-bootstrap-recovery.ts only shared an attempt that was still in flight, so every sequential initialize started a new one, and runtime-bootstrap.ts mapped a 200 session POST + 401 profile probe to retry. Red: with session 200 / probe 401 and a 5 s poll for 120 s, 25 session POSTs (one per poll). Fix: recovery backs off 5/10/20/40/60 s after each failed attempt (first retry after page load is immediate), a click re-arms it; cookie-not-kept stops after two attempts per page load and shows a settings notice plus one console warning; refused (4xx except 408/429, or 503) stops. Backend gap filed as TASK-15628.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
