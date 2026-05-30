---
id: TASK-568
title: Prove live Web search context status in /chat
status: Done
labels:
- chat
- ux
- proof
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a focused /chat live-browser proof for Web search/context activation. Pre-send status-strip/context rail feedback must reflect active Web search/context state, and the send flow must complete or show a recoverable provider/search error without a misleading ready state. Scope is limited to /chat WebUI context/status proof; do not redesign Web search provider settings, extension handoff, long-session, context-limit, compare/export/share.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspect existing /chat Web search/context status-strip and context rail coverage on latest dev.
- [x] #2 Run focused automated coverage or add the smallest missing regression first if live status feedback is not covered.
- [x] #3 Attempt real-server /chat proof for Web search/context activation in the available local environment and record any environment limits separately from product issues.
- [x] #4 Document verification evidence and any remaining follow-up scope in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Existing real-server cockpit proof toggles Web search/context source status and validates live send flows, but did not cover the active `isSearchingInternet` pre-response status. Added a focused status-strip regression and a Playground integration regression so active Web search now overrides `Ready` while search context is being gathered.

Environment note: local backend and mock OpenAI server required escalated loopback binds/probes in this sandbox. The real-server proof itself passed once those local services were available.

Verification:
- Red: `bun run test src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --maxWorkers=1` failed before implementation because the status strip still rendered `Ready` while `isSearchingInternet`/`webSearchInProgress` was true.
- Green: same focused unit command passed, 48 tests.
- Green: `TLDW_WEB_URL=http://localhost:18024 TLDW_WEB_CMD='env NEXT_DISABLE_MEM_OVERRIDE=1 NODE_OPTIONS=--max-old-space-size=4096 bun run dev -- --webpack -p 18024' TLDW_SERVER_URL=http://127.0.0.1:18023 TLDW_E2E_SERVER_URL=http://127.0.0.1:18023 TLDW_API_KEY=smoke-ci-key-12345 TLDW_E2E_API_KEY=smoke-ci-key-12345 SINGLE_USER_API_KEY=smoke-ci-key-12345 NEXT_PUBLIC_API_URL=http://127.0.0.1:18023 bun run e2e:chat-cockpit:real:focused` passed, 5 tests, no skips.
- Green: `git diff --check`.
- Bandit: not applicable; touched frontend TypeScript/tests only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Live /chat Web search/context proof is now covered by the existing focused real-server cockpit suite plus a new narrow regression for the previously misleading active Web search status. The cockpit status strip now shows Searching web with a short reason while Web search context is being gathered instead of presenting the chat as Ready.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
