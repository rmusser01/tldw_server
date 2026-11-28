---
id: TASK-15654
title: Keep cold-start chat and model discovery responsive
status: In Progress
labels:
- bug
- chat
- api
references:
- https://github.com/rmusser01/tldw_server/issues/3135
- https://github.com/rmusser01/tldw_server/pull/3632
- Docs/ADR/069-frontend-api-contract-package.md
- Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the remaining cold-start catalog and startup discovery fixes around the existing server OpenAPI warm-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model metadata degrades safely when provider policy cannot be read, retaining the credential enforcement boundary.
- [x] #2 The shared chat model catalog retries transient discovery failures and retains manual recovery.
- [ ] #3 The live UX harness verifies fresh startup requests and first composer send without prewarming the API.
- [x] #4 Background startup skips the separate OpenAPI drift probe.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Existing decisions cover this fix. ADR-069 governs the OpenAPI contract artifacts and ADR-025 governs the LLM routing and provider policy boundary. The existing server lifespan prepares OpenAPI in a worker thread. The redundant startup drift probe was removed. Chat and clipper capability checks can fetch the schema on cache misses.
Verification passed for 220 focused backend tests, 71 shared UI tests, shared UI type checking, scoped ESLint, and the existing OpenAPI fingerprint and path union. Scoped Ruff and Bandit found zero new findings against dev.
The browser test reproduced an 8.82-second first model read before the MLX directory check moved ahead of the tokenizer import. Run issue-3135-cold-03 streamed a reply on the first send and measured a 1.19-second catalog read in server logs. One repeat reached the reply but hung in Playwright waiting for stream closure. A later repeat measured 6.26 seconds in Chromium while the host CPU was fully busy. The regression times the first rendered reply and compiles the HTML route before opening Chromium.
Run issue-3135-cold-06 was stopped after Next.js spent more than two minutes compiling the initial page on the loaded host. The runner stopped its isolated services. Live confirmation of the final timing test remains pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the redundant OpenAPI drift probe. Local model discovery stays available when cloud provider policy is unavailable. The picker retries transient HTTP failures twice and offers refresh for an empty catalog. MLX discovery checks for a local artifact before importing its tokenizer loader. The first cold browser run passed and the final harness confirmation remains open.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
