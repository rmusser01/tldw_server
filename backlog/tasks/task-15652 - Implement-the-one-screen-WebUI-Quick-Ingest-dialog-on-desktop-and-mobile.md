---
id: TASK-15652
title: Implement the one-screen WebUI Quick Ingest dialog on desktop and mobile
status: Done
labels:
- webui
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Quick Ingest design. Show eligible and excluded sources, processing prerequisites and the immediate destination together, with a persistent mobile action and outcome-specific results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A mixed queue supports source switching and inline link repair with truthful counts.
- [x] #2 The same screen exposes provider prerequisites, run settings and a destination-specific action. Overwrites receive explicit review.
- [x] #3 Mobile keeps the action reachable and results prioritize opening saved sources or retrying failures.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved in chat. ADR required: no. This change uses the existing WebUI modal, session persistence, provider and processing contracts.
Stages: 1. Implement and unit test the source queue, inline configuration and outcome-specific receipts. 2. Integrate pre-submit validation, overwrite review and responsive dialog shell. 3. Run component and browser checks, lint and Bandit. Use the existing session and request contracts throughout.
Verified all three destinations, scoped overwrite review, preserved receipts on retry, provider correction, and the sticky mobile footer. Two independent Chromium import journeys passed through the Media Review handoff. Captured the 1280×800 and 390×844 states; all phone edges, aligned row controls, and the full-width action measured within 1 px.
Verification includes 412 Quick Ingest and launcher tests, 101 related persistence/service tests, three responsive browser cases and two end-to-end import journeys with simulated APIs. All 76 added or changed English labels match the generated locale catalog. Scoped TypeScript passed and ESLint reported no errors. The live API and worker accepted a file and persisted its contents; the Python smoke fixture was updated to use current credential metadata, a temporary job database and the displayed Media title.
A live Chromium upload with an isolated FastAPI server and Media worker passed. The test opened the saved document and found its contents in Media search. Bandit has no new findings in the Python test/fixture. WebUI verification used Chromium desktop and phone viewports; physical device testing and user studies were outside this change.
PR https://github.com/rmusser01/tldw_server/pull/3631 targets dev. The PR embeds the verified desktop and mobile PNGs at Docs/Design/assets/quick-ingest/.
PR #3631 CI follow-up: the failing Chromium artifact showed the one-source receipt and its Open in Media footer action. Updated the journey to use that action and require the destination URL to carry the canonical Media ID returned by ingestion. The existing API assertions still check the record ID, run-specific title and exact uploaded text for both distinct sources. Verified the receipt navigation, canonical ingestion ID and source-grounding suites (112 tests), targeted TypeScript compilation, ESLint and diff whitespace. Bandit does not parse TypeScript. No architecture or runtime behavior changed.
Verified the corrected handoff in Chromium against disposable FastAPI, Media worker and SQLite data: two separate single-file saves opened Media IDs 1 and 2 respectively, with each canonical record matching its uploaded contents. The journey advanced to Knowledge QA.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved one-screen WebUI Quick Ingest flow with eligible source counts, inline link repair and provider recovery, clear destination-specific actions, explicit overwrite review, and outcome-based results. The mobile sheet keeps the action visible, and scoped retries retain completed sources and review drafts. Updated WebUI guides and browser journeys.
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
