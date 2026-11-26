---
id: TASK-15656
title: Implement Quick Ingest extraction, source repair and job status recovery
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15652
references:
- https://github.com/rmusser01/tldw_server/pull/3636
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved first phase of the full modal design review (#3633) on top of the one-screen WebUI (#3631).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extracted text can be read and copied per source on desktop and phone.
- [x] #2 Repair rejected links, missing files and pasted text on the original queue entry without repeating prior successes.
- [x] #3 Bulk cancellation and minimized progress reflect confirmed, pending and unresolved job outcomes.
- [x] #4 Regression tests and desktop/mobile browser evidence are published with a PR against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design in PR #3633; implementation plan IMPLEMENTATION_PLAN_quick_ingest_recovery_15656.md. ADR required: no; existing scoped operations, session persistence and job APIs govern this work. Implement viewer, source repair and status with test-first slices and browser screenshots.
The WebUI reads opt-in ephemeral articles from the authenticated web scrape API and keeps extract-only text in memory. Restored files and pasted text, rejected links, and recoverable server jobs keep their source identity. Background and stop statuses preserve confirmed outcomes and continue tracking unresolved work. The generated API fingerprint follows the existing contract package described in Docs/ADR/069-frontend-api-contract-package.md. No new architecture decision is required.
Verification passed for 612 WebUI tests, 109 API tests, the frontend TypeScript check, scoped ESLint using the repository suppression baseline, generated OpenAPI checks, and Bandit on the changed Python code. Browser coverage exercises the clipboard, file and paste restoration, corrected links, cancellation, background progress and interrupted jobs.
All ten Chromium scenarios passed together. Six screenshots in the PR show desktop and phone extraction, source repair, unconfirmed cancellation and minimized progress. GitHub confirmed the dev base, open PR and six image files. The repository requires the requester to supply their own Change summary before merge.
Integrated the two verified TASK-15652 Media handoff commits and current dev for PR #3636. Regenerated the OpenAPI fingerprint from the combined schema using make openapi-fingerprint. ADR-069 governs this generated artifact. The same integration flows through #3638, #3639, #3640 and #3641.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered the WebUI extraction viewer with exact-text copy, in-place recovery for links, files and pasted text, and server-confirmed cancellation and background status. Opened PR #3636 against dev on top of #3631. The focused WebUI, browser, API, contract, type, lint and security checks passed. The WebUI guide documents session-only extraction and recovery.
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
