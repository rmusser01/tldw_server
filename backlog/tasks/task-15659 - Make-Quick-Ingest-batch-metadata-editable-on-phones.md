---
id: TASK-15659
title: Make Quick Ingest batch metadata editable on phones
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15658
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the responsive source editor from the full-modal review (#3633) on the WebUI from #3638.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Title, speaker, track, tags and inclusion for each source are labeled and reachable at 320 and 390 CSS pixels without horizontal scrolling.
- [x] #2 The desktop editor keeps its columns and each source retains its values and selection across layout changes and submission.
- [x] #3 Publish a PR against dev with desktop and phone screenshots and regression coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new architecture decision. The accepted review calls for a responsive presentation of the same source overrides. Draft persistence and API contracts stay the same.
Approved design in #3633: retain desktop comparison columns and use labeled per-source editors with visible inclusion on phones. Verify via Media → Add media and the submitted collection payload.
Implemented one keyed editor per source. The phone disclosure and inclusion label share a 44px header; each expanded field wraps within the modal. The desktop columns reuse those inputs. Red-to-green component regression and 497 passing scoped tests. Chromium at 320, 390 and 1280px verified geometry, inclusion, keyboard focus across reflow and the collection item payload. The two screenshot widths are 390 and 1280 pixels.
Verified the full Quick Ingest Chromium suite (16 passed), scoped Vitest suite (497 passed), frontend TypeScript, scoped ESLint, translation parity and diff whitespace. Bandit inspected the touched directories; there are no Python code changes. The WebUI source guide and published copy document phone editing. GitHub confirms both screenshot blobs and dev as the base of https://github.com/rmusser01/tldw_server/pull/3639.
Review found that source-only edits do not trigger a collection and native uploads drop metadata. Verify actual collection-item and file-job request payloads for those two supported inputs.
Follow-up saved per-source-only metadata with the default collection name and added durable item bindings for uploaded video files. Mixed URL/file requests retain URL-list binding semantics. A repeated file request reuses the owner-scoped job and original cancellation batch. Regression tests first showed zero planned items and HTTP 422 for a bound native upload. Final checks passed 501 UI tests, 35 API/collection tests (including an actual file job, repeat submit, worker and collection DB read), frontend TypeScript, scoped ESLint and diff whitespace. Bandit found no application issues; the existing B106 in the test fixture and three baseline Ruff import findings are unchanged. User guides updated. Existing ADR assessment still applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phone-width source cards expose title, speaker, track and tags and keep inclusion visible while collapsed. The desktop grid reuses the same inputs. PR #3639 includes desktop and mobile screenshots.
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
