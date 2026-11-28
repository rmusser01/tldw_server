---
id: TASK-15686
title: Make VN asset review inspectable and recoverable
status: Done
labels:
- vn-assets
- webui
- accessibility
references:
- https://github.com/rmusser01/tldw_server/issues/2023
- Docs/Design/2026-11-27-vn-asset-review-workflow.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the independent review-board slice of GitHub #2023. Let reviewers inspect generated images, filter status, and review visible candidates. Preserve selections through refresh and recover cleanly when a mutation fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reviewers can inspect images through the authenticated preview API and filter variants by review status. Preview requests and object URLs are scoped to the current item and pack.
- [x] #2 Bulk review submits visible selected IDs, serializes review writes per pack, and preserves selections on failures. Preferred controls apply to approved variants.
- [x] #3 Targeted tests and desktop/mobile browser checks cover keyboard focus, a populated review queue, and failure recovery.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base commit 9dc8ad084e. ADR required: no. Reuse the existing WebUI API client, verified VN account scope, and asset preview and review routes.
ReviewBoard uses the authenticated preview endpoint, visible status filters, and a scoped selection. The WebUI cancels preview body reads on close and records only image metadata in request history.

The Workbench verifies the active owner before sending, keeps one review write in flight per pack, and settles confirmed writes before reloading details. Account changes detach pending verification and hide unverified board data.

Three Chromium tests passed, including desktop and mobile keyboard review on a 24-item fixture and the existing pack review smoke. The test captures before and after review board screenshots and an inline failure state. TypeScript, scoped ESLint, the Impeccable detector, pre-commit, and independent review passed.

The full VN and API client run passed 400 tests. The two Task69 legacy generation-recovery cases timed out waiting for the storage warning on the account and server variants. The same two cases failed with the same assertion in the broad run on the unmodified dev snapshot at 9dc8ad084e. Production edits use TypeScript.
Pull request [#3666](https://github.com/rmusser01/tldw_server/pull/3666) targets dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The VN review board supports authenticated on-demand previews, status filters and visible selection. Failed reviews preserve keyboard focus and checked items for retry. Owner-scoped writes avoid overlap and stale responses.
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
