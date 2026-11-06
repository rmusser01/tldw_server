---
id: TASK-15430
title: Retarget the shared current-account menu to the dev header
status: Done
created_date: 2026-11-06 23:46
references:
- https://github.com/rmusser01/tldw_server/pull/3400
updated_date: 2026-11-06 23:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update existing draft PR #3400 against dev. Reuse the current composer behavior and preserve both header layouts, notification and account authority, keyboard focus, and existing preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only the public shared menu changes missing from dev remain in the existing draft.
- [x] #2 Verify account-menu, notification, keyboard, and compact composer behavior on dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design and implementation plan: Docs/Design/2026-11-02-header-account-menu.md. ADR required: no. The change keeps the existing authentication, routing, and user-preference contracts.
Dev 2f402bd7e3 already provides persisted Tools disclosure and visible model, attachments and Send on mobile and desktop. Port the original public account component into the shared header control; keep both layouts and test forward/reverse keyboard focus. Reviewed the integration independently. All 184 focused tests in 12 suites passed, as did WebUI typecheck. ESLint: zero errors, one pre-existing share-label warning. Bandit: no Python in the touched directory. No full production build, live sign-in or physical-device verification. The human-authored summary remains a separate merge requirement.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ported the existing draft to dev, preserving its account actions and notification fallback. The current composer already meets the mobile requirement. The draft remains subject to the requester-written change summary.
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
