---
id: TASK-480
title: Add Watchlists audio status locale keys
status: Done
labels:
- watchlists
- webui
- i18n
- review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/1942
modified_files:
- apps/packages/ui/src/assets/locale/en/watchlists.json
- apps/packages/ui/src/public/_locales/en/watchlists.json
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add and verify locale copy for structured Watchlists audio trigger statuses so Reports never renders raw values like queue_unavailable when i18n resources are loaded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and verified Watchlists audio status locale keys so structured audio trigger statuses resolve to human-readable labels instead of raw status strings. Focused Watchlists frontend regression tests and diff checks pass; Bandit is not applicable because only frontend locale/test files changed.
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
