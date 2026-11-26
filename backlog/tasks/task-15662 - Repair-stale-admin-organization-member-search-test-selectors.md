---
id: TASK-15662
title: Repair stale admin organization member search test selectors
status: In Progress
labels:
- ci
- testing
- admin-ui
references:
- https://github.com/rmusser01/tldw_server/pull/3612
- https://github.com/rmusser01/tldw_server/actions/runs/45741754724/job/139715218636
- admin-ui/app/organizations/[id]/__tests__/page.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The admin UI ratchet on PR #3612 flags the empty member-search test. Reproduce the cause in the exact case and repair the test using the current visible member-search control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The current accessible search field filters member rows and shows the no-match message for an unmatched query.
- [ ] #2 Run the organization member-search tests and verify the required frontend check on the updated PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Keep existing admin search behavior and verify it using the current accessible input name.
The GitHub head and exact base both had the same obsolete search placeholder selector. The empty-result case was the only ratchet regression because the two diagnostic stacks differed by Node scheduling frames. Reproduced the two stale selectors with the frozen lockfile (two failures and five passes). Both member-search cases now use the accessible input name and still assert the rendered member rows and no-match message. All seven tests in the changed file passed under the hosted worker and timeout settings. The unchanged strict success validator, targeted ESLint, TypeScript and whitespace checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the two stale locators in the organization member-search test file to use the accessible search field. The existing matching and empty-result assertions passed locally. Hosted verification is pending.
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
