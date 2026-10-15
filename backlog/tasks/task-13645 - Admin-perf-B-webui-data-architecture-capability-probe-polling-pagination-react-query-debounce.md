---
id: TASK-13645
title: "Admin perf B: webui data architecture \u2014 capability probe, polling, pagination,\
  \ react-query, debounce"
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Executed as part of PR #3218 (merged 2026-10-14). Implements F8-F13 per Docs/Plans/2026-10-06-admin-webui-perf-B-data-architecture-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
5/5 stages review-approved. Commits 9c56497036+4400443b98, d5b0ff0147, 54d6cd40ab, a83469c731+2042cc3f77, b48cf4d9e9. Tri-state capability probe (1.4MB spec once/session); polling split + tab-visibility gate; orgs server pagination + api-key remote search (users>100 reachable); react-query reference-data cache; debounced budget input; usage date ranges threaded to all datasets.
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
