---
id: TASK-13687
title: Bind Admin query cache to verified connection and account identity
status: To Do
labels:
- admin
- security
- cache
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Independent PR3229 integration review found static Admin query keys and a module singleton client introduced by merged PR3218. Purging occurs through overview signal loading and compares only server URL. Direct Admin navigation after server or same-server account changes can reuse prior-authority cached data. This is upstream baseline behavior, does not reach chat caches or ownership, and is not worsened by PR3229. Runtime reproduction remains required before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce direct Admin navigation after server and same-server account or org changes using the actual provider and query lifecycle with API I/O doubled.
- [ ] #2 Prior-authority data and held responses cannot publish after authority changes, including A to B to A.
- [ ] #3 Same-authority request deduplication and mutation invalidation remain intact.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Source review by independent reviewerB on e3987dda: apps/packages/ui/src/services/tldw/adminQueries.ts, components/Option/Admin/AdminQueryProvider.tsx, components/Option/Admin/admin-module-signals.ts. Follow-up only; no runtime reproduction or fix claimed. Linked PR3229 integration TASK13686. ADR assessment deferred until design; preserve governing ownership/cache contracts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
