---
id: TASK-12976
title: Implement conservative frontend licensing cutoff
status: In Progress
labels:
- licensing
- frontend
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
- Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved pre-counsel licensing cutoff. Establish the prospective Perimeter path boundary, preserve public history and third-party notices, declare the OpenAPI contract Apache-2.0, pause unlicensed contribution paths, isolate the GPL API image, and suspend protected artifact publishing. Do not add custom post-counsel grants or publish protected binaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root scope map and verbatim legal corpus establish the approved prospective licensing boundary without altering prior public grants.
- [ ] #2 Protected package, repository, UI, contribution, and third-party notices consistently describe the frontend as source-available.
- [ ] #3 The generated OpenAPI contract declares Apache-2.0 while the server implementation remains GPL-3.0-only.
- [ ] #4 The required PR workflow blocks third-party protected, legal-governance, and conservative API declaration changes until later grants exist.
- [ ] #5 The GPL API image excludes protected frontend material and rolling protected image publishing is suspended.
- [ ] #6 All verification gates pass and the result is submitted as a license-only PR into dev before PR #2727.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
