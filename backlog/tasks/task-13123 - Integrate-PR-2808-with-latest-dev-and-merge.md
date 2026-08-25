---
id: TASK-13123
title: Integrate PR 2808 with latest dev and merge
status: In Progress
assignee: []
created_date: '2026-08-25 00:31'
updated_date: '2026-08-25 00:35'
labels:
  - research-workspace
  - merge
  - rebase
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2808'
  - 'https://github.com/rmusser01/tldw_server/pull/2818'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the final merge conflict caused by PR #2818 advancing dev after PR #2808 completed CI, preserve both workstreams, and merge the verified pull request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev is integrated without dropping PR #2808 or PR #2818 behavior.
- [x] #2 The combined OpenAPI fingerprint is regenerated and its drift check passes.
- [ ] #3 Focused overlap tests, repository verification, CI, and review checks pass on the exact pushed head.
- [ ] #4 PR #2808 is merged into dev and the resulting merge commit is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: PR #2818 advanced dev from d3c07a5a to 4091735b after PR #2808 head a025982f completed validation. git merge-tree identified one manual conflict, the generated OpenAPI fingerprint; ChaChaNotes_DB.py and other Notes/Sync overlap auto-merge.

Local integration evidence: merged origin/dev 4091735b into PR head a025982f; the only manual conflict was regenerated from the combined FastAPI schema. Fingerprint now has paths=2039, schemas=3025, sha256=3deff3be1f96... and the canonical OpenAPI drift check passes. Focused overlap suite passed 95 tests with 3 PostgreSQL-service skips: ChaCha task store, SQLite v61 migration, PostgreSQL v61 migration contracts, and sharing schema ownership. Combined ChaChaNotes_DB.py retains both PR #2808 v61 shared-chat migration/store wiring and PR #2818 task-projection drift delegation.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
