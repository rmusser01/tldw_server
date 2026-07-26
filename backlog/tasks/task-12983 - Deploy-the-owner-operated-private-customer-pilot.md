---
id: TASK-12983
title: Deploy the owner-operated private customer pilot
status: To Do
assignee: []
created_date: '2026-07-22 03:51'
labels:
  - deployment
  - licensing
  - customer-pilot
  - operations
dependencies:
  - TASK-12982
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2727'
  - TASK-12982
documentation:
  - Docs/superpowers/specs/2026-07-21-pr-2727-landing-private-pilot-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After PR #2727 lands and its actual merge commit is verified, deploy an access-controlled pilot from an exact dev revision on infrastructure operated by Robert Benjamin Jake Musser while keeping protected reusable artifact publishing frozen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The deployed source commit, backend image digest, frontend build identity, configuration schema, deployment time, and rollback target are recorded and match the verified merge lineage.
- [ ] #2 Database and configuration migrations are assessed for forward and rollback compatibility, a recoverable data backup exists, and restore verification passes before customer data is admitted.
- [ ] #3 Authentication, tenant isolation, secrets handling, customer-data logging, deletion/export behavior, incident response, and exact-artifact smoke/security checks pass in the deployed environment.
- [ ] #4 Only browser delivery for the access-controlled official service is permitted; deployable protected images, extension packages, source bundles, and tagged protected releases are not published.
- [ ] #5 Pilot access and rollback evidence are recorded, including a verified rollback of code, artifacts, configuration, secrets references, and data migration state.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
