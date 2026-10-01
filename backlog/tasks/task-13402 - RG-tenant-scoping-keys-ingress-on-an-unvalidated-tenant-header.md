---
id: TASK-13402
title: RG tenant scoping keys ingress on an unvalidated tenant header
status: To Do
assignee: []
created_date: '2026-09-30 09:45'
labels:
  - rate-limit
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With tenant scoping enabled, deps.derive_entity_key returns tenant:<X-TLDW-Tenant header> (deps.py ~118-122, tenant.py ~38-41) without validating the header against the caller's principal. A client that rotates the header gets a fresh ingress bucket on every request. This predates the RG safety net: R18 kept tenant precedence unchanged, and the auth single-charge skip no longer trusts tenant: entities (commit 832e4a69df). Tenant scoping is opt-in and off by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tenant entity comes only from a validated principal's tenant or org, or an unvalidated header can never mint a new bucket
- [ ] #2 A test proves that rotating the tenant header on one IP shares one bucket
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
