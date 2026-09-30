---
id: TASK-13396
title: RG ingress safety net (spec 1 of 2)
status: To Do
assignee: []
created_date: '2026-09-30 01:39'
labels:
  - resource-governance
  - backend
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Docs/Design/2026-09-29-rg-ingress-safety-net-design.md in three PRs (relief, coverage, switch and docs). Plan: Docs/superpowers/plans/2026-09-29-rg-ingress-safety-net.md. TASK-13395 closes with PR B.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR A merged: safety-net defaults and permanent-429 fixes in both backends
- [ ] #2 PR B merged: resolver, route index, principal identity, audits, route-map lints, WebUI replay; TASK-13395 closed
- [ ] #3 PR C merged: single RG switch, config hygiene, ADR-056, docs
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
