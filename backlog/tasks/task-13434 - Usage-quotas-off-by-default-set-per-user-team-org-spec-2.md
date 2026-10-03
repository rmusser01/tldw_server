---
id: TASK-13434
title: 'Usage quotas off by default, set per user/team/org (spec 2)'
status: To Do
assignee: []
created_date: '2026-10-03 02:41'
labels:
  - quotas
  - backend
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Docs/Design/2026-10-02-usage-quota-posture-design.md in four PRs: A relief (switch + gates), B per-user values and group routes, C storage cut-over and migration, D reporting, docs and ADR-058. Plan for A: Docs/superpowers/plans/2026-10-02-usage-quotas-pr-a-relief.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR A merged: every usage-quota check gated on USAGE_QUOTAS_ENABLED (off by default); billing checks need a wired billing repo
- [ ] #2 PR B merged: quota_resolver, limits.* write path with null-as-delete, team/org override routes, non-storage sites read the resolver
- [ ] #3 PR C merged: storage writers/readers cut over to limits.storage_quota_mb, migration
- [ ] #4 PR D merged: reporting endpoints, ADR-058, docs
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
