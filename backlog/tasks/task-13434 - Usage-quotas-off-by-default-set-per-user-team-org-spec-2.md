---
id: TASK-13434
title: 'Usage quotas off by default, set per user/team/org (spec 2)'
status: To Do
assignee: []
created_date: '2026-10-03 02:41'
updated_date: '2026-10-03 14:09'
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
- [x] #1 PR A merged: every usage-quota check gated on USAGE_QUOTAS_ENABLED (off by default); billing checks need a wired billing repo
- [x] #2 PR B merged: quota_resolver, limits.* write path with null-as-delete, team/org override routes, non-storage sites read the resolver
- [ ] #3 PR C merged: storage writers/readers cut over to limits.storage_quota_mb, migration
- [ ] #4 PR D merged: reporting endpoints, ADR-058, docs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR A opened as #3098 (fix/usage-quotas-off-by-default): switch USAGE_QUOTAS_ENABLED off by default; billing checks need a wired billing repo; one guard per quota choke point; evals daily caps leave the stock RG policy; docs. Subagent-driven: 4 task reviews (3 needed one fix round each for tests that could not fail), Fable final review 'with fixes', one fix wave, re-review clean.

PR A merged as #3098 (ea1eda99f9) on 2026-10-03. PR B in progress: plan Docs/superpowers/plans/2026-10-03-usage-quotas-pr-b-per-user-values.md
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR B merged as #3144 (3700e2d6e7) on 2026-10-04: quota_resolver (user > most generous team > most generous org; 0 blocks; 60 s cache; fail-open), generic limits.* write path with null-as-delete, platform-admin team/org override routes, every non-storage site reads the resolver. Qodo found 17 (14 fixed incl. audio N-1 off-by-one, evals double count, batch undercount; 3 declined). Follow-ups TASK-13435..13437. PR C (storage cut-over + migration) in progress on fix/usage-quotas-storage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
