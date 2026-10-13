---
id: TASK-13434
title: Usage quotas off by default, set per user/team/org (spec 2)
status: Done
assignee: []
created_date: 2026-10-03 02:41
updated_date: 2026-10-03 14:09
labels:
- quotas
- backend
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Docs/Design/2026-10-02-usage-quota-posture-design.md in four PRs: A relief (switch + gates), B per-user values and group routes, C storage cut-over and migration, D reporting, docs and ADR-064. Plan for A: Docs/superpowers/plans/2026-10-02-usage-quotas-pr-a-relief.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR A merged: every usage-quota check gated on USAGE_QUOTAS_ENABLED (off by default); billing checks need a wired billing repo
- [x] #2 PR B merged: quota_resolver, limits.* write path with null-as-delete, team/org override routes, non-storage sites read the resolver
- [x] #3 PR C merged: storage writers/readers cut over to limits.storage_quota_mb, migration
- [x] #4 PR D merged: reporting endpoints, ADR-064, docs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR A opened as #3098 (fix/usage-quotas-off-by-default): switch USAGE_QUOTAS_ENABLED off by default; billing checks need a wired billing repo; one guard per quota choke point; evals daily caps leave the stock RG policy; docs. Subagent-driven: 4 task reviews (3 needed one fix round each for tests that could not fail), Fable final review 'with fixes', one fix wave, re-review clean.

PR A merged as #3098 (ea1eda99f9) on 2026-10-03. PR B in progress: plan Docs/superpowers/plans/2026-10-03-usage-quotas-pr-b-per-user-values.md

PR B merged as #3144 (3700e2d6e7) on 2026-10-04: quota_resolver (user > most generous team > most generous org; 0 blocks; 60 s cache; fail-open), generic limits.* write path with null-as-delete, platform-admin team/org override routes, every non-storage site reads the resolver. Qodo found 17 (14 fixed incl. audio N-1 off-by-one, evals double count, batch undercount; 3 declined). Follow-ups TASK-13435..13437. PR C (storage cut-over + migration) in progress on fix/usage-quotas-storage.
PR C merged as #3199 (5775d3fbbe) on 2026-10-05: per-user storage quota moved from users.storage_quota_mb to limits.storage_quota_mb (user > most generous team > most generous org; null unlimited; 0 blocks). StorageQuotaService reads the resolver; every writer writes the override (same-connection inside request transactions); readers show the enforced value; SQLite migration 100 + one-time PG backfill (marker table authnz_data_backfills) copy non-default values. Fixed along the way: /admin/storage-quotas/users/{id} treated the user id as an org id; check_combined_quota raised TypeError; a 0 quota read as unlimited. No Qodo review (workspace out of credits). PR D (reporting, ADR-058, docs) next.
PR D merged as #3202 (1fc353c3f6) on 2026-10-06: the limits views report the enforced limits.* values (audio daily and monthly minutes from the ledger, evaluations daily caps; null is unlimited); the chatbooks tier tables and endpoint pre-checks are removed; the media 429 sends RateLimit headers; DEFAULT_STORAGE_QUOTA_MB is deprecated; a monthly-minutes denial is now named monthly. Docs: ADR-064, Docs/Operations/Usage_Quotas.md, the Organization_Administration billing rewrite. No Qodo review (out of credits); a Fable whole-branch review stood in. Follow-up TASK-13510 (audit events on the storage quota admin endpoints).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Spec 2 shipped in four PRs: #3098 (A), #3144 (B), #3199 (C) and #3202 (D), merged 2026-10-03 to 2026-10-06. Usage quotas are per-user limits.* values, off unless USAGE_QUOTAS_ENABLED is on, resolved from the user value, then the most generous team value, then the most generous org value; 0 blocks and none is set by default (ADR-064, Docs/Operations/Usage_Quotas.md). Each PR body records its verification: task reviews, a Fable whole-branch review, a rebased ship sweep with every failure triaged, Bandit, ruff tallies against the merge base, the OpenAPI fingerprint, and Postgres tests run rather than skipped. Known skips: Qodo reviewed A and B only (out of credits for C and D, where a Fable review stood in); the 13 Evaluations test_api_endpoints failures (503 credential_store_unavailable) and the MediaIngestion_NEW auth leak under xdist are pre-existing on dev. Follow-ups: TASK-13435, TASK-13436, TASK-13437, TASK-13510.
<!-- SECTION:FINAL_SUMMARY:END -->
