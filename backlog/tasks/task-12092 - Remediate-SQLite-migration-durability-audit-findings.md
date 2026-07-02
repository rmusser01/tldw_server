---
id: TASK-12092
title: Remediate SQLite migration durability audit findings
status: In Progress
created_date: 2026-07-02 03:04
labels:
- audit
- remediation
- db
- migrations
- wave-1
priority: high
references:
- AUDIT-2026-06-27-DB-001
- AUDIT-2026-06-27-DB-002
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md
modified_files:
- Docs/superpowers/plans/2026-07-02-sqlite-migration-durability-remediation.md
- tldw_Server_API/app/core/DB_Management/db_migration.py
- tldw_Server_API/app/core/DB_Management/media_db/
- tldw_Server_API/app/core/DB_Management/migrations/
- tldw_Server_API/tests/DB_Management/
updated_date: 2026-07-02 03:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for the 2026-06-27 SQLite migration durability findings: unsupported legacy Media DB handling, domain-scoped migration packaging, and atomic migration body/ledger/schema updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written before production code changes.
- [ ] #2 Legacy Media DB versions below the supported minimum are upgraded through a tested path or rejected with explicit recovery guidance.
- [ ] #3 Multi-statement migration failure does not leave a successful ledger row or schema_version bump, and avoids partial DDL where SQLite permits rollback.
- [ ] #4 Migration packaging no longer applies incompatible scripts to the wrong database domain.
- [ ] #5 Focused legacy-version, atomicity, migration-scope, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wave 1 reconfirmation on refreshed origin/dev 30495536d3 showed DB-001 and DB-002 still apply. Smallest safe version decision: support fresh DBs and v22-to-v23 migrations in this slice; explicitly reject schema versions 1..21 with backup/recovery guidance unless historical migration bodies become available.
Implementation plan added at Docs/superpowers/plans/2026-07-02-sqlite-migration-durability-remediation.md. Plan locks the supported legacy decision to fresh DBs plus v22-to-v23 automatic migration; schema versions 1..21 get explicit recovery guidance unless historical migration bodies are supplied.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched production paths or skip documented
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
