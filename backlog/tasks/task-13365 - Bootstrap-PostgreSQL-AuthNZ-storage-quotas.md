---
id: TASK-13365
title: Bootstrap PostgreSQL AuthNZ storage quotas
status: Done
assignee: []
created_date: '2026-09-25 19:07'
updated_date: '2026-09-25 19:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A fresh PostgreSQL AuthNZ database created by the production bootstrap lacks storage_quotas, so organization quota writes fail and a live multi-user email upload cannot pass billing enforcement. Add PostgreSQL schema parity with SQLite migration 051 and verify real quota CRUD against the standard isolated PostgreSQL fixture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh PostgreSQL AuthNZ bootstrap creates storage_quotas with org/team foreign keys, scope check and unique non-null org/team indexes
- [x] #2 Real PostgreSQL quota repository upsert/read succeeds using the existing isolated_test_environment fixture
- [x] #3 Packaged PostgreSQL bootstrap schema and runtime ensure remain idempotent and aligned
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direct asyncpg inspection of the fresh isolated AuthNZ database found no storage_quotas relation after production bootstrap. SQLite migration 051 defines it; PostgreSQL runtime ensure and packaged schema omit it. TDD regression will use isolated_test_environment and real AuthnzStorageQuotasRepo.

Runtime and packaged PostgreSQL schemas create scoped storage quotas. Real isolated PostgreSQL quota upsert/read passed; catalog confirmed scope check, two foreign keys, and scoped unique indexes. Bandit 0 findings; fatal Ruff clean. Full live PostgreSQL probe passed. Skip: no production data tested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Runtime and packaged PostgreSQL schemas create scoped storage quotas. Real isolated PostgreSQL quota upsert/read passed; catalog confirmed scope check, two foreign keys, and scoped unique indexes. Bandit 0 findings; fatal Ruff clean. Full live PostgreSQL probe passed. Skip: no production data tested.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
