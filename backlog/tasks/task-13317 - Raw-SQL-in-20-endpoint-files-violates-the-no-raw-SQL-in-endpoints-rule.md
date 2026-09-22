---
id: TASK-13317
title: Raw SQL in 20 endpoint files violates the no-raw-SQL-in-endpoints rule
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
labels:
  - duplication
  - api
  - architecture
dependencies: []
references:
  - 'tldw_Server_API/app/api/v1/endpoints/admin/admin_rbac.py:246'
  - 'tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:596'
  - Docs/Architecture.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs/Architecture.md states "no raw SQL in endpoints". 20 endpoint files violate it; 17 have an existing core owner. Largest: admin/admin_rbac.py (69 statements), jobs_admin.py (56), sync.py (8), prompt_studio_evaluations.py (8), admin_rate_limits.py (8).

Five sites reach past a core module PRIVATE API - jm._connect, jm._pg_cursor, vector_store_batches_db._connect, PromptStudioDatabase._execute, Collections_DB._coerce_bool_flag - so the owners cannot refactor internals. Ten files hand-roll the SQLite/PostgreSQL dialect branch, which is why core/AuthNZ/database.py:1966 _normalize_sqlite_sql exists at all; its docstring calls itself a safety net for when a dollar-style query slips through.

Only 3 orphan tables need new code: a rate_limit_repo for rbac_role_rate_limits + rbac_user_rate_limits, password_history into Users_DB, user_prompts into the prompts owner.

No DDL in endpoints and no injectable interpolation was found - every f-string site was checked individually. Scope is layering, not security.

Owner-only (app/api/v1/**). Source: synthesis F19
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 14 small files route through their existing core owners
- [ ] #2 Three orphan tables get a core owner
- [ ] #3 No endpoint reaches a core private API
- [ ] #4 A lint ratchet prevents new raw SQL in endpoints
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
