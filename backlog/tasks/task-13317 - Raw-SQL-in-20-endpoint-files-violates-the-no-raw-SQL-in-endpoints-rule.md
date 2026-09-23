---
id: TASK-13317
title: Raw SQL in 20 endpoint files violates the no-raw-SQL-in-endpoints rule
status: In Progress
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 16:24'
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
- [x] #3 No endpoint reaches a core private API
- [x] #4 A lint ratchet prevents new raw SQL in endpoints
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC4 DONE in 2b92365ee1: tests/lint/test_no_raw_sql_in_endpoints.py. ACs 1-3 remain.

MEASUREMENT (AST, not grep). Executed raw SQL -- a SQL literal, an f-string whose literal head is SQL, or a module-level name bound to one, passed as the FIRST argument to a call named execute/executemany/executescript/fetchone/fetchall/fetchval/fetch/_execute/execute_query/query/exec_driver_sql/run:

  22 files, 138 statements.

  56  admin/admin_rbac.py                      2  media/debug.py
  32  jobs_admin.py                            2  media/listing.py
   8  prompt_studio/prompt_studio_evaluations.py  2  media/navigation.py
   7  sync.py                                  2  notes.py
   4  admin/admin_ops.py                       2  prompt_studio/prompt_studio_status.py
   4  admin/admin_rate_limits.py               1  admin/__init__.py
   4  admin/admin_tenant_provisioning.py       1  audio/audio_studio.py
   4  users.py                                 1  chat_documents.py
   1  health.py                                1  media/versions.py
   1  outputs_templates.py                     1  prompt_studio/prompt_studio_projects.py
   1  vector_stores_openai.py                  1  watchlists.py

Same files and ordering as the review; the statement counts differ because the review counted differently. A grep-shaped count says 98 files / 470 statements -- that over-counts by catching docstrings, error messages, OpenAPI examples, and text2sql.py, which is an endpoint ABOUT SQL that executes none of its own. Worth knowing before anyone re-measures and thinks the problem grew.

The ratchet was verified to actually fail, not assumed: adding cur.execute("SELECT 1 FROM probe_table") to feedback.py trips the new-file check; an inflated baseline entry trips the improvement check; a bogus entry trips the stale check. Four checks, because a bare count would rot -- new files, growth, stale entries, and unrecorded improvements are each caught.

AC2 GROUNDWORK -- the rbac rate-limit orphan is confirmed and mapped:
  rbac_role_rate_limits and rbac_user_rate_limits have SCHEMA but no repo. Migrations
  exist at AuthNZ/migrations.py:2292,2306 (SQLite) and AuthNZ/pg_migrations_extra.py:1197,1209
  (Postgres), and nothing under core/ reads or writes them. AuthnzRateLimitsRepo is NOT their
  owner -- it owns the `rate_limits` and `account_lockouts` tables, a different concern
  (throttling windows and lockouts).

  Two call sites query them directly, both hand-rolling the dialect branch:
    1. endpoints/admin/admin_rate_limits.py -- the CRUD (4 executed statements).
       Its _POSTGRES_RATE_LIMIT_LIST_QUERIES and _SQLITE_RATE_LIMIT_LIST_QUERIES
       (lines 85-130) are BYTE-IDENTICAL. Verified. There is no dialect difference at
       all in the list path; the split is pure duplication.
    2. API_Deps/auth_deps.py:2206 enforce_rbac_rate_limit -- the PER-REQUEST lookup.
       Here the two branches genuinely diverge in shape: Postgres does TWO round trips
       (fetch role_ids from user_roles, then rbac_role_rate_limits WHERE role_id = ANY($1)),
       while SQLite does ONE JOIN against user_roles. Same semantics, same expiry filter,
       different cost and different row shapes -- Postgres returns dict rows, SQLite tuples,
       reconciled by an isinstance check at :2266-2273.

  A core owner should therefore expose: list_all, upsert_role, clear_role, upsert_user,
  clear_user, and resolve_effective_limits(user_id, resource). Note the two call sites hold
  DIFFERENT handles -- the admin endpoints take get_db_transaction (a transaction connection,
  or a pool adapter in test mode) while auth_deps holds a DatabasePool -- so the owner needs
  to serve both without changing the admin path's transaction semantics. That is the one
  design decision in this piece, and it is why it was not rushed here.

AC3 GROUNDWORK. The five named private reaches are confirmed present:
  jobs_admin.py:595,680,1383,1531  jm._connect()
  jobs_admin.py:600,686,710,1387   jm._pg_cursor(conn)
  prompt_studio/prompt_studio_projects.py:226  db._execute(
  audio/audio_studio.py:866  collections_db._coerce_bool_flag (already carries # noqa: SLF001)
  plus vector_store_batches_db._connect
  Note users.py:260,265 _coerce_bool_flag is a LOCAL function, not a core reach -- do not
  count it.
  A broad AST sweep for private-attribute access from endpoints finds 147 reaches across 54
  files, but most are endpoint-to-endpoint aliases (e.g. admin_mod._is_postgres_backend), so
  that total is not a useful target; the five core-DB reaches are.

NOT RATCHETED, deliberately: the ten files that hand-roll the SQLite/PostgreSQL dialect
branch. A count is the wrong instrument -- the branch is the problem, not its frequency. This
is why core/AuthNZ/database.py:1966 _normalize_sqlite_sql exists; its own docstring calls
itself a safety net for when a dollar-style query slips through.

2026-09-23: AC3 done - the five named private reaches are gone (PromptStudioDatabase._execute -> get_project_by_name; Collections_DB._coerce_bool_flag -> get_live_audio_studio_section_text; vector_store_batches_db._connect -> count_batches; jm._connect/_pg_cursor in prompt_studio_status -> core/Jobs/queue_stats.py and in jobs_admin -> core/Jobs/admin_operations.py; plus jm._update_gauges/_get_queue_flags -> public wrappers). Pinned by test_no_endpoint_reaches_database_internals. Five files left the raw-SQL baseline (jobs_admin, prompt_studio_status, prompt_studio_projects, audio_studio, vector_stores_openai). Finding outside AC3's list: endpoints still import 51 private names from core across 20 files (audio/* the most); not ratcheted.
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
