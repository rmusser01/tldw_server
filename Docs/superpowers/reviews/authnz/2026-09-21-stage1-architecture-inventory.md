# Stage 1 — AuthNZ architecture survey and inventory

Date: 2026-09-21. Read-only.

## Scope

Whole-module survey of `tldw_Server_API/app/core/AuthNZ/`: size and churn distribution, the layering
boundary to `app/api/`, how the module decides which database backend it is talking to, and how schema
evolution is expressed on each backend. Detailed `repos/` helper duplication is stage 2; crypto and
tests are stage 3.

## Code Paths Reviewed

Size and churn (full lists in the `.txt` sidecars):

| File | LOC | commits / 12mo |
| --- | ---: | ---: |
| `migrations.py` | 6,679 | 102 |
| `pg_migrations_extra.py` | 4,219 | 78 |
| `repos/mcp_hub_repo.py` | 4,378 | 37 |
| `byok_runtime.py` | 2,879 | 22 |
| `session_manager.py` | 2,076 | 51 |
| `database.py` | 2,015 | 59 |
| `repos/orgs_teams_repo.py` | 1,829 | 25 |
| `settings.py` | 1,732 | 73 |
| `User_DB_Handling.py` | 1,636 | 73 |
| `initialize.py` | 1,542 | 76 |

Size x churn puts the schema pair (`migrations.py` + `pg_migrations_extra.py`, 10,898 LOC / 180 commits)
first by a wide margin, then `initialize.py`/`settings.py`/`User_DB_Handling.py`/`database.py` as the
bootstrap cluster.

Backend discrimination:

- `database.py:is_postgres_backend (1949-1963)` — the one shared, public discriminator; `getattr(pool, "pool", None) is not None`.
- `database.py:DatabasePool._should_use_postgres (968-1002)` — a *different* rule: URL scheme + `AUTH_MODE` + pytest detection.
- `database.py:DatabasePool.backend_type (1004)` — a third, property-shaped.
- `db_config.py:75-81` — a fourth, URL-scheme string (`"postgresql"`/`"sqlite"`).
- `settings.py:1476-1477` — a fifth, `[AuthNZ] db_type` from config file.
- 22 private per-repo copies of `_is_postgres_backend` / `_is_postgres`, plus
  `byok_rotation.py:_is_postgres_pool (47)`, `rbac_seed.py:_is_postgres_connection (67)`,
  `profile_user_sync_boundary.py:_backend_dialect (13)`, `pg_migrations_extra.py:_looks_like_postgres (3970)`.
  Full list in `2026-09-21-stage2-repo-helper-inventory.txt`.
- Two of the private copies accept a `conn` argument and discard it:
  `repos/orgs_teams_repo.py:32` and `repos/billing_repo.py:27` (`_ = conn  # Compatibility placeholder`).

Placeholder-dialect adapters:

- `database.py:_normalize_sqlite_sql (1966-1976)` — `$N` → `?`, regex `_DOLLAR_PARAM = re.compile(r"\$\d+")` at `database.py:1943`.
- `database.py:_convert_question_mark_to_dollar (1992-2010)` — `?` → `$N`, split-based, with a count guard.
- `app/api/v1/API_Deps/auth_deps.py:_normalize_sqlite_sql (463)` — verbatim duplicate of the first, with its own regex at `:450`.
- Canonical, not imported by AuthNZ: `core/DB_Management/backends/query_utils.py:convert_sqlite_placeholders_to_postgres (39)`, which is the only quote-aware and jsonb-operator-aware converter (`_is_jsonb_operator` at `:84`).

Schema evolution:

- `migrations.py` — 98 numbered `migration_NNN_*(conn: sqlite3.Connection)` functions plus rollbacks, registered as `Migration(version, description, forward, rollback)` entries (registry block at `migrations.py:6180-6520`). SQLite only; zero Postgres branches in the file.
- `pg_migrations_extra.py` — 27 unnumbered idempotent `ensure_*_pg()` coroutines, driven imperatively from `initialize.py:563-626` and `:795`. No version ledger, no ordering contract, no rollbacks.

Layering:

- `User_DB_Handling.py:20` — `from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import oauth2_scheme` (module-level).
- `api_key_audit.py:21` and `:30` — deferred imports from `app.api.v1.API_Deps.Audit_DB_Deps`.
- That is the complete set: 3 sites, 2 files. AuthNZ is one of the *cleanest* core modules on the
  repo-wide 161-import layering problem.

## Tests Reviewed

Located by import-grep, never by path.

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py` | Asserts which repo method runs per backend, against a stub pool | Partly — proves the branch is *taken*, not that the SQL is right |
| `tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py` | Same shape for sessions | Partly, same limit |
| `tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py` | Same shape for api_keys | Partly, same limit |
| `tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py` | Same shape for generated_files | Partly — this is the *only* Postgres-branch coverage for the most-branched file in the module |
| `tests/AuthNZ_SQLite/test_mcp_hub_migrations.py`, `test_mcp_hub_governance_pack_migrations.py`, `test_mcp_hub_capability_adapter_migrations.py`, `test_admin_webhook_migration.py`, `test_users_uuid_schema_drift.py` | SQLite migration application | No — SQLite side only |
| `tests/AuthNZ_Postgres/test_mcp_hub_pg_ensure.py`, `test_user_timestamp_timezones.py`, `test_profile_version_migration_pg.py` | Three of the 27 `ensure_*_pg` paths | Marginally — 3 of 27 |
| `tests/AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py` + `tests/AuthNZ_SQLite/..._sqlite.py` | Column/index parity for one table across both backends | Yes, and it is the **right pattern** — see stage 3 |
| `tests/lint/test_endpoint_auth_deps_import_boundary.py` | AST import ratchet with a ban list + required re-export set | Yes — it is the existing precedent for the fix proposed in F1-5 |

There is **no test anywhere that asserts the SQLite migration set and the Postgres `ensure_*_pg` set
describe the same schema.** Confirmed by grepping all 1,243 `core.AuthNZ`-importing test files for any
comparison of the two modules; only the single-table `llm_usage_log_router_columns` pair does anything
of the kind.

## Validation Commands

```
$ find tldw_Server_API/app/core/AuthNZ -name '*.py' | xargs wc -l | tail -1
   74539 total

$ find tldw_Server_API/app/core/AuthNZ -name '*.py' | wc -l
     121

$ ls tldw_Server_API/app/core/AuthNZ/repos/*.py | wc -l
      39

$ grep -rn "def _is_postgres" tldw_Server_API/app/core/AuthNZ/ | wc -l
      25

$ grep -c "^def migration_" tldw_Server_API/app/core/AuthNZ/migrations.py
98

$ grep -c "^async def ensure_.*_pg" tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
27

$ grep -rn "tldw_Server_API.app.api" tldw_Server_API/app/core/AuthNZ/ | wc -l
       9        # 9 lines across 3 import statements in 2 files

$ grep -o "^def migration_[0-9]*" tldw_Server_API/app/core/AuthNZ/migrations.py | sed 's/def migration_//' | sort -n | uniq -d | tr '\n' ' '
059 060 061

$ grep -rl "core\.AuthNZ" tldw_Server_API/tests | wc -l
    1243
```

## Findings

### FINDING authnz-1 — Five disagreeing answers to "which backend am I on", with an unsound safety net underneath

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       core/AuthNZ/database.py:is_postgres_backend (1949-1963)  [pool-presence]
             core/AuthNZ/database.py:DatabasePool._should_use_postgres (968-1002)  [URL + AUTH_MODE + pytest]
             core/AuthNZ/database.py:DatabasePool.backend_type (1004)
             core/AuthNZ/db_config.py:75-81  [URL scheme string]
             core/AuthNZ/settings.py:1476-1477  [config-file db_type]
             22 private copies: repos/{admin_monitoring_repo.py:38, api_keys_repo.py:96,
               backup_schedules_repo.py:72, billing_repo.py:27, byok_validation_runs_repo.py:50,
               data_subject_requests_repo.py:20, generated_files_repo.py:78,
               maintenance_rotation_runs_repo.py:53, mfa_repo.py:29, monitoring_repo.py:28,
               org_invites_repo.py:42, org_stt_settings_repo.py:20, orgs_teams_repo.py:32,
               prototype_workspaces_repo.py:154, quotas_repo.py:23, rate_limits_repo.py:24,
               registration_codes_repo.py:23, sessions_repo.py:25, shared_workspace_repo.py:67,
               storage_quotas_repo.py:36, token_blacklist_repo.py:26, usage_repo.py:67,
               users_repo.py:29}
             plus byok_rotation.py:_is_postgres_pool (47),
               rbac_seed.py:_is_postgres_connection (67),
               profile_user_sync_boundary.py:_backend_dialect (13),
               pg_migrations_extra.py:_looks_like_postgres (3970)
             safety net: core/AuthNZ/database.py:_normalize_sqlite_sql (1966-1976)
canonical:   core/AuthNZ/database.py:is_postgres_backend (1949) — exists, shared, and bypassed 22 times
destination: n/a — adoption of the existing canonical, not a new module
knowledge:   "the connection I am about to execute against speaks Postgres". Today that is asserted
             from pool presence in 22 repos, from URL+AUTH_MODE+pytest in the pool itself, from the
             URL scheme in db_config, and from a config-file string in settings. Any change to how a
             backend is selected (a new test harness, a proxy DSN, a read-replica URL) must land in
             all five places or they disagree.
scenario:    `_normalize_sqlite_sql` exists, by its own docstring at database.py:1968, as "a safety net
             to avoid aiosqlite warnings when a $-style query slips through the SQLite path" — i.e. the
             authors already know the discriminators can disagree. But the net is unsound: it is a bare
             `_DOLLAR_PARAM.sub("?", query)` with no positional counting and no string-literal
             awareness. `generated_files_repo.py:369-376` builds, on its Postgres branch,
             `(filename ILIKE $N OR original_filename ILIKE $N)` — one placeholder number used twice,
             legal in Postgres, and binds ONE parameter. If a repo's `_is_postgres()` returns True
             (pool object present) while `DatabasePool` routes to SQLite (`_should_use_postgres()`
             False), that query reaches `_normalize_sqlite_sql`, becomes two `?` with one bound
             parameter, and raises `sqlite3.ProgrammingError: Incorrect number of bindings`. The same
             substitution also rewrites `$` inside quoted literals. The canonical converter in
             `core/DB_Management/backends/query_utils.py:39` handles both cases and is not imported here.
impact:      High because the fallback is silent-then-fatal rather than fail-fast, and because the
             discriminator sprawl is what makes the whole class of stage-2 SQL divergences possible:
             23 files each owning their own idea of the backend is why the two SQL bodies drift.
tests:       import-grep reachability, not measured coverage. Four `*_backend_selection.py` unit tests
             under `tests/AuthNZ/unit/` assert which branch runs against a *stub* pool. No test asserts
             the five discriminators agree, and none exercises `_normalize_sqlite_sql` with a repeated
             `$N` or a `$` inside a literal.
effort:      moderate. Replacing 22 private copies with `is_postgres_backend()` is mechanical but the
             private ones are sync and the canonical is async — that is the real work. Hardening
             `_normalize_sqlite_sql` (or deleting it in favour of a loud failure) is cheap and
             independent; do that first.
owner-only:  no for the core changes; yes for the duplicate at api/v1/API_Deps/auth_deps.py:463.
confidence:  confirmed (five discriminators, 22 private copies, the unsound substitution, the repeated
             `$N` generation site); probable-risk (the ProgrammingError, which needs the two
             discriminators to actually disagree at runtime — I did not construct that state).
```

### FINDING authnz-2 — Two backends, two incompatible schema-evolution models, no parity check

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       core/AuthNZ/migrations.py (6,679 LOC, 98 numbered migrations, registry at :6180-6520)
             core/AuthNZ/pg_migrations_extra.py (4,219 LOC, 27 `ensure_*_pg` coroutines at
               :2493, :2519, :2541, :2565, :3288, :3369, :3391, :3409, :3463, :3537, :3548, :3562,
               :3611, :3814, :3851, :3918, :3944, :3980, :4007, :4037, :4066, :4092, :4111, :4134,
               :4155, :4177, :4199)
             driver: core/AuthNZ/initialize.py:563-626 and :795
canonical:   NONE
destination: n/a — this is a parity-enforcement gap, not a helper-consolidation one
knowledge:   the AuthNZ schema. It is currently written down twice, in two languages, with two
             different notions of "has this been applied". SQLite has an ordered, versioned,
             rollback-capable ledger; Postgres has a bag of idempotent `CREATE TABLE IF NOT EXISTS` /
             `ALTER TABLE ADD COLUMN IF NOT EXISTS` calls, ordered only by the sequence of `await`s in
             `initialize.py`. Every schema change must be written twice and there is nothing that
             fails when only one is written.
scenario:    The `sessions.last_activity` column is the worked example. SQLite gets it via
             `migration_098_add_session_last_activity` (migrations.py:6542), registered as version 98.
             Postgres gets it via `ensure_postgres_session_last_activity`
             (core/DB_Management/authnz_session_schema.py:74-86), which runs
             `ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMP` — **nullable, no
             DEFAULT** — then backfills only rows existing at migration time
             (`WHERE last_activity IS NULL`). Meanwhile `repos/sessions_repo.py:create_session_record`
             writes `last_activity` on its SQLite branch (`:121`, `datetime('now')`) and **omits the
             column entirely on its Postgres branch** (`:86-105`, 12 columns). Result: on Postgres,
             every newly created session has `last_activity = NULL` until the first
             `update_last_activity` call. `list_user_sessions` at `:638-641` then runs
             `ORDER BY last_activity DESC`, where Postgres sorts NULLs FIRST — so a user's freshly
             created session is returned at the top of the list with a null "last active" value, while
             on SQLite the same session carries a timestamp and sorts by it. Same endpoint, two
             behaviours, decided by the deployment's database.
impact:      High as a class. The specific `last_activity` consequence is bounded (no idle-timeout
             sweep exists today — I grepped for one and found none, so nothing currently *expires*
             sessions on this column), but the shape recurs: the audit's dual-backend precedent says
             the SQLite side is the tested one, and here the SQLite side is also the only one with an
             ordered, versioned, auditable schema history.
tests:       import-grep reachability, not measured coverage. `tests/AuthNZ_SQLite/` has five migration
             tests; `tests/AuthNZ_Postgres/test_mcp_hub_pg_ensure.py`,
             `test_user_timestamp_timezones.py` and `test_profile_version_migration_pg.py` cover 3 of
             the 27 `ensure_*_pg` functions. Exactly one test pair compares the backends for the same
             object: `tests/AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py` +
             `tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py`, whose assertion
             blocks are byte-identical and whose setup differs only by
             `information_schema` query vs `PRAGMA table_info`. That pair is the template for the fix.
effort:      expensive if attempted as a unification; **cheap for the proportionate fix**, which is a
             parity test generalising the one pair that already exists.
owner-only:  no
confidence:  confirmed (the two models, the 98-vs-27 asymmetry, the missing DEFAULT, the omitted
             column in the PG INSERT, the ORDER BY); confirmed also that no idle-timeout sweep reads
             the column today.
```

### FINDING authnz-3 — Five consecutive migrations are registered under versions three higher than their names, and three numbers are used twice

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       core/AuthNZ/migrations.py, duplicate function-name numbers:
               migration_059_harden_mcp_external_binding_schema (:3418)
                 vs migration_059_create_data_subject_requests_table (:4796)
               migration_060_add_mcp_external_credential_slots (:3480)
                 vs migration_060_create_admin_monitoring_tables (:4848)
               migration_061_add_mcp_path_scope_objects_and_assignment_workspaces (:3620)
                 vs migration_061_create_backup_schedule_tables (:4935)
             registry entries where name and version disagree (migrations.py:6303-6327):
               Migration(62, ..., migration_059_harden_mcp_external_binding_schema)
               Migration(63, ..., migration_060_add_mcp_external_credential_slots)
               Migration(64, ..., migration_061_add_mcp_path_scope_objects_and_assignment_workspaces)
               Migration(65, ..., migration_062_add_mcp_workspace_set_objects)
               Migration(66, ..., migration_063_add_mcp_shared_workspaces)
canonical:   NONE
destination: n/a
knowledge:   "which schema version does this function implement". It is written in two places — the
             function name and the `Migration(...)` version argument — and for five functions they
             disagree by three.
scenario:    The runner is correct today: it dispatches on the explicit integer, so version 64 really
             does run `migration_061_add_mcp_path_scope_objects_and_assignment_workspaces`. The hazard
             is the reader and the next author. Anyone writing the Postgres counterpart, debugging
             "which migrations have applied past 61", or bisecting a schema problem will read
             `migration_061_*` as version 61 — it is version 64, and version 61 is a different function
             (`migration_061_create_backup_schedule_tables`). Separately, the three duplicated name
             prefixes remove the accident-protection the convention was providing: a future
             `migration_059_*` whose suffix happens to match an existing one is silently shadowed at
             module scope, and Python reports nothing.
impact:      Medium. No current defect — I traced the registry and the dispatch is by integer. The cost
             is that a 6,679-line file with 102 commits a year has lost the one cheap invariant that
             made its 98 entries reviewable, on the most-churned file in the module.
tests:       import-grep reachability, not measured coverage. No test asserts
             `Migration(N, ...).forward.__name__.startswith(f"migration_{N:03d}")`. Nothing would have
             caught the drift.
effort:      cheap. One assertion over the registry list, in the style of the existing AST ratchet at
             tests/lint/test_endpoint_auth_deps_import_boundary.py. Renaming the five functions to
             match their versions is a separate, also-cheap, mechanical change.
owner-only:  no
confidence:  confirmed (the duplicate names, the five mismatched registrations, and that dispatch is
             by integer so nothing is currently broken).
```

### FINDING authnz-4 — `core` imports an endpoint dependency for a module-level singleton

```
axis:        encapsulation
class:       n/a
severity:    Low
sites:       core/AuthNZ/User_DB_Handling.py:20 —
               `from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import oauth2_scheme`
             core/AuthNZ/api_key_audit.py:21 and :30 —
               deferred `from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (...)`
canonical:   n/a
destination: `oauth2_scheme` is a FastAPI `OAuth2PasswordBearer` instance — a transport concern. It
             belongs in the API layer with a core-side parameter, not imported downward.
knowledge:   the token-extraction transport contract. `Docs/Architecture.md` is operative here:
             "Clients -> FastAPI endpoints -> Core domain services". A core module importing
             `api/v1/API_Deps/*` is the worst shape of the repo-wide layering inversion, not the mild
             schema-only shape.
scenario:    n/a — encapsulation axis. The concrete cost is import-graph: `core.AuthNZ.User_DB_Handling`
             cannot be imported without pulling in the FastAPI v1 dependency package, which is why the
             sibling site in `api_key_audit.py` had to be made a deferred import.
impact:      Low, and deliberately so. AuthNZ has **3 such sites across 2 files** against a repo-wide
             161 across 107. This module is close to clean; reporting it as a crisis would be wrong.
             It is worth fixing precisely *because* it is nearly free to fix here.
tests:       `tests/lint/test_endpoint_auth_deps_import_boundary.py` is the existing AST ratchet for
             exactly this class; it does not currently cover the core->API direction.
effort:      cheap. `oauth2_scheme` is used for token extraction; the core function can take the token
             as a parameter and let the endpoint supply it.
owner-only:  no for the core side; yes if the re-export in `api/v1/API_Deps/` is touched.
confidence:  confirmed.
```

## Suggested Refactor/Actions

Ordered by ratio of risk removed to effort.

1. **(cheap, no design doc) Add a migration-registry invariant test.** One test iterating the
   `Migration(...)` list asserting the registered version matches the numeric prefix of the forward
   function's `__name__`, and that no two entries share a version. Seed it as a ratchet if the five
   existing mismatches are to stay; otherwise rename those five functions first. Addresses authnz-3.
   Model: `tests/lint/test_endpoint_auth_deps_import_boundary.py`.

2. **(cheap, no design doc) Make `_normalize_sqlite_sql` fail loudly instead of quietly.** Give it the
   same count guard `_convert_question_mark_to_dollar` (database.py:1997-2003) already has, or delete
   it and let the dialect mismatch raise. A silent rewrite of a query the code has already decided is
   Postgres is a worse outcome than an exception. Then delete the verbatim duplicate at
   `api/v1/API_Deps/auth_deps.py:463` in favour of importing the one in `database.py` (owner-only).
   Addresses half of authnz-1.

3. **(moderate, needs a design doc) Generalise the one existing cross-backend parity test.**
   `tests/AuthNZ_{Postgres,SQLite}/test_authnz_llm_usage_log_router_columns_{pg,sqlite}.py` already does
   the right thing for one table — identical assertion block, backend-specific introspection. Turn it
   into a table-driven parity test over the AuthNZ schema: for each table, assert the column set and
   index set match across `information_schema` and `PRAGMA table_info`. Seed it at the tables that
   currently agree so the number of exempted tables can only go down. Addresses authnz-2 without
   attempting to unify `migrations.py` and `pg_migrations_extra.py`.
   Needs `Docs/Design/2026-MM-DD-authnz-schema-parity-design.md` and an ADR entry, because "the two
   schema files are allowed to be separate but must be provably equivalent" is a decision.

4. **(moderate) Collapse the 22 private `_is_postgres_backend` copies onto
   `database.py:is_postgres_backend`.** The blocker is sync-vs-async, so the honest first step is a
   sync sibling on `DatabasePool` (`pool.is_postgres` property) that the 22 repos can call, with the
   async module-level function delegating to it. Do not attempt this before item 2 — the safety net
   must be loud first, or the migration will hide its own regressions. Addresses the rest of authnz-1.

5. **(cheap) Extend the existing import ratchet to the core->API direction** and seed it at the current
   3 AuthNZ sites (repo-wide: 107 files) so the number can only go down, then remove the
   `oauth2_scheme` import by parameterising the core function. Addresses authnz-4.

Item 3 is the one that clears the design-first bar. Items 1, 2, 4 and 5 are small enough not to.
