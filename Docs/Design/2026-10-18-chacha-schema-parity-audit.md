# ChaChaNotes SQLite/Postgres Schema Parity Audit (2026-10-18)

**Task:** TASK-13643 (Batch 3 Stage 1; original plan cited v68/v72 — versions moved, gap shape did not).
**Method:** direct ladder/function analysis of `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` at dev `e1f7770953`.

## Current state

| Ladder | Version | Evidence |
|---|---|---|
| SQLite (`_CURRENT_SCHEMA_VERSION`) | **81** | `:1204` — "Server-issued conversation source incarnation" |
| Postgres (`_POSTGRES_SCHEMA_VERSION`) | **85** | `:1205` |
| Divergence | 4 versions (unchanged shape since the 2026-10-01 plan's v68/v72) | |

No `def _migrate_from_v8x_to_v8y_sqlite` functions exist (`grep -c` = 0): the SQLite ladder genuinely ends at 81. Both upgrade ladders (`:26308`, `:26781` regions, inside `_initialize_schema_postgres` fresh/upgrade paths) call the same `_postgres` migration functions through v85.

## The four PG-only migrations (v82–85) — all RLS/security, by docstring

| Step | Docstring (verbatim) | Classification |
|---|---|---|
| v81→82 `_migrate_from_v81_to_v82_postgres` (:18537) | "Install permanent local receipts with **forced owner RLS** before the version bump." | **DESCOPE-BY-DESIGN** |
| v82→83 (:18517) | "Retain the existing **forced-owner RLS** while adding frozen quota identity." | **DESCOPE-BY-DESIGN** |
| v83→84 (:18495) | "Add explicit **roles** without changing existing forced-owner RLS or rows." | **DESCOPE-BY-DESIGN** |
| v84→85 (:18452) | "Native reference **role migration**" (RLS roles). | **DESCOPE-BY-DESIGN** |

## Why descope-by-design is the working classification

All four migrations implement **PostgreSQL-native security primitives** — row-level security policies, database roles, forced-owner ownership — that do not exist in SQLite. The repo's SQLite security model is **application-level isolation instead**: AuthNZ defaults to SQLite for single-user deployments, and per-user content databases live under `Databases/user_databases/<user_id>/` (AGENTS.md, db_path_utils), so cross-tenant rows never share a SQLite file. Porting RLS to SQLite is not meaningfully possible; the "gap" is two different security architectures, not a missing migration.

## What Stage 2 should therefore do (recommendation)

1. **Do NOT port v82–85 to SQLite.** Record the intentional divergence next to the constants (`:1204-1205`) with a pointer to this audit, so the next agent doesn't re-derive it.
2. **Add a parity guard test**: assert `_POSTGRES_SCHEMA_VERSION - _CURRENT_SCHEMA_VERSION == 4` AND that every PG-only step above SQLite's version is RLS/roles-classified (cheap test over the function docstrings or a registry), failing if a *non-security* PG-only migration ever appears — that would be a true port gap.
3. **Update `Docs/Database_Migrations.md`** with the two-architecture note.
4. If any code path gates on `version >= 82` for *feature* (not security) behavior reachable from SQLite deployments, that specific feature needs an explicit SQLite story — none was found in this pass (the gates found at `:26308`/`:26781` are inside postgres-only init paths).

## Open questions for the owning decision

- Whether "frozen quota identity" (v82→83) carries quota semantics that SQLite deployments should enforce at the application layer today (if quotas are enforced in code uniformly, no action; if Postgres RLS is the only enforcement, single-user SQLite is unaffected by definition).
- TASK-13644–13646 (Scheduler `migrate_schema` stubs, Evaluations PG adapter, ChromaDB TODOs) are unaffected by this finding and remain as planned.
