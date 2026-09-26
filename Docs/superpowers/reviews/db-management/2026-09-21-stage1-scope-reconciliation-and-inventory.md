# Stage 1 (2026-09-21): Scope, Prior-Findings Reconciliation, Inventory

## Scope

Second-pass audit of `tldw_Server_API/app/core/DB_Management/` (233,293 LOC, 265 Python files),
extending — not replacing — the 2026-04-07 stage1–stage5 ledger and the 2026-04-15 rebaseline in this
directory. This pass is a duplication / correctness / efficiency audit under the repo-wide 2026-09-21
briefing, with two explicit obligations the April pass did not discharge:

1. `ChaChaNotes_DB.py` (45,292 LOC, 408 commits/12mo) and `PromptStudioDatabase.py` (7,426 LOC) were
   listed in the April stage 1 inventory and then **never reviewed** in stages 2–5. Confirmed by
   `grep -c` over the April stage files (below). Both are covered here.
2. The 2026-09-21 cross-user isolation audit established that the SQLite/PostgreSQL split is only
   tested on SQLite. Every dual-backend duplicated pair found here is checked by import-grep for
   whether tests exist on both backends or one.

Out of scope for this pass: everything the April ledger already covered and this stage re-verifies as
addressed (see Reconciliation).

## Code Paths Reviewed

Re-verification of the eight April findings:

- `backends/pg_rls_policies.py:850` (`raise DatabaseError(f"{name} RLS statement {index} failed: ...")`)
- `db_migration.py:293-320` (loader), `db_migration.py:840` (`Missing migration versions`), `:895-898`
  (contiguity range check)
- `UserDatabase_v2.py:1000-1400` (bootstrap/normalization/seed region; one `except` at `:1019`, guarded
  and re-raising)
- `content_backend.py:157-175` (`backend_target_key`), `:178-186` (cache globals +
  `_retired_backend_finalizers`), `:189-198` (`_close_backend_pool`), `:201-230` (`_retire_backend_pool`)
- `db_path_utils.py:205-215` (`os.path.realpath` containment), `:287-292`, `:383`, `:394`
  (`is_relative_to`)
- `media_db/api.py:170-191` (`read_media_by_id` via `MediaLookupRepository` / `_require_read_method`)
- `backends/sqlite_backend.py:33` and `backends/postgresql_backend.py:38` (both now import
  `FTSQueryTranslator`)
- `migrate_db.py:82`, `:101`, `:205-207`, `:236` (`create_backup=not args.no_backup`)

New-scope entry points read in full or structurally in stages 2–3:

- `PromptStudioDatabase.py:275-7426` (whole file, AST census + targeted reads)
- `ChaChaNotes_DB.py:222-45289` (structural census; targeted reads of the backend adapter block
  `:448-726`, the migration step table `:8583-8608`, the SQLite ladder `:20349-21169`, the PostgreSQL
  ladder `:24881-25300`, the v67–v70 migrations `:17700-17830`, the WAL close path `:8088-8100`, and the
  `setattr` delegation tail `:44809-45218`)
- `chacha/` (21 modules, 31,053 LOC) and `media_db/` (31,846 LOC) package boundaries
- `Moderation_Review_DB.py:26-100`, `:380-430`, `:630-675`
- `Personalization_DB.py:80-100`, `:464-512`; `Sync_DB.py:9310-9355`, `:1428-1436`
- `Collections_DB.py:80-95`, `:4332-4360`; `Evaluations_DB.py:89-165`, `:1300-1340`, `:1825-1845`
- `Workflows_DB.py:266-360`, `:1663-1690`; `transaction_utils.py:55-95`
- `sqlite_policy.py:1-150` (whole file)
- `media_db/runtime/rows.py` (whole file), `media_db/runtime/email_search_cursor.py` (whole file),
  `chacha/shared_workspace_chat_store.py:1212-1275`

## Tests Reviewed

Located by import-grep, never by path (`grep -rl "core\.DB_Management" tldw_Server_API/tests` = 1,270
files).

| Test file | Protects | Downgrades risk? |
| --- | --- | --- |
| `tests/DB_Management/test_content_backend_cache.py` | superseded-pool close/retire semantics | Yes — closes April finding 4, the last item the 2026-04-15 rebaseline listed as live. 22 passed. |
| `tests/DB_Management/test_pg_rls_policies_contract.py` | RLS installer fail-closed | Yes — closes April finding 1 |
| `tests/DB_Management/test_db_migration_loader.py`, `test_db_migration_planning.py` | malformed-artifact rejection, version contiguity | Yes — closes April finding 2 |
| `tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py` | auth bootstrap fail-closed | Yes — closes April finding 3 |
| `tests/DB_Management/test_db_path_utils.py` | realpath trust boundary | Yes — closes April finding 5 |
| `tests/DB_Management/test_media_db_api_error_contracts.py` | DB-error propagation from `media_db.api` | Yes — closes April finding 6 |
| `tests/DB_Management/test_database_backend_fts_normalization.py` | cross-backend FTS normalization | Yes — closes April finding 7 |
| `tests/DB_Management/test_migration_cli_integration.py` | `--no-backup` passthrough | Yes — closes April finding 8 |
| `tests/prompt_studio/test_database.py` | `_SQLitePromptStudioDatabase` CRUD against a real temp SQLite file | Partially — it is the only real-DB PromptStudio test, and it never calls `list_optimizations`, `get_prompt(include_deleted=…)`, `delete_signature(id, True)` or `create_bulk_test_cases(client_id=…)`. See stage 2 finding 1 and 3. |
| `tests/prompt_studio/integration/test_api_endpoints.py:369-380`, `tests/prompt_studio/unit/test_optimization_endpoint_error_mapping.py:32-37` | the optimizations endpoint's pagination/error mapping | **No** — both substitute a stub object defining `list_optimizations(self, *_args, **_kwargs)`. The stub hides the fact that the default SQLite implementation has no such method. |
| `tests/DB_Management/test_deck_owner_name_migration_postgres.py`, `test_character_owner_name_migration_postgres.py` | per-owner name uniqueness migrations | **No** — PostgreSQL only; there is no SQLite counterpart, and SQLite never receives the policy. See stage 2 finding 2. |
| `tests/DB_Management/test_local_keyword_merge_survivor.py`, `test_local_keyword_survivor_migration.py` | `merged_into_sync_id` tombstone on both `keywords` (SQLite) and `chacha_keywords` (PostgreSQL) | Yes for that one column — this is the pattern the deck/character migrations should have followed. |
| `tests/DB_Management/test_prompt_studio_sync_log_backends.py` | PromptStudio sync-log rows on both backends | Partially — one method family only, out of 59 paired methods. |

## Validation Commands

```
$ find tldw_Server_API/app/core/DB_Management -name '*.py' | xargs wc -l | sort -rn | head -1
   233293 total
$ find tldw_Server_API/app/core/DB_Management -name '*.py' | wc -l
      265
$ git log --since='12 months ago' --name-only --pretty=format: -- tldw_Server_API/app/core/DB_Management \
    | grep '\.py$' | sort | uniq -c | sort -rn | head -7
 408 ChaChaNotes_DB.py
 119 Media_DB_v2.py          # deleted file; churn is the decomposition itself
  97 Collections_DB.py
  96 Sync_DB.py
  66 db_path_utils.py
  49 Watchlists_DB.py
  41 PromptStudioDatabase.py
$ grep -c 'ChaChaNotes_DB' Docs/superpowers/reviews/db-management/2026-04-07-stage[2-5]*.md
  0  0  0  0
$ grep -c 'PromptStudioDatabase' Docs/superpowers/reviews/db-management/2026-04-07-stage[2-5]*.md
  0  0  0  0
$ grep -rl "core\.DB_Management" tldw_Server_API/tests | wc -l
     1270
$ python3 -m pytest -q tldw_Server_API/tests/DB_Management/test_content_backend_cache.py
======================= 22 passed, 13 warnings in 2.75s ========================
$ python3 -m pytest --collect-only -q tldw_Server_API/tests/DB_Management 2>&1 | tail -1
=================== 2961 tests collected, 4 errors in 4.73s ====================
$ python3 -m pytest --collect-only -q tldw_Server_API/tests/DB_Management 2>&1 | grep -E '^E  ' | sort -u
E   ModuleNotFoundError: No module named 'hypothesis'
E   ModuleNotFoundError: No module named 'psycopg'
$ python3 -m pytest --collect-only -q tldw_Server_API/tests/ChaChaNotesDB 2>&1 | tail -1
=================== 1086 tests collected, 4 errors in 3.62s ====================
$ python3 -m pytest --collect-only -q tldw_Server_API/tests/prompt_studio 2>&1 | tail -1
======================== 1152 tests collected in 1.27s =========================
```

Test-file counts per hot module (import-grep, **reachability not coverage**):

```
$ for m in PromptStudioDatabase ChaChaNotes_DB Moderation_Review_DB Personalization_DB \
    Workflows_DB transaction_utils Collections_DB Sync_DB Evaluations_DB sqlite_policy; do
    echo "$m: $(grep -rl "DB_Management[./]$m\|DB_Management import.*$m" tldw_Server_API/tests | wc -l)"; done
PromptStudioDatabase: 34
ChaChaNotes_DB: 478
Moderation_Review_DB: 1
Personalization_DB: 44
Workflows_DB: 47
transaction_utils: 1
Collections_DB: 67
Sync_DB: 58
Evaluations_DB: 26
sqlite_policy: 8
```

Machine-generated inventories live in the sidecars:
`2026-09-21-stage1-source-inventory.txt`, `2026-09-21-stage1-churn-baseline.txt`,
`2026-09-21-stage2-promptstudio-pair-inventory.txt`,
`2026-09-21-stage2-chachanotes-backend-pair-inventory.txt`.

## Findings

### Reconciliation of the 2026-04-07 / 2026-04-15 findings

All eight April findings re-verified against today's tree. **Eight of eight are ALREADY-ADDRESSED.
None are still live.** This includes the one the 2026-04-15 rebaseline itself listed as unresolved.

| April finding | Status 2026-09-21 | Evidence |
| --- | --- | --- |
| 1 High — PG RLS installers partially fail yet report success | **Addressed** | `backends/pg_rls_policies.py:850` raises `DatabaseError(f"{name} RLS statement {index} failed: {exc}")`; `tests/DB_Management/test_pg_rls_policies_contract.py` exists |
| 2 High — migration loader fails open, non-contiguous upgrade | **Addressed** | `db_migration.py:308,317` raise `MigrationError` on malformed artifacts; `:840` raises `Missing migration versions: {missing_versions}`; `:895-898` builds the expected contiguous range |
| 3 High — `UserDatabase_v2` bootstrap swallows normalization/seed failures | **Addressed** | file is now 2,018 lines; the cited `:1097-1267` region holds no swallowing handler — the single `except` in `:1000-1400` is at `:1019`, immediately narrowed by `sqlite_profile_version_connection_invalid(exc)`; `test_userdatabase_v2_bootstrap_failclosed.py` exists |
| 4 Medium — superseded content backends replaced without closing pools | **Addressed** (was the only item still live on 2026-04-15) | `content_backend.py:180` `_retired_backend_finalizers`, `:189` `_close_backend_pool`, `:201` `_retire_backend_pool`; `pytest tests/DB_Management/test_content_backend_cache.py` = **22 passed** |
| 5 Medium — trusted path containment is lexical, symlink escape | **Addressed** | `db_path_utils.py:205` `candidate_real = os.path.realpath(...)`, `:287` parent realpath, `:383`/`:394` `is_relative_to` |
| 6 Medium — `media_db.api` collapses backend failures to `False`/`None`/`[]` | **Addressed** | `media_db/api.py:175-191` now routes through `MediaLookupRepository` / `_require_read_method` and propagates; `test_media_db_api_error_contracts.py` exists |
| 7 Medium (probable) — backend FTS surface not syntax-parity | **Addressed structurally** | both `backends/sqlite_backend.py:33` and `backends/postgresql_backend.py:38` now import `FTSQueryTranslator`; `test_database_backend_fts_normalization.py` exists. Confidence: probable — parity is asserted by that test, not re-derived here. |
| 8 Low — `migrate_db.py --no-backup` not wired | **Addressed** | `migrate_db.py:236` `migrate(args.db_path, args.version, create_backup=not args.no_backup)` |

That is a clean close-out of the April pass. It also means the April ledger's `## Test Gaps` list is
stale and should not be carried forward: every gap it names now has a named test file.

### F-DB-0 — the April review's own scope gap is the reason the two biggest files are unaudited

- axis: duplication (process finding, reported once so later stages need not repeat it)
- class: n/a
- severity: Low (as a finding), but it is the enabling condition for findings 1–5 in stage 2
- sites: `Docs/superpowers/reviews/db-management/2026-04-07-stage1-review-artifacts-and-inventory.md`
  (inventory lists both files) vs `2026-04-07-stage2…md`, `…stage3…md`, `…stage4…md`, `…stage5…md`
  (zero mentions of either)
- knowledge: stage selection in the April pass was organised by *subsystem concept* (backends,
  factories, paths, migrations, media_db) rather than by *size × churn*, so the 45,292-line / 408-commit
  file and the 7,426-line dual-backend file were both inventoried and then skipped.
- impact: the highest-blast-radius file in the repository has never been reviewed. Every finding in
  stage 2 is in one of those two files.
- effort: cheap — this stage fixes it.
- owner-only: no
- confidence: confirmed

## Suggested Refactor/Actions

1. Mark all eight April findings closed in whatever tracker carries them, citing the tests above. Do
   not re-open finding 4 — the 2026-04-15 rebaseline's "still live" note is superseded.
2. Treat `2026-04-07-stage5-test-gaps-and-synthesis.md#Test Gaps` as historical. The live gaps are the
   ones in stage 3 of this pass.
3. Adopt size × churn as the stage-ordering heuristic for any future pass over this module; the churn
   sidecar is regenerable in one command.
