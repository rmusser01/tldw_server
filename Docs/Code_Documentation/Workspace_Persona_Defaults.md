# Workspace Persona Default Choices

Stage 2A of [#2950](https://github.com/rmusser01/tldw_server/issues/2950) adds durable opt-out to the existing Workspace Persona defaults. It does not add auto-provisioning, startup provenance, strict retry handling, Buddy behavior, or UI changes. The [approved design](../Design/2026-09-13-persona-workspace-choice-provenance-design.md) covers those later stages separately.

## Read And Write Contract

Workspace management responses expose `assistant_defaults_explicit_none` as a read-only boolean. Continue using the existing version-checked PATCH command to change the choice:

| Operation | Stored defaults | Opt-out |
| --- | --- | --- |
| Normal Workspace creation | SQL NULL | false |
| PATCH omits `assistant_defaults` | unchanged | unchanged |
| PATCH `assistant_defaults: null` | SQL NULL | true |
| PATCH valid Persona defaults | reference object | false |
| Existing legacy SQL NULL at upgrade | unchanged | true |
| Existing legacy non-null at upgrade | unchanged, including malformed values | false |

Clearing an already-empty default still records opt-out and advances the Workspace version. Defaults and opt-out change in one optimistic-locking update; conflicts leave both untouched. Renames, upserts, archive/unarchive, and soft deletion retain the choice. There is no public reset-to-unset command. Sending the derived field directly in PUT/PATCH is rejected with 422, including null.

The migration's conservative opt-out does not prove historical user intent. Corrupt non-null defaults, including a default paired with true, remain unavailable in the effective management projection (`invalid_default`); explicit owner clear/save repairs the pair. Persona references remain user-owned and reference-backed, with no snapshots.

Clone snapshots and the current Research Workspace import format do not represent a trustworthy portable Persona choice. New clone and import targets therefore opt out. Importing into an existing Workspace preserves its choice; clone publication preserves the target's opt-out. No cross-owner Persona reference is copied.

## Required Maintenance Upgrade

Schema v68 adds a non-null choice bit with a false default (SQLite integer with a 0/1 constraint; PostgreSQL boolean), then backfills SQL NULL defaults to true in the registered migration transaction. Fresh databases use the same registered migration chain. SQLite completes legacy initialization and source-catalog checks through v67 before starting the final migration transaction: legacy script helpers can implicitly commit, so none may run after the new writes. Historical schemas and validation guards are unchanged.

1. Schedule an offline maintenance window. Block incoming writes and drain active requests/jobs. Stop every old API process, worker, and direct database writer, including idle processes holding cached handles. Inventory every affected per-user ChaChaNotes database and PostgreSQL deployment before proceeding.
2. Take a consistent, restorable pre-upgrade backup while writes are stopped. Use the deployment's existing database backup procedure, including SQLite WAL handling or PostgreSQL backup tooling. Do not copy only an active SQLite main file.
3. Deploy the compatible binary and initialize each affected database through its normal `CharactersRAGDB` configuration/ownership path, with user traffic and workers still stopped. Use one migration initializer per database during maintenance. Initialization runs v67-to-v68 transactionally; do not patch schema markers or add columns manually. Inspect failures before proceeding to another database.
4. Verify each database reports schema v68. Check representative legacy-null rows are opted out, configured rows retain their references, and malformed non-null storage remains unchanged. In a disposable Workspace, test clear/save/read across a process restart and stale-version rejection.
5. Restart only compatible API and worker binaries and reopen traffic after all checks pass. Keep the maintenance window closed if writer quiescence or database coverage cannot be guaranteed.

Initialization-time version checks reject an older binary opening migrated data. They do **not** fence an older process's already-open handle: its former clear statement can write `(NULL, false)`, and its save can write `(Persona, true)`. This release does not support mixed-version rolling upgrades. The backend regression includes this cached-writer hazard as an explicit deployment limitation, not a claim that runtime fencing exists.

## Recovery And Verification

A failed v67-to-v68 migration transaction retains the v67 schema and choice data. Failures in preceding legacy validation/initialization must not apply the v68 column or backfill; this is not a new atomicity guarantee for upgrades between older historical versions. After successful migration, prefer a compatible fix-forward binary. Never downgrade the schema marker, drop the opt-out column, or run old writers against v68. Restoring the full pre-upgrade backup in an offline recovery loses subsequent writes, so reconcile those separately with explicit operator approval; an old-binary rollback that retains post-upgrade choice data is not supported.

Backend regressions are in `tldw_Server_API/tests/DB_Management/test_workspace_persona_optout_v68.py`; they use the repository's PostgreSQL fixture and SQLite database initialization. They cover fresh/legacy storage, transitions, restart, lifecycle, rollback, and old cached writers. API regressions remain in `tests/Workspaces/test_workspace_assistant_defaults_api.py`. A skipped PostgreSQL fixture is not PostgreSQL rollout evidence: require a successful live run before deploying that backend.

Stage 2A was verified on PostgreSQL 18.6 on 2026-09-13: all six PostgreSQL opt-out cases passed. The combined SQLite/PostgreSQL opt-out, defaults DB/API, PostgreSQL clone lifecycle and v67 migration regression passed 114 tests with no failures or skips. Tests used the existing `pg_database_config` fixture with `TLDW_TEST_POSTGRES_REQUIRED=1`, which fails rather than skips if the server is unavailable. This validates the tested backend paths; deployments still require the per-database offline checks above.
