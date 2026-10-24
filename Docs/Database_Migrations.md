Database Migrations Overview

This project maintains multiple logical databases with distinct migration registries:

- Content databases (Media/ChaCha/etc.)
  - Registry: `schema_version` (single-row integer) and `schema_migrations` (for legacy tracking via the DatabaseMigrator).
  - Location: Managed primarily within the package-native `tldw_Server_API/app/core/DB_Management/media_db` package and the `db_migration.py` helper.
  - Backends: SQLite (default) and PostgreSQL (optional). PostgreSQL migrations are applied inline via helper methods on `MediaDatabase`.

- Evaluations/Audit databases
  - Registry: `schema_version` and `migrations` tables for applied steps and verification.
  - Location: `migrations_v5_unified_evaluations.py` and `migrations_v6_audit_logging.py` maintain their own registries to reflect the independent lifecycle of these stores.

Why two registries?

- Content schema updates are tightly coupled to runtime application boot (for both SQLite and PostgreSQL) and often need inline verification (FTS, RLS policies). The single version integer plus inline checks keep the boot path fast and predictable.
- Evaluations/Audit introduce separate, self-contained features and may be operated as distinct databases. Their modules maintain a `migrations` history to aid exporting, auditing and verification tasks without coupling to content store versioning.

Guidance

- When changing Media/ChaCha schema: update `MediaDatabase._CURRENT_SCHEMA_VERSION` and add inline migration helpers or SQLite migration SQL files (under `DB_Management/migrations/`).
- When changing Evaluations/Audit schema: update the appropriate migration module and record the step in the module’s registry tables.
- Keep tests aligned with the registry logic: content tests should validate `schema_version` and FTS/RLS; evaluation tests should validate both `schema_version` and `migrations` integrity.

## SQLite migration transactions

`DatabaseMigrator` runs each migration body, its successful `schema_migrations`
entry, and its `schema_version` update in one `BEGIN IMMEDIATE` transaction.
A failure rolls back that migration's schema and data changes before recording
the failure. Earlier successful migration versions remain applied; a sequence
of versions is not one transaction. Backups remain available for recovery.

New SQL migrations should omit transaction controls. For compatibility, the
runner accepts one outer `BEGIN` / `COMMIT` (or `END`) pair in existing scripts
and owns that boundary itself. Shipped migration files stay unchanged so their
checksums continue to match previously recorded upgrades. Transaction controls
and foreign-key setting changes inside the body are rejected by SQLite's
authorizer, including statements containing comments. Trigger bodies and
semicolons in string literals remain valid.

Place `PRAGMA foreign_keys = OFF` / `ON` (or the equivalent `foreign_keys(OFF)` /
`foreign_keys(ON)` syntax) before and after the migration body, outside any
legacy wrapper. The runner executes these connection settings outside its
transaction. Other setting spellings in the body fail instead of being silently
ignored. These scripts are intended to run through `DatabaseMigrator`.

## SQLite permissions and failed-upgrade recovery

Run upgrade preflight as the account that will perform migrations, after stopping
all server and worker writers. That account needs write access to each database,
its containing directory and any existing `-wal`, `-shm` or `-journal` files.
Readable data can pass an integrity check while these permissions still prevent
migration. Preflight reports the file or directory that needs attention; it does
not change permissions. Filesystem access can change after preflight, so a passing
check is not a guarantee that a later migration will complete.

If a permission failure interrupts startup, keep the original backup untouched.
With all users of the affected database stopped, correct ownership and access for
the migration account across the database, directory and existing SQLite sidecars,
then resume on the disposable upgrade target and compare the saved records.
Do not delete WAL files to clear the error: they can contain committed data.
Alternatively, restore the verified backup into a separate empty target, apply
the upgrade there and compare records before switching deployments. Do not run an
older binary on the migrated copy or merge a backup over a live data directory.

## Legacy SQLite Media DB recovery

Automatic upgrades of existing file-backed Media DBs start at schema version
22. Versions 1–21 are rejected before invoking the packaged migrator, because
the package does not contain a supported earlier Media DB migration chain.
An empty database is initialized directly at the current version. This boundary
does not describe other logical databases or PostgreSQL migrations.

If startup reports an unsupported legacy version:

1. Stop the server and workers using that database. Preserve an untouched backup
   of the database and any SQLite `-wal` / `-shm` sidecars, plus the associated
   files and configuration. Work from a separate copy.
2. Use the previous server release that successfully opened this database to
   export the content it supports. If that release provides Chatbooks, see the
   [Chatbook guide](User_Guides/WebUI_Extension/Chatbook_User_Guide.md). Export
   coverage varies by release; inventory media, transcripts, notes, metadata,
   and source files before choosing a format. Keep the original backup even
   when export succeeds.
3. Start the current server with a separate, empty destination database/data
   directory. Import the supported export, or re-ingest preserved source files.
   Do not point the current server at the only copy of the legacy database.
4. Compare record counts and representative content, transcripts, relationships,
   and files with the original. Switch to the destination only after checking
   the data you need. Retain the old database for anything the export omitted.

This is a recovery procedure, not a guaranteed lossless automatic conversion.
If no compatible release can export the required data, preserve the backup and
seek a schema-specific migration before proceeding. Do not manually increase
`schema_version` to bypass the check or apply migrations for another database.

## Validated upgrade starting points

The 2026-10-23 X01 validation demonstrated upgrades from the published v0.1.32
package (Media schema23), and from genuine v0.1.43 and v0.1.46 source
installations. Each starting point covered SQLite/PostgreSQL content and Jobs
with single-user and multi-user authentication, old-data inventory, whole
configured backups, separate restores, owner-run migrations, ordinary runtime
access, new writes and saved-Chat reload. In v0.1.32 single-user mode, AuthNZ
remains SQLite even when content and Jobs use PostgreSQL; that is a hybrid
deployment, not PostgreSQL AuthNZ coverage. Prompts retain their SQLite sidecar.

v0.1.32 is the earliest tested published-package starting point. This does not
certify every intervening release or establish a lossless path for Media
schemas1–21. The v0.1.43/v0.1.46 results cover source deployments; their failed
published-wheel checks are not accepted by those source results.
Keep the separate backup and compare the complete configured deployment before
switching. See the [X01 acceptance record](Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#x01-supported-upgrade-selected-closure)
for exact coverage and recovery limits.

ADR assessment: no new support policy or migration architecture is introduced;
this records executed coverage within the existing migration boundary above.

## Temporary document upload handoffs

Short-lived document handoffs use
`<USER_DB_BASE_DIR>/document_upload_drafts.db`, shared by workers within that
configured deployment. This remains a SQLite store when content uses PostgreSQL.
Give separate deployments separate data roots.

After upgrading from the installation-wide draft store, recreate any pending
upload handoff by selecting its document again. The old
`Databases/document_upload_drafts.db` is neither read nor migrated automatically:
its numeric owner IDs cannot be safely attributed to a configured deployment.
Saved sources and Chats retain their normal migration and backup procedures.
