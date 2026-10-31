# Task 2 report

## Status

Complete. ChaCha schema v55 now provides stable organization identities for
keywords, keyword collections, and note folders on SQLite and PostgreSQL, with a
backend-neutral projection seam for resource and relationship Sync envelopes.

Committed as `6fe539a302 feat(notes): add stable organization identities`.

## Takeover and authorization

This task was taken over with the original implementer's uncommitted RED tests
and partial SQLite migration preserved. The prior attempt paused because relayed
authorization was not accepted for the automatic PostgreSQL migration.

The inherited human turn directly authorized the planned automatic PostgreSQL
v54-to-v55 migration. That migration executes inside the initializer's existing
transaction, adds nullable columns, backfills active and deleted rows with
application-generated canonical UUIDv4 values, verifies null/blank and duplicate
counts, then applies `NOT NULL`, the three explicitly named unique indexes, and
schema version 55. It does not delete rows, integer IDs, relationships, or schema
version history.

The coordinator also authorized two narrow Task 2 scope exceptions after an
extra API regression probe found a stale reconstruction path:

- `test_notes_restore.py` now gives its keyword response fixtures canonical
  UUIDv4 `sync_id` values.
- `notes.py` passes that stored ID into its existing `KeywordResponse`
  reconstruction during note restore.

No other Notes restore behavior changed.

## Changes made

- Advanced ChaCha to schema v55 and registered the SQLite `54 -> 55` migration.
- Added non-null stable IDs and named unique indexes for keywords, collections,
  and folders in fresh schemas and upgrades.
- Added the authorized transactional PostgreSQL nullable/backfill/verify/
  constrain/version migration using the `chacha_keywords` physical mapping and
  PostgreSQL parameter style.
- Generated UUIDv4 IDs before all new product-store inserts and retained them
  across reads, lists, mutations, soft deletion, and restoration.
- Added the frozen organization resource, relationship, and snapshot models plus
  `NotesOrganizationSyncStore` resource/relationship apply and snapshot methods.
- Recalculated folder subtree paths in one transaction, with preflight checks for
  self-parenting, cycles, missing/deleted parents, conflicts, and 500-character
  path limits.
- Preserved parent pointers and membership/link rows during soft deletion.
- Added `sync_id` to the three additive response schemas and the note-restore
  keyword reconstruction.
- Corrected the collection FTS update/delete triggers so restoring an already
  soft-deleted collection does not issue an invalid FTS delete command.

## TDD evidence

### RED: v54 migration

The inherited focused migration tests initially failed two tests because schema
v55, its migration-map entry, and the `note_folders`/`sync_id` upgrade behavior
did not exist (`sqlite3.OperationalError: no such table: note_folders`).

### GREEN: v54 migration

```text
python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py
2 passed, 2 warnings in 14.45s
```

The fixture covers active and deleted rows, canonical UUIDv4 formatting,
per-table uniqueness, preserved integer IDs/relationships/deletion/version
fields, schema version 55, migration-map registration, and reopen idempotency.

### RED: product stores and projection seam

The initial focused keyword/folder run failed during collection because
`organization_sync_store` did not exist. After the first implementation pass,
four focused tests exposed two concrete defects: `sqlite3.Row` was treated as a
dict with `.get()`, and restoring a soft-deleted collection caused the inherited
FTS trigger to report `database disk image is malformed`.

### GREEN: product stores and hierarchy

```text
python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k "sync_id or hierarchy or soft_delete"
8 passed, 17 deselected, 2 warnings in 14.06s

python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py
25 passed, 2 warnings in 21.82s
```

### RED/GREEN: PostgreSQL migration and projection parity

The PostgreSQL contract initially failed because
`_migrate_from_v54_to_v55_postgres` did not exist. After implementation, the
mock contract verifies statement ordering, `%s` migration parameters, the
`chacha_keywords` table, the same transaction connection, canonical generated
UUIDv4 values, null/duplicate verification, named unique indexes, `NOT NULL`,
and final version advancement. Projection checks cover all three resources,
relationship application/snapshotting, ignored bootstrap routing metadata, and
one transaction per public apply call.

```text
python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
2 passed, 2 warnings in 11.70s

TLDW_TEST_NO_DOCKER=1 python -m pytest -q -p no:cacheprovider -rs tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py
1 skipped: Postgres not reachable; skipping Postgres-backed tests
```

### RED/GREEN: additive note-restore response

The extra focused API probe first failed because the test fixture omitted the
new required field, then remained RED because `notes.py` manually reconstructed
`KeywordResponse` without forwarding the stored `sync_id`. The two explicitly
authorized narrow changes made the same test green:

```text
python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_restore.py::test_restore_note_with_keywords
1 passed, 2 warnings in 10.39s
```

### Static verification

```text
python -m ruff check --no-cache <new and focused Task 2 modules/tests>
All checks passed!

git diff --check
exit 0
```

The monolithic `ChaChaNotes_DB.py` retains pre-existing repo-wide Ruff findings
outside the Task 2 lines; the new module and focused touched files are clean.

## Self-review

- All Sync identities are opaque, canonical UUIDv4 strings; none derive from
  integer IDs, names, timestamps, or paths.
- PostgreSQL constraint hardening happens only after both active and deleted rows
  are backfilled and null/duplicate verification passes, on the caller's one
  transaction connection.
- Integer IDs remain the local primary/foreign keys and every existing pointer or
  membership/link relationship is preserved by resource tombstones and migration.
- Folder path conflicts and invalid ancestry are detected before any subtree
  mutation; temporary paths avoid uniqueness conflicts during valid moves.
- `routing_metadata.bootstrap_capture` and origin provenance are intentionally
  not interpreted in this seam, reserving that behavior for Tasks 6 and 9.
- No later-task materializer, enrollment, bootstrap, or provenance behavior was
  added.

## Files changed

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py`
- `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- `tldw_Server_API/app/api/v1/schemas/notes_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/notes.py` (authorized one-line exception)
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py`
- `tldw_Server_API/tests/Notes/test_notes_restore.py` (authorized fixture exception)

## Concerns

- The optional live PostgreSQL integration remains skip-safe and was skipped in
  this environment because no PostgreSQL server was reachable. The server-free
  PostgreSQL migration/projection contract is green.
- Focused output contains pre-existing log-buffer permission noise and warning
  output; neither affected test results.

## Fix Round 1

### Authorization and scope

The inherited human turn directly approved ADR-033's suppression-table design
and the corresponding automatic SQLite/PostgreSQL schema-v55 changes. This fix
round implements that decision and the two Important review findings only:

- canonical `notes.folder_link` tombstones now suppress effective manual and
  source-backed membership without deleting source provenance;
- folder descendant discovery now rejects repeated or missing nodes before any
  path mutation, including moves to the root (`parent_sync_id=None`).

The controller-authored ADR-033, ADR index, design, implementation-plan, and
Backlog-task updates were preserved unchanged in intent and included with the
fix. The only API-test edit adds the already-required folder `sync_id` to one
stale mock/expected response so the existing effective-folder-read endpoint can
reach its assertion; no endpoint production behavior was added.

### TDD evidence

#### RED: fresh and v54-to-v55 suppression schema

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py
1 failed, 1 passed, 2 warnings in 11.44s
```

The new fresh-schema assertion observed no suppression columns. The upgrade
fixture also proved the table did not yet exist before the v55 implementation.

#### GREEN: fresh and v54-to-v55 suppression schema

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py
2 passed, 2 warnings in 11.45s
```

Both paths now create the composite-primary-key suppression table and its folder
index; the migration test also proves duplicate `(note_id, folder_id)` rows are
rejected.

#### RED: canonical folder-link suppression

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py::test_folder_link_tombstone_suppresses_effective_membership_without_deleting_source_provenance
3 failed, 2 warnings
```

Manual-only state had no suppression table and source-only/mixed state still
returned the tombstoned leaf through source membership. After the single-note
read and snapshot were fixed, the bulk-read assertion was added under strict
TDD by temporarily removing its production filter:

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py::test_folder_link_tombstone_suppresses_effective_membership_without_deleting_source_provenance
2 failed, 1 passed, 2 warnings in 11.97s
```

The source-only and mixed cases showed the suppressed leaf in the bulk result.

#### GREEN: canonical folder-link suppression

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py::test_folder_link_tombstone_suppresses_effective_membership_without_deleting_source_provenance
3 passed, 2 warnings in 10.67s
```

Manual-only, source-only, and mixed state each exercise idempotent tombstone and
upsert calls. The assertions cover single-note reads, bulk reads, snapshots,
manual projection, suppression cardinality, and preservation of source
membership/source-key rows.

#### RED: pre-existing descendant cycle

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py::test_folder_hierarchy_rejects_preexisting_descendant_cycle_before_mutation
1 failed, 2 warnings in 12.94s
```

`pytest-timeout` interrupted the unbounded `queue.extend(...)` loop. The test
sets `parent_sync_id=None` and captures both paths before the operation.

#### GREEN: pre-existing descendant cycle and rollback

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py::test_folder_hierarchy_rejects_preexisting_descendant_cycle_before_mutation
1 passed, 2 warnings in 9.13s
```

The bounded visited-set traversal rejects the repeated node before temporary or
final path updates, and the test proves both stored paths remain unchanged.

#### GREEN: affected contracts and regressions

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
2 passed, 2 warnings in 9.13s

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py
11 passed, 2 warnings in 11.92s

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k 'sync_id or hierarchy or soft_delete or suppression'
9 passed, 20 deselected, 2 warnings in 12.08s

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_restore.py::test_restore_note_with_keywords
1 passed, 2 warnings in 9.15s

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_api_integration.py -k folders
1 passed, 50 deselected, 2 warnings in 9.22s

TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider -rs tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py
1 skipped: Postgres not reachable; skipping Postgres-backed tests
```

The live PostgreSQL skip is explicitly permitted by the Task 2 brief. The
server-free PostgreSQL contract verifies both fresh DDL and the v54-to-v55
migration use the same transaction connection.

### Exact warning and log evidence

Focused output included these exact lines:

```text
system_log_buffer append failed: PermissionError
SINGLE_USER_API_KEY uses a legacy format; generate a new server-generated key for improved security.
USER_DB_BASE_DIR not configured in tests, using isolated fallback: <temporary test path>
character_cards_fts out of sync (active=1 indexed=0); rebuilding.
```

None originates in the touched fix-round code. The first comes from the
pre-existing buffered logging sink trying to append outside the sandbox. The
second comes from `app/core/AuthNZ/settings.py`; the third from
`app/core/DB_Management/db_path_utils.py`; and the fourth from the existing
`ChaChaNotes_DB.py` character-card FTS self-heal path, outside the modified
schema/read methods. Pytest's two framework warnings are the existing passlib
`'crypt' is deprecated and slated for removal in Python 3.13` warning and
existing Pydantic field-shadow warnings such as
`Field name "schema" in "ResponseFormatJsonSchemaSpec" shadows an attribute in parent "BaseModel"`.

During the first migration RED, an assertion containing a set literal was
passed through the existing transaction logger and its braces were interpreted
by Loguru, producing the exact secondary error `KeyError: "'created_at'"`.
That logger is baseline code outside the Task 2 changes; the final test no
longer routes a brace-bearing assertion message through it.

### Files changed in Fix Round 1

- `Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md`
- `Docs/ADR/README.md`
- `Docs/superpowers/plans/2026-08-08-notes-organization-sync-implementation-plan.md`
- `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`
- `backlog/tasks/task-13003 - Synchronize-Notes-keywords-collections-and-folders.md`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py`
- `tldw_Server_API/tests/Notes/test_notes_api_integration.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py`

### Fix-round self-review

- Effective single-note reads, bulk reads, and snapshots subtract suppressions.
- Upsert clears suppression and idempotently ensures one manual projection;
  tombstone removes only the manual projection and idempotently ensures one
  suppression row.
- Source membership and source-key rows are never deleted by either operation.
- The suppression schema is present for fresh SQLite/PostgreSQL databases and
  existing v54 databases upgraded to v55.
- Descendant traversal is bounded and validates every queued node before any
  path mutation; invalid cyclic or missing descendants cannot partially repath
  the hierarchy.
- No bootstrap, enrollment, materializer, or later-task behavior was added.
