# ChaChaNotes DB Corruption Investigation And Recovery Runbook

Date: 2026-05-09

## Scope

This note records the non-destructive investigation of the malformed default
ChaChaNotes database found during the character-chat WebUI walkthrough, and the
recovery process to use before any restore is attempted.

Do not overwrite the original database as part of diagnosis. Treat SQLite
`.recover` output as a candidate salvage artifact until it passes app-level
schema and workflow checks.

## Evidence Summary

Original DB:

- Path: `Databases/user_databases/1/ChaChaNotes.db`
- Forensic copy: `/private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.original.db`
- SHA-256 for both source and forensic copy:
  `5c159820e8eb1954f1c04ed2ddb606371b3924d2a1993d36a13304d32bd5cb92`
- Size: `19685376` bytes
- Original file times: birth `Jan 12 20:54:26 2026`, modified `Apr 18 15:29:18 2026`, changed `May 8 20:46:14 2026`
- SQLite file header is present and reports WAL read/write file format mode:
  writer version `2`, read version `2`, SQLite writer version `3046001`,
  page count `4806`, free pages `13`
- No `ChaChaNotes.db-wal` or `ChaChaNotes.db-shm` sidecars were present beside
  the source file at investigation time.
- `lsof` showed no process with the source DB open at investigation time.

Immutable checks on the forensic copy failed before schema inspection:

```text
sqlite3 'file:/private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.original.db?immutable=1' 'PRAGMA quick_check;'
Error: in prepare, database disk image is malformed (11)
```

The same malformed error occurred for `PRAGMA integrity_check` and basic
`sqlite_master` inspection. This confirms the failure is not just an application
migration error; SQLite cannot reliably prepare reads against the original DB.

## Recovery Candidate

The recovery SQL was generated from the forensic copy:

```text
sqlite3 /private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.original.db .recover
wc -l /private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.recover.sql
29892
```

Importing that SQL into a clean database produced an integrity-clean SQLite
file:

```text
PRAGMA integrity_check;
ok
```

Recovered schema version:

```text
rag_char_chat_schema|44
```

Critical recovered counts:

```text
character_cards|451
conversations|915
messages|2123
writing_themes|5
writing_wordclouds|0
lost_and_found|368
sync_log|11919
```

Sample sanity checks found character cards, recent conversations, and
user/assistant messages with non-empty content. The recovered candidate therefore
contains meaningful chat and character data.

However, the recovered candidate is not directly restore-ready. Initializing
`CharactersRAGDB` against a copied recovered DB failed app-level schema
verification:

```text
SchemaError: Missing required FTS tables for 'rag_char_chat_schema': flashcards_fts
```

That means the candidate should be used for salvage, comparison, or a repair
workflow, not copied over the production file without additional FTS/table
rebuild validation.

## Lost Root Pages

The recovered `lost_and_found` table contains 368 rows:

```text
rootpgno|rows
87|12
88|24
89|332
```

The recovered schema maps those root pages to writing-related objects:

```text
87|table|writing_themes
88|index|sqlite_autoindex_writing_themes_1
89|table|writing_wordclouds
```

The damage is therefore concentrated around writing-table pages, while the
character, conversation, and message tables were substantially recoverable.

## Root-Cause Assessment

Confirmed evidence:

- The source DB file itself is malformed under SQLite immutable checks.
- The header is a valid SQLite header and indicates WAL file format mode.
- The source directory did not contain WAL/SHM sidecars at investigation time.
- `.recover` salvaged a schema-v44 database with substantial character-chat data.
- `lost_and_found` rows map to writing tables and their index pages.
- Current `DB_Backups.py` full backup code uses the SQLite backup API, and the
  incremental path uses `VACUUM INTO`; the current code is not a raw DB-file copy
  path.

Hypothesis status:

- Interrupted write or checkpoint: plausible. WAL-mode header, absent source
  sidecars, and localized btree damage fit a write/checkpoint interruption, but
  no direct OS/process event was found.
- Missing sidecar WAL after unsafe copy/restore: plausible. The source file is
  WAL-mode and no sidecars were present. Current backup/restore code argues
  against the current app backup path as the cause, so this remains possible only
  through older code, external tooling, or manual file movement.
- Migration-time writing-table damage: possible but unproven. The damaged root
  pages map to writing objects, but the relevant writing migrations predate the
  observed April file modification time and no migration log tying the event to
  corruption was found.
- Concurrent access: unknown. No current process had the file open during
  investigation. Past concurrent app/test access cannot be reconstructed from
  the available evidence.
- Filesystem or external tooling interruption: unknown. The file has macOS
  provenance metadata and a later ctime, but that is not evidence of data-page
  mutation by itself.

Most likely class: an interrupted SQLite write/checkpoint or an unsafe
copy/restore of a WAL-mode DB without a complete sidecar/checkpoint state. The
available evidence does not prove a single cause.

## Startup Guardrail

The dependency layer now performs a read-only `PRAGMA quick_check` before opening
an existing per-user ChaChaNotes SQLite DB through `CharactersRAGDB`. If SQLite
reports corruption signatures such as `database disk image is malformed`,
`malformed database schema`, or `file is not a database`, initialization fails
closed with:

```text
HTTP 503
ChaChaNotes DB corruption detected; repair or restore required
```

The user-facing error intentionally avoids leaking local file paths. The health
snapshot records the safe classification `sqlite_corruption` for diagnostics.

This guardrail prevents a malformed existing file from being treated as a normal
schema-initialization failure and avoids silently creating or overwriting user
data.

## Manual Doctor And Recovery Flow

Use this flow until there is a dedicated admin repair command.

1. Stop the server and any tests or scripts that may touch the same user DB.
2. Copy the DB and any sidecars into a dated forensic directory:
   `ChaChaNotes.db`, `ChaChaNotes.db-wal`, and `ChaChaNotes.db-shm` if present.
3. Record `shasum -a 256`, `stat`, `file`, sidecar listings, and `xxd -l 128`
   for the copy.
4. Run immutable checks on the copy:
   `PRAGMA quick_check`, `PRAGMA integrity_check`, and a minimal
   `sqlite_master` query.
5. If immutable checks fail, run `.recover` from the forensic copy into SQL and
   import that SQL into a new candidate DB.
6. Validate the candidate with `PRAGMA integrity_check`, schema version, key
   table counts, sample character/conversation/message rows, and
   `lost_and_found` root-page mapping.
7. Run an app-wrapper smoke check against a copy of the candidate in a temporary
   allowed DB directory.
8. If the wrapper reports missing FTS or schema objects, do not restore the
   candidate directly. Rebuild missing virtual tables through an explicit repair
   path or export/import salvage data through app-level APIs.
9. Restore only after creating a final backup of the current live DB and after a
   human approves the exact candidate file, validation output, and rollback path.
10. Restart the server and verify character list loading, conversation list
    loading, opening a recovered conversation, and creating a new character-chat
    message.

## Verification Commands Recorded

Focused regression tests:

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py -q
```

Recovery validation commands:

```text
sqlite3 /private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.recovered.db 'PRAGMA integrity_check;'
sqlite3 /private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.recovered.db 'SELECT * FROM db_schema_version;'
sqlite3 /private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.recovered.db 'SELECT rootpgno, COUNT(*) FROM lost_and_found GROUP BY rootpgno ORDER BY rootpgno;'
```
