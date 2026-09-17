# UAT213 / TASK13260.152 and UAT214 / TASK13260.153

## Frozen review-ready result

Four owned paths: two production modules and two new permanent backend regression files. Exact bytes, baseline hashes and byte-identical review snapshots are recorded in owned-manifest.json. owned.patch includes the complete unit and no concurrent agent changes. Parent owns native acceptance, task finalization and commit.

### UAT213 — persisted settings were read as absent

Native logs record `QueryResult has no attribute fetchone`. PostgreSQL backend.execute returns QueryResult with `.first` and named dictionary columns. The settings getter called `.fetchone()` and then attempted tuple unpacking, caught the error and returned None. The actual GET settings route consequently returned200 with an empty settings document despite a persisted document.

The PostgreSQL branch now uses `.first` and named `settings_json`, `settings_version`, and `last_modified` fields. SQLite, SQL, JSON decoding, missing/error fallback and backend execution are unchanged. A real pool-return observation proves both standalone getter checkouts return IDLE; no lifecycle change was necessary. Caller rollback remains intact. This does not add new read-your-uncommitted-settings semantics to the existing separate backend execution path.

### UAT214 — character WorldBook reads failed before executing SQL

Native logs record that BackendConnectionWrapper does not support the context-manager protocol. get_character_world_books entered `with self.db.get_connection()` before its SELECT. The populated HTTP route also immediately calls get_entry_counts_for_world_books, which had the same defect; fixing only the first method would leave populated responses failing.

Both pure reads now call the existing `execute_query(..., read_only=True)` helper. SQL text, parameters, enabled/deleted filters, priority/name ordering, response fields, normalized entry-count subset and zero-fill are unchanged. The helper completes an initially idle standalone PostgreSQL read and leaves caller-owned transactions alone. The same removal also prevents the old SQLite connection context from committing a caller's pending writes; the causal rollback regression covers both backends.

No generic wrapper protocol, schema, initialization, authorization or global helper change. UAT176 initialization stays separate and its real backend regressions pass.

## Causal RED and final GREEN

Required PostgreSQL execution used the existing official runner and pg_database_config fixture, never a running application database or manual database setup.

- **Causal RED: 14 failed /16 passed /0 skipped, 36.66s.** UAT213:5 PostgreSQL failures, including persisted/empty document roundtrip, actual GET settings200 returning `{}`, caller transaction control, and standalone-read observation. UAT214:8 PostgreSQL failures from the unsupported context in the two reads, including actual empty/populated route500;1 SQLite caller-rollback failure because a pure read committed the pending write. Tests were not changed after this causal run.
- **Focused GREEN:30 passed /0 skipped,32.98s.** Same two files. Settings missing/empty/populated/version/timestamp/invalid JSON, actual HTTP GET, rollback and real PG returned-connection IDLE. WorldBooks empty/populated, enabled book AND enabled attachment, deleted exclusion, priority/name order, actual HTTP rows/counts, count normalization/zero-fill/disabled-entry semantics, caller read-own-pending/rollback and standalone PG IDLE.
- **Adjacent GREEN:69 passed /0 skipped,28.70s.** Real WorldBook initialization transaction controls, conversation update contracts, delegated conversation store, actual SQLite CharacterAssociation, settings merge contracts and character WorldBook permission/error mappings.
- **Ruff:**9 baseline/9 current findings, identical file/rule/message signatures after line shifts; no findings in either new test. Existing import order/SIM118/UP032/TRY203 remain unchanged.
- **Bandit:** both production files zero findings/errors; new tests zero findings/errors with B101 assertions excluded. Baseline production also zero.
- **Compilation:** all four owned Python files pass. Added-line whitespace check passes. AST method comparison confines changes to get_conversation_settings, get_character_world_books and get_entry_counts_for_world_books.

## Reproduce

From repository root, with network access for the official isolated fixture:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat213-214-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_conversation_settings_read_backends.py tldw_Server_API/tests/DB_Management/test_character_world_book_reads_backends.py -q --tb=short
```

Adjacent command:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat213-214-adjacent-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py tldw_Server_API/tests/DB_Management/test_conversation_updates_backends.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py::TestCharacterAssociation tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py -q --tb=short
```

## Limits and residual source evidence

This is not a complete PostgreSQL WorldBook workflow certification. Other WorldBook CRUD/context methods still use the legacy `with get_connection()` pattern (including create_world_book, get_world_book, attach/detach and get_entries). They were not silently converted or claimed fixed. The populated read fixture seeds actual tables through real DB transactions to isolate the two approved reader contracts; it does not disguise a successful PostgreSQL create/attach API workflow. Native residual confirmation/association is the parent's separate decision.

The patch does not add an owner policy or claim raw-SQL isolation. Existing character-detail authorization and HTTP permission mapping remain unchanged. No browser action, live runtime/model call, deployment, task edit or git action was performed. Native post-repair acceptance remains pending. The copied error excerpt contains only the two relevant already-redacted failure categories.
