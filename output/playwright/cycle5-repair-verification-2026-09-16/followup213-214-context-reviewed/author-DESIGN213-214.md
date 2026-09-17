# Approved bounded UAT213 / TASK13260.152 and UAT214 / TASK13260.153

213: PostgreSQL get_conversation_settings consumes QueryResult.first and reads named settings_json/settings_version/last_modified fields. Keep backend.execute lifecycle, SQLite branch, JSON and error/None semantics unchanged. Real official PG/SQLite missing/empty/populated/version/invalid JSON, caller rollback and actual settings GET controls. Observe actual pool-return IDLE state, without replacing backend execution.

214: get_character_world_books and actual populated route follow-up get_entry_counts_for_world_books use existing execute_query(read_only=True) for pure reads. Keep SQL, bound filters, metadata, enabled/deleted semantics, priority/name order, normalized count subset and zero-fill. Real official PG/SQLite fixture, populated/empty route controls, caller rollback/read-own-pending and standalone PG IDLE. Test seed uses real DB transactions because unrelated WorldBook writer methods share the unsupported context pattern; no writer claim from seed.

No backend wrapper protocol changes, schema, authorization, generic execute helper, or blanket manager conversion. Other manager CRUD/context methods retain the pattern and are explicitly outside this bounded observed read-path repair. Preserve their existence as residual evidence. Parent owns runtime/native/tasks/git.
