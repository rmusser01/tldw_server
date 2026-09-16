# Independent UAT168 review — initial seven-substitution batch

Reviewer: source013_diagnosis (not the UAT168 author).

Reviewed production SHA256: e80da46d26fcc34ec51ac9904ef6d2a8cd102c6e566f86789c3fb649934c0c9e.
Reviewed new test SHA256: 68868cfec799d4cff7073ae22dc169e48311078633589639584dae0f411570ad.

## Seven substitutions

The substitutions use exact SELECT column names: cnt for COUNT(DISTINCT f.id) AS cnt; version for deck version; id/version for mutable card update; id, uuid and card_id for asset reconciliation. SQLite uses sqlite3.Row and PostgreSQL dict_row, so named access is supported by both. SQL, transaction boundaries, optimistic-version checks, asset cross-card rejection and attach/detach operations are unchanged. No finding against the seven changed lines.

The new backend tests use real CharactersRAGDB and official pg_database_config, covering empty count, total/deck-filter count, and actual asset row attach/foreign-card rejection/detach. The asset fixture bytes are storage-only bytes, not an image validation test; this matches the DB-method boundary. The shared UAT167 real-router suite independently covers deck/card updates and bulk save. I inspected the root-produced required-PG combined receipt: 36 passed / 0 skipped in 25.43 seconds, retained as .tmp/fresh-uat-recovery-20260916/uat167-168-green.redacted.log. I did not claim to rerun that command.

## Remaining in-scope branch for author

update_flashcard still reads row[0] in its no-mutable-fields branch when expected_version is supplied (near line 37368). The query is SELECT version; PostgreSQL returns a mapping. FlashcardUpdate accepts a request containing only expected_version, _prepare_flashcard_update removes that field and leaves an empty update dictionary, and the route calls this branch. Thus the function repair leaves the same mapping assumption on an allowed version-only request. This was reported to root as a source-level residual; I have not yet independently run an actual PostgreSQL reproduction. Recommend adding backend no-op/version-conflict controls and using the same named version column if reproduced. No other-method audit or broader row rewrite is requested.

Final independent clearance is held for this branch decision/reproduction. No production/test edits, runtime/browser action, or database setup were performed by this reviewer.
