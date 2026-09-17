# UAT182 — named Flashcard asset binary column

Task TASK13260.119. Baseline causal official-fixture probe: populated asset add/metadata succeed; PostgreSQL getter raises KeyError0 at row[0], while SQLite succeeds. The content endpoint delegates to the same getter. Parent owns native acceptance, task records and commits.

## Approved design
Replace only `blob = row[0]` inside get_flashcard_asset_content with `blob = row["image_data"]`. Both sqlite3.Row and PostgreSQL dictionary rows expose that selected name. Keep the SELECT, read_only lifecycle flag from181, bytes conversion, memoryview branch, missing-row None, soft-deleted filtering and defensive null handling unchanged. No migration/schema/auth/image-validation change.

## Permanent tests
Use official isolated pg_database_config and real SQLite controls for exact populated bytes and metadata, real router upload/content HTTP response including MIME and repeat GET, and missing/deleted404. Use a small separate row-boundary control for bytes/memoryview/empty-bytes/None conversion. Both real schemas declare image_data NOT NULL, so null is intentionally a defensive unit boundary, not an invented live DB state. No manual test DB creation or app DB use.

## Stages
1. Permanent causal RED: complete,6fail/6pass before production182 edit.
2. Parent applies isolated one-line patch after181 flags freeze: complete. This agent does not modify shared production.
3. Required PG/no skips + SQLite/control tests, source-scoped Ruff/Bandit, independent review packet: complete,28testsPASS/no skips. Native image-upload/content/reload is separate parent acceptance.
