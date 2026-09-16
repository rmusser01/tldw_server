# UAT180 / TASK-13260.117 — flashcard lifecycle row access

## Diagnosis and approved investigation

`soft_delete_flashcard` reads the selected `id`, `version`, and `deleted` fields by integer positions; `reset_flashcard_scheduling` similarly reads `id` and `version`. SQLite rows support these positions; the PostgreSQL adapter returns named mappings. The adjacent ordinary update path already uses named columns. The existing real SQLite endpoint tests cover these operations, but do not exercise the PostgreSQL row shape.

## Proposed minimal production scope

After fresh real-PostgreSQL RED and parent release of the shared ChaCha file, replace the two destructuring statements with named accesses. Preserve all SELECTs, mutation fields, version checks, idempotent-delete behavior, transaction ownership and endpoint mappings. No general row adapter, exception change or other mutation cleanup.

## Permanent tests

New `test_flashcard_lifecycle_backends.py` uses real SQLite and the official `pg_database_config` fixture, with actual Flashcards router requests. Cover successful delete and repeated deletion; scheduling reset with explicit HTTP version and omitted DB-level version; HTTP omission remains422; missing and stale versions; reset defaults and unchanged content; outer rollback after each real mutation. PostgreSQL must run with zero skips through the existing owned-cluster runner. No AuthNZ pool or manual database provisioning.

## Ownership and stages

Parent owns task/tracker/shared design, native acceptance, commits and runtime. Parent committed UAT177/178 as f817f4aa968d9d5ec5c01c5b4db53c924867c4fe and released ChaCha for the two statements after the real RED. This packet and the new regression are independently owned. Stages1 diagnosis,2 RED,3 minimalGREEN,4 static verification complete; independent review/native pending.
