# UAT174 / TASK-13260.111 — private uniqueness classification

## Root cause
The actual native duplicate deck POST returns500. PostgreSQLBackend.execute intentionally raises a new generic DatabaseError outside the driver catch to remove raw message/cause/context. ChaCha add_deck catches that error and calls _is_unique_violation, which currently recognizes only message fragments. The privacy boundary therefore erases the fact needed for the existing ConflictError→HTTP409 mapping.

## Proposed bounded implementation
1. Add a payload-free UniqueConstraintError subclass of the existing backend DatabaseError. Existing DatabaseError catches remain compatible. Its type is the complete structured category; it stores no SQLSTATE, values, SQL, diagnostics, or constraint name.
2. In PostgreSQLBackend.execute only, recognize trusted psycopg driver SQLSTATE23505 while handling the original error. Retain only a boolean. Keep existing rollback/logging behavior and raise the safe typed error with the unchanged generic message outside the catch. All other failures continue to raise ordinary DatabaseError. No execute_many or transaction-manager redesign.
3. In ChaCha _is_unique_violation only, recognize the safe subclass before retaining the existing SQLite/message compatibility behavior. No other ChaCha methods, HTTP handlers, schemas, or UI changes.

## Existing patterns studied
- backends/base.py uses DatabaseError subclasses for safe backend categories (NotSupportedError).
- PostgreSQLBackend.execute and unit/test_postgresql_error_redaction.py require generic message and empty driver cause/context, including rollback-failure privacy.
- ChaCha's workspace saved-view helper already recognizes SQLSTATE23505, but also inspects constraint identity and cause chains; this repair must not use that pattern across the redaction boundary.
- flashcards.create_deck and map_db_error_to_http already map ConflictError409 and other database errors500; no new endpoint mapping is needed.

## Verification / stages
- RED before implementation: official pg_database_config; actual duplicate-deck endpoint/handler chain returns500 today, should return409. Assert original committed row unchanged and no extra row. Real non-unique database failure remains500; successful later operation proves rollback/recovery.
- Privacy controls at backend boundary: uniqueness category without query/values/constraint name, __cause__, or __context__; generic failures remain generic. Run existing source-query redaction and transaction-manager suites unchanged. Verify SQLite duplicate behavior.
- GREEN: minimal three-production-file change. Run focused real-PG + unit/SQLite controls, scoped Ruff/Bandit and diff checks. Freeze source/tests/patch/manifest for independent review before parent-owned native duplicate/save acceptance.

## Ownership and status
No production edits until parent releases ChaChaNotes_DB.py after171 commit. Parent owns browser/runtime, tracker/tasks, shared design integration, staging/commits. Agent owns only proposed base.py, postgresql_backend.py, ChaCha uniqueness helper/import, and focused tests after release. Account handling and all other ChaCha callers remain outside this repair.

## Handoff status
Parent released ChaCha after171/172 commitcc0f12a5d8 and approved this exact marker design. RED and GREEN stages complete; frozen final29/4pass, Ruff0, Bandit0findings/0errors. Independent review and native acceptance remain parent-owned. See IMPLEMENTATION.md and owned-manifest.json.
