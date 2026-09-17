# UAT181 complete Notes bootstrap repair

TASK13260.118. Ready for independent source/test review. Exactly four `read_only=True` flags across three production files; one new permanent test file. Source and tests are frozen. Parent owns native acceptance, task tracking and commits.

## Causal finding and reassessment

The Notes page starts keyword listing (including note counts), collection listing (including contents) and ordinary Notes listing. The first two routes used unscoped reads on the same event-loop-thread database connection. Empty results still opened a transaction. Later correctly scoped Notes reads preserved that pre-existing transaction, accumulating folder locks. Earlier171/181 tests did not prepend these actual neighboring handlers; the cached PG health probe was already scoped and was not the cause at that site.

The latest native overlap independently retained old PID18641 locks on keywords/collections/Notes/folders and a new initializer waiting for note_folders. That attempt returned200 after the old process exited; this packet makes no native500 claim for that attempt. Previous native receipts with a restricted relation allowlist were not an exhaustive inventory.

This third bounded stage adds full-page chain coverage, instead of one further isolated caller. The ownership boundary remains unchanged because implicit/explicit caller-work preservation is required. If this full bootstrap or next native acceptance fails again, parent explicitly requires stopping patches and reassessing architecture before another repair.

## Exact change

- `CharactersRAGDB._list_generic_items`: pure keyword/collection listing; exactly two production callers.
- `KeywordStore.count_keyword_collections`: pure active collection count.
- `KeywordStore.get_keywords_for_collection`: pure membership contents read.
- `NoteStore.get_note_counts_for_keywords`: pure active-note usage aggregate.

Each query retains its SQL, parameters, filters, ordering and return shape. No global execute_query/default, connection policy, endpoint/auth, write, schema or migration change. AST verification removes the four newly added literal-true keywords and obtains exact baseline AST equality for all three files.

## Permanent RED and GREEN

New `tldw_Server_API/tests/DB_Management/test_chacha_postgres_notes_bootstrap_lifecycle.py` directly invokes the real async Notes handlers with supplied fixture DB/identity/rate limiter, preserving the real database operations. It exercises both keyword→collection and collection→keyword predecessor orders, empty and populated data, the actual include_note_counts=true/include_keywords=true flags, and then ordinary Notes listing. The original connection stays alive while a real second CharactersRAGDB initializes the same official isolated PostgreSQL database with a100ms lock budget. Per-handler diagnostics record transaction state and all public ordinary/partitioned table relation names; no production query parameters are collected.

Independent starter tests isolate five entry points corresponding to the four SQL sites, avoiding a first leaked read masking a later one. Data assertions check actual membership, note counts, title/content and the original note. Controls cover Notes-only replacement, pending implicit/raw-BEGIN/explicit/nested/backend writes with both commit and rollback, first reads inside explicit scopes, and SQLite data/rollback.

| Execution | Result |
| --- | --- |
| Same44 tests, baseline production |14 failed,30 passed,zero skips;64.81s |
| Four approved flags, unchanged tests |44 passed,zero skips;83.75s |
| Existing Notes lifecycle +KeywordStore +keyword management |47 passed,zero skips;64.63s |
| Existing graph/keyword-count control including deleted note |1 passed,zero skips;1.34s |

Total final verification: **92 passing tests across5 files,zero skips**. The14 failures were10 individual retaining reads and4 live replacement failures after correct handler outputs. All caller/SQLite/Notes-only controls passed before implementation. No harness failures or test weakening occurred. `red-test.py` matches the final permanent test byte-for-byte.

All PostgreSQL work used the existing official DB_Management fixture through `run-pg-tests.mjs`, required PostgreSQL enabled, no Docker autostart, no manual database setup or live app requests. These tests certify actual handler/database behavior, not native login or network scheduling.

## Reproduce

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-notes-bootstrap-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_notes_bootstrap_lifecycle.py -q --tb=short
```

Exact adjacent commands and results are retained in the matching `*-command.json` and redacted logs. Baseline copies, production.patch, owned.patch, source/evidence manifests and review snapshots accompany this report.

## Static checks and limits

Three production files: Ruff0 baseline/current,0 added; Bandit0 findings/errors baseline/current. New test: Ruff0, formatter check and Python compilation pass; Bandit0 with only pytest assertion ruleB101 excluded. Bandit emits existing comment/nosec warnings while reporting no findings/errors. Owned whitespace check passes. No broader lint/test/backend behavior claim.

Native Flashcards/Buddy/Notes reads followed by replacement remain the parent's final acceptance gate; task stays In Progress. No runtime, browser, model, configuration, task/tracker or git mutations were performed by this author.
