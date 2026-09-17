# Independent review: UAT181 complete Notes bootstrap

**Clear for the bounded source repair; no actionable findings.** Independent current-source run: **44 passed, zero skipped, 69.24 seconds** (42 PostgreSQL cases and 2 SQLite cases). The official isolated PostgreSQL runner required PostgreSQL and provisioned its own fixture databases; no native application requests or runtime changes were made.

## Scope and ownership

All four current source/test hashes match the author's manifest and exact review snapshots. An independent line/AST comparison verifies exactly four literal `read_only=True` additions across three production files. Removing those four keywords gives identical baseline ASTs. SQL, parameters, result conversion, filtering, ordering, writes, helper defaults and error behavior are unchanged. The permanent test is byte-identical to the retained RED test.

| Site | Inspected operation |
| --- | --- |
| `CharactersRAGDB._list_generic_items` | Active keyword/collection list. The only two production callers supply keywords and keyword_collections. |
| `KeywordStore.count_keyword_collections` | Active collection COUNT. |
| `KeywordStore.get_keywords_for_collection` | Collection membership JOIN and ordered active keywords. |
| `NoteStore.get_note_counts_for_keywords` | Active-note usage aggregate; an empty ID list still executes the aggregate. |

These are side-effect-free SELECTs with no locking clauses or mutating functions. The unchanged helper owns only PostgreSQL reads entered from IDLE with both wrapper/backend transaction depths zero. Implicit caller transactions, raw BEGIN, explicit and nested scopes are preserved. SQLite is unchanged.

## Causal and regression coverage

The tests call the actual async Notes, keyword and collection handlers with real database operations, injecting only fixture identity/database and an allowing rate limiter. Both keyword→collection→Notes and collection→keyword→Notes orders are covered with the UI's include-note-count/include-collection-contents flags. Empty means absent keywords/collections; both variants retain a real note and folder.

The first connection stays open while a second real CharactersRAGDB initializes the same isolated database under a bounded lock timeout. State checks include all public ordinary/partitioned table relation locks, not a hand-picked Notes-only allowlist. Five independent read starters prevent one inherited transaction from concealing another unscoped query. Data assertions retain memberships, counts, note content/title and folder data.

The reviewed unchanged RED receipt has 14 failures/30 passes: four actual replacement failures after correct handler results and ten retaining-read failures. The final suite independently passes every case. Caller controls preserve pending writes across implicit/raw-BEGIN/ChaCha/nested/backend ownership with commit and rollback; explicit first-read scopes retain their transaction until exit. Notes-only and SQLite data/rollback controls also pass. These are direct handler/database tests, not an ASGI auth or browser-scheduling simulation.

## Static evidence and limits

Author static receipts report Ruff 0 baseline/current and Bandit production 0 findings/errors baseline/current. New test Bandit excludes only assertion rule B101. Source hashes match those receipts; independent review checked the exact scope and query semantics rather than rerunning unchanged static scans.

This accepts the specified complete ordinary Notes bootstrap chain, not every read in the application or a whole-database lifecycle guarantee. The latest reported native overlap eventually returned 200 after the old process exited; this review does not relabel it as a 500. Native Flashcards/Buddy/Notes→replacement acceptance remains the parent's gate. If that gate fails again, the documented architecture reassessment requirement still applies.

## Reproduce

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-notes-bootstrap-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_notes_bootstrap_lifecycle.py -q --tb=short
```

Exact hashes, changed lines, AST results and author packet hashes are in `source-verification.json`; the independent log is retained alongside this report. No production/test, runtime, browser, task/tracker or git mutations were performed by this reviewer.
