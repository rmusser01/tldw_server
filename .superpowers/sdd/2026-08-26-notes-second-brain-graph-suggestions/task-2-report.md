# TASK-13138 Task 2 Report

## Implementation Summary

Task 2 adds deterministic, bounded, source-grounded lexical retrieval on top
of Task 1's owner- and dataset-scoped v64 graph store. It provides canonical
content fingerprints and field-bounded evidence references, backend-portable
FTS retrieval, direct-relationship and exact-fingerprint suppression, bounded
tag collection, and projection freshness reporting. It does not make provider
calls or add Jobs, state-machine, API, UI, or schema-version changes.

## Files

- `tldw_Server_API/app/core/Notes_Graph/suggestion_content.py`
- `tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py`
- `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py`
- `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py`
- `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py`
- `.superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md`
- `backlog/tasks/task-13138 - Implement-first-class-Notes-graph-workspace-and-reviewable-AI-suggestions.md`

## Analogous Patterns Reused

1. `note_store.py::search_notes` supplies the backend split for parameterized
   SQLite FTS5 `MATCH ?`/`bm25` and PostgreSQL `notes_fts_tsv`/`ts_rank`
   retrieval.
2. `note_graph_projection_store.py` supplies owner-bound graph reads, direct
   relationship checks, and deterministic identifier tie-breaking.
3. `KeywordStore` backend-table mapping is reused for the PostgreSQL keyword
   table rather than assuming the SQLite table name.
4. `tests/_plugins/postgres.py::pg_database_config` supplies the established
   live-PostgreSQL integration fixture and its unavailable-only skip behavior.

## TDD And Verification

### Initial RED

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py -q
```

Output: `2 collection errors`; the expected
`suggestion_content` and `suggestion_retrieval` modules did not exist.

### Bounded-Estimate RED Then GREEN

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py::test_retrieval_estimates_only_bounded_evidence_windows -q
```

RED output: `1 failed, 2 warnings`; the prior full-note estimate exceeded the
24,000-token bound. The implementation was changed to count only capped
evidence-window spans. The same command then reported `1 passed, 2 warnings`.

### Final Required Task 2 Suites

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py -q
```

Output: `20 passed, 2 warnings in 13.41s`.

The integration suite exercised SQLite and live PostgreSQL through
`pg_database_config`; no backend tests were skipped.

### Directly Affected Existing Tests

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py::TestNoteStoreSearch tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k test_search_notes -q
```

Output: `3 passed, 57 deselected, 2 warnings in 8.57s`.

### Static And Security Checks

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Notes_Graph/suggestion_content.py tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py
```

Output: `All checks passed!`

Command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Graph/suggestion_content.py tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py -f json -o /tmp/bandit_task_13138_task_2.json
```

Output: Bandit JSON result contains `0` findings. It reports acknowledged
parameterized SQL `# nosec B608` locations in the store; no new findings were
introduced.

## Constraint Self-Review

- Canonical bytes are exactly ASCII `notes-graph-content-v1`, NUL, LF/NFC
  normalized title UTF-8, NUL, normalized content UTF-8. Fingerprints use the
  resulting SHA-256 bytes.
- Evidence offsets are Unicode code-point, half-open ranges. References retain
  a field discriminator and reconstruct only when the current fingerprint and
  field-local offsets are valid, so no reference crosses title/content.
- Source and candidate byte limits run in SQL before text transfer. SQLite uses
  exact `length(CAST(COALESCE(... ) AS BLOB))`; PostgreSQL uses
  `octet_length(COALESCE(...))`. A selected oversized source raises; oversized
  candidates are excluded and aggregate-counted.
- FTS queries and derived term expressions are bound parameters. Retrieval code
  logs neither queries, terms, nor note content. Missing FTS structures fail
  closed with no table-scan fallback.
- Binding limits are enforced: 24 terms, 60 backend overfetch, 30 selected
  candidates, 100 tags, four source windows, two candidate windows, 480 code
  points per window, and bounded-window token estimation below 24,000.
- Eligibility removes only the selected note, trash, direct manual/projected
  wikilink relationships in either direction, and exact source/fingerprint
  rejection pairs. Unit coverage confirms shared tags and source membership
  remain eligible.
- Backend rank order is retained only within the backend execution and uses an
  identifier tie-breaker. Raw backend scores are never returned or compared
  across engines.
- Every read receives a dataset ID and executes with owner scope through the
  Task 1 store. PostgreSQL also sets the dataset setting for RLS, but explicit
  application predicates remain present. No v64 migration/versioning changed.

## Concerns

No task-specific concerns remain. The successful pytest commands each report
two environment/configuration warnings unrelated to the retrieval behavior.
