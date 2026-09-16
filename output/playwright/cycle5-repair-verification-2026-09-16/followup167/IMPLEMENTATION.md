# UAT167 / TASK-13260.104: PostgreSQL Flashcards timestamps

## Root cause and minimal design

Native POST /api/v1/flashcards/decks completed the write, then FastAPI rejected PostgreSQL-aware datetime values in created_at and last_modified because the public Deck schema requires optional strings. Deck create/list/update and card single/bulk save/list/read/update use the same raw DB rows. The approved repair normalizes datetime values at the two response models, preserving their string schemas. Existing string values and nulls pass through untouched; unrelated invalid types remain rejected. The normalization retains the datetime offset and precision using isoformat(), consistent with existing boundary conversion patterns. No database write, ownership, preference, source-provenance, route, or request-schema behavior changed.

Production changes are only two before-field validators (12 lines) in app/api/v1/schemas/flashcards.py, on Deck created_at/last_modified and Flashcard created_at/last_modified/due_at/last_reviewed_at. Source and exact tests are retained in review-snapshot; hashes are in owned-manifest.json.

## Test boundary

New permanent tests use the actual FastAPI router with real CharactersRAGDB databases. Only the database dependency is bound to a test-owned instance. PostgreSQL instances use the official pg_database_config -> pg_temp_db plugin on the existing owned cluster, with TLDW_TEST_POSTGRES_REQUIRED=1 and Docker autostart disabled. SQLite instances use pytest tmp_path. No LLM call or application runtime/browser is involved. The synthetic note-shaped generated-card payload checks source_ref_type/source_ref_id and canonical saved content.

The suite independently covers deck create/list/update, card single/bulk save, card read/update/list, aware UTC/non-UTC/naive datetimes, unchanged strings/nulls, idempotence, and invalid input types. It does not claim a full multi-user authentication test: owner/auth policies are unchanged, and root owns native Alice/Bob acceptance. There is no standalone deck GET route; canonical get_deck supplies the create/update responses and list_decks supplies the read collection.

## RED and current GREEN evidence

- Initial sandbox run could not reach the local PostgreSQL port. Required fixtures failed; no skips. This is an environment receipt, not a product RED.
- Escalated official-fixture RED: 14 failed / 14 passed / 0 skipped. Four real PostgreSQL API timestamp response failures plus six timestamp schema failures. Four additional failures were concrete row-position KeyErrors separately assigned UAT168: count_flashcards, update_deck, update_flashcard, reconcile_flashcard_asset_refs (bulk save). Full redacted receipt: pg-red.log.
- After only the schema repair: 24 passed / 4 failed / 0 skipped. All ten timestamp failures now pass. The same four UAT168 DB failures remain, not masked, skipped, or changed here. Receipt: pg-schema-green.log.
- Scoped Ruff: zero findings (ruff.json). New test import formatting was corrected after initial check.
- Bandit on the touched production Python schema: zero findings and zero parsing errors (bandit.json/bandit.log).
- git diff --check on owned scope: PASS.
- Existing SQLite endpoint/schema regression suite: 184 passed / 0 skipped (4 existing warnings), 132.34 seconds. Receipt: sqlite-regressions.log.

Commands (repository root, activate .venv first):

    TLDW_UAT_EVIDENCE_LABEL=uat167-pg-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_flashcards_timestamp_contract.py -q --tb=short
    TLDW_UAT_EVIDENCE_LABEL=uat167-pg-schema-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_flashcards_timestamp_contract.py -q --tb=short
    python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_scheduler_schema.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -q --tb=short
    python -m ruff check tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/tests/Flashcards/test_flashcards_timestamp_contract.py --output-format json
    python -m bandit tldw_Server_API/app/api/v1/schemas/flashcards.py -f json -o .tmp/uat167-repair-20260916/bandit.json

## Combined verification supplied by root

After separately owned UAT168 fixes, the full permanent timestamp suite (28 cases) plus the eight real-backend count/asset cases passed: 36 passed / 0 skipped, 25.43 seconds. Receipt: combined-green.log, run label uat167-168-green. This independently inspected receipt completes all eight UAT167 PostgreSQL endpoint cases, their SQLite counterparts, and timestamp schema controls. The author did not rerun identical tests without source changes. Independent UAT168 review separately raised its same-function version-only update branch; that does not change the schema patch or the passed UAT167 cases.

## Final combined verification

The final frozen eight-line UAT168 release plus UAT167 passes all 38 cases with zero skips; see combined-final-green.log. The additional two cases prove the repaired version-only update on SQLite and PostgreSQL. Independent UAT168 review and exact hashes are retained in ../uat168-repair-20260916/independent-review.md and independent-manifest.json.

## Handoff limits

Source/test batch frozen for independent review. The final combined run is GREEN: 38 passed / 0 skipped, 28.32 seconds (combined-final-green.log, label uat167-168-final-green). This supersedes the initial 36-case receipt. The separate UAT168 version-only update finding was reproduced on actual PostgreSQL and resolved by its author; independent review is now clear. Root owns native saved-draft recovery and backend restart; no native acceptance claim is made here. No tracker/plan/git staging/commit/browser/service edits performed.
