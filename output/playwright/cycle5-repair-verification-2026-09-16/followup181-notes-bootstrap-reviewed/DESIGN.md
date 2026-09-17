# UAT181 Notes bootstrap causal regression design

TASK13260.118; baseline1872951315dc96c06faf1da23aa6520de3866565. Parent authorized causal tests/design, then approved exactly four flags after retained RED14 failed/30 passed/zero skips. This is the third bounded181 stage; a further whole-bootstrap/native failure requires stopping patches and reassessing architecture before another fix.

The previous39+6 stages covered named Flashcards/Buddy/default/persona reads and Notes in isolation. They did not represent the complete Notes page bootstrap: keywords with note counts, collections with inline keywords, and Notes list. The latest native overlap now confirms locks on chacha_keywords/keyword_collections as well as Notes/folders, with a replacement waiting for note_folders. That attempt ultimately returned200 as the old process exited; it is not an observed native500.

## Test boundary

Add one permanent `test_chacha_postgres_notes_bootstrap_lifecycle.py`. Use the official DB_Management PostgreSQL fixture; one committed note/folder, with empty or populated keyword/collection data. Invoke the real async Notes handlers with the page's include_note_counts=true/include_keywords=true flags, in both predecessor orders, then Notes. Keep the first raw connection alive while constructing a real second CharactersRAGDB against the same isolated DB with a short lock budget. Record transaction state and all public table relations after each handler, excluding query values. Assert expected counts/memberships/content, not just absence of locks.

Isolate the actual list/count/contents reads reached by those handlers to identify every retaining statement, including reads masked by the first open transaction. Preserve pending writes under implicit, raw-BEGIN, explicit, nested and backend-owned transactions; exercise first-read explicit ownership and SQLite parity/rollback. No provider, auth runtime or live application requests. Auth/rate identity is supplied to direct real handlers; this does not certify login.

## Approved production scope

Four pure read statements across three existing files: `_list_generic_items` (only callers are keyword and collection listing), `KeywordStore.count_keyword_collections`, `KeywordStore.get_keywords_for_collection`, and `NoteStore.get_note_counts_for_keywords`. Each independently retained INTRANS in both empty/populated RED controls. Exactly four read_only flags now reuse existing initial-IDLE/depth ownership; no global execute_query default, autocommit, generic rollback, request-finalizer sweep or caller transaction settlement.

## Reassessment and acceptance

The owning boundary remains sound for known pure reads and preserves caller intent; the failed acceptance exposed an incomplete call-chain inventory. Test the complete observed page bootstrap before applying any new flags, rather than adding another named read without its adjacent counts/contents. If these controls show an additional unowned starter outside the four statements, stop and report it before widening scope. Native replacement and the full existing Flashcards/Buddy/Notes acceptance remain parent-owned.
