# UAT192 / TASK13260.130 named tag columns

Two production expressions change: set_flashcard_tags uses selected row['id']; _sync_flashcard_keyword_links uses selected r['keyword_id']. PostgreSQL mapping rows and SQLite Row both support these names. SQL, transaction ownership, normalization and version updates remain unchanged. Existing synthetic PostgreSQL FTS test now returns {'id':1} rather than a tuple; all SQL assertions are retained.

Initial test harness called a nonexistent keyword-list method, yielding unrelated SQLite errors. Retained initial-harness-test.py and initial runner log are not causal RED. Corrected actual-db regression uses the public get_keywords_for_flashcard and yields8 expected PostgreSQL failures/12 passing controls/0skip (22.75s). This includes HTTP POST returning200 while silently leaving no links, HTTP PATCH/PUT returning500, and direct mutation/outer rollback failures. Missing/deleted and SQLite controls pass.

Minimal repair:20 new cases plus8 existing FTS cases =>28 passed/0skip (21.46s), official PostgreSQL fixture and real SQLite. Tests cover first/existing/empty tag replacements, normalized JSON and actual link membership, version increments, unchanged front/back, outer rollback, missing/deleted card no keyword creation. Final new test was formatting-only normalized after GREEN; independent rerun is required on those bytes.

Scoped Ruff is0 for baseline/final production and new test. Bandit scans frozen baseline/final full ChaCha with0 findings/errors; tests separately with only pytest B101 excluded,0findings/errors. No new security boundary or SQL interpolation is introduced. Baseline/final snapshots preserve192 scope while181 architecture work may separately modify the shared source.

Independent review and native tag acceptance remain pending. The live no-reload API still runs earlier committed code; no native repair claim is made.
