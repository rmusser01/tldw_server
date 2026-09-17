# Reopened UAT210 / TASK13260.148 — legacy tag-edge keyword JOIN

The original five-method PostgreSQL graph compatibility unit is committed47e23bd5f3. A later populated/empty-edge legacy `get_note_tag_edges` call still joined literal `keywords`; PostgreSQL uses the existing mapped `chacha_keywords` table. The bounded residual uses `_map_table_for_backend("keywords")` for this one JOIN. It does not change the generic mapper or the previously repaired named row accesses.

Permanent controls add four cases (SQLite/PostgreSQL × empty/populated edges) to `test_note_graph_query_backends.py`. Each invokes the query with a real note, so the empty-edge case does not take the no-input early return. `uat210-tag-edge-residual-red`:2 PostgreSQL failures/2 SQLite pass,0skip6.60s. `uat210-tag-edge-mapping-green`:4pass,0skip9.42s. Final frozen combined suite includes all24 graph cases:202pass/0skip/2warnings284.51s (`uat209-210-217-frozen-green`), all six integrated hashes unchanged.

`owned.patch` contains only the mapped-table variable/JOIN and these four tests. The later owner predicates in this same function are UAT209 and are excluded from this compatibility patch. `method.before.txt` and `method.mapping-only.txt` preserve the intermediate boundary. Shared final bytes/hashes live in209's review snapshot/manifest. No browser/native claim; root owns acceptance and integration.
