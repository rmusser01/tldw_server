# UAT210 — PostgreSQL Notes graph read compatibility

TASK13260.148. Two paths frozen for parent independent review. Root approved the exact five graph methods, including backend-mapped keyword joins, before production changes. UAT209 owner predicates remain separate; no209 production change is included in this snapshot.

## Cause and minimal repair

PostgreSQL result rows are mappings, but five legacy graph methods consumed numeric offsets. All-ID/tag-ID/source-ID queries now read `id`; count_user_notes uses an explicit `cnt` alias and named result access; counts_per_tag reads `keyword_id` and `cnt`. Existing row/ordering/deletion/limit behavior stays intact. Two tag/count queries also used SQLite's literal `keywords` table instead of PostgreSQL's existing `chacha_keywords` mapping. They now reuse `_map_table_for_backend("keywords")` in the local query only. No generic translator, ownership, transaction, schema, or RLS change.

An AST comparison in ast-scope.json proves only those five methods differ. The patch and exact source/test snapshots are in owned.patch and review-snapshot/. Source SHA b92ee2e69786957822bb7a66365fcdbec7169c93b13f8d464f4b473d195b34db; new test SHA1dac2aa5601c81a271831b5fd2095b47c0eec7f3e4bb6cc6f6ebd3c4e4f96b68.

## Causal evidence and verification

- Original private single-owner real-backend20 controls: **8 PostgreSQL failures /12 controls pass /zero skips**,23.88s. Tag/count queries fail earlier at their unmapped table; ID/source/count reads demonstrate KeyError(0).
- Private source-only table-mapping qualification (no repository mutation): **2 expected populated mapping-row failures /6 pass /12 deliberately filtered**,9.28s. The empty tag/count cases pass once only their existing table mapping is used; populated results then expose KeyError(0). First private import-path error is preserved as harness evidence and did not collect cases.
- Permanent exact20 controls before production: **8 failures /12 pass /zero skips**,26.58s (`uat210-permanent-red.redacted.log`). Tests use a single owner so209 cross-owner leaks do not confound this compatibility contract.
- Final formatted source/tests plus existing graph suite: **46 passed /zero skips /4 warnings**,33.17s (`uat210-final-green.redacted.log`). Covers empty/populated results, both deletion modes, canonical source/tag filters, and original SQLite graph behaviors. The initial pre-format GREEN46 is retained separately.
- Ruff production baseline0/current0, new test0. Bandit production0findings/0errors; new test0findings/0errors with test assertion B101 only excluded. Static JSON receipts remain in the diagnosis packet; summary here. No broad formatting of production was done.

Exact final/independent command from repository root (official fixture runner; network escalation for the owned local test cluster):

```sh
source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=uat210-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_note_graph_query_backends.py tldw_Server_API/tests/Notes_Graph/unit/test_graph_db_queries.py -q --tb=short
```

New permanent test: `tldw_Server_API/tests/DB_Management/test_note_graph_query_backends.py`. Existing SQLite graph test unchanged. Official fixtures exclusively created/dropped isolated test databases; no native runtime/database/browser/config/provider action, task/tracker edit, stage or commit by this author. Parent owns native acceptance and integration. This fixes the exact graph read compatibility unit; it is not a Notes owner-isolation or whole graph acceptance claim.
