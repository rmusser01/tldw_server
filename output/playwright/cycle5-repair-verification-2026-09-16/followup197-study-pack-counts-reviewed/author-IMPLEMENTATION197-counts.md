# UAT197 / TASK13260.135 — Study Pack append-count repair

## Frozen minimal correction

The actual StudyPack service successfully parses fixture model output and persists cards, then `add_study_pack_cards` fails on PostgreSQL mapping access `before_row[0]`. The same failure is independently reproduced at `after_row[0]` after correcting only the first access. Both unaliased `COUNT(*)` queries return the PostgreSQL column `count`; SQLite retains its existing positional access.

Exactly the two integer conversion expressions now select `row["count"]` on PostgreSQL and `row[0]` on SQLite. SQL text, parameters, duplicate handling, existing transaction scope, null-row fallbacks and return delta are unchanged. `ast-equivalence.json` and function before/after snapshots prove this exact scope. No generic row helper, transaction flag or migration change belongs to197.

Owned paths:

- Two count reads in `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py::add_study_pack_cards`.
- New `tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py`, SHA256 `d18f3e5e983f303c615313022a08d6276e0ae786d6269e7f318fa89564236ab2`.

The combined shared ChaCha snapshot SHA256 is `40869b8d462980a7cee43ea30db6e13af64cd82da59fdd1b308dafa7c274736c`. Sidebar's disjoint193/194 schema/Character work is not attributed to this unit. `owned.patch` constructs only the two approved expressions against the retained baseline; it excludes unrelated concurrent shared-file changes. Function-level identity remains the attribution boundary if those other regions change later.

## RED → GREEN

All logs below are under `.tmp/fresh-uat-recovery-20260916/` and have `.redacted.log` suffix.

- `uat197-counts-red`: **3 PostgreSQL failures / 5 controls / 0 skips**, 10.21s. Every nonempty PostgreSQL append fails on the first count access; all four SQLite controls and empty PostgreSQL control pass.
- `uat197-after-count-red`: **1 PostgreSQL failure / 0 skips**, 2.54s, on the existing append case after only the first correction; traceback proves `after_row[0]` is independently invalid.
- `uat181-worker-197-counts-green`: **29 passed / 0 skips**, 42.23s: unchanged 8 count controls plus 21 actual worker controls. Successful PostgreSQL worker generation now reaches real persisted pack/deck/card/membership and returns usable results; repeated cached jobs pass.
- `uat181-study-pack-adjacent-green`: **48 existing tests passed / 0 skips**, 20.63s, covering worker, generation, storage and source resolution. No existing tests were edited.

The 8 new cases use actual SQLite or official isolated PostgreSQL fixture databases. They exercise empty input; first append; all-duplicate and mixed existing/new batches; independent per-pack membership; caller transaction rollback; invalid foreign-key batch rollback preserving existing members. No result rows, transactions or persistence paths are mocked.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat197-counts-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat197-after-count-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs 'tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py::test_append_counts_new_members_only_and_scopes_each_pack[postgresql]' -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat181-worker-197-counts-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py -q --tb=short
```

## Static and handoff

Scoped Ruff baseline/current 0/0; test Ruff format passes; Python AST/compile passes; diff check clean. Bandit of both touched production Python files: 0 findings / 0 errors; new tests also 0/0 excluding B101 assertions only. Receipts are shared in `.tmp/uat181-study-pack-repair-20260917/` and copied here. The large existing shared file emits nosec-comment warnings but no findings or parse errors.

Author source/tests frozen; independent review requested separately from worker181. No native StudyPack failure/acceptance or real-model-quality claim. No live databases, runtime, browser, tracker, staging or commits were changed. Root owns integration. The actual lifetime change that enables job cleanup is181, not197.
