# Independent residual UAT181 review

TASK13260.118. **Clear for parent integration/native reacceptance; no source changes requested.**

## Frozen scope verified
All four current files match the author's manifest and exact review snapshots. Independent source reconstruction and AST comparison prove the only production changes are six `read_only=True` arguments:

| Method | SELECTs inspected |
| --- | --- |
| `ensure_character_tables_ready` | Normal readiness and post-recovery verification, each `SELECT 1 FROM character_cards LIMIT 1`. |
| `get_character_card_by_name` | Normal and post-recovery lookup using the same parameterized name/deleted filter. |
| `list_persona_profiles` | Parameterized owner/deleted/active filter and bounded list ordering/pagination. |
| `list_persona_buddies` | Parameterized owner/persona IDs and profile visibility join. |

Every flagged statement is a side-effect-free SELECT. No SQL, filters, defaults, writes, explicit commit/rollback, generic helper, RLS or dependency scheduling change is present. PersonaStateStore forwards `execute_query` to the existing DB helper. The helper starts an owned transaction only when the raw PostgreSQL connection is IDLE and both ChaCha and backend transaction depths are zero. Existing transactions, including implicit pending writes, remain caller-owned; SQLite remains unchanged.

## Independent tests
Exact requested official-fixture command:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-residual-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_shell_read_lifecycle.py -q --tb=short
```

**31 passed,0 skipped,4 warnings,52.44s**:29 actual PostgreSQL cases and2 SQLite controls. The existing runner requires PostgreSQL and uses official disposable databases on the owned test cluster. No live application database, browser, runtime or model was operated.

The test meaning is adequate for the reported regression. Seeded existing default-character and persona identities avoid accidentally substituting creation for reads. The real default-maintenance function and persona endpoint (including buddy projection) run on the same pinned connection before actual Buddy/Notes reads. A second real CharactersRAGDB constructor runs production initialization against the same database while the first remains alive; only lock budgets are shortened. Separate no-predecessor controls discriminate the residual starter from downstream reads. The real asynchronous default-maintenance executor is also exercised, with executor connection cleanup only afterward.

All four isolated starters reach IDLE. Twelve explicit/implicit/nested caller-write outcomes preserve uncommitted state until caller commit/rollback; six first-read explicit-scope controls preserve scope ownership. The SQLite chains preserve reads and caller rollback. These are behavioral lock/visibility controls, not mirrored flag assertions. Author RED9 failures/22 controls is retained separately; the broader184-case author GREEN was not independently rerun.

## Native corroboration and limits
The retained metadata at2026-09-17T01:19:58.855104+00:00 shows session7234 idle in transaction with a granted persona_profiles AccessShareLock, and session12217 active with an ungranted AccessExclusiveLock on that relation. Only those metadata fields are copied into `native-metadata-excerpt.json`; no queries, user content or credentials are included. Process ownership/overlap comes from the parent's native receipts, not this review's own runtime observation.

The31 tests cover the normal retained-data paths. The two recovery SELECTs were source-reviewed as equivalent pure reads; these tests do not force PostgreSQL missing-table recovery. Recovery's existing schema initialization/connection-close behavior is unchanged, so the whole recovery method is not claimed to be transaction-neutral. The older Buddy-only session's exact first statement remains unknown. This bounded repair does not certify all repository read starters or complete native replacement acceptance.

## Static evidence
Independent Bandit on the three Python production files:0 findings/0 errors (existing nosec-comment warnings remain). Independent Ruff reports the same five semantic baseline diagnostics,0 added; the author's new-test lint/format and compilation checks are retained in its packet. This is not a claim of globally clean Ruff. Hash, six-site AST and baseline comparisons are in `source-check.json` and `static-check.json`.

This reviewer wrote only private evidence under this folder. No production/test/task/tracker edits, staging, commits, browser or runtime actions.
