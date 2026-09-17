# TASK13260.168 frozen persistence core

Exactly three production paths and two new suites are frozen. Adjacent existing test fixture maintenance is still being attributed separately; it will not change these five bytes without reviewer coordination.

## Contract
Nullable portable immediate survivor on keyword tombstones; fresh/historical NULL, delete/not-self CHECK; SQLite67→68 and PostgreSQL69→70. Existing row data/indexes/triggers/RLS remain. Merge source CAS and all four membership families share the existing transaction. Ordinary add, canonical keyword restore and PG flashcard tag restore clear the redirect. Owner-aware resolver follows at most100 UUIDv4 identities, current live rows first, no label guessing; invalid/deleted/foreign/cycle fails closed. Locked resolver requires caller connection, sorts discovered IDs, locks and compares full row snapshots; changed chain raises ConflictError. Existing local merge parent locks use that same order. Canonical Sync heads/payloads unchanged.

## Independent command
```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat168-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/DB_Management/test_local_keyword_merge_survivor.py tldw_Server_API/tests/DB_Management/test_local_keyword_survivor_migration.py -q --tb=short
```
Author62PASS0skip71.37s, uat168-frozen-core-green.redacted.log. Production Ruff0/Bandit0; tests Ruff0/Bandit0 excludingB101. AST-owned-scope lists exactly changed methods.

## Causal attribution
- Actual merge/reopen2FAIL/4controls on original source.
- Resolver expansion36FAIL/8controls contains2 invalid flashcard fixture payloads; corrected to add_flashcard then actual set_flashcard_tags. Other missing resolver/column outcomes retained.
- Genuine migration8RED before implementation.
- Opposing real PG merges1RED: one success plus backend deadlock error, not expected version ConflictError. Stable parent ordering control nowGREEN.
- First candidate48PASS/5FAIL:2 wrong constraint exception assertions,2 flashcard fixture mistakes, original lock ordering was imported before patch. Next56PASS/2FAIL: backend intentionally strips original SQLSTATE/cause; catalog-bound rejection replaces that incorrect diagnostic expectation. Corrected4constraint controlsPASS.
- An invalid race gate held an unrelated keyword write while awaiting same-owner worker completion. Interrupted owned pytest gracefully; receipt retained, not product finding. Corrected5bounded PG racesPASS8.24s, with test-local worker timeouts.
- Four-family links/failure rollback, all three restores, same-ID remerge, historical migration rollback, current chain/rename, foreign hop, SQLite device labels, restricted-role real writes and sorted locks are covered. StageB actual decision/publication lifecycle belongs to Retry; its4consumerRED receipt is separate. No native or whole lifecycle completion claim.
