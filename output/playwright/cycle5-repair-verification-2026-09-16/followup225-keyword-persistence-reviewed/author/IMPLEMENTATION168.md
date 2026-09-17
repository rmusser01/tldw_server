# UAT225 local keyword survivor persistence — TASK13260.168

## Reviewable result
Seven frozen paths: three production files, two new real-backend suites, two existing migration-test maintenance files. The five-file core was independently reviewed: 62 PASS / 0 skips / 70.58s, exact source hashes stable. Its independent report is `.tmp/uat168-independent-20260917/REVIEW168-CORE.md`. The remaining two maintenance files independently passed17/0skip26.60s; final seven-file review is CLEAR. Reviewer report `.tmp/uat168-independent-20260917/final-seven/REVIEW168-FINAL.md` SHA18c3123da68db4ed22a5335d13678113d208074873c08971b1b2a6febaebb510; final reviewer manifest81b8b26fe0d1d458710098a60949d310828c61bb9218df5715a0dc5416713cea.

Local merges now retain their immediate portable survivor ID atomically with the source tombstone and all four membership moves. Restore clears it through ordinary add, organization keyword upsert and PostgreSQL flashcard tagging. Resolution follows exact IDs to the current live keyword, bounded at100 identities, with PG owner filtering at every hop and unchanged SQLite per-file/device-label semantics. Missing, plain-deleted, malformed, foreign, cyclic and excessive chains fail closed. No label guessing, new authority table, immutable receipt rewrite, Sync wire field or canonical-head change.

SQLite67→68 and PG69→70 add one nullable tombstone column and a deleted/not-self CHECK. Genuine old schemas lack the new column before upgrade. Historical active/deleted data, IDs, versions, timestamps, indexes, triggers and RLS metadata remain unchanged; old tombstones become NULL without inferred backfill. Failure rolls back column and version; reopen is idempotent.

## Owned implementation
- `ChaChaNotes_DB.py`: two schema constants, two next migrations, existing dispatch/initializers, direct PG flashcard keyword restore assignment.
- `keyword_store.py`: add/restore metadata, local merge source CAS target, existing parent locks ordered by ID, and `resolve_merge_survivor(sync_id, *, conn=None, for_update=False)`.
- `organization_sync_store.py`: only keyword active upsert clears local redirect. Existing canonical serialization remains explicit and unchanged.
- Locked resolver requires caller connection, discovers chain, acquires sorted row locks and revalidates full observed rows. An intervening merge/delete/restore/rename raises existing `ConflictError`; it does not commit or retry caller work. Standalone reads use the existing read-only helper.
- New tests cover real writes, reopen, owner/device behavior, all three restores, chains/current live identity/rename, plain delete/idempotence/remerge, foreign hop, invalid/cyclic/over-limit chains, four link families and injected failure rollback, caller rollback, restricted NOSUPERUSER/NOBYPASSRLS service writes, stable locks and race revalidation.

## Verification and attribution
All test commands use the official runner, requiredPG=true, Docker autostart disabled, the existing owned cluster55475 and official disposable fixtures. No live app/browser/config/model action.

| Receipt label in `.tmp/fresh-uat-recovery-20260916` | Outcome |
|---|---|
| `uat168-persistence-red` | Actual merge+reopen2FAIL /4controlsPASS, 0skip |
| `uat168-resolver-red` | 36FAIL/8PASS; two failures were incorrect flashcard fixture payloads; all retained |
| `uat168-migration-red` | 8FAIL, absent new column/migrators/constraint |
| `uat168-lock-order-red` | Opposing actual PG merges1FAIL: success plus backend deadlock error |
| `uat168-first-green` | 48PASS/5FAIL:2 diagnostic assumptions,2 flashcard setup errors,1 run importing old lock order |
| `uat168-expanded-green` | Interrupted invalid test gate228.59s; no product finding |
| `uat168-race-gate-green` | Corrected5PASS/0skip8.24s |
| `uat168-complete-green` | 56PASS/2FAIL: tests assumed driver SQLSTATE that production intentionally erases |
| `uat168-constraint-green` | Actual rejected writes + exact validated catalog4PASS/0skip5.38s |
| `uat168-frozen-core-green` | Final62PASS/0skip71.37s |
| `uat168-adjacent-first` | 309PASS/4FAIL/0skip371.55s; only stale current-head expectations |
| `uat168-head-maintenance-green` | All17 affected migration testsPASS/0skip28.95s |

The invalid race gate inserted an unrelated pending keyword before synchronously awaiting a same-owner mutation. Read-only disposable-database metadata showed the worker waiting on the caller transaction. Only owned pytest2571 was gracefully interrupted; fixture teardown completed. Removed that interfering write from this race and retained independent caller rollback controls; worker lock/statement timeouts bound the final test. Original opposing-merge RED is separate and remains the basis for stable parent ordering. The raw gate snapshot and interrupted receipt are retained.

Constraint tests use a real failing database update and inspect the exact validated CHECK catalog: the PostgreSQL backend intentionally removes original driver message, cause and SQLSTATE. No production error relaxation was introduced.

## Narrow existing test maintenance
Only current-head assumptions changed. Character migration still runs genuine capped67→68 and all original row/constraint guards. Its separate current SQLite-head assertion is68. Deck migration now explicitly executes capped68→69 and asserts data+owner/name uniqueness before reopening the current head; normal reopen expectations use the PostgreSQL target. Exact source68 rejection/rollback controls remain68. All original data/catalog assertions retained.

The separate v55/v57 historical fixture suite fails3/39 on exact pre168 modules too: latest-schema relabeling collides with attachment/task guards. Root associated UAT228/TASK13260.169 before its two-file fixture repair. These are not silently skipped or counted as passed; original source and failures remain in this packet. Their fixes are excluded from168 owned.patch/manifest.

## Static checks and boundaries
Production Ruff0 baseline/current; Bandit0 findings/errors baseline/current. New tests and two maintained test files Ruff0 and Bandit0 findings/errors (B101 assertion checks excluded only). Test SQL suppression comments refer only to closed backend table mappings/literal relation inventories; values remain bound. All seven files compile. AST inventory proves only the named ten methods plus two constants and validation imports changed in production. Existing large production files are not broadly reformatted.

Source/test manifest: `owned-manifest.json` SHA414c940622184bb028bb0e347be877bc3c1e8c87448251a97a7c5dc99c30e2ae; exact copies under `review-snapshot`, attributable `owned.patch`, original three-file source in `baseline`.

## Explicit remaining scope
This persistence core does not closeUAT225. Retry owns StageB local publication/decision/Jobs/enrollment integration. Its actual merge-before-acceptance and merge-before-publication4RED receipt is retained separately (`uat225-merge-consumer-red`); subsequent consumer acceptance/races and native acceptance require their own evidence. Canonical Sync decisions still resolve immutable heads. No global nonHTTP/transaction lifecycle or rawSQL tenant-isolation guarantee is added.
