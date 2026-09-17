# Independent review — UAT193 / 194 / 195 and UAT183 follow-up

## Verdict

**Clear for integration within the approved scope.** No unresolved production finding. Independent mandatory-PostgreSQL verification: **242 passed / 0 skipped / 5 warnings / 340.67s**, covering the frozen 96 new and 146 adjacent controls. The additional OSCE file passes **5 / 0 skips / 4 warnings / 3.85s** after the approved test-only correction. All nine final source/test files still match their author snapshots.

One adjacent test finding was reported before correction: normal head initialization now returns PostgreSQL68, but the existing OSCE fresh-schema test asserted literal67. The unchanged file reproduced **1 failure / 4 passes / 0 skips**, 4.47s. Under reopened183, the author changed only that current-head expectation to `_POSTGRES_SCHEMA_VERSION`; the exact capped66→67 routing and OSCE schema/data controls remain. Its original RED and independent GREEN are retained. No production guard or historical migration expectation was weakened.

## Frozen identities and attribution

Author packet: `.tmp/uat193-diagnosis-20260917/`.

- Final nine-file manifest SHA256: `708d1bfa83bb10da5c174681d2a770df65f71fe848d3b5bb1cddaa1b72560145`.
- CharacterStore SHA256: `93ed045b9561eb119d1ce279e643c8ebf819989f54e2e40135cb7be8a7fdc952`.
- Combined ChaCha SHA256: `b1c8b144c2ce3914c10652d817ad93550733820f1f624455e60016195db1eb6e`.
- Corrected OSCE test SHA256: `612866fd431f8b1dd26435b2aa98c9a1419ba1b81df1873dbeeed4839eaa98aa`.

`source-before.json`, `source-midrun.json`, `osce-before.json` and `source-after.json` pin the execution windows. The original eight files stayed unchanged; OSCE was added as the ninth separately frozen test file. `author-final-manifest.json` records all paths and hashes. The author snapshot byte comparison passed for all nine.

My previously authored UAT197 count repair is **excluded from this independent review**. Its function AST is unchanged and the author's `owned.patch` contains none of that method delta. Reviewed shared-file changes are only the PostgreSQL head, new migration and initializer registration. The separately committed StudyPack owner is also outside this review.

## Source and regression assessment

**Ownership:** PostgreSQL owner fragments use fixed SQL and bound `client_id` values. They apply before card pagination/counts and to ID/name/batch/setup/search/tag paths, including requested deleted rows. Card mutation preflights, final UPDATE predicates, conflict reads and idempotent paths carry the same owner boundary. Correlated conversation filters/count ordering use the card owner. SQLite continues to treat `client_id` as a sync-device marker within a per-user file; its positive shared-file control remains.

**Children and transactions:** exemplar ownership comes from the parent via EXISTS. INSERT has its own owned/active-parent condition, so the stale-parent-preflight regression tests actual write authorization. Child reads/update/delete include owner predicates; existing deleted-parent/child behavior is preserved. Actual owned CRUD, foreign rejection, deleted visibility, rollback of nested card/exemplar writes, and no partial conversation after rejected chat creation are covered. No generic SQL rewriting, transaction policy, factory relaxation or redundant child owner field was introduced.

**Actual dependency/factory boundary:** the cold test begins with an empty real cache and invokes real uncached dependency initialization, default maintenance, list routes and chat creation for two owners. The conversation factory remains unchanged and checks ownership. The immutable snapshot regression creates through the real route, changes the owned character afterward and compares persisted canonical snapshot bytes. Backend configuration and auth principals are fixtures; this is meaningful dependency/route coverage, not a full AuthNZ or native-browser acceptance.

**Migration:** actual v67 construction through registered migrations precedes v68 reopen. The new step runs in the existing schema transaction and requires v67. It validates the single-column name constraint by relation/attribute identity, adds owner/name uniqueness, quotes only the validated old identifier and drops without CASCADE. Final verification catches remaining global name uniqueness, including a standalone index, before version advancement. Tests retain all rows/IDs/content/defaults/tombstones, same-owner reservation, unrelated uniqueness, same/different-owner concurrent bootstrap and repeat initialization. Injected failure after catalog replacement and unexpected catalog controls prove rollback of catalog, rows and version. PostgreSQL advances to68; SQLite remains67.

**Typed search:** nullable emotion/scenario parameters are explicitly TEXT in each existing branch. Bound query/filter/owner order matches the SQL. Real PostgreSQL and SQLite tests exercise omitted, supplied and empty filters for browse and text search, deleted rows, rhetorical filtering, total-before-pagination and no-match results. This addresses the actual42P18 failure without changing the filter contract.

**Historical183 fixture:** the SQLite test constructs real v21 from V4 plus registered steps, verifies future exemplar/attachment tables are absent, and seeds historical columns. It checks exact21→22 and retained data before normal current-head reopening. All prior table assertions remain. The three separate pre-existing mock SQL-argument expectations correctly add the owner parameter while retaining their backend/FTS assertions.

## Role and acceptance limits

The original194 exposure/mutation was under a configured superuser/BYPASSRLS role accepted by current startup. The approved store predicates provide defense in depth for that accepted configuration. The unchanged forced character RLS rejects foreign SQL reads/updates under a verified NOSUPERUSER/NOBYPASSRLS fixture role. **This review does not claim ordinary enforced-RLS role leakage**, or native restricted-role deployment qualification. Name uniqueness is independently relevant across tenants regardless of RLS visibility. ROLE-CONTRACT.md states these distinctions accurately.

No model, browser, live runtime, live profile or live database was used. All PostgreSQL work used the official isolated DB_Management fixtures through the existing required-PG runner. Native default-selection/chat acceptance remains the parent's gate. This is not an audit of every shared-PG store or every historical migration test.

## Verification and security

Exact independent commands are retained in `uat193-195-independent-command.json` and `uat183-osce-independent-green-command.json`, with corresponding redacted logs. Both use `.tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs`, PostgreSQL required, owned cluster55475 and Docker autostart disabled. The combined command runs the three new Character backend files plus existing CharacterStore, exemplar DB/tag search/PG FTS, exemplar API and immutable behavior snapshot suites.

Ruff replay via actual logical filenames: **17 baseline / 17 current / 0 added / 0 removed**; the added OSCE path has0. Both production files Bandit **0 findings / 0 errors**. Seven touched test paths Bandit **0 / 0** excluding B101 assertions only. Nine files parse/compile without execution, and scoped diff whitespace check passes. Existing nosec-comment warnings in the large shared source are not security findings or parse errors. The review identified no new credential logging, dynamic unbound values, permission bypass or RLS relaxation.

Only this private review packet was written by the reviewer. No production/test/source/task/tracker/runtime/browser/git mutation was performed.
