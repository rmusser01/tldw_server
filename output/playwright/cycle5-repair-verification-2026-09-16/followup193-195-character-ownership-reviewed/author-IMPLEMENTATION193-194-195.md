# UAT193 / UAT194 / UAT195 repair packet

Associated tasks: 13260.131, 13260.132, 13260.133. The separately attributed UAT183 historical-fixture correction belongs to 13260.120. Parent owns integration, task updates and native acceptance.

## Result and limits

Frozen candidate: **96 focused tests passed, zero skipped** (144.10s), plus **146 adjacent tests passed, zero skipped** (220.02s), and the five separately reviewed OSCE controls passed (4.03s). The official DB_Management PostgreSQL fixtures ran with PostgreSQL required and Docker autostart disabled. SQLite controls ran alongside them. No live UAT database, browser, service, provider, runtime configuration or model was changed by this agent.

The original native193 case used a configured PostgreSQL role with both superuser and RLS-bypass privileges. Character194 exposure/mutation was reproduced under that same kind of privileged fixture role. The existing forced character RLS rejects foreign SQL reads/updates under a verified ordinary NOSUPERUSER/NOBYPASSRLS role. **No ordinary enforced-RLS role leakage is claimed.** The approved repair adds local ownership checks because startup accepts the configured service role and the Notes owner contract already requires this protection under privileged roles. See ROLE-CONTRACT.md.

The new cold-dependency regression executes real uncached get_chacha_db_for_user initialization, builtin/default seeding, cache publication/health, actual character list and POST chat factory for two users. Authentication principals and backend configuration are fixture-controlled; this is not a browser login, token-security or deployment-role acceptance test. A nonmutating baseline-store replay gives one PostgreSQL failure and one SQLite pass. Current source passes both.

Native193 default selection/chat acceptance remains the parent's gate. UAT195 has a deterministic database regression, not a claimed native failure. Independent review is in progress. The reviewer identified one existing OSCE fresh-head assertion outside the146 list. Its sole hardcoded67 expectation is now backend-specific; all five controls pass, including unchanged historical66→67 and live table/data checks.

## Production changes

1. CharacterStore applies parameter-bound owner predicates only on PostgreSQL. Card ID/batch/name/list/query/setup/search/tag reads and card update/delete/restore preflight, idempotency and final SQL share that boundary. Correlated conversation filters/count ordering use the same owner. SQLite's per-file sync-device client_id behavior stays intact.
2. Exemplar reads and mutations derive owner from the linked character. The INSERT also checks ownership in SQL, so a stale parent preflight cannot authorize it. Existing child/deleted-parent behavior remains; no redundant owner column or global query-helper policy was added.
3. PostgreSQL schema67→68 replaces the catalog-identified global name constraint with UNIQUE(client_id,name). The transaction preserves rows, IDs, content and tombstones, verifies catalog and version, rejects conflicting destination constraints or remaining standalone global-name uniqueness, and rolls back on injected failure. SQLite head remains67. Earlier migrations and factory ownership are unchanged. Deps required no changes.
4. UAT195 adds CAST(? AS TEXT) at the existing nullable emotion/scenario predicates. The actual driver error was IndeterminateDatatype42P18, parameter$4, for both omitted and supplied filters. Listing the owned exemplar is a positive control; the earlier hypothesis that explicit search filters would pass was disproved and retained. Search, filtering, ordering, totals and pagination remain covered.

Shared-file attribution: ChaChaNotes_DB.py includes Source013's separate UAT197 two named-count changes. `chacha-attribution-ast.json` proves that compared with committed07e0abf1c4, only my PG schema constant/new migration/initializer and the separate add_study_pack_cards method changed. `owned.patch` excludes the197 method delta; the snapshot/hash intentionally records actual combined bytes. Commit053ff74116 subsequently integrated197 without changing these working bytes.

## Tests and causal evidence

| Evidence | Outcome |
| --- | --- |
| Initial shared-owner RED | 8 failures /16 controls pass /0skip |
| Expanded ownership RED v2 | 20 ownership/uniqueness failures, plus separate search error /34 controls pass; first tuple-return assertion mistake retained and corrected |
| Initial PG schema contracts | 4 expected failures /2 controls pass /0skip, including passing restricted-role and same-owner concurrency controls |
| Atomic/catalog controls before migration | 3 failures; no upgrade, rollback checkpoint or conflict rejection yet |
| UAT195 owned optional-filter RED | 11 PG failures /11 SQLite passes /0skip |
| Additional global-name-index guard RED | 1 failure; migration incorrectly completed while extra global index remained; guarded now |
| Actual cold dependency baseline replay | PostgreSQL1 failure /SQLite1 pass |
| Frozen focused suite | 96 passed /0skip |
| Adjacent baseline replay | 3 owner-argument controls pass; historical21 fixture fails the same v59 registry guard |
| Frozen adjacent suite | 146 passed /0skip |

Harness corrections are explicit: the first migration fixture prematurely closed its shared backend; one owned NOLOGIN role cleanup failure was recovered through a constrained official-fixture cleanup and retained. The initial86-case GREEN attempt had83 passes and3 harness failures: the constructor wraps the expected migration errors, and the reopen test closed the shared pool within its loop. Corrected tests assert the exact inner cause and retain instances until verification ends. No production guard was weakened to accommodate any harness issue.

The three existing SQL-argument assertions now include the bound owner (and the fake PG FTS database has a client_id). Their pre-change baseline tests pass. UAT183 is in `uat183-test-only.patch`: the test creates real schema21 through V4 and real registered4→21 migrations, verifies no future exemplar/attachment tables, seeds retained data, executes the exact21→22 step, then restores normal initialization and checks current SQLite67 plus data preservation. It does not relabel a head database or weaken registry-collision guards.

## Reproduction

From repository root, activate the project virtual environment before each invocation. The official runner resolves private fixture credentials without printing them.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=review-character-focused node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
 tldw_Server_API/tests/DB_Management/test_character_shared_owner_contract.py \
 tldw_Server_API/tests/DB_Management/test_character_owner_name_migration_postgres.py \
 tldw_Server_API/tests/DB_Management/test_character_exemplar_search_backends.py -q --tb=short
```

Adjacent files are the six paths in `uat193-195-adjacent-final-command.json`: CharacterStore, exemplar DB, tag search, PostgreSQL FTS, actual exemplar API and immutable behavior snapshot tests. Both exact commands and unmodified redacted receipts are retained in this packet.

## Static checks and review boundaries

- Bandit: both production files0 findings/errors. Seven touched test files0 findings/errors with B101 excluded for pytest assertions. Initial concatenation findings were reviewed: ownership fragments are fixed SQL and all values are bound; local annotations document remaining static-fragment false positives. No global suppression.
- Ruff: same17 baseline diagnostics (3 CharacterStore,12 existing PG FTS test,2 existing store test); zero added diagnostics. New tests, migration source and corrected historical fixture are clean.
- All nine frozen Python files compile; whitespace diff check passes.
- `final-owned-manifest.json` and `final-review-snapshot/` contain exact frozen bytes. `verification-artifacts.json` records original/retained receipt hashes. `secret-scan-final.json` records an in-memory comparison against known owned PostgreSQL password/DSN and vision key without emitting values.
- No claim of a clean full repository, whole-domain authorization audit, all migration tests or native recovery. Existing unrelated stores and policies are outside this unit.

## Final OSCE fixture follow-up (UAT183)

Reviewer causal run: one fresh-head assertion failure (68 versus67), four controls pass. The only edit changes that assertion to CharactersRAGDB._POSTGRES_SCHEMA_VERSION; exact historical66→67 cap and table/data/owner assertions stay unchanged. Author same-file run: five passed, zero skipped. This correction is appended only to uat183-test-only.patch. The prior independent183162+4 evidence is not replaced.

Final nine-path manifest SHA256: `708d1bfa83bb10da5c174681d2a770df65f71fe848d3b5bb1cddaa1b72560145`. All nine bytes match that manifest. The manifest's status records its creation time while the five-test OSCE run was still in flight; this report and retained receipt provide its completed result. No production source changed after the96/146 freeze.
