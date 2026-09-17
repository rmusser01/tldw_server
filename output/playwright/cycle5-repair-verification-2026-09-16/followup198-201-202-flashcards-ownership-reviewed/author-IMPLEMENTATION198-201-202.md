# UAT198 / UAT201 / UAT202 — frozen author handoff

Tasks: TASK13260.136, TASK13260.139, TASK13260.140. Source and tests are frozen for independent review; native acceptance is pending. This packet does not claim a clean whole-worktree snapshot or whole-product tenant isolation.

## Result and scope

The selected PostgreSQL Flashcards repository now enforces its `client_id` owner on deck/card reads, mutations, parents/workspaces, assets, tags, rating, and review-session lifecycle. Predicates are parameter-bound and applied before pagination, aggregation, or mutation. SQLite keeps its existing per-user-file boundary and cross-device sync-client semantics. The HTTP router, generic SQL execution, caller transaction ownership, authentication, workspace sharing policy, and unrelated Study storage were not changed.

UAT201 advances only PostgreSQL from schema 68 to 69, replacing the validated global deck-name constraint with `(client_id, name)`. IDs, data, tombstones, unrelated constraints, same-owner name conflicts, and rollback are preserved. SQLite remains version 67. UAT202 changes deleted-deck restoration to named `id`/`version` fields and retains the same deck identity; its lookup and update also receive the UAT198 owner boundary.

Author source: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, SHA256 `6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7`.

Eight-path frozen manifest: `owned-manifest.json`, SHA256 `c87fe3776978277d64ebc5954657597464ac3f6c5f23c75c02429e8210d0d35b`. Each file has a byte-identical `review-snapshot/` copy. `owned.patch` includes the three new tests and four bounded existing-test updates. `changed-methods.json` lists 48 changed/added methods, including local helpers and schema initialization; the approved 44-method inventory describes the reviewed resource boundary rather than the number of edited functions.

## Native and role evidence

The retained native Bob identity events (`id=3`) at 04:30:02.099/.340/.598 UTC precede successful private Alice (`client_id=2`) deck/card responses. Earlier Alice events in the same file are positive controls. Native used the accepted privileged/BYPASSRLS configuration; no native foreign mutation was attempted.

The official isolated two-owner PostgreSQL fixture independently reproduced reads and writes. The catalog also proved decks/flashcards have no RLS policies, and a verified NOSUPERUSER/NOBYPASSRLS SQL role could read foreign rows. The approved repair is **application persistence isolation**, not raw SQL isolation. The original raw-SQL negative probe is retained in `raw-sql-policy-diagnostic.py` and the baseline test snapshot. No permanent test is skipped or deselected to hide that limit.

The final restricted-role control executes actual repository calls under `SET LOCAL ROLE`, verifies both privilege flags false, checks private list/detail isolation, performs an owned edit, rejects a foreign edit, and proves Alice's raw row unchanged. It grants the tables/sequences needed by actual persistence triggers. This is separate from native authentication, not a claimed restricted-role browser run.

## Important behavior preserved

- The owner comes from the selected DB, not the ambient recipient context. A real `SharedWorkspaceAccessService` handoff test loads Alice's actual content DB for a permitted recipient, reads cards there, and verifies denied membership prevents another owner load. Only the service's AuthNZ membership/user-return seams are stubbed; this is not a real AuthNZ login or membership-SQL test.
- Workspace IDs, public/team/org visibility labels, and deck-share records do not themselves authorize ordinary private routes. Explicit owner-side share administration and lookup remain supported.
- Owned cards linked to a soft-deleted own deck remain directly readable. Creation/reparenting require live owned parents. Malformed cross-owner deck/keyword/review links cannot disclose foreign metadata or distort owned analytics.
- Assets include metadata and bytes, unattached uploads, reconciliation, cleanup, and foreign attachment rejection.
- Identical `due:global` review scopes remain independent for two owners. Session reads, completion, stale cleanup, aggregate repair, reviewed rows, and ratings remain owner-local.
- Same tag text resolves within the current owner via a Flashcards-local insert/select path using existing `(client_id, LOWER(keyword))` uniqueness. Generic KeywordStore helpers were not rewritten.

## Causal evidence and review corrections

| Receipt label | Result | Meaning |
| --- | --- | --- |
| `uat198-owner-boundary-red` | 14 failed / 18 passed / 0 skipped | Private reads, foreign mutations, workspace boundary, and separate raw-SQL policy diagnosis |
| `uat198-resource-edges-red` | 7 failed / 7 passed / 0 skipped | Assets, rating/session/parent plus separately associated deck-name and restore failures |
| `uat198-tag-edge-corrected-red` | 1 PostgreSQL failed / 1 SQLite passed | Same-tag actual card PATCH/read-tags selects foreign owner metadata; earlier wrong-envelope harness attempt retained |
| `uat198-closure-red` | 21 failed / 23 passed / 0 skipped | Derived reads, grants, parents, sessions, cleanup, malformed links, restricted-role calls |
| `uat201-migration-red` | 6 failed / 1 passed / 0 skipped | Historical68 upgrade/catalog/data/rollback/name behavior before migration |
| `uat198-final-edges-baseline-red` | 6 PostgreSQL failed / 6 SQLite passed | Nonmutating baseline-method replay of additional resource boundaries |
| `uat198-malformed-child-red` | 2 PostgreSQL failed / 2 SQLite passed | Foreign tag cannot qualify rating scope; foreign review cannot alter owned analytics |
| `uat198-ancestor-review-red` | 2 PostgreSQL failed / 2 SQLite passed | New/reparented deck accepted owned parent whose ancestor belongs to another owner |
| `uat198-explicit-queue-review-red` | 3 PostgreSQL failed / 3 SQLite passed | Foreign malformed child selected before LIMIT1 hides the owner's valid learning/review/new card |
| `uat198-final-green` | 112 passed / 0 skipped | Earlier complete author suite, before the final review additions |
| `uat198-review-corrections-green` | 97 passed / 0 skipped | Final reviewer corrections, positives, and all adjacent compatibility corrections |

Independent reviewer Retry identified the last two source defects before final release. Both were reproduced permanently before editing production. PostgreSQL ancestry traversal now rejects an absent/foreign ancestor, including creation; SQLite behavior is unchanged. All three explicit-deck queue branches apply owner filtering before LIMIT1. Positive controls retain valid ancestry and learning→review→new priority.

The first migration GREEN attempt had six passes and one fixture failure from closing a shared pool before a final assertion; moving that assertion before close corrected the harness, with the failure retained. No migration guard was removed.

## Existing-test integration, separately attributed

The first identical 322-case adjacent run returned 313 passed / 9 failed / 0 skipped. Its assertions are retained in `adjacent-first-failure-assertions.txt`.

1. **UAT201 integration:** Three Character migration assertions used ordinary current-head constructors but expected literal68. The historical67→68 test now uses a capped real-v68 subclass and still asserts exactly68 with unchanged catalog/data checks. Unrestricted reopen/retry expects `_POSTGRES_SCHEMA_VERSION` (69). Historical failure rollback remains exactly67.
2. **UAT198 integration:** Four binary row-conversion cases used an `object.__new__` fake with no backend/owner metadata. It now explicitly supplies PostgreSQL backend type and a synthetic owner; bytes/memoryview/empty/null assertions are unchanged. The deck-locking fake similarly supplies a synthetic owner and retains all lock-count/SQL assertions. Its preexisting import-order warning was sorted.
3. **Baseline fixture defect:** The session rollback helper changed the stale timestamp via public `execute_query(commit=False)`, read its own uncommitted seed, then rolled that seed back with the tested outer transaction. `uat198-session-rollback-baseline` nonmutating replay of the original production methods reproduced the exact failure. Only the setup UPDATE now uses `commit=True` before the separate rollback-under-test; no production transaction logic or assertion was weakened.

## Final verification

**Final permanent release: 126 passed, zero failed/skipped/deselected, 195.90s. Final identical adjacent release: 322 passed, zero failed/skipped/deselected, 270.00s.** Both used official required-PostgreSQL fixtures plus SQLite controls. The separate 97-case correction run also passed with zero skips. These overlap and are not presented as 545 unique tests. Four existing warnings per run are retained in the receipts; they are not test skips or failures. All eight frozen source/test hashes remained unchanged through these runs. Independent review and native acceptance are separate gates.

Final static checks on all eight owned files: Ruff 0 findings; Python compile 8/8; `git diff --check` 0. Bandit: production 0 findings/errors; all seven tests 0 findings/errors with only B101 excluded for normal pytest assertions. Baseline production Ruff/Bandit were also clean. B608 inline exclusions cover only fixed internal owner SQL fragments; values remain bound and aliases are fixed/validated, not user-supplied.

The exact final required-PG command is:

```sh
source .venv/bin/activate &&
TLDW_UAT_EVIDENCE_LABEL=uat198-201-202-final-release node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
  tldw_Server_API/tests/DB_Management/test_flashcard_shared_owner_contract.py \
  tldw_Server_API/tests/DB_Management/test_flashcard_owner_resource_edges.py \
  tldw_Server_API/tests/DB_Management/test_deck_owner_name_migration_postgres.py -q --tb=short
```

The safe `uat198-final-adjacent-release-command.json` gives the identical twelve-file adjacent list, including lifecycle, scheduler/review sessions, assets, tags, analytics, deck hierarchy/sharing, HTTP routes, basic cards, and exact Character migration contracts. Official fixtures require PostgreSQL and disable Docker autostart; no manual application database setup or live runtime mutation occurred.

## Limits and next gate

Native acceptance is still pending and belongs to the parent. Source predicates do not install RLS or protect arbitrary SQL clients with table grants. No broad Study/Quiz isolation claim is made: templates, StudyPack/citations, source-review plans, and direct shared assistant storage are outside this unit. The HTTP assistant's existing scoped card fetch benefits from the getter repair, without certifying all assistant storage.

Source013 separately reproduced the cold StudyPack adopter selecting a synthetic `study-pack-worker-2` owner instead of canonical owner2; it is an upstream selected-owner identity issue under a separate parent-owned task, not a reason to weaken this contract. No factory/dependency change is included here.

The parent's prior181 operation ownership,192/197 named-row reads,199 ConversationStore,200 PersonaStateStore, and183 historical fixture changes are preserved and excluded from this unit's attribution. All comparison baselines and frozen hashes are retained. Independent reviewer Retry has the release; parent must integrate and perform native Bob identity/read negatives plus owned positives and separately qualified migration/restore acceptance.
