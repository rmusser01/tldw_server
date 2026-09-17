# Independent review — UAT198 / UAT201 / UAT202

**Verdict: clear for the bounded application-persistence repair on the exact eight frozen files below.** Two defects found during independent source review were reproduced and corrected before this release. No remaining actionable defect was found in this scope. Native acceptance and the separately tracked UAT204 caller-identity repair remain pending parent-owned gates.

Reviewer: `retry031_repair`, reviewing `sidebar155_review`'s implementation under TASK13260.136 / .139 / .140. This review changed no production/test source, task, tracker, git state, browser, or native runtime. Test databases were created and removed only by the official fixtures through the required-PostgreSQL runner.

## Exact source and evidence

- Author frozen manifest: `../uat198-diagnosis-20260917/owned-manifest.json`, SHA256 `c87fe3776978277d64ebc5954657597464ac3f6c5f23c75c02429e8210d0d35b`; copied here as `reviewed-source-manifest.json`.
- Production: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, SHA256 `6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7`.
- `pre-run-hashes.json` and `post-run-hashes.json` prove all eight working files and author snapshots matched their declared hashes before and after the independent run. `reviewed.patch` is the frozen eight-file delta.
- The other seven files are the three new owner/resource/migration suites and the four existing fixture integration files listed in the manifest and command below. No whole-worktree cleanliness claim is made.
- `PROVISIONAL-REVIEW.md` and `first-candidate.*` retain the earlier review state. Their pending disposition is superseded by this report, not evidence that final approval was given to the earlier candidate.

## Independent verification

**183 passed, 0 skipped, 0 deselected, 4 warnings, 246.84 seconds.** Exit 0. This includes the full 126-case new owner/migration suite and all four adjusted existing fixture files. The runner required PostgreSQL, disabled Docker autostart, reused the existing owned test cluster on port 55475, and used official per-test isolation. Exact command receipt and redacted output are retained here as `uat198-independent-final-command.json` and `uat198-independent-final.redacted.log`. The compact runner log reports four warnings without their details; they are not being represented as zero warnings.

```sh
source .venv/bin/activate &&
TLDW_UAT_EVIDENCE_LABEL=uat198-independent-final node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
  tldw_Server_API/tests/DB_Management/test_flashcard_shared_owner_contract.py \
  tldw_Server_API/tests/DB_Management/test_flashcard_owner_resource_edges.py \
  tldw_Server_API/tests/DB_Management/test_deck_owner_name_migration_postgres.py \
  tldw_Server_API/tests/DB_Management/test_character_owner_name_migration_postgres.py \
  tldw_Server_API/tests/DB_Management/test_flashcard_asset_content_backends.py \
  tldw_Server_API/tests/Flashcards/test_flashcard_deck_hierarchy_locking.py \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py \
  -q --tb=short
```

Fresh independent static checks: Ruff on all eight files exited 0 with zero findings; Bandit on production exited 0 with zero findings/errors; Bandit on the seven test files exited 0 with zero findings/errors, excluding B101 only for ordinary test assertions. All eight files parsed. An AST comparison to the author's baseline found exactly the declared 48 changed/added class methods. Details are in `static-summary.json`, `ruff.json`, and `bandit-*.json`. These checks do not establish security outside the exercised/reviewed boundary.

The author subsequently reported 126 new-suite passes (195.90s) and 322 adjacent-suite passes (270.00s), both zero skips on the same frozen hashes. Their final report SHA256 is `406b7a57a5820374e99b6a7e71c77cfbc5b135ebfb320327f3ac17d6e8db0f07`. Those author runs are separate evidence; this verdict does not substitute their counts for the independent 183-case result.

## Findings corrected before approval

### Owned parent with a foreign ancestor

The first candidate scoped the immediate parent but returned early during creation, and treated a missing owner-qualified ancestor as successful traversal during reparenting. An existing owned parent linked to a foreign ancestor could therefore admit another owned descendant. I reported this source path before the final release. The author retained permanent causal controls: `uat198-ancestor-review-red` produced **2 PostgreSQL failures / 2 SQLite passes** before correction. That author RED and its command are copied here; I did not independently rerun the old source.

The final `_validate_deck_parent_locked` at line 34841 traverses PostgreSQL ancestry during creation and reparenting and rejects unavailable/foreign ancestors. SQLite's per-file semantics remain unchanged. The independent final run exercises both negatives and the positive owned three-level hierarchy.

### Explicit-deck queue filtering after selection

The first candidate scoped the global queue but omitted owner filtering in its three explicit-deck learning/review/new SELECTs. A malformed foreign child sorting first could be rejected only after LIMIT 1, hiding a later valid owned card. I reported this source path before the final release. `uat198-explicit-queue-review-red` retained **3 PostgreSQL failures / 3 SQLite passes**. That author RED and its command are copied here; I did not independently rerun the old source.

The final `get_next_review_card` at line 36428 applies the existing bound owner predicate before ordering/LIMIT in all three explicit-deck branches. The independent run passes malformed-child negatives and ordinary learning → review → new priority controls.

Both findings belong to the already approved UAT198 ancestor/queue scope. They are review findings with isolated test evidence, not additional native observations.

## Source review and distinct attribution

**UAT198:** The selected PostgreSQL DB's `client_id` is the owner boundary. The fixed-alias `_flashcard_owner_filter` at line 34798 binds values, and reviewed callers apply it before pagination, aggregation, or mutation. The review covered deck/card detail/list/search/batch/export and queue paths; parent/workspace references; share administration; assets including bytes, reconciliation and cleanup; owner-local tag lookup; rating and review-session creation/reuse/read/completion/abandon/repair. Actual statements and metadata joins carry owner checks. Caller-owned transactions and scheduling algorithms are preserved. SQLite omits this owner predicate and retains cross-device client IDs within its per-user file.

The restricted-role control runs actual repository reads and writes under a verified NOSUPERUSER/NOBYPASSRLS `SET LOCAL ROLE`, allows an owned edit, rejects a foreign edit, and checks the other owner's stored row. This is a real disposable PostgreSQL persistence control, not a native browser login or direct-login runtime test.

The positive authorized-sharing test runs real `SharedWorkspaceAccessService` with real owner databases while ambient context is the recipient. Membership is checked before owner loading, the selected owner's card remains readable, recipient edits remain denied, and denied membership prevents a further owner load. Membership/user-result and loader seams are stubbed; this does not certify real AuthNZ membership SQL. Workspace IDs, visibility labels and share records alone do not grant ordinary routes access.

**UAT201:** `_migrate_from_v68_to_v69_postgres` at line 17747 validates the expected global name-uniqueness shape, adds `(client_id, name)`, safely quotes the catalog-provided old constraint name, drops it, verifies the replacement/no remaining simple global name index, and updates the version transactionally. Tests use a real historical v68 schema and preserve IDs, values, tombstones, same-owner conflicts and unrelated constraints. Occupied destination/extra uniqueness fail closed; injected post-DROP failure restores prior catalog/data/version. PostgreSQL head is 69; SQLite remains 67. This is not a compatibility promise for arbitrary customized catalogs outside the validated shapes.

**UAT202:** `add_deck` at line 34899 restores owner-local tombstones using named `id`/`version` fields. Real PostgreSQL mapping and SQLite row controls preserve the deck identity and reject foreign restoration. The named-row correction is distinct from UAT198 owner predicates.

## Existing-fixture integration

All four changes were read and independently exercised:

1. Character migration tests cap the exact historical 67→68 test at a real v68 constructor and retain literal 68/catalog/data assertions. Ordinary current-head reopen/retry now expects the class head (69). Failure rollback still checks version 67. This preserves the earlier UAT193 contract.
2. Synthetic asset conversion instances now explicitly identify the PostgreSQL backend and owner. Existing bytes/memoryview/empty/null assertions remain unchanged.
3. Synthetic hierarchy lock instances now have an owner; the existing query-count and locking assertions remain. The final import sort is formatting only.
4. The review-session rollback fixture commits only its stale-timestamp seed before taking the baseline snapshot and opening the separate transaction under test. The deliberate outer rollback and unchanged-stale-row/no-replacement assertions remain. The author's `uat198-session-rollback-baseline` reproduces the preexisting failure by replaying selected baseline methods, not a whole-module rollback; I inspected that receipt and replay loader and independently ran the corrected real PostgreSQL/SQLite controls. No production transaction behavior was adjusted to accommodate the test.

## Explicit limits and next gates

- **Raw SQL isolation remains absent for these tables:** Flashcards/decks have no RLS policy, and a granted SQL role can directly read foreign rows. That retained diagnosis is outside this approved application-persistence contract and is not counted as green application coverage.
- This unit assumes the caller selects the canonical owner DB. **UAT204 / TASK13260.142** separately proves a cold StudyPack caller passes `study-pack-worker-2`, caches that ID, and persists inaccessible content. Source013 owns its minimal caller correction. It does not justify weakening UAT198 predicates. Parent will wait for UAT204 review before native reload.
- Test routers select owner dependencies directly; these controls do not certify actual JWT/API-key identity derivation. Parent-owned native Bob identity/read negatives and owned positives are still required.
- Templates, StudyPack/citation storage, source-review plans and direct Study assistant storage are outside this unit. No whole Study/Quiz/product tenant-isolation claim is made.
- No native mutation, provider inference, full fresh matrix, or production upgrade was run by this reviewer. Prior operation-ownership, row-access, ConversationStore, PersonaStateStore and historical-fixture work is preserved, not attributed to this repair.

With those explicit boundaries, the exact frozen UAT198/201/202 source and integration changes are independently reviewed clear. Parent owns staging, integration and native acceptance.
