# UAT198 — shared PostgreSQL Flashcards owner boundary

Tasks: TASK13260.136 (UAT198), TASK13260.139 (UAT201), TASK13260.140 (UAT202). The parent approved the bounded application-owner design after independent Source013 review, then separately approved the PG68→69 deck-name migration and named restore-row fix. The final implementation is frozen for independent review. No browser, runtime, configuration, application database, task, or git mutations were performed by this agent.

## Evidence and causal boundary

The native capture shows Bob identity events (`id=3`, `username=bob`) at 04:30:02.099/.340/.598 UTC, followed by successful deck/card responses containing Alice `client_id=2` private rows. Earlier Alice responses in the same file are controls, not cross-owner evidence. See `native-allowlisted-summary.json` and the parent's `uat198-native-read-failure.json`. The native role is privileged/BYPASSRLS; no native foreign mutation was attempted.

The actual router/DB fixture uses two owner instances over one official isolated PostgreSQL backend; SQLite uses distinct per-user files. The dependency override supplies the real owner DB without mocking SQL, routes, scheduling, or assets. It does not replace the native authentication evidence or claim to test an actual AuthNZ login.

| Run | Result | Established failure |
| --- | --- | --- |
| `uat198-owner-boundary-red` | 14 failed, 18 passed, zero skipped, 52.49s | Private deck/card list, totals, search, detail, explicit deck/workspace filters; card edit/delete/reset/tags/insert and deck edit/delete modify foreign rows; missing raw-SQL RLS |
| `uat198-resource-edges-red` | 7 failed, 7 passed, zero skipped, 22.98s | Foreign asset bytes return 200; foreign asset attachment transfers owner; foreign rating changes schedule/owner; foreign parent assignment; another owner's `due:global` session reused; separate name collision and restore row-shape failure |

All 21 failures are PostgreSQL. The 25 passing controls include each corresponding SQLite behavior, owned operations, workspace presentation, owner-side explicit share lookup, and same-file SQLite sync devices. These are two non-overlapping runs, not a claimed combined 46-case rerun. All production hashes in `source-during.json` still match after the runs. The first snapshot was taken after the first run; it is not a pre-launch receipt.

## Root cause

The shared PostgreSQL owner instance carries the authenticated owner as `self.client_id`, but the Flashcards persistence methods still assume SQLite's physical per-user file boundary. Many reads filter only an ID or workspace, and writes select by ID then assign `client_id=self.client_id`, transferring another owner's row. `_flashcard_visibility_filter` is a navigation filter, not authorization. List-only repair would leave direct bytes, mutation, rating, and session paths exposed.

Unlike Character UAT194, Flashcards has no RLS policy at all: the actual catalog shows `relrowsecurity=false`, `relforcerowsecurity=false` for `decks` and `flashcards`, and no policies. A disposable verified NOSUPERUSER/NOBYPASSRLS role granted SELECT reads Alice rows with `app.current_user_id=3`; changing only runtime privilege does not repair it. `restricted-role-catalog.json` retains this SQL-level proof. This is not a native application run under the restricted role.

## Proposed application repair

Keep the owner contract on the selected database instance, matching the approved CharacterStore and Notes precedents. Add parameter-bound **PostgreSQL-only** owner predicates within the Flashcards persistence methods. SQLite continues using its per-file user boundary and permits different sync-device client IDs.

Use a small Flashcards-local owner-clause helper with fixed internal aliases if it reduces repeated backend branches; do not alter generic SQL execution, transaction ownership, authentication, ambient scope, or workspace-sharing policies. Keep presentation filters separate. Put predicates in actual writes and locked parent lookups, not just HTTP preflight. Preserve optimistic versions and caller-owned transactions.

`proposed-method-inventory.json` records the exact 44-method review boundary and individual source hashes. It is a review inventory, not a mandate for 44 edits: methods already closed by an authorized local helper can retain their implementation.

| Boundary | Required behavior |
| --- | --- |
| Deck CRUD and shares | List/get/by-name/include-deleted rows belong to the DB owner. Owner-only update/delete/restore and share administration; explicit recipient-share lookup continues inside the already-authorized owner DB. No empty-update success that implies a foreign deck exists. |
| Card CRUD, batches, queue, tags, totals, exports, analytics | Scope all rows before search, LIMIT/OFFSET, aggregation, and mutation. Include owner deckless cards. Card/parent joins must not expose a foreign deck's name/settings/workspace for malformed cross-owner links. Do not silently alter valid owner-card visibility when its own deck is soft-deleted. |
| Parent references | Creation/reparent/bulk operations require a live owned deck; deck parent ancestry must stay inside owner scope. A workspace ID is not a grant: validate the selected live workspace against the DB owner at this boundary, not through the existing unqualified `get_workspace` existence check. Keep all-or-nothing behavior. |
| Asset metadata/content/reconciliation/cleanup | Direct asset metadata and bytes are owner-only, including unattached uploads. A card cannot acquire another owner's asset; reconciliation and stale cleanup cannot mutate foreign assets. Preserve byte conversion, attachment conflicts, and existing upload validation. |
| Rating and session lifecycle | Own card plus own session must be required atomically. Active session lookup, duplicate cleanup, stale abandonment, completion, rollup repair, reviewed-card lists, latest review, and analytics remain owner-local. Identical `due:global` keys for two owners must produce separate sessions. |

Preserve existing authorized owner-DB selection: SharedWorkspaceAccessService checks active recipient membership before loading the owner's DB and denies recipient edits; the study job accessor separately authorizes owner selection. Do not reinterpret bare `workspace_id`, `visibility=public/team/org`, or a share record as permission in the ordinary owner routes. No recipient-sharing feature is proposed.

## RLS decision — explicit separate contract

Recommended first repair is the above application persistence contract under both privileged and restricted runtime roles. It does **not** promise that an arbitrary SQL client with SELECT grants is tenant-isolated. The raw-SQL policy test is diagnostic evidence and cannot be claimed GREEN from application predicates.

Installing forced policies on this resource family is a separate explicit design decision, not an incidental addition. It needs owner-loader/background scope controls: an authorized owner DB may differ from the ambient recipient GUC, and workers can run without a request GUC. If parent includes this work in198, define those rules before production and test SELECT/USING and mutation/WITH CHECK on the exact resource tables. Do not install a global policy sweep or modify unrelated domains. Otherwise retain the raw-SQL exposure as an explicit tracked residual and move that diagnostic into the private probe packet instead of weakening it or shipping a deliberately failing test.

## Distinct reproduced deck defects requiring association

1. Two owners cannot use the same deck name: `add_deck` raises `ConflictError` from the global `decks.name` unique constraint. This remains true even if RLS hides the other owner's row. Proposed separate versioned PG68→69 migration replaces only validated global name uniqueness with `(client_id, name)`, preserves IDs/data/deleted-name reservations, remains atomic, and leaves SQLite67 unchanged. Confirm catalog/index variants and rollback controls before writing migration code; this is not authorized merely by this proposal.
2. Restoring an owned deleted PostgreSQL deck via `add_deck` raises `KeyError(0)` at the `deleted_row[0]`/`[1]` unpack. Proposed separately attributed named `id`/`version` reads; this also needs an owner predicate so repair cannot restore another owner's tombstone.

The exact causal traces are retained in `uat198-resource-edges-red-assertions.txt`. Neither failure is a new native observation.

## Tests before release

Extend the existing permanent router/DB tests with explicit owned/foreign/missing/deleted direct methods; batch all-or-nothing; share administration; parent/workspace guards; malformed cross-owner links; owned soft-deleted parent read compatibility; asset cleanup; session completion/stale cleanup/rollup and caller rollback. Add actual persistence calls under a restricted role with sufficient table/sequence/trigger grants, separate from the diagnostic raw SQL test. Verify sharing through an explicitly authorized owner DB, not a fabricated recipient bypass. Keep same-file SQLite sync controls.

For a separately approved name migration: actual historical v68 catalog and seeded rows, fresh v69/reopen, same-owner conflict, two-owner same name, owner-local tombstones, unexpected global-index guard, and transaction rollback. Existing historical66/67/68 migration assertions must remain exact.

Run required-PG with zero skips, adjacent Flashcards/deck/sharing/scheduler/assets/session/export tests, scoped Ruff baseline, and Bandit. Freeze source and tests for independent review before parent restarts the native API. Native acceptance must repeat Bob identity plus private list/detail negatives and owned positives; no live foreign mutation is needed.

## Named limits

This boundary does not certify all Study/Quiz tables: templates, study packs/citations, source-review plans, and direct shared Study-assistant storage are excluded pending separate concrete findings. The assistant HTTP path already fetches the card through the proposed scoped getter, but that does not prove its entire storage domain. Parent/source reviewers must not label this a whole-product tenant-isolation audit.

## Implementation stages

1. **Causal diagnosis — complete.** Retain native identity/read proof, official32+14 cases, role catalog, source inventory and separate findings.
2. **Scope closure — complete.** Parent selected application persistence predicates, explicitly retaining the raw-SQL no-policy limit, and associated UAT201/202. Tag resolution, parent/workspace, assets, review/session, restricted-role persistence, and authorized owner-loader controls were added. The original raw-SQL diagnostic remains in the private packet; no skipped/deselected permanent test is used for this limit.
3. **Minimal persistence/schema repair — complete.** PostgreSQL owner predicates, Flashcards-local keyword resolution, validated PG68→69 uniqueness migration, and named restore row reads are frozen. SQLite and transaction ownership behavior remain covered. Independent source review identified ancestry traversal and explicit-deck LIMIT ordering gaps; both received permanent causal RED before narrow corrections.
4. **Verification/review/native handoff — in progress.** The final correction run passed 97 tests with zero skips; all eight files compile and pass Ruff, and production/test Bandit has no findings (pytest assertion check B101 excluded for tests only). Final permanent126 and full adjacent322 runs passed with zero skips. Independent review is underway. Native acceptance remains exclusively with the parent. See IMPLEMENTATION198-201-202.md for final results and exact manifests.
