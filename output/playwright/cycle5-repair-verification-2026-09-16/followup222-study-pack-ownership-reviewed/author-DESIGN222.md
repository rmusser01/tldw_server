# UAT222 / TASK13260.160 — selected-owner StudyPack/provenance boundary

Status: concrete design for parent review; no production edit yet. The native original job5/pack2/deck10 and all native data remain untouched. UAT223 separately owns `StudyPacks/provenance.py` route construction and its existing test file; this unit will not edit those paths.

## Evidence and cause

The native authenticated receipt shows Alice2 job5/detail/cited-card assistant200, Bob3 job5 correctly404, but Bob pack2 detail200 with Alice client_id2 and private source bundle. The selected privileged/BYPASSRLS PostgreSQL service role makes application predicates necessary; this is not evidence about a restricted-role deployment.

The detail route directly calls `get_study_pack`, whose query filters only id/deleted. Regenerate calls the same getter before copying the source bundle into a real new Jobs entry. Job-status authorization is separate and already selects the job owner's database when an authorized administrator reads another owner's job; that contract must remain intact.

Clean permanent initial17 controls: **8 PostgreSQL failures /9 passing controls /0 skips**,22.86s. Failures are actual detail200 instead of404, regenerate202 instead of404, four foreign store reads, and two actual owned-card assistant responses containing foreign child citation/pack metadata. The assistant rejects a wholly foreign card first, but that guard cannot authorize every joined child. Malformed mixed-owner children were created through the current public persistence methods in the isolated fixture; no raw fixture SQL was required to demonstrate these gaps. All SQLite account-file controls and same-file device-label controls pass. The initial11-failure candidate included3 fixture setup errors and is preserved separately, not counted as product evidence.

Regenerate regression uses the existing real disposable SQLite-backed JobManager. It only queues a fixture job: no worker or model is started. No native generation occurs. Tests override actor/database dependencies, so they establish route/storage behavior rather than repeat native authentication acceptance.

## Proven read inventory

| Method in ChaChaNotes_DB.py | Current gap | Bounded PostgreSQL rule |
| --- | --- | --- |
| `get_study_pack` (~39195) | Foreign pack row returned by id | Require selected owner on pack; returnNone when hidden. Existing detail/regenerate routes then404. |
| `list_study_pack_cards` (~39253) | Foreign pack/membership/card metadata returned | Require selected owner on pack, membership and referenced card. Preserve order, include_deleted selection, and same-file SQLite behavior. |
| `list_flashcard_citations` (~39495) | Foreign citation rows and foreign-card citations returned | Require selected owner on citation and its referenced card. Preserve ordinal/id order and existing include_deleted behavior. |
| `get_study_pack_for_flashcard` (~39556) | Owned card can expose foreign pack through mixed-owner membership | Require owned card, membership and pack before applying existing order/LIMIT1. A later valid owned membership must remain eligible if an earlier foreign membership is filtered. |

Use existing `_selected_owner_filter(self.client_id, alias)` for fixed internal aliases and bound values. Compare existing `get_flashcard`, `get_flashcard_asset`, and `get_note_folders_for_note`/selected parent checks. SQLite predicates remain empty because client_id labels sync devices there. Keep current deleted/status semantics; do not invent a new global visibility mode, cache policy, RLS migration or SQL translator.

## Directly implicated writes and parent validation

The following exact methods share the same rows and are the proposed bounded write scope; causal write controls are now retained:10 PostgreSQL failures /10 SQLite controls passed /0 skips (17 prior cases deselected),28.93s. The seven mutation cases alter owner rows; the three parent cases wrongly accept a foreign destination deck, card or replacement pack. The earlier probe missing its exception import is preserved as a harness failure, not a product count. No unrelated Notes, Chat, scheduler or source-resolution methods are included.

| Method | Proposed validation |
| --- | --- |
| `create_study_pack` | Validate non-null referenced destination deck with existing `_validate_flashcard_deck_locked`; preserve existing workspace behavior and JSON/source-bundle contract. Validate a supplied superseded_by pack as an owned live pack. |
| `add_study_pack_cards` | Within existing transaction, require owned live pack and every referenced live owned card before inserting any membership. Preserve duplicate/no-op counting and rollback. |
| `add_flashcard_citations` | Require owned live parent card before appending rows. Empty-input no-op stays unchanged. |
| `replace_flashcard_citations` | Require owned live parent card, scope citation updates to selected owner, keep replacement atomic. |
| `replace_flashcard_citations_and_source_reference_summary` | Require owned live card; scope both card mutation and citation mutation. Preserve atomicity and public row/version/legacy-summary contract. |
| `set_flashcard_source_reference_summary` | Scope the parent-card lookup and update; a foreign UUID must not reassign its client_id to the caller. |
| `soft_delete_study_pack` | Scope initial/version/recheck/update to owner; retain same-owner stale-version/idempotent-delete semantics. |
| `supersede_study_pack` | Require both packs owned; preserve sorted PostgreSQL lock order, optimistic version, no self-supersession, live replacement, and rollback. |

For the repeated pack/card transaction checks, the smallest established mechanism is extending the existing `_require_selected_owner_row` closed table allowlist with `study_packs` and `flashcards` only, choosing the fixed `uuid` lookup column only for flashcards and the existing `id` for all other rows. Keep its PostgreSQL-only behavior, lock ownership, error semantics and caller transaction. Do not alter existing Notes/message branches or introduce a generic query framework. Existing `get_flashcard` is a read contract, not a substitute for the in-transaction locked parent check.

The production proposal is therefore **12 existing StudyPack/provenance methods plus the narrow existing helper allowlist/identifier extension, all in ChaChaNotes_DB.py**. No endpoint, provenance-service, cache, schema, policy or migration edits. Read-only flags and caller transaction ownership remain unchanged. All SQL data stays bound; inserted identifiers remain a closed set.

## Compatibility controls required before final GREEN

- Both actual backends: own/missing/deleted/version/duplicate membership/ordering/normal helper/HTTP controls, genuine detail/regenerate and existing completed-job/admin-owner resolution.
- Real PostgreSQL two selected owners: every foreign read/mutation and both directions of parent references; include_deleted does not disable ownership; old mixed-owner membership/citation rows remain filtered. After writer checks are introduced, explicitly seed historical malformed rows through fixture-only raw storage so read defense remains tested independently of newly rejected writes; retain the original public-method RED as proof of previous reachability.
- Owned-card assistant: retain normal citations and primary selection, source/deck/card/pack identities, filter only foreign rows, and choose later valid owned membership. UAT223 link-route semantics stay separate.
- SQLite same file: multiple device labels can read/update their own shared local resources; different account files stay isolated. Do not infer user ownership from SQLite client_id.
- Caller-owned transaction tests: denied operation and injected write failure leave no partial membership/citation/source-summary or sync changes; ordinary nested operations still roll back with their caller. Preserve existing generation-service outer transaction and optimistic versions.
- Parent metadata: use valid owner-controlled destination deck; preserve the existing workspace contract and source bundle data, without trying to authorize every historical source reference through a new resolver.
- Existing required-PG StudyPack storage/generation/worker/response/provenance regressions and pertinent prior owner tests; Ruff, Bandit, exact method-only patch/AST attribution and independent review. Native same-pack Alice/Bob readback remains parent-owned after review.

## Scope alternatives

Getter-only would fix the native detail and enqueue seam but leave two demonstrated owned-card assistant leaks and writable foreign parent relationships. It cannot satisfy the task's private metadata/read-write boundary. The bounded family repair above is recommended. A broad schema/RLS/cache redesign is unnecessary for these proven cases and is excluded.
