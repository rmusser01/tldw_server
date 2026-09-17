# UAT209: selected-owner Notes persistence design (proposal; production held)

TASK13260.147. Parent supplied the original native actor3→owner2 private Notes read and six-read official-fixture RED. This packet extends the diagnosis using only official temporary fixture databases. No native request or production change was made.

## Causal evidence

| Run | Result | Meaning |
| --- | --- | --- |
| Parent initial catalogue/detail/count/trash/batch/search | 6 PG failures / 6 SQLite pass | Shared selected-owner Notes read leak |
| Corrected private mutation/keyword/graph edges | 11 PG failures / 11 SQLite pass, zero skips,32.61s | 10 owner failures plus1 separate graph row-shape failure |
| Organization and real restricted role | 6 PG failures / 7 pass, zero skips,18.57s | Folder/collection/keyword owner errors;6 SQLite controls and1 restricted-role control pass |

The mutation cases directly call the real persistence boundary and receive success for another owner's update, soft-delete, hard-delete and restore. Note mutations can assign the caller's client_id to the foreign row. Keyword list/id/text and both note→keyword parent combinations expose or create foreign links. Folder list/name/linked read and foreign-note folder synchronization also succeed across owners. Same-text keyword creation reuses the foreign keyword rather than producing an owned row. Initial private test had two SQLite harness mistakes (separate-file integer-ID collision and expected FK error wrapper); original source/log remain and harness-correction.md records their correction. No production fix was applied to obtain the corrected RED.

`get_all_note_ids_for_graph` independently raises KeyError(0) on a populated PostgreSQL row at note_store.py1820. This is not an ownership assertion and must receive separate task association/repair before changing positional accesses. Adjacent graph count/tag/source methods have similar positional expressions but are not yet individually reproduced; do not claim them proven.

## Intended authority and existing working patterns

HTTP Notes uses `ChaCha_Notes_DB_Deps.get_chacha_db_for_user`; the real dependency requests the cache with user_id and canonical `str(current_user.id)`. UAT204 already removed the StudyPack custom owner label. Sync NotesMaterializer uses `_projection_client_id(note_db) == str(note_db.client_id)`. No new cache/accessor policy is justified by this failure. Permanent coverage should still use actual cold HTTP dependencies for two actors to bind request principal to selected persistence owner.

PostgreSQL Notes, keywords, folders and related tables already have RLS, including FORCE where required; this is not Flashcards' missing-policy case. A real NOSUPERUSER/NOBYPASSRLS role under owner GUC3 passed actual list/detail Notes, keyword and folder service reads. The accepted native runtime uses a privileged role that bypasses those policies. Therefore the proposed fix adds application persistence predicates as defense in depth; it does not promise isolation for arbitrary bypass-role SQL. The role control uses SET LOCAL ROLE after normal fixture bootstrap, not a cold restricted-login initialization claim.

Existing `NoteStore.get_source_note_projection`, `get_note_source_info`, `owns_note_id`, Notes Studio/task/attachment stores, graph projection/live-edge queries and the recently reviewed Flashcards/CharacterStore owner guards provide precedents. They use PostgreSQL selected-owner predicates and preserve SQLite's per-file device/client labels. Notes-specific organization schema already has owner-qualified keyword text, collection name and live folder path indexes; no new uniqueness migration is indicated.

Legitimate shared workspaces authorize membership before loading the owner's database. Their `workspace_notes` table is distinct from private `notes`. Preserve that access service/owner-loader contract; do not treat a supplied workspace ID as permission to read another owner's private Note. This unit does not alter shared-workspace routing or add sharing to private Notes.

## Proposed minimal boundary

Three production files only: NoteStore, KeywordStore and the existing ChaCha facade/generic/folder helpers. `proposed-method-inventory.json` records exact named review boundaries, including already-scoped controls; it is not an automatic edit list.

1. **Private Notes:** parameterized PostgreSQL client_id predicates on catalogue, exact/batch/trash lookup, counts, FTS and fallback search. Versioned update/delete/restore must scope their mutation SQL and status/version readbacks, including idempotent/no-op paths. Foreign rows appear absent and keep their original contents/version/owner; existing ConflictError/false contracts remain. Guard optional conversation/message parents against foreign/deleted sources in the same transaction when relevant. Preserve existing caller connection/transaction behavior, read_only flags and sidecar/projection updates.
2. **Organization resources:** scope keyword/collection CRUD, same-name reuse/restore, merge/search/count and folder path construction/reuse to the selected PostgreSQL owner. Reuse current generic implementations by adding an explicit optional owner argument from the relevant KeywordStore callers; defaults must remain unchanged for all unrelated callers. The alternative is hard-coded table scoping in generic helpers, but explicit caller scope is easier to audit and avoids silently changing unrelated generic behavior. SQLite callers retain current arguments and label semantics.
3. **Linked resources:** note/keyword, collection/keyword, conversation/keyword and folder memberships require both referenced parents to belong to the selected owner. Read joins must hide malformed legacy cross-owner links even when one parent is owned. Mutations use owner-qualified predicates/parent locking inside the existing transaction, not only a separate preflight; preserve link idempotence, timestamps and sync-log behavior. No generic SQL classification, request-lifetime, migration, automatic commit or RLS rewrite.
4. **Graph and child boundaries:** scope the legacy graph seed/count/tag/source joins at NoteStore, while preserving already-scoped manual/projection edges, Studio/tasks/attachments. Source/store inventory identifies these existing controls. Prove malformed child/reference behavior before touching any additional store. Sync upsert/tombstone are explicitly included as preservation/bypass tests because they write by object ID: accepted materializer authority remains canonical; any proven foreign-object bypass must be reported and minimally guarded without changing envelope/version contracts.

## Test-first gates before implementation approval/freeze

Expand the existing permanent test boundary after parent review: actual two-owner HTTP list/detail/export/search and representative PATCH/DELETE/restore; own positive, foreign/deleted, version conflict, no-op/idempotence, raw/caller/nested transaction rollback, concurrent parent ownership guard, and empty/batched results. For linked resources: both foreign-parent directions, same text under two owners, owned tombstone reuse, malformed pre-existing cross-owner links, merge/rename isolation, folder hierarchy/path reuse, and no foreign version/content changes. Exercise both the supported privileged service role and the existing restricted RLS role explicitly.

Retain SQLite per-file old device labels (different from canonical numeric owner), current Sync materializer semantics, actual cold dependency/cache identity, shared owner-loader controls, and previous171/181 operation ownership/lifetime suites. Run the separate graph positional regression under its own task; do not hide its failure as an owner test skip. Finally scoped Ruff/Bandit, baseline attribution for existing diagnostics, exact frozen snapshots and independent review precede parent native acceptance.

## Review decision requested

Approve the three-file selected-owner persistence boundary and explicit optional generic scope design, or split organization CRUD into a separately reviewed task if preferred. Production remains untouched pending that decision. Existing Notes Studio/task/attachment and shared-workspace lifecycles are preservation controls, not a broad rewrite. No native/general completion claim is made.
