# Local keyword merge survivor: read-only design

Date: 2026-09-17. Parent requested a separate bounded design before assigning a repair task. No product/test edits or fixture/native executions were performed for this investigation. The disjoint local adapter checkpoint and Stage A shared-file lease are unchanged. This is a source diagnosis and proposed test contract, not a reproduced acceptance defect or an approved schema change.

## Finding and existing mechanisms

The Notes graph spec explicitly requires merge acceptance to use the surviving tag, with actual note membership as the success condition (`Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md:708–713`). Plain deletion is a different outcome.

- `chacha/keyword_store.py:462–595` performs the local merge in one transaction. It checks source/target ownership and versions, moves four relationship families, and tombstones the source with the next source version. The survivor appears only in the returned dictionary. The source row and legacy sync-log payload do not retain it.
- `Sync/v2/notes_organization_coordinator.py:970–1055` plans canonical relationship moves and the source tombstone. Every step carries `notes_keyword_merge_response`; `:1058–1091` restores the response from the immutable manifest. This is an existing durable outcome **for canonical merges only**.
- `Notes_Graph/suggestion_decisions.py:253–304` follows canonical current-head merge metadata, resolving its local target ID through an owner-bound resource lookup. It already bounds chains to 100 visited identities. Live resources take precedence over old merge history.
- `ChaChaNotes_DB.py:1687–1726` keyword sync triggers record row/delete state, not a target ID. They cannot distinguish an ordinary delete from a merge. Reconstructing a survivor from moved memberships or labels would be an inference, not a durable result.
- A merge response is not stored by the local Notes API (`endpoints/notes.py:7298–7350`). Only its active canonical branch uses durable request replay.
- `note_graph_suggestion_store.py:1574–1592,1833–1873` publication resolves existing tags from active keyword rows. A provider candidate naming a keyword merged while generation was in flight is currently filtered at publication. Acceptance-only repair would therefore miss a real boundary; it needs a causal control before modifying that boundary.

Search scope: the keyword/organization stores, generic keyword CRUD and schema definitions, Notes API merge/delete routes, Notes graph generation/publication/decision stores, canonical organization coordinator/materializer/bootstrap, and existing keyword/API/acceptance tests. No alternate local durable merge-outcome storage was found in this scope. Media DB keywords are a separate resource and are excluded.

## Options

| Option | Benefit | Cost / correctness limit |
| --- | --- | --- |
| Nullable portable survivor identity on the existing keyword tombstone | One nullable field; survives restart; shared by publication and acceptance; records merges even when no suggestion exists yet | A real schema migration and explicit reset semantics at every restore writer are necessary. Historical local tombstones cannot be backfilled safely. |
| Retarget existing suggestion rows at merge time | Avoids adding a keyword field | Misses provider results not staged yet, changes review identity and revision/replay/suppression semantics, spans every dataset/run state, and couples ordinary keyword merge to optional suggestion tables. A late-result fix would still need durable merge history. |
| Reuse canonical current heads/manifests | Already supports canonical merge/replay and chains | Inactive Sync has no dataset/head to write. Creating a fake canonical binding or synthetic Sync history violates the approved local/canonical transition design. Preserve this path for canonical merges. |
| Add a separate merge-outcome journal | Could preserve every historical transition | More schema, indexing, retention, and version-selection logic than required. Existing behavior resolves current live identity first, rather than exposing a historical merge ledger. |

**Recommendation:** one nullable `merged_into_sync_id TEXT` on the existing keyword row, used as current local tombstone metadata. Keep canonical head resolution as it is. Do not rewrite pending suggestions or receipts at merge time and do not introduce a new namespace/table.

## Proposed persisted contract

1. New and ordinary deleted rows have `NULL`. A successful local merge atomically writes the target's **portable** `sync_id` in the same source CAS update that tombstones it and moves memberships. The response, target version semantics, source version increment, and outer transaction ownership stay unchanged.
2. Store the immediate target, not a compressed chain. Resolve A→B→C by exact identities; stop at the first active row. Missing, plain-deleted, self-referential, cyclic, malformed, or over-limit chains fail closed as missing/stale. No label or membership guessing. No hard-delete cascade or FK that would remove/retarget existing data.
3. Restore makes the old identity active again and clears its redirect in the **same update**. Thus A→B, restore A, then ordinary delete A must not resurrect the obsolete A→B redirect. This matches the current canonical resolver's live-row-first behavior. Re-merging restored A creates a new redirect under its new version.
4. An idempotent ordinary delete of an already merged tombstone does not erase its merge result or increment a version. A successful plain delete of an active row leaves NULL. Source/target must both be valid active rows at merge; preserve source/optional-target CAS conflicts.
5. PostgreSQL traversals and writes must use the selected owner predicate at **every hop**, including corrupt historical foreign targets; this cannot rely on RLS. Same-normalized-label foreign data must neither satisfy a redirect nor be mutated. SQLite keeps its existing per-file/device-label behavior; do not introduce a global `client_id == actor` rule into ordinary SQLite keyword CRUD. The new local resolver must explicitly respect this backend distinction; existing canonical organization ownership remains unchanged.
6. Treat this as local product metadata, not a new Sync wire field. No legacy sync-log or canonical payload shape change is necessary for the local fix. Canonical merges continue using their existing heads/manifests. Local review state is retired on canonical enrollment, so no cross-dataset redirect/replay migration is promised. Do not consult an old local redirect in preference to a canonical head.
7. Existing tombstones migrate to NULL without changing identity, versions, timestamps, names, memberships, owner labels, FTS, policies, or triggers. No reliable local historical target exists to backfill. That limitation must remain explicit.

## Exact restore seams that prevent a one-line-only fix

- `KeywordStore.add_keyword:43–71` calls `_add_generic_item`. Passing this field as a NULL additional column lets the existing undelete update clear it while retaining `sync_id`; `_add_generic_item:32657–32685` already skips replacement of `sync_id`. Avoid a generic CRUD rewrite.
- `organization_sync_store.py:647–793` can restore a keyword through an upsert. Clear the local field when a keyword becomes active, without changing folder/collection operations or canonical merge metadata.
- `ChaChaNotes_DB.py:_flashcard_owned_keyword_id:38003–38027` restores PostgreSQL keywords directly; clear the field there too. SQLite flashcard tagging already calls `add_keyword`.
- Prefer a narrow schema CHECK that permits a non-NULL redirect only on a deleted row and rejects self-identity. This makes overlooked restore writers fail rather than silently leave a stale redirect. Exact portable-ID validation belongs to the existing portable-identity validator/application boundary; do not add an ad hoc parser. Migration controls must establish the chosen CHECK on both backends.

## In-flight work and transaction rules

Keep the original suggestion's portable identity and immutable decision/replay envelope. Resolve the current survivor when publishing and when accepting/reconciling local work. Publication can use the existing staged normalization/filtering stage to display the survivor and deduplicate it; it must not manufacture an accepted result or mutate completed receipts. An already-accepted response remains an immutable historical response even if a later ordinary merge moves the membership.

For acceptance, an unlocked early resolution is only a proposal. Recheck the exact chain inside the existing product mutation guard and finalize only after the selected note has the resolved active survivor membership. A concurrent merge/restore/delete must either serialize or release/retry the existing acceptance; it must not add membership to a tombstone or falsely report success. Local authority/suggestion guard locks remain before keyword/product locks, preserving the approved enrollment fence.

Avoid acquiring an arbitrary chain in traversal order. Proposed bounded protocol: discover the at-most-100-row chain, acquire its keyword rows in stable local-ID order in the product transaction, then re-read/validate the same identities, versions and redirect edges. If changed, return the existing retryable conflict/release result; do not add an unbounded retry loop. The merge writer should acquire its two parent rows in that same stable order before the existing checks. SQLite uses its existing write transaction. Prove the actual concurrent paths before choosing additional locking code; no owner-wide lock service or generic framework is proposed.

## Bounded files / stages for a separately associated task

1. **Causal tests only:** new `tests/Notes_Graph/integration/test_suggestion_local_keyword_merge.py` and a focused DB migration/keyword suite. Use the real local adapter/decision boundary when integrated; meanwhile isolate the real local writer plus proposed resolver without mocking the missing persistence. Demonstrate pending merge→accept and generate→merge→publish separately. Preserve canonical positive and plain-delete controls. No production edits until this RED is retained.
2. **Persistence:** `ChaChaNotes_DB.py` only schema-version/migration dispatch + the direct flashcard restore assignment; `chacha/keyword_store.py` add/merge/read helper; `chacha/organization_sync_store.py` keyword restore clear. Current inspected heads are SQLite67 and PG69; next steps would be SQLite67→68 and PG69→70 **only if still current when assigned**. Use proper historical fixture construction, not relabelling a current schema. Fresh/reopen/rollback must all agree.
3. **Local consumers:** existing new local adapter/decision integration in `Notes_Graph/suggestion_local_mutations.py` and `suggestion_decisions.py`, plus `note_graph_suggestion_store.py` local publication resolution. Share the bounded owner-aware product resolver, not separate divergent algorithms. No endpoint/schema response, canonical coordinator/head format, model/provider, or Jobs changes.
4. **Review:** exact hunks and source hashes, official required-PG+SQLite results, existing canonical acceptance/merge API/keyword/flashcard regressions, Ruff/Bandit and migration static checks. Parent controls task, schema authorization, source lease, native acceptance and integration.

## Causal control plan

- Actual local merge A→B, reopen, accept pending A: only B membership; exact replay unchanged. A→B→C and renamed C work; unrelated equal-text tag does not.
- Provider snapshot contains A; merge before staging/publication; actual publication retains a reviewable survivor candidate. Merge after publication but before acceptance uses the same current survivor contract. No model inference; deterministic provider boundary only.
- Plain delete, missing target, foreign target, malformed/cycle/over-limit, source/target stale versions and missing/deleted parents produce no product/receipt success. Include owner-positive/foreign-negative with same normalized label under two PG owners; SQLite differing device labels remain compatible.
- Restore A through ordinary add, guarded organization upsert and PostgreSQL flashcard tagging; then delete or re-merge. Restore B within A→B→C resolves current B; subsequent plain delete B must not reuse B→C.
- Real concurrent PG merge vs acceptance, target merge vs acceptance, restore/delete vs acceptance and enrollment vs acceptance: bounded completion, no deadlock/false acceptance, rollback leaves caller-owned work intact. SQLite deterministic interleavings and two-connection contention controls where supported.
- Caller rollback and injected post-membership/finalization failure roll back redirect/product/decision together; keyword-only partial creation semantics stay intact.
- Genuine SQLite67 and PG69 historical fixtures preserve seeded active/deleted rows and all schema metadata across migration, reopen, repeated initialization, and injected migration failure. No invented backfill. Test constraints and restricted runtime role.
- Existing canonical `test_existing_tag_rename_and_merge_follow_current_portable_identity`, ordinary deletion, immutable batch replay, and unrelated flashcard/keyword tests remain positive controls. Existing canonical publication filtering of a merge during generation is a source candidate; do not silently change that separate path or claim it covered without its own causal case.

## Limits

No new test or executable probe was run in this read-only task. Source observations establish the missing durable local outcome, but actual acceptance and concurrency failure counts remain to be collected after task association. Stage A containment and the 18 passing disjoint-adapter controls are separate evidence; neither proves this merge behavior. No raw native logs, credentials, runtime data, or provider calls were used.
