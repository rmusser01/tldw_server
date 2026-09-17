# UAT225: inactive-Sync lifecycle and transition design

Status: bounded direction approved by parent after Sidebar's source challenge; shared integration remains held until Stage A freezes/reviews. Full lifecycle correction is associated with TASK13260.164; TASK13260.163/UAT225 remain open. Source013 owns the separately bounded Stage A read/capability repair and its tests. Original `DESIGN225.md` is retained as the earlier containment proposal, not a complete fix. Existing TASK13138 also promises this behavior; parent owns task updates.

## Recommendation

Implement the missing local decision adapter and owner-local suggestion lifecycle, then **retire the local review dataset when the actual canonical Notes default is enrolled**. Reuse the existing canonical `note_task_scope_authority` row as the irreversible local-write fence; insert only the real validated canonical dataset with all three unrelated graph flags explicitly false. Do not insert `legacy:<owner>` into that table. Do not introduce a stable review namespace, a new authority table, a Sync domain, or a Job/receipt rekey protocol.

Retirement means an atomic authority change immediately forbids old-scope admission, worker advancement/publication, and product acceptance. Bounded maintenance subsequently records stale/cancelled outcomes and drains cleanup obligations. It does **not** mean scanning every historical row in one unbounded enrollment transaction. Immutable Job payloads and completed receipt envelopes remain byte-for-byte under their original dataset key. Already committed accepted links/tags remain ordinary product state and are captured by existing bootstrap.

### Why retirement is consistent with the existing contract

- The spec's Sync Boundary (lines 378–397) explicitly promises inactive-Sync mutations through existing legacy paths and explicitly keeps provisional review state out of Sync.
- Receipts are owner/**dataset** scoped (602–634); runs and rejection sets are also dataset scoped (586, 636–641). There is no stated requirement to migrate provisional rows, current dismissal suppression, or replay keys across a change of selected dataset authority.
- Thus a new canonical dataset may start a new suggestion history. The old dataset remains inaccessible through the ordinary selected-authority routes; an old key does not replay against the new namespace. Retention of an old immutable receipt is not a promise to bypass current authority in order to expose it.
- This deliberately does not delete accepted product links/tags or pretend their acceptance was undone. It also does not expire current rejections merely due to age: local rejections become obsolete because their entire authority scope was retired. That retirement condition needs an explicit retention test and documentation.
- My earlier stable-namespace suggestion was stronger continuity than the spec requires and is withdrawn. It would expand route identity, admission, replay, merge, and schema contracts unnecessarily.

## Confirmed evidence and limits

- Native owner 1 has no Sync storage and zero authority rows: `../uat225-native-20260917/admin-scope-metadata.json`. This is parent-retained read-only metadata, not agent access to native DB/config.
- Actual route/factory/store regression on official PostgreSQL and SQLite: `uat225-frozen-diagnosis-red.redacted.log`, 4 expected failures / 4 registered-scope controls / 0 skips. Exact exception is `NotesGraphDatasetScopeError("notes_graph_dataset_scope_invalid")`, preceding provider/worker readiness. Original test/manifest are retained; Source013 now owns the test for Stage A.
- No new runtime, model, browser, native DB, or fixture execution was performed for this follow-up. Concurrency, inactive acceptance, and transition proposals below remain test obligations, not verified fixes.
- TASK13138 was read through official CLI. Its Tasks 6–8 history describes guarded **Sync** materialization and canonical acceptance; no intentional exclusion of inactive Sync was found. Its checked AC9/10/21 and the linked spec still require ordinary mutation invariants and guarded reconciliation.

## Existing mechanisms and exact gaps

Paths below are repo-relative. Line numbers describe the inspected source and may move during Source013's disjoint Stage A edits.

| Boundary | Reusable mechanism | Missing work |
|---|---|---|
| Route authority | `app/api/v1/endpoints/notes_graph_suggestions.py:103–111` resolves actual selected Notes authority; inactive/no explicit dataset yields exact `legacy:<owner>`; foreign/explicit inactive datasets are non-enumerating 404 | Keep this selected-authority behavior. No public arbitrary legacy alias after canonical activation. |
| Store scope | `core/DB_Management/chacha/note_graph_suggestion_store.py:249–277` sets transaction-local PG dataset and currently requires shared authority | Permit exact owner-local scope only while no canonical authority exists. Acquire the existing SHARE lock/recheck before local work. After reservation, only a closed internal retirement/cleanup path may access the old key. |
| Existing local links | `endpoints/notes_graph.py:557–574` -> `ChaChaNotes_DB.py:25875` -> `note_link_store.py:283–419` | The wrapper chooses a random edge ID and has no guard argument. Use the same lifecycle store through a small local adapter with stable suggestion-derived ID and its existing before/after callbacks; do not invoke the random-ID wrapper repeatedly. |
| Existing local tags | `endpoints/notes.py:7037–7048,7497–7519` uses canonical coordinator when available, otherwise ordinary keyword APIs | `NotesOrganizationSyncStore.apply_resource:645` and `apply_relationship:1265` already provide validated owner checks, normalized collision serialization, and same-product-transaction before/after guards. They can be reused without constructing fake Sync services or writing SQL in suggestion service code. |
| Decision orchestration | `Notes_Graph/suggestion_decisions.py:129–429` owns claim, renewal, exact postcondition finalization, deterministic keys, two-step new-tag acceptance, and reconciliation | `suggestion_service.py:39–68` returns None without Sync. Add real local mutation collaborators; keep shared claim/finalization behavior. `_resolve_keyword:250–306` directly consults Sync heads and must explicitly distinguish local authority. |
| Acceptance race fence | `note_graph_suggestion_store.py:3022–3120` rechecks scope in `_require_acceptance_fence_locked` inside the actual product transaction | Make this recheck reject retired local scope too. The local adapter must acquire the authority lock **before** link/note/keyword locks, rather than waiting until its before callback; preserve the callback as a second exact-fence check. |
| Jobs | `suggestion_jobs.py:78–92,104–145,180–296` binds content-free payload, run UUID enqueue idempotency, owner/domain/queue/type, completion receipt | Preserve all identity and one-attempt machinery. Enrollment can occur between run commit, external Jobs enqueue, and bind. Retired `admitting` rows must retain the obligation to find a late/unbound Job by the existing run-ID idempotency lookup and cancel it. |
| Maintenance | store `list_maintenance_dataset_ids:652`, core `suggestion_maintenance.py:98–120`, service `notes_graph_suggestions_maintenance.py:90–110` | Include exact local scope with durable state, even after canonical reservation. Retired scope must use a retirement branch, never normal success publication or product acceptance reconciliation. |
| Note mutation invalidation | store `invalidate_for_note_change:4361` is called in NoteStore's actual note transaction | Include exact local scope; current enumeration reads only authority rows. Keep source/target fingerprint invalidation atomic with note update/trash/delete. |

## Canonical enrollment and concurrency gate

### Audited entrypoints

1. `Sync/v2/profile.py:279–407`, `SyncV2ProfileManager.bootstrap_profile`: validates mode/domains/encryption; registers device; obtains actual owner default at line 357; then organization bootstrap at 359–372, link bootstrap at 373–386, attachment and task setup afterward. **Insert a narrow service-owned product scope reservation/retirement call immediately after the default dataset is resolved and before either bootstrap begins.** It must execute on repeated/resumed profile bootstrap too.
2. `Sync/v2/service.py:3473–3595`, generic `enroll_dataset`: rejects reserved organization/link domains and default-personal/client-family metadata. It cannot create this first qualifying Notes default. It can later activate task domains on an existing default; regression-test same-target task binding after all-false reservation. Do not add an unrelated generic enrollment hook.
3. `notes_organization_bootstrap.py:80–228` and `notes_link_bootstrap.py:53–149` take product snapshots and verify captured state. Their lower-level direct invocation is used in tests/internal code. Ensure the new precondition is also enforced at these direct entry boundaries (a shared idempotent preparation call, not copied SQL), or prove callers cannot bypass it. The test contract includes direct invocation, retry, already-ready, and interrupted bootstrap.
4. `profile.py:_bind_personal_context_dataset:682–729` is a second real default-creation path: lines 696–699 obtain/create a default before binding Personal Context. `bootstrap_personal_context:530` can also supply a dataset from the existing guard. The common profile-manager preparation boundary must run for both newly created and supplied/resumed owner defaults here, as well as ordinary bootstrap. Merely patching `bootstrap_profile` is insufficient. Preserve Personal Context's custody, bootstrap guard, selected store, and opaque binding semantics.
5. `server_origin.py:293–307` treats the real default's existence as active while its domains are initializing. This is a cross-database boundary, not an atomic combined Sync/ChaCha transaction. A failure after Sync default creation must fail closed and resume reservation before snapshots; it must never reopen local writes as fallback.

The service already owns `NotesMaterializer.note_db` through `Sync/v2/factory.py:135–187`. `service.py:2489–2507` has an existing domain-materializer-to-product-store pattern. Use that explicit existing collaborator for a narrow preparation method; do not reach into bootstrapper private members, create a new DB, or infer an owner from a device label.

### Reservation algorithm to prove first

- In one existing ChaCha transaction, take the same table-level authority lock family as existing binders (conflicting with local SHARE holders), then load the exact owner row.
- No row: insert actual validated canonical dataset with `task_graph_bound=False`, `moodboard_graph_bound=False`, `studio_graph_bound=False`. Existing equal row: preserve every flag. Different row/owner/malformed state: abort, preserving both product and authority.
- Current v61 DDL permits all three flags false (`ChaChaNotes_DB.py:13072–13088`, PG additions14419–14421); policy is owner-scoped (`16115`). No schema change is required for this reservation. Actual restricted-role insertion and later binders still require real fixture proof.
- Commit the authority fence before bootstrap snapshots. Every local mutating transaction acquires SHARE and checks absence **before any product/run row lock**. PostgreSQL reservation waits for an earlier valid local product transaction; once it commits, later local work fails. SQLite must show the corresponding transaction ordering/rollback behavior using actual connections, without pretending row locks exist.
- The fence is the atomic retirement point; per-row state draining is bounded. No accepted local product may be committed after the fence. A product committed before it is included in ordinary snapshot/bootstrap. A crash between new-tag keyword creation and membership can leave the valid keyword, but cannot manufacture an accepted membership.
- If another existing task/moodboard binder creates the canonical row first, the same absence rule already retires local authority. Cleanup discovers that fact independently; the profile hook is not the only detector.

### Old-scope cleanup authority

Allow only exact `legacy:<bound owner>` with a real different canonical owner row, through internal closed store operations. Preserve PG `app.current_user_id` and set the original dataset setting for each operation; never disable RLS. This capability permits bounded cancellation lookup/binding, stale finalization, receipt closure, evidence removal, and retention only. It must not permit admission, running/provider progression, staging, publication activation, rejection/reset from public routes, or creation/finalization of a product acceptance.

Use existing leases/revisions, cancellation receipts, active/archive Jobs lookup, and budgets. `admitting` without job_id remains reconcilable across the existing ten-minute missing-Job grace: a late enqueue can be found by run UUID, verified against the exact immutable payload, attached for cancellation only, and cancelled. Do not immediately mark it terminal and abandon a later enqueue. Completed terminal receipts remain unchanged; in-progress ones receive one stable terminal retirement outcome. Never label a dataset change as a note fingerprint edit merely to reuse a code.

Pending/accepting/staged review rows become stale under bounded CAS. Accepted rows/products remain accepted. Old rejected rows/rejection sets are obsolete due to retired dataset authority, with bounded retention. Maintenance may discover already committed product postconditions for audit, but may not create a link/tag or reactivate a retired run. Jobs calls stay outside ChaCha transactions.

## Bounded phases and file ownership proposal

Parent associated these phases with TASK13260.164 and authorized new disjoint local-adapter module/tests before shared integration. Shared store/API/service/profile files remain held while Source013 completes Stage A; UAT225 remains open until all required phases pass.

### Phase B1 — local scope, admission, worker, invalidation, maintenance

Production: `chacha/note_graph_suggestion_store.py`; `Notes_Graph/suggestion_service.py`; `Notes_Graph/suggestion_maintenance.py`; `Notes_Graph/suggestion_jobs.py` only if needed to reuse the existing exact late-enqueue lookup/cancellation path. No Jobs manager/backend rewrite. Service wrappers should remain unchanged unless an actual ownership/cleanup regression proves otherwise.

Tests: new actual-factory `tests/Notes_Graph/integration/test_suggestion_legacy_lifecycle.py`; extend pertinent Jobs/lifecycle tests only for shared contracts. Real SQLite + official required PostgreSQL, real Jobs fixtures, a bounded fake provider (no inference), real retrieval/source notes. Prove capability/admission202, one content-free Job, worker completed receipt then publication, list/evidence, reject/reset/replay, cancel/retry, provider-independent cleanup, source/target edits/trash/delete. Cross-owner/arbitrary dataset negatives and restricted-role controls are mandatory. No manually seeded authority on the local positive path.

### Phase B2 — local guarded acceptance

Production: `Notes_Graph/suggestion_decisions.py`; factory in `suggestion_service.py`; one small `Notes_Graph/suggestion_local_mutations.py` adapter if separating the concrete local operations is clearer. Reuse `NotesLinkStore` and `NotesOrganizationSyncStore`; preserve their validation/transactions. No new general coordinator framework, fake Sync store, direct product SQL, or unrelated endpoint rewrite. API capability readiness becomes available only when this real local decision path is present.

Tests: real route + actual local stores for related note, existing tag, new tag, same-request replay, different-key same logical identity, normalized tag collision, rename/delete, concurrent accept/reject, lease expiry, stale fingerprint, caller rollback, injected finalizer failure, and crash after keyword creation. Assert product+suggestion+receipt atomicity and unchanged unrelated rows. Keep existing canonical acceptance suite.

**Concrete tag-merge obstacle:** the spec at708–713 promises following the existing canonical merge result. Local `KeywordStore.merge_keywords:462–595` moves memberships, tombstones source, and returns the source/target mapping; it does not persist a portable merge redirect. Current decisions recover redirects from a Sync head. There is no trustworthy local redirect to read. Do not infer a target from tag text or existing membership. First add an actual local-merge causal control and decide a separately reviewable narrow merge-outcome persistence/invalidation contract with the parent. A plain delete may be stale; silently treating a demonstrated local merge as an ordinary delete would not establish full parity. No keyword/schema edits are proposed as an incidental part of this report.

### Phase B3 — retire on canonical enrollment

Production: shared store preparation/retirement helpers; `Sync/v2/profile.py` and `service.py` for the explicit owner-validated common hook used by both `bootstrap_profile` and `_bind_personal_context_dataset`; the two Notes bootstrappers for direct-entry preconditions only if necessary after caller tests. Reuse existing table and flags; **no schema migration/new namespace**. Closed retirement logic in the existing maintenance module/store. Add a new public reason only if a real route requires one; do not fabricate source-change telemetry.

Tests: `tests/Notes_Graph/integration/test_suggestion_legacy_transition.py` with both backends and actual profile/service/bootstrap. Barriers at admission commit→enqueue→bind, acceptance claim→product guard, keyword→membership, provider result→stage, completed Job→publication, and enrollment snapshot. Assert only the transaction winning before the fence may create product state; no late publication; original Jobs/receipts unchanged; late/unbound jobs cancelled; repeat/crash/resume enrollment; matching/mismatched existing authority; all flags preserved; later task/moodboard/Studio binding correct; direct lower-level bootstrap entry; unsupported generic enrollment unchanged; cleanup bounded, owner-safe, provider-independent, and never performs product acceptance. Exact supported source ordering, not sleeps, should drive barriers.

### Phase B4 — integrated review and native acceptance

Run relevant existing suggestion retrieval/lifecycle/jobs/acceptance/API tests, Sync profile/link/organization bootstrap and binding controls, and service maintenance controls. Required PG cannot skip. Include Ruff, Bandit, compile and exact source manifests. Then independent review and parent-owned native acceptance: actual authorized local capability/read/one deliberately requested generation and decision lifecycle if configured; otherwise explicitly retain worker/provider configuration limitations. Stage A's truthful disabled capability alone does not close the promised feature.

## Alternatives rejected / remaining decisions

- Inserting `legacy:<owner>` into shared authority poisons immutable later binding; rejected.
- Repointing/rekeying runs or Jobs/receipt payloads breaks durable identity; rejected.
- Globally allowing legacy scope in the common helper after canonical enrollment can revive old workers/publication/acceptance; rejected.
- Stable owner namespace/new binding table gives seamless review continuity not demanded by this dataset-scoped spec; unnecessary here.
- Reserving the **real canonical** row with all flags false uses the current authority schema and keeps later domain adoption explicit. This is the preferred minimal candidate, but concurrency, restrictive-role and direct-entry tests must precede production.
- Local merge continuation lacks persisted authority. This is an actual inspected mechanism gap, not a newly reproduced native failure; parent must bound its causal test/repair separately rather than claim it covered.
- No whole-feature pass, model quality result, native worker proof, or cross-database atomicity is claimed by this design.

## Review assistance

Sidebar independently inspected the retirement policy, all-false schema, both default-creation entrypoints and late-enqueue/acceptance races. Their source observations informed the final comparison; they have not reviewed an implementation. Parent approved the direction and created TASK13260.164. No final implementation verdict is implied.
