# UAT225 / TASK13260.163 — independent design review

## Recommendation

**Option A is acceptable as a bounded read/capability repair, with the refinements below. It does not satisfy the complete inactive-Sync suggestion lifecycle and must not be presented as doing so.** The broader workflow is a documented contract, not a hypothetical feature request. Under the user's fix-all direction, retain a linked, active repair for that contract before claiming overall Suggestions/fresh-install acceptance. Parent has agreed and assigned independent investigation of the smallest complete lifecycle path.

The native metadata now corroborates the source diagnosis: owner 1, Sync storage absent, no owner authority row, no legacy authority row; the metadata transaction was read-only verified at 09:41:08.357. Together with the actual-router official-PG/SQLite 4-failure/4-control causal receipt, this establishes a missing fresh scope path rather than a provider-readiness error. This review inspected that receipt/design; it did not rerun tests or access the native database.

## Contract assessment

- `Docs/API/Notes_Graph_Suggestions.md`, Capability Preflight, promises HTTP 200 with unavailable capabilities for expected provider/FTS/worker conditions. The current scope exception occurs before those checks. Replacing that generic 503 with truthful reads/readiness is valid.
- The design spec's Sync Boundary explicitly states that inactive Sync uses the existing legacy mutation path while preserving the same product invariants. `core/Notes_Graph/README.md` also says inactive omission preserves the legacy product path.
- Actual `build_suggestion_decision_service` returns None when Sync is inactive; the API factory substitutes `_UnavailableDecisionService`. Thus no read-only allowance can make accept/reject/reset and complete generation/review lifecycle available.
- The existing manual Notes endpoint paths already create links and keyword relationships while Sync is inactive (`notes_graph.create_manual_link`, `notes.link_note_keyword`). This confirms an intended legacy product path. However, these direct fallback calls alone are not a complete replacement for suggestion acceptance: its current coordinator provides deterministic mutation identity, a guarded product transaction, acceptance fencing/finalization and replay. Simply invoking an endpoint or bypassing the fence would violate the suggestion contract.

Therefore the new UI disclosure should identify unavailable suggestion decisions, not assert that enabling Sync is a newly required user setup step. A missing implementation path must not be relabeled an intentional operator prerequisite.

## Five-read eligibility

| Store method | Inspected behavior | Disposition |
| --- | --- | --- |
| `load_source_note` | Reads live `notes` with selected-owner predicate and SQL byte cap; distinguishes owned oversize from unavailable. | Eligible. Preserve404/422 behavior and owner binding. |
| `ensure_fts_ready` | Verifies PostgreSQL catalog/index or SQLite FTS table/trigger definitions; performs no repair/DDL. | Eligible. Preserve real FTS/schema errors. |
| `list_suggestions` | Owner+dataset+source+fingerprint filters; joins same-owner/dataset succeeded run; bounded keyset page. | Eligible. Do not fabricate an empty page or expose staged/failed-run rows. |
| `list_suggestion_evidence` | Owner/dataset joins, source fingerprint and succeeded-run gating; reconstructs only live owned current-fingerprint note evidence. | Eligible. Include populated and stale/foreign evidence controls. |
| `get_rejection_set` | Exact owner/dataset/source/fingerprint SELECT; returns None without inserting when absent. | Eligible. Preserve stored counts/revisions. |

These are pure data reads. The PostgreSQL helper's transaction-local dataset `set_config` remains necessary scope setup; it is not permission to create product/authority rows. Keep the existing transaction/lifetime boundaries. A private, default-off helper option used only by these five call sites is reviewable; do not weaken the generic strict validator or propagate the allowance to admission, publication, decisions, maintenance or invalidation.

Eligibility must be checked each time against the exact canonical selected owner and exact server-derived `legacy:<owner>`, with **no authority row for that owner**. An unrelated owner's authority row must not block this owner's fresh read. A conflicting binding for this owner, arbitrary dataset, foreign legacy key or malformed owner must fail closed. Do not cache an unbound authorization decision across later binding. No app-facing request parameter may enable the internal allowance.

## Immutable binding risk

Do **not** insert a legacy authority row as a convenience fix. `note_task_scope_authority` has an owner primary key and shared task/moodboard/studio flags. `moodboard_sync_store.bind` rejects changing an existing owner's dataset and validates complete scope state before binding. Registering `legacy:<owner>` would consume that owner's immutable binding and can prevent later canonical activation for unrelated products. Setting a flag false does not make the dataset mutable.

Existing task storage demonstrates a distinct local-unbound state while preserving canonical authority. It is a useful design reference for the later complete lifecycle, not permission to automatically enroll Sync, rewrite authority or rekey suggestion state inside a GET. Suggestion maintenance and note-change invalidation currently enumerate registered authority datasets; any later legacy write implementation must address those enumerators and eventual canonical transition together.

## Required Option A refinement: truthful actions as well as generation

`generation_available=false` is insufficient when decisions are unavailable. Inspector and Relationships enable Accept/Reject/Reset using `allowed_actions`, independently of generation availability. For a facade with no real decision authority, advertise no unsupported actions. On the strict unbound path every mutation is unavailable, so an empty action list is coherent. Preserve normal registered canonical behavior, including already-supported per-action permission and not-ready checks; do not globally remove actions merely because a provider is temporarily unavailable.

Use explicit factory readiness (`decisions is not None` or an equivalent typed input), not probing `_UnavailableDecisionService` attributes: its `__getattr__` raises. Choose and test deterministic reason priority against feature/worker/provider/FTS readiness. Register the safe reason in the actual strict frontend capability parser and localized inspector disclosure; a backend string alone will otherwise become an invalid response. Preserve provider disclosure/ETag handling and prove direct admission revalidates readiness rather than treating an old ETag as a grant.

## Required controls before accepting the read stage

1. **Actual fresh routes:** official required PostgreSQL and SQLite, actual router→dataset resolver→factory→facade→store, owner note present, no authority rows. Capabilities 200/unavailable and list 200; no Sync storage, authority, run, Job, receipt or product write.
2. **Eligibility and transition:** exact own legacy key allowed; wrong owner's key/arbitrary key/conflicting own canonical binding rejected; another owner's binding irrelevant. Registered canonical and existing exact registered scopes retain their contract. After a real supported canonical binding, stale legacy reads are rejected and canonical reads succeed; no flags or dataset rows silently change.
3. **Owner/source safety:** actual cold/warm selected-owner factory, foreign/missing/deleted source 404, owned oversized source 422, strict scope predicates on both backends; restricted-role PG behavior where applicable, without claiming privileged-role tests prove RLS.
4. **Populated read controls:** published suggestion plus current evidence and rejection state read truthfully; source/target edits, foreign evidence, failed/staged run rows and owner/dataset-bound cursors stay filtered/rejected. An empty-only fixture would miss both read and action-advertising regressions.
5. **Truthful capabilities:** healthy provider+worker cannot produce available generation or usable actions without decision authority; exact expected error reason reaches actual frontend parsing and visible UI. Disabled feature, worker/provider/FTS limitations and registered ready path remain distinct. No false Generate/Accept/Reject/Reset affordance for strict unbound state.
6. **No mutation escape:** real POST admission and each decision family remain denied without enqueue, provider invocation, authority insertion or partial product mutation. Preserve replay/conflict behavior for supported registered paths. Explicit/nested caller transactions, rollback and transaction-local dataset state retain existing ownership.
7. **Real failures:** unexpected DB faults still fail safely; FTS not-ready becomes the intended capability state without silently repairing schema. Source authorization happens before capability/source disclosure.

## Complete lifecycle follow-on boundary

The follow-on should prove the actual inactive-Sync create→one-attempt worker→receipt-gated publication→read→accept/reject/reset path and maintenance/invalidation, with deterministic IDs/replay and transactional fences. Preserve later canonical activation without stealing the immutable shared binding or abandoning outstanding state. Reuse existing product mutation semantics; do not weaken publication receipts, owner predicates or explicit transaction decisions. A focused fake-provider fixture is appropriate for causal backend controls; any promised native model journey remains a separate acceptance step.

No production/test edits, native requests, runtime/browser/config/database operations, git or task updates were performed. This private review and its hash manifest are the only written artifacts. The document gives design approval for a bounded stage, not a GREEN implementation or complete feature verdict.
