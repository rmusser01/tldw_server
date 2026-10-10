# Atomic edited-target admission — TASK-13569

Protect Edit and resend against changes to the excluded user row between the final client capture and admission. The retained ancestry already has H1 validation; this unit checks the target that a before-message selection deliberately excludes. It preserves the original message graph and the edited proposal on refusal.

ADR assessment: ADR-049 governs selected-source ownership, ADR-050 governs immutable operation receipts, and ADR-059 governs Backlog tracking. No new ADR is required: the strict optional version1 precondition enforces existing ownership and transaction rules without introducing a new owner, storage table, queue or recovery architecture. Reassess if implementation changes those rules.

Scope: history wire and ordinary/character request schemas; both owner-transaction admission methods; ordinary and character endpoint/service forwarding; exact server capability detection; captured Edit intent through ordinary, character and compact shared callers; OpenAPI/API documentation; real SQLite/PostgreSQL, authenticated endpoint, client and schema tests. Reuse opaque message revisions and the existing owner lock. Do not replace ancestry-tolerant generic H1 validation.

## Stage 1: Qualify the excluded-target race
**Goal**: Reproduce target-only changes after capture using the real database and authenticated admission paths.
**Success Criteria**: Unchanged controls admit; content, metadata/image, role, parent, deletion and edit-Undo changes refuse before row insertion or provider dispatch. The original graph and proposal survive.
**Tests**: Existing DB admission fixture with real update_message after capture; ordinary/character HTTP routes with provider-call assertions; invariant cases for target and retained-parent combinations.
**Status**: Complete

## Stage 2: Enforce the typed precondition inside admission
**Goal**: Carry a strict optional edited-target precondition into both existing owner transactions.
**Success Criteria**: Live settled user ID and original opaque revision match before append; that revision binds the raw parent. The validated interpretation binds the logical predecessor, which can differ from raw parent for reviewed legacy rows. Protected operation requires an explicit branch and the corresponding before-message selection. A committed existing-ID replay compares immutable original intent including the precondition before fresh target validation; conflicting intent refuses. Generic Send, Branch and Regenerate stay valid.
**Tests**: Real transactional red/green tests, schema validation, replay after later edits, rollback and concurrent mutation controls; official PostgreSQL fixture when available.
**Status**: Complete

## Stage 3: Require qualified client support
**Goal**: Send the retained precondition through all protected Edit consumers only on capable servers.
**Success Criteria**: Exact primary schema capability is required before journal or mutation. The original target facts survive preparation; older servers give a useful refusal and retain edited input. Ordinary and character payloads carry the same typed contract; no unguarded fallback is added.
**Tests**: Shared capability positive/negative schema cases; actual ordinary/character mode dispatch and held-source tests; both client compilers.
**Status**: Complete

## Stage 4: Verify and continue the finite UX queue
**Goal**: Review, document and commit the bounded root fix, then continue Roleplay/D7 and persona walkthroughs.
**Success Criteria**: Affected tests, both client types, static checks, Bandit for touched Python, canonical task/diff checks and independent review pass. Latest dev ancestry is verified. Runtime, packaged, physical AT and participant evidence remain explicitly qualified.
**Tests**: Appropriate connected DB/API/client suites, changed-scope Bandit and lint, normal commit hooks, current native Edit journey when the current bundle is available.
**Status**: In Progress

Implementation begins after the current TASK-13549 consumer unit is verified and committed. This plan records the approved next issue; no test result or completion is claimed here.


2026-10-10 scope refinement after both independent read-only reviews: use exactly `{version:1,message_id,message_revision}` as the optional strict precondition. Require an explicit branch and matching before-message selection; original H1 already binds owner/conversation/ancestry, so do not duplicate raw parent or change the H1 digest. Reflect original conversation/id/revision/selection_digest in the typed definitive no-admission refusal and match all echoes in ordinary, Character and saved-chat write tracking. Strip the private field from provider parameters and normalized extra body/header controls. Require the actual POST primary JSON schema before journaling; old-server unsupported remains explicit.

The shared public metadata writer can restore the old opaque revision on same-timestamp A→B→A. Repair its existing message version on changed existing semantic metadata under the same owner transaction; preserve initial admission/settlement revision1, true no-op, API pin edits and Sync/import writers that already own versioning. Qualify the PostgreSQL merge placeholder separately. Metadata/role races must use real mutation paths: update_message ignores sender, and API pinned edits also change settings, so neither is a valid isolated role/metadata proof. Causal authenticated admission tests are distinct from missing schema/signature prerequisites.

ADR required: no; ADR049/050/059 govern this bounded enforcement of the existing selected-source and operation lifetime rules. Latest fetched dev0c53973f17eede4074f4aa7894d0b42322340614 is included. Current Scene commit and bounded bundle are verified; this unit now begins with real failure controls. No implementation result or overall completion claimed.


Verification checkpoint: backend261 passed/104 official PostgreSQL skips; client577 passed, with final263 passing after diagnostic repair; reviewed legacy final8 passed/8 skips. Both client compilers exit0,20 TS scopes introduce0 lint findings,10 Python scopes introduce0 Ruff findings. Bandit retains3 baseline findings and adds only required pytest assertion reports; no new production or other test security finding. Both independent source reviews CLEAR. Current native Edit and packaged/physical AT/participant evidence remain pending; Stage4 and overall acceptance remain open. Evidence prefix `/private/tmp/chat-atomic-edit-target-13569-`.
