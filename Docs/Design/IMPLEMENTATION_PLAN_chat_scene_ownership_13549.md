# Qualified scene ownership, original-read Save and accepted bind — TASK-13549

Continue the approved all-issues/remediation/root-cause scope after scene-integrity commit092911f514, on dev4d6a85c4c693c4c83480d0f7c8046cfda0634c55 included by742cb26271. Shared scene/profile shape/version and post-I/O certification are repaired. Bare scene/profile IDs and scratch, Save without original-record comparison, unqualified global projections and missing accepted first-bind remain roots.

ADR assessment: no new ADR. Derived private device locators, existing bucket records/Web Locks and original-read CAS enforce ADR049 and the approved D7 ownership design; ADR059 governs tracking. ADR050 is a no-blind-replay precedent only. Add no public capability/authority, protocol, background replay, new queue or dependency. Search/assessment: Docs/ADR/README.md and accepted049/050. Reassess if implementation requires a new durable authority rule.

## Scope and fixed contracts

- Inventory all eight existing service consumer files plus compact ordinary preparation and independent private-popup routes, actual shared scene/profile service, Actor store/projection, draft transition/accepted receipts, shared bucket and both client exports/types/tests. Reuse original verified H1/controller account, source/draft/view leases, existing credential-free full account hash and exact scope/address.
- Native scene address requires matching verified account, workspace/global scope, exact conversation and accepted H1 owner. Fresh scene requires verified account plus unique fresh address. Genuine local scene requires exact local profile and conversation from its local owner, independently of backend account availability. Character defaults require verified account plus exact card ID; they are shared within that account rather than chat-specific.
- Qualified records use existing durable unpinned bucket CAS; strict reads certify persistence after I/O, preserve malformed/unsupported raw records and reject unknown absence. Retain old bare keys untouched; never silently adopt, migrate or delete ambiguous owner data.
- An editor retains the original hydrated scene record (or certified absence) and a cloned proposal. Save compares that original record under the existing Web Lock; it cannot refresh/adopt a newer record at Save time. Missing locks refuse conditional writes. An explicit character-default Save may read the current default at command entry, retaining its frozen proposal and original lease through CAS. Ensure replacement timestamps exceed the original record to prevent same-clock ABA.
- Atomic applied settings/preview/tokens are stamped by their qualified source and independent account/H1/draft lease. Staged edits and inherited defaults do not become Applied during hydration. Closing an editor retires its operation/presentation, not an independently applied scene. Foreign/retired caches are masked. An unverified profile is unavailable rather than a successful no-default read.
- Freeze original fresh-scene receipt before asynchronous dispatch. Copy only after an actually accepted bind receipt identifies its qualified destination; retain source and refuse independently populated destinations. Identical exact copies may settle idempotently. Switch/review flows cannot infer permission to transfer ambient scene data. Preserve existing unsupported native actor/context refusals.
- All full/compact/native send/startup/editor/default/profile readers and writers must use the captured qualified source; no compatibility fallback to bare IDs may remain in the delivered change. Shared service repair precedes consumer migration; do not claim namespace closure while production callers remain on old storage.

## Stage 1: Shared durable read prerequisite and source qualification
**Goal**: Make the reused storage primitive certify reads, and define original owner/address qualification.
**Success Criteria**: Held strict reads after backend loss reject before presence/absence; advisory/pinned behavior stays compatible. Captured target tests distinguish account/server/auth-source/org/user/workspace/fresh/local/card collisions and reject unavailable/mismatched/retired owners without storage IO.
**Tests**: Actual bucket red/green plus existing claims/recovery/tabs; pure and actual-service source controls.
**Status**: In Progress

## Stage 2: Qualified records and original-read editor commitment
**Goal**: Replace ambiguous scene/profile storage with private qualified records and original hydration CAS across three editors/default controls.
**Success Criteria**: Account/chat/local-profile collisions isolate, original-record conflicts preserve newer data/staged proposal, unknown reads cannot Save, held lease retirement cannot acknowledge, and supported metadata remains compatible. All actual editor/default calls carry captured source/receipt; read fallback does not certify an unverified profile.
**Tests**: Actual service records, held IO/lock/context/ABA, real scene editors and explicit defaults/Retry.
**Status**: Not Started

## Stage 3: Applied projection and dispatch preparation
**Goal**: Keep staged/read defaults separate from owned Applied state; every dispatch prepares from its original qualified scene.
**Success Criteria**: Atomic projection matches current account/H1/address; editor close preserves applied values; staged hydration/Cancel do not publish; replacement/fresh/local/native and independent popup read their actual source. Unsupported native actor/context behavior remains explicit. No bare production read/save fallback remains.
**Tests**: Actor store/actual Form/editor/normal/Compare/native/compact/greeting/startup preparation and source retirement controls.
**Status**: Not Started

## Stage 4: Accepted first-bind transfer
**Goal**: Carry a frozen fresh scene only to its actually accepted new local/native destination.
**Success Criteria**: Original read/owner/address and accepted destination receipt fence transfer; independent destination changes refuse, source remains, duplicate exact accepted completion is idempotent, and switch/review/cross-account/retired binds do not copy.
**Tests**: Actual draft transitions, local/native send/creation acceptance, held source/destination writes, reload and collision controls.
**Status**: Not Started

## Stage 5: Verify, review, commit and continue master queue
**Goal**: Complete this root set with credible shared-service and consumer evidence while preserving remaining all-ten work.
**Success Criteria**: Both client types, affected behavior/static/security/task gates and two independent source reviews pass. Normal incremental commits, owned plan retirement on actual completion. Native/packaged/physical AT/participant evidence is distinct from doubles; remaining master tasks stay current.
**Tests**: Focused causal controls, appropriate affected suites, final type/lint/runtime-equivalent formatting/diff/task checks and bounded native qualification when a current bundle is available. Bandit touched Python only.
**Status**: Not Started


2026-10-09 design reviews: both approved independent reviewers confirm the shared bucket post-read prerequisite. Do not copy ordinary-settings bare local-ID qualification. Original local H1/profile plus address must govern scene IO without a backend account predicate; account verification remains mandatory for defaults. Certified local absence stays editable when optional defaults are unavailable. Compact ordinary preparation currently omits actorSettings and is part of the functional repair. Replacement selection uses an explicit scene or proposed new fresh target, never the departing scene. H1 onLoaded is synchronous void: retain its actual accepted bind receipt there, then await transfer after loadConversation before dispatch. A reviewed retained-input switch cannot inherit the original bind transfer. Independent popup uses its private handle and verified snapshot, not ambient full-page state.

Stage1 prerequisite evidence: held actual bucket read red2 failures/2 positives becomes all bucket claim19/recovery11/tabs3 and settings ACK20/scopedSave34 green87. Existing creation/promotion consumers26 pass, total seven suites113 checks. Both types and two-scope lint delta0 pass; owned formatting2 is runtime-equivalent. Both independent source reviews CLEAR. Evidence prefix `/private/tmp/chat-scene-ownership-13549-bucket-`. Bandit N/A TS/Markdown-only. This repairs strict-read certification, not namespace/CAS integration or historical crash durability/cross-key atomicity; Stage1 and all subsequent scene stages remain open.
