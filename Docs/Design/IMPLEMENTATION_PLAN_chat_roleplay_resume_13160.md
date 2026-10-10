# Roleplay Resume implementation — TASK-13160

Continue the accepted snapshot/fenced-completion contract in `Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md` Stages4/5. Reopened chats must consume the saved behavior and authoritative history, preserve input across conflicts, and retain generated content if another writer invalidates saving.

ADR assessment: no new ADR. `backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md` governs this public contract; ADR049/050 govern selected owner and operation lifetime, ADR059 governs task editing. Keep Personal Context authority separate. Latest dev0c53973f17eede4074f4aa7894d0b42322340614 is included.

Reuse the current snapshot/materialized settings store, owner transaction, message writers and immutable owner-only message authority. No parallel conversation store, current-card substitution, implicit branch, or capability inferred from OpenAPI. Preserve legacy requests when the protected contract is absent. Streaming support requires its separately qualified feature; the base contract includes non-stream persistence.

## Stage 1: Exact append and request boundaries
**Goal**: Accept one caller-selected user ID with an immutable original request and complete prior-state precondition.
**Success Criteria**: Identical retry returns the original receipt; conflicting reuse and scope/policy/settings/history/tail drift write nothing. Strict models reject unknown or malformed controls.
**Tests**: Real database unchanged/replay/ancestor-edit/delete/append/settings/missing/invalid/cross-owner controls on supported backends; authenticated route and schema checks.
**Status**: Complete

## Stage 2: Snapshot-only generation
**Goal**: Freeze original settings, all behavior resources and exact branch history inside the owner transaction before provider dispatch.
**Success Criteria**: Source-card/preset/lore/exemplar/memory changes or deletion cannot alter provider inputs; all preconditions and exact acknowledged input match. Overrides fail before dispatch.
**Tests**: Real stored behavior with current-source loaders forbidden; every pre-dispatch fence mismatch; retained original provider/model; bounded history/assets.
**Status**: In Progress

## Stage 3: Fenced commit and exact capability
**Goal**: Persist generated assistant text only against the original complete fence and advertise only verified required features.
**Success Criteria**: Concurrent mutations preserve generated text with saved=false and no assistant/branch. Unchanged commit returns stable identity/parent; active Sync remains fail-closed. Optional stream grants cannot use client-known credentials.
**Tests**: Blocked provider plus second-connection mutation; rollback and idempotent persist; authenticated capability; strict grant/owner/context controls if stream feature is enabled.
**Status**: Not Started

## Stage 4: Shared consumers and native qualification
**Goal**: Wire both chat layouts to the exact server contract and complete real character-card reopen/continuation journeys.
**Success Criteria**: No mutation before original scoped capability/readiness proof; proposal/recoveries and scene/resource settings stay honest. Affected tests, types, static/Bandit, independent review and normal hooks pass.
**Tests**: Full/compact shared callers, imported PNG/JSON cards, greeting/two-turn/reopen; exact runtime requests and saved state. Packaged/physical AT/uncoached participant evidence remains separately qualified.
**Status**: Not Started
