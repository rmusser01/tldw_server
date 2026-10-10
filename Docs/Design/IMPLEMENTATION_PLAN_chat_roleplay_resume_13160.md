# Roleplay Resume implementation — TASK-13160

Continue the accepted snapshot/fenced-completion contract in `Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md` Stages4/5. Reopened chats must consume the saved behavior and authoritative history, preserve input across conflicts, and retain generated content if another writer invalidates saving.

ADR assessment: no new ADR. `backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md` governs this public contract; ADR049/050 govern selected owner and operation lifetime, ADR059 governs task editing. Keep Personal Context authority separate. Baseline dev0c53973f17eede4074f4aa7894d0b42322340614 is included; refreshed origin/dev3b68f2dbc2 will be integrated after the qualified foundation commit.

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
**Status**: Complete

## Stage 3: Fenced commit and exact capability
**Goal**: Persist generated assistant text only against the original complete fence and advertise only verified required features.
**Success Criteria**: Concurrent mutations preserve generated text with saved=false and no assistant/branch. Unchanged commit returns stable identity/parent; active Sync remains fail-closed. Optional stream grants cannot use client-known credentials.
**Tests**: Blocked provider plus second-connection mutation; rollback and idempotent persist; authenticated capability; strict grant/owner/context controls if stream feature is enabled.
**Status**: In Progress

## Stage 4: Shared consumers and native qualification
**Goal**: Wire both chat layouts to the exact server contract and complete real character-card reopen/continuation journeys.
**Success Criteria**: No mutation before original scoped capability/readiness proof; proposal/recoveries and scene/resource settings stay honest. Affected tests, types, static/Bandit, independent review and normal hooks pass.
**Tests**: Full/compact shared callers, imported PNG/JSON cards, greeting/two-turn/reopen; exact runtime requests and saved state. Packaged/physical AT/uncoached participant evidence remains separately qualified.
**Status**: Not Started

Execution detail for Stages 3/4:

1. Overlay the verified v1/four-feature capability only on canonical authenticated owner discovery. Keep public docs intact; use both actual route policies and actual registered handlers, expected-user assertion and no-store. No invented Resume toggle or affirmative stream feature.
2. Extend existing scoped capability discovery rather than inferring support from OpenAPI. A write proof uses its original ServicePromptRequestScope and bypasses cache/in-flight reuse; generic anonymous discovery remains available. Preserve strict numeric/feature checks and scope retirement errors.
3. Preserve and strictly validate the existing detail snapshot/eligibility/settings/history/tail projection. Extend the actual Object.assign-installed chat API methods and only the required scoped paths. Freeze original caller IDs and exact append/complete bodies before handoff; no current card/model/sampling/Actor overrides in protected completion.
4. Add the nonstream protected path at the shared Full/Compact/native/private character seam. Require the accepted H1 view to match the Resume tail/path; refuse Edit/regenerate/branch adaptation. Keep original source/account/command retirement and canonical follow guards. Read-only historical identity must survive current-card deletion.
5. Extend the existing recovery record with genuine Roleplay intent/receipt, preserving H1 admission as a distinct authority. Pre-handoff refusal retains input; acknowledged append plus completion refusal is accepted_unsent; generic failure/Stop/malformed response remains uncertain; saved=false retains generated text; saved=true requires exact original IDs/fences before following canonical rows. Retry replays the original operation, never adds another user or silently branches.
6. Qualify actual scoped requests, two turns/reopen after source changes/deletion, malformed/cross-context receipts, held discovery/detail retirement, Stop/New/account changes and saved/unsaved recovery in both layouts. Refresh the owned backend/bundle and reuse the single existing browser session; preserve original drafts and unrelated conversation data.

Existing original-scope D7 settlement/reattachment/import/view-only/Sync and extension optional loading remain separate bounded units after this contract. Packaged/physical AT and uncoached-participant timing are not inferred from source or automated tests.
