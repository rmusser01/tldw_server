# Character setup preflight and ownership transition plan

**Task/spec:** TASK-13454; approved Chat UX design, Stage 1. Draft association/recovery and greeting-only commitment remain separate required deliveries before Stage 1 completion.
**Goal:** Validate the full proposed native character setup before changing the source conversation, and give Apply callers a fenced destination at the actual transition boundary.
**Baseline:** origin/dev `75ab224081bf140ef52017c1a9b0a04f6878d488`; branch HEAD `37fea2a173`.
**Architecture:** Extract the native sender's configuration checks; retain input/admission/active-owner checks at dispatch. Project existing preset/prompt/model/context/scene inputs before root Apply. Reuse the model store's pure commit normalization for projected target settings, including undefined/global inheritance; activate the target scope at acceptance before existing settings setters. Reuse controller reset, qualified route replacement, source/action/account guards and existing setters. Add a narrow destination callback to the existing Apply contract so asynchronous preflight cannot make scene saving target the source. The live tracked picker delegates to the same root; staged and overlay selectors retain their semantics. No new conversation/settings framework or backend API.
**ADR assessment:** No new ADR. Existing ADR-049 ownership and accepted native snapshot limits remain in force. Client scene/behavior extensions stay blocked until TASK-13160 supplies their supported contract.

## Stage 1: Reproduce unsafe setup boundaries
**Goal:** Show missing local-source detachment, preflight-before-mutation, full saved-setup projection and deferred destination capture.
**Success Criteria:** New cases fail on those behaviors before production changes; current relevant suites establish the baseline.
**Tests:** Native history-actions, mounted PlaygroundForm, prompt-template, real Drawer and AssistantSelect behavior suites.
**Status:** Complete

- [x] Run the relevant existing suites.
- [x] Add bounded regressions through real root Apply and Drawer/picker boundaries; run red.

## Stage 2: Share validation and project intended setup
**Goal:** Native sending and setup reject the same unsupported configuration.
**Success Criteria:** Proposed model/provider/settings/scene/context are checked without interpreting a local source as the destination's owner; invalid setup changes no source state.
**Tests:** Existing native guard matrix plus setup-introduced unsupported fields/model/provider/context.
**Status:** Complete

- [x] Extract configuration validation; preserve captured store/tool identities, explicit-zero/default handling, user-input and active-owner send guards.
- [x] Resolve saved prompt/preset/model/context/scene before identity Apply and use that projection for the accepted commit.
- [x] Fence deferred preparation with source, caller lifetime, model/tools and captured account scope; retain drafts/source on failure/cancellation. Temporary source content remains protected rather than silently converted.

## Stage 3: Commit a fresh owner and report the destination
**Goal:** A valid local-to-character transition retires only its source selection; asynchronous Apply saves scene to its destination.
**Success Criteria:** Controller reset and IDs agree, callers receive a current destination before subsequent awaits/rendering, and the live tracked picker cannot bypass preflight.
**Tests:** Owned local source transition, stale preparation, matching identity/overlay, delayed Drawer Apply, saved setup and direct picker acceptance/rejection.
**Status:** Complete

- [x] Reuse the qualified route replacement and active controller reset without invoking the generic clear's model-reset/overlay-destruction effects.
- [x] Replace early destination snapshots in Drawer/startup-preview Apply with a notification at the root commit boundary; preserve existing synchronous callback compatibility.
- [x] Route live tracked picker through root Apply and wait for acceptance before closing; preserve staged/overlay paths.
- [x] Keep source text visible/recoverable while the per-conversation switch/bind draft delivery is unfinished; do not claim successful-transition source-draft association closed by this unit.

## Stage 4: Verify, review and record
**Goal:** Reviewed incremental change with preserved native guards.
**Success Criteria:** Relevant suites/typecheck pass, touched lint adds no errors/warnings, two reviews resolve important findings, working local commit and canonical task notes.
**Tests:** Complete affected suites, WebUI typecheck, baseline-relative lint/formatting, diff checks and Backlog normalization. Bandit not applicable to TS/docs-only scope.
**Status:** Complete

- [x] Run checks and independent reviews using the existing two authorized reviewers.
- [x] Archive the completed mode-cancellation plan in `37fea2a173`, remove only that plan, record verification and commit.
- [x] Continue draft switch/bind, uncertain-create recovery, greeting commitment and the remaining approved stages; keep broader acceptance criteria open until their evidence exists.

## Verification and limits

Six affected suites pass (165 tests), including native send guards, root cancellation/configuration/account drift, real scoped settings inheritance, Drawer destination saving and picker lifetime. WebUI typecheck passes. No new touched lint errors or warnings; existing baseline warnings and formatting retained. The new validator passes Prettier. Diff checks and canonical task normalization pass. Both independent reviewers found no remaining Critical/Important issue in this unit. Bandit is not applicable to TS/docs-only changes. Live backend and extension journeys are still required in Stage 5. This unit keeps source text visible; per-conversation association and retained fresh/temporary recovery handles are the next delivery. Temporary transcript conversion, uncertain-create recovery and greeting-only persistence are not claimed fixed. Broader task acceptance criteria remain open.
