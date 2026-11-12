# C03 saved result provenance and export

Associated task: TASK-13260.278.18.83.45. Human approval covers all43 UAT workflow fixes.

## Scope and design
The actual saved C03 run retains its scores but omits evaluated input/expected values and dataset lineage; the UI has no export or history-to-runs action. Capture the actual standard/pipeline execution's evaluation and dataset content hashes in the existing result JSON. Copy input/expected values before evaluator execution and preserve them for success/error/timeout rows. Reuse the recipe content-hash helper and downloadBlob. Show captured dataset identity/version and explicit unavailable provenance for old results. Export the owner-filtered detail response; never export raw provider config or guess missing historical data. History opens the selected evaluation's Runs tab and clears stale run/recipe selection.

ADR required: no. This corrects result inspectability using existing JSON persistence, hash and download conventions; ADR-012 resource IDs and existing ownership filters remain.

## Stage 1: Reproduce missing retention
**Goal**: Real runner and endpoint read regressions fail for absent snapshots; UI regressions fail for absent export/history actions.
**Success Criteria**: Failures are at the missing behavior, not setup.
**Tests**: Saved owner-scoped detail/list after dataset removal/evaluation edits; dataset override identity; actual batch error row input retention; UI saved/legacy detail exports and history navigation.
**Status**: Complete

## Stage 2: Minimal repair and affected verification
**Goal**: Persist actual execution provenance, expose/export it and navigate from history.
**Success Criteria**: New regressions and genuinely affected suites/configured client types pass; scoped baseline/current security has no new findings; independent new-delta review clear.
**Tests**: Real existing runner/progress/read ownership suites, both configured UI suites and typechecks, lint/compile/security.
**Status**: In Progress

## Stage 3: Owned live qualification
**Goal**: One new declared bounded run on corrected owned source, followed by reload/history/export readback.
**Success Criteria**: Two real literal outputs, one pass/one fail; exact IDs/hash/input/expected/actual persisted; one actual JSON export matches the visible saved run. Preserve old run and all paused heartbeat/native/installer/model/other UAT STOPs.
**Tests**: Same owned multi-user SQLite WebUI; one build and one owned source load after fresh guards. No replays of old controls or proof bundles.
**Status**: Not Started

## Current checks and review follow-up
- Real new Python causal red:4failed/23excluded. UI causal red:4failed/4excluded. Minimal runner capture/export/history changes authored.
- Affected UI8passed and WebUI8passed separately. Python58passed plus3official PostgreSQL fixture setup failures before assertions; the owned cluster reports a read-only filesystem. This is not a61-pass/PG acceptance claim and no unchanged replay follows.
- UI configured typing0; Web typing found the new history functional-setter mismatch. Correct to the common URLSearchParams argument contract; retain all navigation assertions.
- Independent first-delta review found a cached previous-principal export window. Three real QueryClient regressions reproduced cached private rows during owner switch/unresolved authority and late former-owner quota publication (3failed/3excluded). The shared detail/comparison query now uses the existing verified connection/principal authority and its owned read lease; prefix invalidations remain. Final affected UI14 and WebUI14 passed separately; both configured client typechecks passed. Independent follow-up source/history/account-boundary review is clear. These mocked authority/HTTP regressions do not establish live browser privacy acceptance.
- Full C03 family and other modes/clients remain open; child TASK-13260.278.5.2 covers only the earlier completed two-example criterion.

## Current build and live gate — 2026-10-09

- Both changed Python files compile; scoped baseline/current Bandit has0 findings/0 errors. Six-file lint reports have0 errors and retained inherited warnings. The comparator command itself failed on warning-embedded line numbers; equivalent signatures were verified from the full reports and affected source. No unchanged fourth lint invocation.
- ONE declared frontend build returned wrapper exit1 after116.126seconds within600. Failure occurred during optimized Next webpack compilation; the returned Node native stack omits the leading diagnostic, so exact cause remains unconfirmed. No speculative application or memory-setting repair.
- All6890 build source inputs stayed equal, next-env was restored exactly and private config was preserved. Token/bundle checks did not run. The build is unqualified.
- The declared owned API/frontend source reloads and one new saved-run/reload/history/export were not executed; the existing runtime is retained. Stage2 remains In Progress; Stage3 Not Started. PostgreSQL read-only setup and full C03/43-family acceptance remain open. No retry/build/window/tab/proof artifact or publication followed.
