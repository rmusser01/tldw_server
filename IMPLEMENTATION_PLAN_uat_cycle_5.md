# Cycle 5 repair and verification plan

Parent TASK13260. Design: [cycle5 repairs](Docs/Design/2026-09-16-uat-cycle-5-repairs.md). Running tracker: [fresh single/multi UAT](Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md).

## Stage 1: Finish and preserve the frozen cycle5 matrix

**Goal:** Account for every authoritative workflow row in both modes without changing product `ab527eb3b4`.
**Success Criteria:** Each row has passed, failed or blocked evidence; all observed issues are tracked; credentials are excluded; source and dev ancestry are recorded.
**Tests:** Native named workflows, real provider responses, canonical API reads, account isolation, offline/reconnect and natural token expiry. Read-only reproductions may run in private harnesses.
**Status:** Complete

- Both ordinary two-turn Chat/reload and Biology five-card loops pass. Natural multi token expiry passes. Several source/analysis/Retry workflows pass with explicit adaptations and external limits.
- Open repairs:126–135 and reopened068. Preserve source, runtime, private browser and evidence ownership until both executors finish.
- Execution ended around09:32UTC. Root audited both final reports against canonical captures, preserved failures and bounded claims, and verified the unchanged product diff. All four owned runtimes paused; evidence retention and hashes accompany the final checkpoint.

## Stage 2: Implement bounded repairs with permanent regressions

**Goal:** Correct all confirmed cycle5 failures while preserving related successful behavior.
**Success Criteria:** Each unit has a demonstrated failing regression, a minimal fix, passing relevant controls and independent review.
**Tests:** Actual router/handoff tests126; StrictMode/reattach/session tests127; changing multi-card queue and re-rate tests128; canonical auth and tenant tests129; real catalog-shape/model-owner tests130; actual picker/reset/route/loader/ownership tests068; creation/mirror/send identity tests131; actual SQLite FTS fallback132; final provider-bound Retry order133; actual readiness transport/refresh/polling134.
**Status:** In Progress

Expected disjoint ownership after the freeze is released:

- Controller:126 and128 in Flashcards, sequentially to avoid shared test/route interference.
- Existing Chat executor:068 character route transition, then131 greeting identity after diagnosis confirms scope.
- Existing account executor:129 Prompt auth mode, then132 FTS fallback and133 Retry context, as sequential backend units.
- Existing integration reviewer:127 ingest lifecycle, then130 analysis catalog identity,134 readiness refresh and135 outage diagnostics, as sequential frontend units with shared connection ownership.

Assignments are dispatched explicitly only after Stage1 closes. No worker broad staging, commits, runtime changes or inference; controller integrates exact reviewed files. Independent reviewers must differ from the author. A demonstrated scope collision is coordinated before edits.

## Stage 3: Verify repairs together and in the preserved native profiles

**Goal:** Show the actual repaired workflows work together before another full UAT.
**Success Criteria:** Relevant combined tests pass; no new compiler/lint/Bandit findings; each confirmed repair has targeted native evidence and honest remaining limits.
**Tests:** Affected frontend/backend suites, comparison to90 existing compiler diagnostics, scoped lint and Python Bandit, independent review, exact native acceptance from the design.
**Status:** Not Started

- Stop/restart only identified owned runtimes when source changes require it. Preserve the prior no-restart setup evidence and all profile data.
- Serialize real model inference. Do not change product while collecting native acceptance.
- Reconcile Backlog criteria from evidence; retain unresolved coverage limits rather than closing them as native passes.
- Commit working reviewed units with tracking and validation; never bypass hooks.

## Stage 4: Run another full fresh workflow matrix

**Goal:** Recheck the authoritative journeys on new configuration/data/browser state after all confirmed repairs.
**Success Criteria:** Both mode matrices account for every required row on one frozen source, with any newly observed issue tracked immediately. No blanket sign-off while confirmed product failures remain.
**Tests:** The twelve-row named-journey protocol, real model generation, canonical persistence, user/permission isolation, connection recovery and natural expiry controls.
**Status:** Not Started

- Check fetched dev ancestry before freezing; preserve the truthful original-baseline correction.
- Reuse dependencies transparently; do not claim clean-machine installation.
- Keep exact Wikipedia and other external/tool limits explicit.
- Preserve full evidence and review the final report. Continue bounded repairs for any new confirmed issue. Remove only this plan when its work is actually complete.
