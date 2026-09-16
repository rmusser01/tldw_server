# Cycle 5 repair and verification plan

Parent TASK13260. Design: [cycle5 repairs](Docs/Design/2026-09-16-uat-cycle-5-repairs.md). Running tracker: [fresh single/multi UAT](Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md).

## Mandatory entry gate for another full UAT

**Gate: BLOCKED — user reaffirmed repair and verification before rerun on 2026-09-16.** Continue bounded repairs and targeted acceptance checks. Stage 4 cannot begin merely because implementation or automated suites pass.

- Reconcile all 150 unique findings, UAT-001 through UAT-150, against their latest evidence. The recent PostgreSQL checks cover explicit runtime selection, canonical policy validation, sequence ownership, MCP health, Collections bootstrap and cached Chat/Notes dependencies. Historical failure text and stale task status must not substitute for a current disposition. The previously quoted48 formally closed issues is a bookkeeping count, not a complete verified-fix count.
- For each issue, record the repair revision or exact patch identity, relevant regression result, original-scenario acceptance evidence, and any remaining blocker. Separate verified fixes, implemented fixes awaiting acceptance, unresolved defects, and blocked checks. Count reopened issues once and exclude them from verified closures.
- Keep UAT-068, UAT-126 through UAT-135, outstanding hidden-tab acceptance UAT-114, image-recovery acceptance UAT-118, and PostgreSQL validation TASK13260.75 unresolved until their required evidence is recorded. This list does not waive reconciliation of older findings.
- Required PostgreSQL checks must execute successfully through the repository fixtures. A fixture skip, unavailable database, or Docker access failure blocks acceptance; it does not certify PostgreSQL behavior.
- Fix and retest every defect found during targeted acceptance before another full run. Unsupported or inaccessible coverage stays explicitly blocked; it cannot silently become passed or disappear from the requested scope.
- Record a dated gate decision, evidence-backed counts, remaining coverage limits, and the exact source revision before starting Stage 4. Any remaining known defect or required verification gap keeps the gate blocked.

This gate update is tracked in TASK13260. It does not mark any product repair complete or replace the active implementation work.

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
**Status:** Complete

All bounded cycle5 repairs and UAT136's test-only PostgreSQL correction are independently reviewed and committed through `3c30685611`. Targeted native acceptance remains Stage3; these implementation commits do not close native findings.

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
**Status:** In Progress

- Combined shared UI3000/128 and WebUI312/16 pass (overlapping suites are separate runs); full TypeScript retains exactly90 existing diagnostics,0added/removed. Bandit10production Python paths has0findings/errors. Combined backend689pass/1pre-existing streaming-concurrency skip with mandatory PostgreSQL; no PostgreSQL skips. Independent bounded required-PG32backend+2AuthNZ checks pass with0skips on18.6.
- Targeted native acceptance uses preserved server data and new persistent repair browsers after original CLI sessions closed. The native freeze ended15:09:54UTC after134 terminal invalidation. Both owned Next frontends are paused.137/138 are reviewed and committed as010b864500;140 as7c7f4093df.013 and139/141/142 now pass independent review and focused regressions; native acceptance remains pending for all follow-ups. Expanded combined coverage exposed143's six additional failing Chat suites, which must be diagnosed and corrected before a green rerun is claimed.
- Reopened013 after an actual Home source send loses grounding and answers from older history. TASK13260.3 source-context acceptance is not complete; diagnosis identifies the missing effective retrieval flag and implementation is underway. Count013 once within140. UAT095's extra browser-sentinel requirement was independently corrected to its original actual-transport acceptance scope.

- Dated15:10 gate reconciliation:129verified,4older exact acceptance gaps (020/024/031/064),5unresolved (013/137–140),2blocked (114/118),140total. PostgreSQL native matrix remains separately required. Retained native single/multi bundles and independent audit follow-up back these counts.

- Final follow-up frontend verification:3143sharedUItests/150suites and332WebUItests/17suites pass separately; TypeScript retains90existing signatures,0added/removed.143's six stale suites are corrected and independently reviewed. Targeted normal-runtime PostgreSQL setup then exposes144: explicit single-user PG selection is ignored by the pool and bootstrap fails. TASK13260.83 must repair and verify this boundary before native PostgreSQL workflows; the earlier required-PG tests remain valid within their narrower pytest scope.

- Expanded144 verification exposes145/TASK13260.84: three existing PostgreSQL bootstrap fixtures directly insert user rows and are rejected before assertions by the canonical write safeguard. The same failures reproduce on the prior production code. Correct fixtures through the existing canonical helper, retain all scenario assertions and rerun mandatory PostgreSQL controls without skips.

- Follow-up144/145 verification passes42tests with0skips, independent12/12 and4/4. Actual normal AuthNZ initialization passes on a second fresh official PostgreSQL profile without SQLite fallback or test flags. Full API startup then fails146/TASK13260.85 because its validator requires removed legacy media policies. Align validation with canonical current policies, retain missing-policy rejection, verify on real PostgreSQL and repeat normal API startup before browser workflows.

- Both normal PostgreSQL APIs now complete startup and return health200 with reviewed146 source; author56 and independent30 required-PG checks pass0skips. New startup failures147–149 require bounded repairs before workflows: ChaCha/Media initialization deadlock through the wrong keywords table, SQLite-only MCP writable probe, and Collections duplicate-column backfills. Preserve profiles, repair/test/review each unit, then repeat normal startup and original affected subsystem checks. Health200 alone is insufficient acceptance.

- Fresh r3 startup verifies147 warm-up and148/149 initialization. Authenticated MCP health and reading-digest schedules pass in both modes. Actual cached Chat/Notes reads reveal150/TASK13260.89: the dependency runs SQLite PRAGMA on PostgreSQL and leaves an aborted connection. Repair with backend-appropriate liveness/cleanup, required-PG regressions and independent review, then repeat preserved r3 authenticated reads before browser workflows. Original failures remain retained.

- 150 is reviewed/committed and actual repeated authenticated reads pass in both modes.147 remains open after another real restart deadlock reveals reciprocal Media sequence maintenance of foreign ChaCha tables. Retain the valid ChaCha ownership fix; restrict Media maintenance to owned pairs with positive inventory/advancement and negative foreign-sequence/lock controls. Current gate136verified/11implemented-pending/2blocked/1unresolved across150unique findings; no PostgreSQL browser workflow has run.

- At17:11UTC, reciprocal147 is independently reviewed and committed473e17be93. Author/independent137 tests across5 files each pass with0skips; canonical39-pair inventory and scoped Ruff/Bandit are clean. Three consecutive normal startups per PostgreSQL mode pass Media initialization, structured connection/write health and repeated authenticated Characters/Chats/Notes reads. Exact020 pre-provider warning acceptance also passes in the fresh r3 single browser. Current gate138verified/10implemented-pending/2blocked/0unresolved across150findings. Full fresh UAT remains blocked by the remaining acceptance gaps; r3 runtimes stay available for targeted work.

- Stop/restart only identified owned runtimes when source changes require it. Preserve the prior no-restart setup evidence and all profile data.
- At17:37UTC, targeted native pass on1ea5402c83 ends. Exact064 account-switch history and140 warning checks pass; original013 routes correctly through scoped RAG but times out at10seconds while the backend succeeds after19.689seconds. New151/TASK13260.90 tracks stale private deck labels, and152/TASK13260.91 tracks Settings' short generation defaults. Stop the four verified UAT frontends, preserve APIs/data, repair both bounded units with RED/GREEN and independent review, then repeat their original scenarios before another full UAT.
- Serialize real model inference. Do not change product while collecting native acceptance.
- Reconcile Backlog criteria from evidence; retain unresolved coverage limits rather than closing them as native passes.
- Commit working reviewed units with tracking and validation; never bypass hooks.
- PostgreSQL is required acceptance under TASK13260.75 after the user's explicit correction. Resolve official fixture/Docker startup, run affected live PostgreSQL controls with `TLDW_TEST_POSTGRES_REQUIRED=1`, and retain any failures. Earlier skipped checks are gaps, not completed validation.

## Stage 4: Run another full fresh workflow matrix

**Goal:** Recheck the authoritative journeys on new configuration/data/browser state after all confirmed repairs.
**Success Criteria:** Both mode matrices account for every required row on one frozen source, with any newly observed issue tracked immediately. No blanket sign-off while confirmed product failures remain.
**Tests:** The twelve-row named-journey protocol, real model generation, canonical persistence, user/permission isolation, connection recovery and natural expiry controls.
**Status:** Not Started

- Check fetched dev ancestry before freezing; preserve the truthful original-baseline correction.
- Explicitly record and exercise PostgreSQL as well as SQLite backend configurations in the fresh single-user and multi-user matrix. Reused dependencies and test infrastructure remain transparent.
- Reuse dependencies transparently; do not claim clean-machine installation.
- Keep exact Wikipedia and other external/tool limits explicit.
- Preserve full evidence and review the final report. Continue bounded repairs for any new confirmed issue. Remove only this plan when its work is actually complete.
