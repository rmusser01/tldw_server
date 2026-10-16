# Shared UI coverage recovery — TASK-13817

Baseline: origin/dev bfafa07604. Fix the existing native package coverage path without removing tests or weakening production guards.

## Stage 1: Reproduce and trace
**Goal**: Identify fixture errors, per-test deadlines, and suite runtime separately.
**Success Criteria**: Reproductions and native test manifest retained privately; minimal causes documented in task.
**Tests**: Research import full file; Skills full file; roleplay failures; full coverage shard timings.
**Status**: Complete

## Stage 2: Repair roots
**Goal**: Repair missing shell context and asynchronous test boundaries; make complete coverage execution fit bounded runners.
**Success Criteria**: Existing behavioral assertions remain; no test exclusions or arbitrary deadline inflation; coverage includes all source files.
**Tests**: Red/green reproductions; full native coverage across disjoint shards and merged summary.
**Status**: In Progress

## Stage 3: Review and integrate
**Goal**: Independent review, new PR, latest-dev integration, passing checks and requester Change summary before merge.
**Success Criteria**: Review findings addressed; every coverage shard and required status passes; merged commit verified.
**Tests**: Relevant lint/type/CI validation, security validation for touched executable scope.
**Status**: In Progress

ADR check: ADR required: no. This repairs fixture shell context and the existing coverage runner's execution layout, preserving package ownership, full source scope, test isolation, and advisory coverage percentages. No public API, persistence rule, or merge policy changes. ADR-059 governs task editing.

Evidence: research import baseline 7/68 failures traced to missing Router; corrected 68/68 pass under native coverage. Skills server-switch reproduces a 5,000ms timeout without coverage; all 83 cases pass with the existing required shard budget of 15,000ms. Roleplay fails different cases near 5,000ms across local/CI runs. Native discovery finds 2,519 test files; monolithic CI completed 30 before its 900s cap. Coverage is split by Vitest's disjoint hash shards; native blob merge is guarded against missing reports. A measured native eighth had completed only 150/315 files after 9m32s, leaving insufficient CI margin; use sixteen shards with at most eight runners at once. Native blob export/merge smoke passes (57 tests, full uncovered source denominator retained). First hosted full coverage completed all 2,519 files, exposing stale fixtures across previously unexecuted suites. Those fixtures are repaired without production changes; final hosted 16-shard rerun and merged report remain required before merge. CPU profiling traced pathological selector work to jsdom 28; compatible pinned jsdom 29.1.1 reduced representative coverage suites from 76s/214s to 5s/33s. A 16-input native blob merge memory smoke passed with the full source denominator retained. Workflow admission and immutable-head contracts pass (46 tests).

Final dependency assessment: jsdom 29's rewritten computed-style inheritance caused deep accessible-name queries to exceed existing deadlines; jsdom 26 avoids that rewrite but regresses bulk selectors and motion capability. Both were rejected. Preserve pinned jsdom 28.1.0 and its original CSS dependency graph, updating only DOM selector to 7.1.1 via an explicit workspace override. The called selector APIs remain compatible and its Node 20.19 floor matches jsdom's existing requirement. A clean frozen install verifies actual selector 7.1.1 and cssstyle 6.0.1; existing CSS calc serialization passes. Full native coverage and compatibility checks remain mandatory. The complete run also exposed an ordinary apiSend mutation refresh/replay regression; preserve current account guards but restrict refresh/replay to the guarded readiness GET, with a successful-refresh adversarial regression test. ADR required: no; this restores existing documented transport behavior rather than changing its architecture.

Root fix selected after the selector-only run still timed out: a native Bun package patch backports jsdom 29.1.1's conservative stylesheet subject prefilter into jsdom 28.1.0. Use selector 7.1.1's string-keyed subject extraction cache instead of a per-rule cache, preserving selectorText mutations; XML and quirks retain full matching. Regression red: 18/19 pass, 1,830 matcher calls for 500 unrelated rules. Patched green: 19/19, only 57 calls; computed style, hidden inheritance, logical/grouped/escaped selectors, CSSOM mutation, XML and quirks behavior retained. No global CSS or accessibility mocks. This avoids both the original stylesheet matching hot path and jsdom 29's computed-style inheritance regressions. Keep the original CSS graph; all dependency behavior must survive frozen installation and full hosted coverage before merge.

Verification: patched root coverage 163/163 passes (123.22s), including Skills83, prompt settings51, WorldBooks bulk3, Quiz3, history1 and prefilter19. Original WorldBooks entry fixture additionally passes3/3 under coverage; the temporary paste fixture was removed because the central fix is sufficient. Unchanged compatibility coverage136 passes/2existing skips across16files in67.59s versus207.64s without prefilter. Review-expanded semantic prefilter regression24/24 passes. Independent patch review found no Critical/Important issue; upstream MIT notice retained in apps/patches/README.md. Final44edited-suite combined coverage and full hosted16-shard rerun remain to verify before merge.
