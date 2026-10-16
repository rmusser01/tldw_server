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
**Status**: Not Started

ADR check: ADR required: no. This repairs fixture shell context and the existing coverage runner's execution layout, preserving package ownership, full source scope, test isolation, and advisory coverage percentages. No public API, persistence rule, or merge policy changes. ADR-059 governs task editing.

Evidence: research import baseline 7/68 failures traced to missing Router; corrected 68/68 pass under native coverage. Skills server-switch reproduces a 5,000ms timeout without coverage; all 83 cases pass with the existing required shard budget of 15,000ms. Roleplay fails different cases near 5,000ms across local/CI runs. Native discovery finds 2,519 test files; monolithic CI completed 30 before its 900s cap. Coverage is split by Vitest's disjoint hash shards; native blob merge is guarded against missing reports. A measured native eighth had completed only 150/315 files after 9m32s, leaving insufficient CI margin; use sixteen shards with at most eight runners at once. Native blob export/merge smoke passes (57 tests, full uncovered source denominator retained). Hosted full coverage remains to verify before merge.
