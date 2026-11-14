# B07 saved report source counts and manual-monitor health

Associated Backlog task: TASK-13260.278.18.83.45. Root recorded this distinct unit before edits.
Workspace: uat-native-regenerate-20261008/tldw_server2, c700e42.
Scope: outputMetadata.ts and watchlists-overview.ts, plus two new behavioral test modules. The four extraction-unit paths remain frozen.
Root owns runtime, browser, profiles, combined types/build and live recovery.

## Diagnosis and design

The automatic briefing producer retains per-item provenance with numeric source_id values but does not emit source_count. Reports currently default that missing count to zero. Preserve any explicit count accepted by the existing count validator, including zero and digit strings. Otherwise count distinct positive safe numeric source IDs from the actual provenance array. Ignore malformed entries and never derive selected-source counts from source_counts status totals. Do not rewrite outputs or invent alerts, snapshots or readiness.

The public WatchlistJob contract supports a manual active monitor without schedule_expr or next_run_at. The wizard returns no schedule for manual mode; backend create registers only jobs with an expression. Existing wf_schedule_id can also identify a supported scheduler link. Overview currently flags every active monitor without a parseable next-run time. Require a nonblank expression or scheduler link before flagging absent/invalid next_run_at. Preserve inactive behavior and source/run/delivery/audio failure totals.

ADR check: ADR required: no. Searched Docs/ADR/README.md and traced the current public job and saved provenance contracts. These are display corrections within existing contracts, with no changed persistence, scheduling policy, module boundary, authority or wire format. No new durable architecture decision; root owns the associated task note.

## Stage 1: Causal display regressions
**Goal**: Demonstrate incorrect counts and manual-monitor attention on unchanged source.
**Success Criteria**: Normal configured tests reach the real metadata helpers and overview aggregate service; controls preserve explicit counts, scheduling links and other health signals.
**Tests**: Retained automatic provenance, duplicates/malformed IDs, explicit counts, legacy readiness; manual/scheduled/linked/inactive monitors and source/run/output failure totals.
**Status**: Complete

## Stage 2: Minimal display corrections
**Goal**: Render authoritative source totals and distinguish manual monitors from broken schedules.
**Success Criteria**: Causal tests and genuinely affected old helper/service consumers pass; no backend or saved-output changes.
**Tests**: New behavioral modules plus existing outputMetadata and watchlists-overview modules, normal UI/Web configurations.
**Status**: Complete

## Stage 3: Freeze and independent/live qualification
**Goal**: Hand off a reviewable display-only correction.
**Success Criteria**: Scoped baseline/current ESLint and syntax, independent source review, root combined types/build and actual corrected display recovery.
**Tests**: Installed Node20.19.5/Vitest4.0.18, normal configs and 8192MB; outer600 guard. Bandit N/A: TypeScript-only scope.
**Status**: In Progress

## Attempt ledger

Maximum three failed issue attempts, then stop and reassess. Distinguish operator/fixture failures from causal failures. Capture bounded natural exits/results in memory only; no logs/proof artifacts. No unchanged unrelated tests or whole types/build/runtime actions.

Pre-test static operator failure: Node synchronous stdin read returned EAGAIN before lint. Switched only the in-memory wrapper to normal async stdin reading; corrected configured baseline ESLint completed naturally0 in2.446s, both production paths0errors/0warnings. Config prints inherited missing root pages-directory advisory; no source lint finding. Complete pre-unit bytes/hashes retained in memory.

Causal attempt1: unchanged production natural1 in1.912s outer /1.45s Vitest; 12failed18passed across30new cases. All failures reached intended source-count/manual-attention assertions; no setup/fixture failure. Seven provenance count failures returned0 instead of1/2, and five manual monitor failures added one false job/health warning. Eighteen explicit-count/fallback refusal/inactive/scheduled-link/valid-next-date controls passed. Root accepted the exact two-function correction. Preserve those controls and qualify affected old consumers in the corrected-source phase.

Corrected-source UI affected command: natural1,4.546s outer /3.97s Vitest,56passed4failed across60. All15 count cases,11 overview controls and all30 existing affected helper/service cases passed. Four new manual cases reached an incorrect additional oracle: existing getEarliestActiveJob returns the active manual monitor with nextRunAt:null/timezone:UTC, not null. The baseline four mixed assertions are not clean sole source-red claims; the separate baseline other-health case independently proved false jobs attention, and retained-provenance failures remain causal. Correct only nextActiveJob expectation to the actual monitor identity, preserving every attention/status/scope/count assertion. Production unchanged. Third bounded command will run only those four failed parameter cases; retain56 successes. No full UI replay. First Web qualification remains open.

Focused fixture-only completion: only the four previously failed manual expression variants passed, natural0,1.879s outer /1.36s Vitest. Eleven other overview cases were intentionally filtered by the selector, not rerun. Retain56 prior corrected-source successes. UI evidence is60distinct passes split56+4, not one60-case green command.

First Web affected command:60/60passed,4files, natural0,2.912s outer /2.23s Vitest; no filtered/skipped cases. Scope is the two new modules30cases and two existing affected modules30cases. Existing tests unchanged. Exact normal positional filters retained in memory (Vitest substring matching resolves the two old src/... tails to the shared UI modules).

Static qualification: configured ESLint baseline both production paths0errors/0warnings; current four TS paths0errors/0warnings, natural0 in3.227s. Same inherited missing root pages advisory. TypeScript transpile syntax four paths0diagnostics, natural0 in0.308s. Bandit N/A: only TS code/tests changed. No whole semantic typing/build/runtime run by this agent; root owns those final gates.

Frozen source scope: outputMetadata.getSourceCount alone now preserves explicit accepted source_count, else counts unique valid positive safe numeric source IDs from retained provenance. Overview job attention adds only active scheduled-expression/link admission before the existing next-run parser. No new backend rule, count math from status totals, alerts, snapshot/readiness, saved output mutation or scheduler policy. Full new30 bodies self-reviewed against actual producer/job/aggregate contracts. Both existing modules unchanged. Scope limits: helper/real aggregate service with external API seams, not rendered browser/HTTP/auth/backend scheduler execution or full page layout acceptance. Root independent source review and actual corrected-build recovery remain open.

The four preceding arXiv extraction paths retain their exact frozen hashes; no extractor or root task mutation in this unit.
