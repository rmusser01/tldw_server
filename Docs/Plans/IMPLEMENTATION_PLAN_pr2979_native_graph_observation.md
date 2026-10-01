# Native retained Prompt Studio graph observation implementation plan

> For agentic workers: use the existing isolated worktree and execute the single diagnostic unit with test-first checks and independent immutable review. Root owns tasks, documentation and Git.

**Goal:** Observe actual tracked-object reference structure missing from UAT569, without correcting or accepting the final native shutdown gate.
**Architecture:** Reuse the existing stdlib observer. Stream a tracked-object graph before the first original unconfigure call and during the earliest registered ordinary atexit callback, after later application callbacks and before native module clearing. Emit anonymous dense node indices, type labels and directed tracked-node indices; only attested loaded source-module names label roots. Release temporary object references before cleanup proceeds. No application/dependency/workflow changes.
**Tech stack:** Python3.12 stdlib gc/json/types; existing pytest synthetic controls and shell-free native sampler.
**Spec:** Authorized heartbeat retained graph/type/referrer ownership observation and in-chat bounded design, 2026-10-01; TASK-13260.278.18.83.59.

## Constraints and disclosure

- Tested source immutableb38; existing diagnostic ci.yml/actions/env/pytest/plugins/warnings/case300s/job60m unchanged.
- Do not call collect/disable/enable/freeze/set_debug/set_threshold; do not alter callbacks/configuration to bypass GC or cleanup.
- No values, keys, repr, locals, exception messages or raw addresses in graph artifacts. Dense IDs are only within each snapshot; cross-snapshot identity is unavailable.
- Trusted exported Python class/module symbols only; unexported/dynamic type symbols remain anonymous. No custom object attribute access or iteration for graph traversal: use C tp_traverse through get_referents.
- gc.get_objects holds temporary strong references; capture allocations can perturb automatic GC. Measure overhead and release all temporary references before original cleanup. Diagnostic result never substitutes uninstrumented acceptance.
- Graph only includes collector-tracked nodes and C tp_traverse edges. Untracked/extension references and state after native module clearing are unavailable. A snapshot shows structure, not a proven continuous native stall or causation of previous events.
- Seven original synthetic test bodies/assertions preserved; original delegate result/exception/status/ownership/native-sampler/raw-privacy behavior retained.

## Stage1: Evidence and isolated tracking
**Goal:** Preserve source54 and officially track before edits.
**Success criteria:** Native UAT569 evidence reviewed; official duplicate search empty; freshrefs and new recovery; same logical child59 created through official CLI in isolated/root corpora.
**Tests:** Freshrefs/clean source/task existence.
**Status:** Complete

## Stage2: Causal synthetic and minimum graph capture
**Goal:** Observe a real retained cycle with anonymous edges while preserving cleanup.
**Success criteria:** Original observer fails the new actual-launcher missing-graph assertion; new capture records the real cycle and source roots; privacy sentinel/raw IDs absent; weak-reference instance releases after capture; unavailable artifact/read failures preserve original7/RuntimeError/SystemExit11.
**Tests:** Add test_native_graph_records_private_cycle_without_values first, run against source54, then implement graph_snapshot(phase) and limited initialize/first-unconfigure wiring. Add privacy/no-retention/fault controls; run all seven original synthetic checks plus new cases without app imports/whole Prompt/PG.
**Status:** Complete

### Single reviewable diagnostic unit
- [x] Add an actual-launcher retained SimpleNamespace cycle test and assert complete graph artifacts at both fixed phases, a two-node cycle with real type labels, dense indices and private sentinel absence.
- [x] Run focused source54 causal red and retain natural exit/log/XML.
- [x] Implement graph_snapshot(phase: str) for fixed phases pre-unconfigure/pre-module-clear only, using gc.get_objects/get_referents, bounded source-symbol labels, streamed node rows/footer and incomplete status. Guard failures and release lists/maps before delegating; preserve original functions outside initialize/pytest_configure.
- [x] Add a poisonous repr/private dynamic symbol control and weakref release assertion; add module-local graph-read/write failure controls preserving all original cleanup statuses and unchanged GC configuration.
- [x] Run complete diagnostic synthetic module, retain prior failures and actual natural controller/pytest completion. No whole-local Prompt.

## Stage3: Preservation, static checks and independent review
**Goal:** Qualify exact immutable candidate.
**Success criteria:** Compile/Ruff/diff pass; Bandit findings/errors retained and dispositioned; original7 test ASTs, unchanged helper bodies and exact workflow/setup/env/budget equivalence verified. Independent source/evidence review clear.
**Tests:** AST/hash/source assertions; stdlib compile; Ruff/private cache; Bandit JSON; existing actionlint with shellcheck integration disabled; immutable patch/manifest hashes and review.
**Status:** Complete

## Stage4: Separate publication and one native observation
**Goal:** Observe actual native retained graph before any corrective edit.
**Success criteria:** Freshrefs/recovery/reviewedhash verified, expected-absent separate branch publication, one explicit ci.yml dispatch on exact diagnostic commit/frozenb38. Actual graph/parity/natural exit OR original maximum retained with missing information disclosed; ownership/causal/final gates remain open until actually proven.
**Tests:** Exact run/job API, source/helper/config/version1190case/95skip parity, graph phase completeness/types/reference structure/capture overhead, original timeout or natural exit.
**Status:** Not Started

No ADR: diagnostic observation preserves existing architecture. Official Python3.12 documentation: https://docs.python.org/3.12/library/gc.html .

## Local qualification evidence — 2026-10-01

UAT570 local graph qualification: source54 causal red fails only missing graph; synthetic child natural0. Final15 synthetic cases pass/zero skips/errors/failures49.02s; actual controller/pytest natural0/source unchanged. Seven original cases/four original test function ASTs and seven original helper ASTs preserved; only initialize/pytest_configure wiring changes. Fixed pre-unconfigure/pre-module-clear snapshots stream anonymous tracked tp_traverse adjacency, static source-attested type/module labels, incomplete status/capture overhead; private values/raw addresses/custom-metaclass hooks excluded, temporary extra instance references released, GC configuration/callbacks unchanged. Compile/Ruff/diff/unchanged-workflow Actionlint0 (shellcheck integration disabled). Bandit actualexit1/59LOW48B101+2B404+9B603/zero errors/MEDIUM/HIGH; new21B101+3B603 are synthetic assertions/fixed shell-free trusted interpreter scripts, no suppression/security-pass claim. Retain initial controls-red/14pass1fail symbol-label precursor/initial Ruff findings, corrected final checks; private AST-format/directory/read-only artifact guards excluded from CI/source/test failure. Immutable source patch /private/tmp/pr2979-uat570-candidate-v1.patch SHA53d0f33186ba201fc5bc78eb229334204e9dfa11ae9e1b21cb1a1b402539159e; root verified158-artifact manifest /private/tmp/pr2979-uat570-final-evidence-v1.json SHA814a9b082431c26047d1b881273960c2191c8c2d7dbfba97fb889d6529488c50. Independent review pending; Stage2Complete/Stage3InProgress/Stage4NotStarted. No actual UAT570 native graph or causal/final gate acceptance, whole-local Prompt/PG/app import, source repair, dependency/CI/budget/GC/cleanup/warning/exit changes. Scope/labels remain partial; capture allocations/census can perturb automatic GC, shallow memory is not peak RSS.

## Final immutable review — 2026-10-01

UAT570 final independent immutable review CLEAR: /private/tmp/pr2979-uat570-independent-review.json SHA915e85ca96ee993f38a3f2088719c400569162bbe92cf15699f5794c613fec1f. Reviewer independently verifies exact patch/158 artifact hashes/source54 AST/workflow equivalence and graph privacy/ranges/real synthetic module-to-cycle/FinalizationHold ownership. Additional narrow late node/footer/close/event-MemoryError and custom-metaclass descriptor checks preserve original7/instance release/GC configuration; no own broad pytest/Prompt/PG/app import. This supersedes prior pending review only. LocalAC1/2 and DoD2-6 checked, nativeAC3/DoD1 OPEN. Stage1-3Complete/Stage4NotStarted; task remains InProgress until batch finalization. No actual native UAT570 result, retained actual hosted owner attribution, corrective source edit or final uninstrumented acceptance.

UAT570 final frozen review binding supersedes the provisional pre-freeze872756... receipt: final reviewer report mode0444 SHA915e85ca96ee993f38a3f2088719c400569162bbe92cf15699f5794c613fec1f, clear for separate diagnostic publication/no actionable P1/P2. The report was finalized after the first private receipt read; exact source/patch/verdict remain unchanged. Final supplement /private/tmp/pr2979-uat570-review-completed-supplement-v2.json binds the immutable final report, replacing the prior receipt only. No source/CI/test failure or additional source change; nativeAC3/DoD1/final uninstrumented gate remain open.
