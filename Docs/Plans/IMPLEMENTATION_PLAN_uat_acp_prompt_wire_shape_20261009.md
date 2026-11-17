# C05 ACP authored prompt to content-block wire contract

Associated task: TASK-13260.278.18.83.45. Root recorded the actual connected OpenCode session's one normal UI README Send returning Invalid params/Error with zero messages/tokens/effects before edits. Root approved the shared transport design, including early endpoint validation after owner admission and before bootstrap/governance. Root owns runtime, private profile, browser, provider and actual live acceptance.

## Design and ADR assessment

ADR required: no new decision. [ADR-016](../ADR/016-acp-session-and-orchestration-persistence.md) governs the unchanged owned ACP storage and bootstrap boundary. This is an adapter correction between the existing authored role/content application contract and the existing ACP v1 content-block contract; it changes no public app input, storage, owner, permission, cancellation, worker, provider or bootstrap settlement rule.

The actual UI hook and REST client send role/content, as do Workflow and Scheduler callers. The direct runner initializes protocol version 1 and raw-forwards prompt through the Go runner. The separate sandbox prompt transport must share conversion. [Official ACP v1 prompt turn](https://agentclientprotocol.com/protocol/v1/prompt-turn) requires ContentBlock[], and [content](https://agentclientprotocol.com/protocol/v1/content) requires text blocks with type/text. ACP has no prompt system/assistant role field.

Preserve authored messages for governance, persistence, mentions and fork-bootstrap construction. Validate after actual endpoint owner admission before bootstrap/governance; convert a detached wire copy before direct/sandbox prompt admission/governance/transport. Preserve native block structure/order, exact authored text and valid metadata without coercion. Retain legacy extra fields under a namespaced ACP _meta entry only where required to preserve caller data. Reject malformed/ambiguous input before prompt effects. Existing native-history resumability remains honestly unchanged. Permission/cancellation functions and metadata agent's disjoint GET/accessor changes remain untouched.

No reusable ACP schema is present locally. The existing MCP gateway projection changes metadata and optional-field contracts, so it cannot preserve ACP native blocks exactly. Use a focused required-shape guard rather than duplicating a broad schema validator or adding a dependency. Pre-unit source bytes are retained only in memory; no proof or baseline files.

## Stage 1: Establish causal wire failure
**Goal**: Prove the actual role/content failure through normal initialized stdio lifecycle.
**Success Criteria**: A strict controlled native ACP peer rejects unchanged role/content after the real Go runner forwards it; native content-block controls are accepted. Real authenticated REST/WS paths retain actual owner/storage admission and exact no-effect assertions.
**Tests**: New scoped wire module reuses the existing normal compiled-runner and official cookie fixture; direct/sandbox boundary and bootstrap/governance contracts use real production logic with external transport peers only. No live runner/model request or fabricated README answer.
**Status**: Complete

## Stage 2: Correct the shared boundary
**Goal**: Minimal existing-helper and transport correction.
**Success Criteria**: Legacy system/user/assistant strings become ordered native text blocks; valid native blocks remain exact copies. Malformed/ambiguous data cannot dispatch, consume bootstrap, append messages or tokens. Original authored governance/storage inputs and denial, ownership, admission/cancel/permission behavior remain.
**Tests**: New direct/REST/WS/sandbox/native/order/metadata/malformed/bootstrapping/governance cases, plus genuinely affected existing prompt and wire controls selected from current modules.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Scoped current-source checks and independent handoff.
**Success Criteria**: Natural configured Python3.12 red/green results and accurate scope/limitations; compile/Ruff and baseline/current Bandit with no new findings; exact hashes and unit deltas reported directly. Root independent review precedes runtime reload/live acceptance.
**Tests**: Activated supported fixall environment, normal configured pytest plugins, outer600/case300; primary3.11 environment is static Bandit only. No global frontend checks/build, backend runtime/profile edits, dependency installation, task edits, Git, model actions or proof files.
**Status**: In Progress

## Attempt ledger

Attempt 1, unchanged production: activated Python3.12 normal pytest configuration, outer600/case300, natural exit1 in105.885s. New module22failed/7passed, no setup/teardown errors. Six native text/image/audio/link/embedded-resource cases and the governance-denial control passed through real initialized Go/stdin transport. Authored direct/REST/WS/mixed/sandbox/bootstrap cases failed; malformed and early-effect controls exposed missing validation. Captured-output tail omitted detailed early failure excerpts, so retain the exact aggregate/nodeids without claiming a complete retained traceback or replaying established reds. Baseline static Bandit on all four production paths: zero findings/parse errors. Green qualifies only the22failed cases, retaining seven passing controls.

Attempt 2, minimal production correction: natural exit1 in81.306s;19 passed/3 failed/7 filtered, no setup/teardown errors. All direct, malformed, mixed, WebSocket and sandbox cases passed. Compact failure details establish that the three REST cases were refused by actual CSRF403 before endpoint execution. Withdraw causal attribution for those three attempt1 failures: they are fixture prerequisites, not source-red evidence. The existing official cookie fixture requires its actual csrf_token cookie in X-CSRF-Token for mutations. Add only that header to these three cases (and their fork calls), preserving every original expectation and all real auth/CSRF middleware. Sort new-test imports only. Attempt3 selects these three and seven genuinely affected native/denial current-source controls; the19 green bodies are retained without replay.

Attempt 3: natural exit0 in43.732s;10 passed/19 filtered, no errors. Corrected real-cookie REST/storage/fork-bootstrap and malformed-before-preparation cases now pass, as do the six native block shapes and denial control on current production. Retained19 from attempt2 plus10 from attempt3 qualify29 distinct new cases; there is no combined29-green execution. This exhausts the three-attempt ledger; no further new-module replay. Final test-only import grouping follows installed Ruff, without changing fixtures or bodies.

Genuinely affected existing scope:13 mention preprocessing controls and9 compiled-runner permission/admission/Stop/deadline/ownership cases. Normal plugins and an owned supported JOBS_DB_PATH remain through official fixture teardown. No old governance import shims, Scheduler/Workflow/provider replay or live runner/model request.

Affected existing qualification: natural exit0 in40.583s;22 passed/no skips (13 mention controls and9 compiled-runner permission/admission/Stop/deadline/ownership cases). Together with the split new green results,51 distinct affected/new cases are qualified; no combined51 execution or live/provider acceptance claim.

Static qualification: compile all four production paths and the new module without bytecode writes. Baseline Bandit production0 findings/parse errors; current same0. The new module's47 B101 findings are pytest regression assertions, not production authorization. Root requested narrow explained per-assert B101 annotations matching repository practice; all47 are individually annotated, with no blanket/config/production suppression. Static AST equality proves every assertion/body unchanged (AST SHA2569e53e27650573fae35928172fbd432ebe441b6b51033e550ec40b4d190a4a3b4); no causal replay for comments. Final test Bandit0 findings/parse errors with47 specifically skipped assertion checks. Ruff baseline has helper I001, runner B009 and sandbox I001/UP022; current helper/new test are clean, runner/sandbox findings remain inherited. New module imports only are grouped by installed Ruff; all fixture references and test bodies unchanged.

Qualification limits: the downstream peer implements controlled native ACP request acceptance/echo, without a model, README reading or tool effect. The sandbox manager prompt method uses an actual initialized stdio client under an injected owned handle; it does not provision a sandbox runtime. Governance denial uses the external decision seam while retaining real prompt admission/cleanup. Native prompt-history resumability/storage support is unchanged. No private runtime/provider action, actual OpenCode result, API reload or whole-UAT acceptance is established by these tests. Root independent source review and actual live acceptance remain Stage3 gates.

Maximum three attempts per distinct issue; classify fixture errors separately, retain passing controls and stop/reassess at the cap. No unchanged older suite replay.
